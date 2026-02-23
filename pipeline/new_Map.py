import torch
import numpy as np
import cv2
from PIL import Image
import requests
from model_unet import UNet 
from tkinter import filedialog, Tk
import elevation
import rasterio 
import os
import tempfile

MAP_DEGREE_EXTENT = 0.005

def load_trained_model(path="geoai_model.pth"):
    model = UNet(in_ch=5, out_ch=4)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def get_location_from_user():
    location = input("Type a city/location : ")
    return location

def get_coordinates(location):
    """
    Convert location name to coordinates (lat, lon) using Nominatim.
    """
    geocode_url = f"https://nominatim.openstreetmap.org/search?format=json&q={location}"
    headers = {"User-Agent": "phoenix/1.0 (ShahadAlbolaki@gmail.com)"}
    
    try:
        response = requests.get(geocode_url, headers=headers)
        response.raise_for_status() # Raises an HTTPError for bad responses (4xx or 5xx)
        info = response.json()
        
    except requests.exceptions.RequestException as e:
        # Handle connection or HTTP errors
        raise ValueError(f"Geocoding API Request Error: {e}") from e
    
    if len(info) == 0:
        # No location found for the input query
        raise ValueError(f"Location '{location}' not found by the geocoding service.")

    try:
        # Access the first, most relevant result
        first_result = info[0]
        lat = float(first_result["lat"])
        lon = float(first_result["lon"])
        
        return lat, lon 
        
    except (KeyError, ValueError, IndexError) as e:
        # Catch if 'lat' or 'lon' keys are missing, or if conversion to float fails
        print(f"DEBUG: Problem parsing coordinates from API result: {e}")
        print(f"DEBUG: Malformed API response snippet: {info[0]}")
        raise ValueError(f"Geocoding API returned unparseable data for '{location}'.") from e 

def compute_slope(elev_array):
    """
    Compute slope from 2D DEM array.
    """
    gy, gx = np.gradient(elev_array)
    slope = np.sqrt(gx**2 + gy**2)
    return slope

def get_elevation_and_slope_channels(lat, lon, raster_size=256):
    """
    Downloads high-resolution DEM using 'elevation.clip()', reads it with rasterio,
    and returns the resized Elevation and computed Slope channels.
    """
    print(f"  📡 Fetching high-resolution DEM for Lat: {lat:.4f}, Lon: {lon:.4f}...")
    
    # Define the bounding box [lon_min, lat_min, lon_max, lat_max]
    lon_box_min = lon - MAP_DEGREE_EXTENT
    lon_box_max = lon + MAP_DEGREE_EXTENT
    lat_box_min = lat - MAP_DEGREE_EXTENT
    lat_box_max = lat + MAP_DEGREE_EXTENT
    bounds = (lon_box_min, lat_box_min, lon_box_max, lat_box_max)
    
    print(f"  Bounding box: {bounds}")

    # Use a temporary file to store the downloaded DEM
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
        dem_filepath = tmp.name

    try:
        # 1. Download DEM using the 'elevation' package
        print("  Downloading DEM data...")
        elevation.clip(
            bounds=bounds, 
            output=dem_filepath, 
            product='SRTM1'  # Use 30m resolution data
        )
        
        # Check if file was created
        if not os.path.exists(dem_filepath) or os.path.getsize(dem_filepath) == 0:
            raise Exception("DEM file not created or is empty")

        # 2. Read the GeoTIFF using rasterio
        print("  Reading DEM file...")
        with rasterio.open(dem_filepath) as src:
            # Read the entire raster into a NumPy array
            dem_array = src.read(1)
            print(f"  Original DEM shape: {dem_array.shape}, min: {dem_array.min():.1f}, max: {dem_array.max():.1f}")
            
        # 3. Elevation Channel: Resize to the required 256x256 input size
        if dem_array.size == 0:
            raise Exception("DEM array is empty")
            
        elevation_img = Image.fromarray(dem_array)
        elevation_data = np.array(elevation_img.resize(
            (raster_size, raster_size), Image.Resampling.BICUBIC))
        elevation_data = elevation_data.astype(np.float32)
        
        # Replace any NaN values with 0
        elevation_data = np.nan_to_num(elevation_data, nan=0.0)

        # 4. Slope Channel: Compute slope
        print("  Computing slope...")
        slope = compute_slope(elevation_data) 
        
        print(f"  ✓ Elevation range: {elevation_data.min():.1f} to {elevation_data.max():.1f}")
        print(f"  ✓ Slope range: {slope.min():.4f} to {slope.max():.4f}")
        
        return elevation_data, slope
        
    except Exception as e:
        print(f"  ❌ Error fetching DEM data: {e}")
        print("  ⚠ Falling back to synthetic elevation and slope...")
        
        # Create synthetic terrain instead of just zeros
        # Generate some hills/valleys based on lat/lon
        x = np.linspace(-2, 2, raster_size)
        y = np.linspace(-2, 2, raster_size)
        X, Y = np.meshgrid(x, y)
        
        # Deterministic synthetic elevation based on coordinates
        seed_val = int(abs(lat * 1000 + lon * 1000)) % 10000
        np.random.seed(seed_val)
        
        # Create some terrain features
        elevation_synth = (
            100 * np.exp(-0.1 * (X**2 + Y**2)) +
            50 * np.sin(0.5 * X) * np.cos(0.5 * Y) +
            30 * np.random.randn(raster_size, raster_size) * 0.1
        )
        
        # Add a base elevation
        base_elev = 50.0 + (hash(f"{lat:.2f}{lon:.2f}") % 500) / 10.0
        elevation_synth = elevation_synth + base_elev
        
        elevation_synth = elevation_synth.astype(np.float32)
        slope_synth = compute_slope(elevation_synth)
        
        print(f"  ⚠ Using synthetic elevation: {elevation_synth.min():.1f} to {elevation_synth.max():.1f}")
        
        return elevation_synth, slope_synth
        
    finally:
        # Clean up the temporary file
        if os.path.exists(dem_filepath):
            try:
                os.remove(dem_filepath)
            except:
                pass

def get_user_image():
    root = Tk()
    root.withdraw()  # Hide full window
    root.attributes("-topmost", True)

    filepath = filedialog.askopenfilename(
        title="Choose a map image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.tif *.bmp")]
    )

    if not filepath:
        raise ValueError("No image selected!")

    img = Image.open(filepath).convert("RGB")
    print("User selected:", filepath)

    return  img

def preprocess_input_image(img, lat, lon):
    """
    Returns a 5×256×256 tensor
    """
    target_size = 256
    
    # Load RGB image
    img = np.array(img)
    img = cv2.resize(img, (target_size, target_size))
    

    elevation_data, slope = get_elevation_and_slope_channels(lat, lon, raster_size=target_size)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    road = cv2.Canny(gray, 100, 200) / 255.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    water = cv2.inRange(hsv, lower_blue, upper_blue) / 255.0

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    green = cv2.inRange(hsv, lower_green, upper_green) / 255.0

    # Stack channels
    x = np.stack([elevation_data, slope, road, water, green], axis=0)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    return x_tensor

def generate_city(img, lat, lon):
    x = preprocess_input_image(img, lat, lon)
    with torch.no_grad():
        y_pred = model(x)  # shape (1,4,256,256)
    return y_pred.squeeze(0).cpu().numpy()


def render_output_city_map(y, threshold=0.5):

    # Convert prediction scores (0 to 1 range, typically) to a binary mask (0 or 1)
    y_binary = (y >= threshold).astype(np.float32)
    
    c0_res = y_binary[0]  # Residential
    c1_com = y_binary[1]  # Commercial
    c2_ind = y_binary[2]  # Industrial
    c3_roads = y_binary[3] # Roads
    
    # 2. Fixed Color Definitions (Solid Map Look)
    # Residential (c0): Light Beige/Pink
    COLOR_RES = np.array([0.90, 0.75, 0.75])
    
    # Commercial (c1): Bright Blue/Purple (Business, Offices)
    COLOR_COM = np.array([0.40, 0.40, 0.95])
    
    # Industrial (c2): Dark Gray/Brown
    COLOR_IND = np.array([0.55, 0.50, 0.45])
    
    # Roads (c3): Light Gray/White
    COLOR_ROAD = np.array([0.95, 0.95, 0.95])
    
    # 3. Layered Composition with Overlap Priority
    rgb = np.zeros((256, 256, 3), dtype=np.float32)
    
    # Order: Industrial > Commercial > Residential > Roads (Roads should cover all)
    
    # Industrial (Lowest Priority Background Area)
    mask_ind = np.stack([c2_ind] * 3, axis=-1)
    rgb = np.where(mask_ind, COLOR_IND, rgb)
    
    # Commercial (Overlaps Industrial)
    mask_com = np.stack([c1_com] * 3, axis=-1)
    rgb = np.where(mask_com, COLOR_COM, rgb)
    
    # Residential (Overlaps Commercial/Industrial)
    # This is the primary urban area color
    mask_res = np.stack([c0_res] * 3, axis=-1)
    rgb = np.where(mask_res, COLOR_RES, rgb)
    
    # Roads (Highest Priority - Should appear on top of everything)
    mask_road = np.stack([c3_roads] * 3, axis=-1)
    rgb = np.where(mask_road, COLOR_ROAD, rgb)
    
    # 4. Final Processing and Saving
    # Convert to 8-bit integer (0-255) for saving
    rgb = np.clip(rgb, 0, 1)
    rgb_8bit = (rgb * 255).astype(np.uint8)

    out_path = "generated_city_map5.png"
    
    # OpenCV uses BGR by default, so we convert RGB to BGR before writing
    cv2.imwrite(out_path, cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2BGR))

    print("City map generated at:", out_path)
    
    return out_path

if __name__ == "__main__":
    print("Loading model...")
    model = load_trained_model("geoai_model.pth")

    location = get_location_from_user() 
    lat, lon = get_coordinates(location) 
    print(f"Coordinates found: Lat={lat:.4f}, Lon={lon:.4f}")

    print("Select an input image...")
    user_img = get_user_image()

    print("Generating new city layout...")
    output_tensor = generate_city(user_img, lat, lon)
    out_path = render_output_city_map(output_tensor)

    print("City generated at:", out_path) 

    
