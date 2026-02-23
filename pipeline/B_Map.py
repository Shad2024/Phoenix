import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
import numpy as np
from affine import Affine
import torch
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import os
import cv2
import json
from owslib.wcs import WebCoverageService
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
import requests
from GeoAI_Constraint_Generator import get_slope_channel, download_city_osm, rasterize_gdf 
from model_unet import UNet 

RASTER_SIZE = 256
OUTPUT_DIR = "geoai_generated_maps12"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_trained_model(path="geoai_model.pth+"):
    """Load PyTorch model"""
    model = UNet(in_ch=5, out_ch=4)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Updated lookup table with working URLs
DAMAGE_MASK_LUT = {
    "Beirut": { 
        "city": "Beirut",
        "event": "Port explosion 2020",
        "year": 2020,
        # Using a more reliable source
        "url": "https://github.com/UNHCR-GeoAI/damage-assessment/raw/main/data/beirut_damage_sample.geojson",
        "format": "geojson",
        "bounds": {"W": 35.505, "S": 33.895, "E": 35.515, "N": 33.905}
    },
    "Gaza": {
        "city": "Gaza",
        "event": "Conflict damage assessment",
        "year": 2023,
        "url": "https://raw.githubusercontent.com/UNHCR-GeoAI/damage-assessment/main/data/gaza_damage_sample.geojson",
        "format": "geojson",
        "bounds": {"W": 34.445, "S": 31.505, "E": 34.455, "N": 31.515}
    },
    "Kyiv": {
        "city": "Kyiv",
        "event": "Conflict damage assessment 2022",
        "year": 2022,
        "url": "https://raw.githubusercontent.com/UNHCR-GeoAI/damage-assessment/main/data/kyiv_damage_sample.geojson",
        "format": "geojson",
        "bounds": {"W": 30.505, "S": 50.405, "E": 30.515, "N": 50.415}
    },
    "Palu": {
        "city": "Palu",
        "event": "Earthquake & tsunami damage",
        "year": 2018,
        "url": "https://raw.githubusercontent.com/UNHCR-GeoAI/damage-assessment/main/data/palu_damage_sample.geojson",
        "format": "geojson",
        "bounds": {"W": 119.845, "S": -0.895, "E": 119.855, "N": -0.885}
    }
}

def build_transform(north, south, east, west, width, height):
    """Build affine transform for rasterization"""
    x_res = (east - west) / float(width)
    y_res = (north - south) / float(height)
    transform = Affine(x_res, 0.0, west, 0.0, -y_res, north)
    return transform

def create_sample_damage_mask(raster_size, location="Beirut"):
    """Create a realistic sample damage mask when real data is unavailable"""
    print(f"Creating realistic damage mask for {location}")
    mask = np.zeros((raster_size, raster_size), dtype=np.float32)
    
    if location.lower() == "beirut":
        # Simulate port explosion pattern - circular blast pattern
        center_x = raster_size // 2 + 20
        center_y = raster_size // 2 - 10
        
        # Main blast zone
        y, x = np.ogrid[:raster_size, :raster_size]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Concentric damage rings
        mask[dist < 40] = 1.0  # Severe damage
        mask[(dist >= 40) & (dist < 60)] = 0.7  # Moderate damage
        mask[(dist >= 60) & (dist < 80)] = 0.4  # Light damage
        
        # Add some linear damage along roads
        mask[center_y-5:center_y+5, :] = np.maximum(mask[center_y-5:center_y+5, :], 0.6)
        mask[:, center_x-5:center_x+5] = np.maximum(mask[:, center_x-5:center_x+5], 0.6)
        
    elif location.lower() == "gaza":
        # Simulate conflict damage - multiple scattered damage points
        for _ in range(15):
            center_x = np.random.randint(20, raster_size-20)
            center_y = np.random.randint(20, raster_size-20)
            radius = np.random.randint(5, 15)
            
            y, x = np.ogrid[:raster_size, :raster_size]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            mask[dist < radius] = np.maximum(mask[dist < radius], 0.8)
            
    else:
        # Generic earthquake/conflict pattern
        # Cluster damage in certain areas
        clusters = [(raster_size//3, raster_size//3), 
                   (2*raster_size//3, 2*raster_size//3),
                   (raster_size//2, 2*raster_size//3)]
        
        for cx, cy in clusters:
            for _ in range(3):
                offset_x = np.random.randint(-30, 30)
                offset_y = np.random.randint(-30, 30)
                center_x = cx + offset_x
                center_y = cy + offset_y
                radius = np.random.randint(10, 25)
                
                y, x = np.ogrid[:raster_size, :raster_size]
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                mask[dist < radius] = np.maximum(mask[dist < radius], np.random.uniform(0.5, 1.0))
    
    return mask

def fetch_damage_mask(damage_data, raster_size, bounds):
    """Fetch real damage mask or create realistic sample"""
    if damage_data is None:
        print("No damage data provided, creating sample mask")
        return create_sample_damage_mask(raster_size, "generic")
    
    url = damage_data["url"]
    fmt = damage_data["format"]
    location = damage_data.get("city", "unknown")

    try:
        if fmt == "geojson":
            print(f"Attempting to download damage data from: {url}")
            
            # Try to download with timeout
            try:
                gdf = gpd.read_file(url, timeout=10)
                print(f"Successfully loaded damage data: {len(gdf)} features")
                
            except Exception as download_error:
                print(f"Download failed: {download_error}")
                print("Creating realistic sample damage mask instead")
                return create_sample_damage_mask(raster_size, location)
            
            # Use provided bounds
            if bounds is None:
                print("No bounds provided, using data bounds")
                data_bounds = gdf.total_bounds
                west, south, east, north = data_bounds
                bounds_dict = {'W': west, 'S': south, 'E': east, 'N': north}
            else:
                bounds_dict = bounds
                
            west, south, east, north = bounds_dict['W'], bounds_dict['S'], bounds_dict['E'], bounds_dict['N']
            transform = build_transform(north, south, east, west, raster_size, raster_size)
            
            # Filter geometries
            shapes = [(geom, 1.0) for geom in gdf.geometry 
                     if geom is not None and not geom.is_empty]
            
            if not shapes:
                print("Warning: No valid geometries found, creating sample mask")
                return create_sample_damage_mask(raster_size, location)
                
            arr = rasterize(
                shapes=shapes,
                out_shape=(raster_size, raster_size),
                transform=transform,
                fill=0,
                all_touched=True,
                dtype=np.float32
            )
            
            print(f"Damage mask created: shape={arr.shape}, min={arr.min()}, max={arr.max()}, sum={arr.sum()}")
            
        elif fmt == "geotiff":
            print(f"Attempting to download raster damage data from: {url}")
            try:
                with rasterio.open(url) as src:
                    # Resample to target size
                    data = src.read(
                        1, 
                        out_shape=(raster_size, raster_size),
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                    arr = data.astype(np.float32)
                    
                    # Normalize if needed
                    if arr.max() > 1.0:
                        arr = arr / 255.0
                        
                    print(f"Raster damage loaded: shape={arr.shape}")
                    
            except Exception as raster_error:
                print(f"Raster download failed: {raster_error}")
                return create_sample_damage_mask(raster_size, location)
                    
        else:
            print(f"Unknown format: {fmt}, creating sample mask")
            return create_sample_damage_mask(raster_size, location)
            
        return arr
        
    except Exception as e:
        print(f"Error in damage mask processing: {e}")
        print("Creating realistic sample damage mask")
        return create_sample_damage_mask(raster_size, location)
    

def fetch_satellite_damage(bounds):
    """
    Fetch satellite-derived damage polygons (UNOSAT-style)
    Returns: GeoDataFrame (EPSG:4326)
    """
    try:
        # Example: UNOSAT sample (you can swap source later)
        url = "https://raw.githubusercontent.com/UNHCR-GeoAI/damage-assessment/main/data/gaza_damage_sample.geojson"
        gdf = gpd.read_file(url)

        # Clip to bounds
        bbox = gpd.GeoSeries.from_bbox((
            bounds["W"], bounds["S"], bounds["E"], bounds["N"]
        ), crs="EPSG:4326")

        gdf = gdf[gdf.intersects(bbox.unary_union)]

        return gdf[["geometry"]]

    except Exception as e:
        print(f"Satellite damage fetch failed: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def fetch_satellite_water(bounds):
    """
    Fetch satellite-derived water bodies (JRC Global Surface Water)
    Returns: GeoDataFrame
    """
    try:
        # Sample water polygons
        url = "https://raw.githubusercontent.com/gee-community/geemap/master/examples/data/world_water.geojson"
        gdf = gpd.read_file(url)

        bbox = gpd.GeoSeries.from_bbox((
            bounds["W"], bounds["S"], bounds["E"], bounds["N"]
        ), crs="EPSG:4326")

        gdf = gdf[gdf.intersects(bbox.unary_union)]

        return gdf[["geometry"]]

    except Exception as e:
        print(f"Satellite water fetch failed: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    

def fetch_ghsl_builtup(bounds, raster_size=256):
    """
    Fetch GHSL built-up raster for a small neighborhood on-demand
    bounds: dict with keys 'W', 'S', 'E', 'N'
    raster_size: output raster size (square)
    Returns: numpy array (H, W) in [0,1]
    """
    # Connect to GHSL WCS
    wcs_url = "https://ghsl.jrc.ec.europa.eu/ghs_bu_s_2018/wcs"  # official endpoint
    wcs = WebCoverageService(wcs_url, version='2.0.1')

    coverage_id = "GHS_BUILT_S_E2018_GLOBE_R2023A"

    # Prepare subset
    subset = [
        ('Long', bounds['W'], bounds['E']),
        ('Lat', bounds['S'], bounds['N'])
    ]

    # Get coverage as GeoTIFF
    response = wcs.getCoverage(
        identifier=coverage_id,
        format='image/tiff',
        subsets=subset
    )

    # Read raster in memory
    with MemoryFile(response.read()) as memfile:
        with memfile.open() as src:
            # Resample to target size
            data = src.read(
                1,
                out_shape=(raster_size, raster_size),
                resampling=rasterio.enums.Resampling.bilinear
            ).astype(np.float32)

    # Normalize GHSL values [0,100] -> [0,1]
    data = np.clip(data / 100.0, 0, 1)
    return data

        

def fuse_osm_with_satellite(
    roads_gdf,
    buildings_gdf,
    constraints_gdf,
    sat_damage_gdf,
    sat_water_gdf
):
    """
    Correct OSM layers using satellite-derived truth
    """

    # ---- DAMAGE ----
    if not sat_damage_gdf.empty:
        buildings_gdf["damaged"] = buildings_gdf.geometry.intersects(
            sat_damage_gdf.unary_union
        )

    # ---- WATER / CONSTRAINTS ----
    if not sat_water_gdf.empty:
        constraints_gdf = gpd.GeoDataFrame(
            geometry=pd.concat(
                [constraints_gdf.geometry, sat_water_gdf.geometry]
            ),
            crs=constraints_gdf.crs
        )

    return roads_gdf, buildings_gdf, constraints_gdf

def fuse_osm_with_builtup(buildings_gdf, sat_builtup, bounds):
    """
    Correct building layer using GHSL built-up raster
    """
    
    # Convert raster to polygons
    transform = from_bounds(bounds['W'], bounds['S'], bounds['E'], bounds['N'], sat_builtup.shape[1], sat_builtup.shape[0])
    
    # Rasterize buildings from GHSL
    builtup_polygons = rasterize(
        [(geom, 1) for geom in buildings_gdf.geometry if geom is not None and not geom.is_empty],
        out_shape=sat_builtup.shape,
        transform=transform,
        fill=0,
        all_touched=True
    )

    # Fuse: take max of OSM buildings and GHSL built-up
    fused_buildings = np.maximum(builtup_polygons, (sat_builtup > 0.1).astype(np.float32))
    
    return fused_buildings



def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("         GEOAI REBUILD PROJECT - MAP GENERATOR")
    print("="*60)
    
    # Get user input
    available_locations = list(DAMAGE_MASK_LUT.keys())
    print(f"Available locations: {', '.join(available_locations)}")
    location = input(f"\nType a city/location: ").strip()
    
    # Normalize input (capitalize first letter)
    if location.lower() in [loc.lower() for loc in available_locations]:
        for loc in available_locations:
            if loc.lower() == location.lower():
                location = loc
                break
    
    # Initialize variables
    damage_data = None
    bounds = None
    
    # Check if location exists in lookup table
    if location in DAMAGE_MASK_LUT:
        damage_data = DAMAGE_MASK_LUT[location]
        bounds = damage_data.get("bounds")
        print(f"\n✓ Location found: {damage_data['city']}")
        print(f"  Event: {damage_data['event']} ({damage_data['year']})")
        print(f"  Bounds: {bounds}")
    else:
        print(f"\n⚠ {location} not found in database")
        print("Using Beirut as default location")
        location = "Beirut"
        damage_data = DAMAGE_MASK_LUT[location]
        bounds = damage_data.get("bounds")
    
    print("\n" + "="*60)
    print("DOWNLOADING AND PROCESSING DATA")
    print("="*60)
    
    try:
        # Channel 1: Slope
        print("\n[1/5] Getting terrain slope data...")
        ch1_slope = get_slope_channel(RASTER_SIZE, bounds)
        print(f"   ✓ Slope data: {ch1_slope.shape}, range: [{ch1_slope.min():.3f}, {ch1_slope.max():.3f}]")
        
        # Channel 2: Damage mask
        print("\n[2/5] Getting damage assessment data...")
        #ch2_damage = fetch_damage_mask(damage_data, RASTER_SIZE, bounds)
        sat_damage_gdf = fetch_satellite_damage(bounds)
        ch2_damage = rasterize_gdf(
                sat_damage_gdf,
                RASTER_SIZE,
                bounds,
                burn_value=1.0
                )

        print(f"   ✓ Damage mask: {ch2_damage.shape}, range: [{ch2_damage.min():.3f}, {ch2_damage.max():.3f}]")
        
        # Channels 3-5: OSM data
        print("\n[3-5/5] Getting urban infrastructure data...")
        city_name = location.split("_")[0] if "_" in location else location
        
        try:
            
            #1 Fetch OSM
            roads_gdf, buildings_gdf, constraints_gdf = download_city_osm(location)

            # 2️ Fetch satellite-derived layers
            sat_water_gdf = fetch_satellite_water(bounds)
            sat_builtup = fetch_ghsl_builtup(bounds, RASTER_SIZE)

            # 3️ Fuse satellite + OSM
            roads_gdf, buildings_gdf, constraints_gdf = fuse_osm_with_satellite(
            roads_gdf,
            buildings_gdf,
            constraints_gdf,
            sat_damage_gdf,
            sat_water_gdf
            )

            # 4️ Rasterize (UNCHANGED)
            ch3_roads = rasterize_gdf(roads_gdf, RASTER_SIZE, bounds, all_touched=True)
            #ch4_buildings = rasterize_gdf(buildings_gdf, RASTER_SIZE, bounds)
            ch4_buildings = fuse_osm_with_builtup(buildings_gdf, sat_builtup, bounds)
            ch4_buildings = np.maximum(ch4_buildings, rasterize_gdf(buildings_gdf, RASTER_SIZE, bounds))
            ch5_constraints = rasterize_gdf(constraints_gdf, RASTER_SIZE, bounds)

            
            print(f"   ✓ Roads: {ch3_roads.shape}")
            print(f"   ✓ Buildings: {ch4_buildings.shape}")
            print(f"   ✓ Constraints: {ch5_constraints.shape}")
            
        except Exception as osm_error:
            print(f"   ⚠ OSM data error: {osm_error}")
            print("   Creating synthetic urban data...")
            
            # Create synthetic urban features
            ch3_roads = create_road_network(RASTER_SIZE)
            ch4_buildings = create_building_layout(RASTER_SIZE)
            ch5_constraints = create_constraints(RASTER_SIZE)
            
    except Exception as e:
        print(f"\n❌ Error in data processing: {e}")
        print("Using synthetic data for all channels...")
        
        # Create all synthetic data
        ch1_slope = create_synthetic_slope(RASTER_SIZE)
        ch2_damage = create_sample_damage_mask(RASTER_SIZE, location)
        ch3_roads = create_road_network(RASTER_SIZE)
        ch4_buildings = create_building_layout(RASTER_SIZE)
        ch5_constraints = create_constraints(RASTER_SIZE)
    
    # Stack all 5 channels
    input_tensor = np.stack([ch1_slope, ch2_damage, ch3_roads, 
                            ch4_buildings, ch5_constraints], axis=-1)
    
    print(f"\n✓ Input tensor created: {input_tensor.shape}")
    print(f"  Data type: {input_tensor.dtype}")
    print(f"  Value range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    
    # Save as .npz file
    npz_path = os.path.join(OUTPUT_DIR, f"{location}_input.npz")
    np.savez_compressed(npz_path, 
                       X=input_tensor, 
                       bounds=bounds, 
                       location=location,
                       channels=['slope', 'damage', 'roads', 'buildings', 'constraints'])
    print(f"\n✅ Input tensor saved: {npz_path}")
    
    # Model inference
    print("\n" + "="*60)
    print("MODEL INFERENCE")
    print("="*60)
    
    # Check for models
    pytorch_model_path = "geoai_model.pth"
    tensorflow_model_path = "geoai_model.h5"
    keras_model_path = "geoai_model.keras"
    
    output_tensor = None
    
    # Try PyTorch first
    if os.path.exists(pytorch_model_path):
        print("Loading PyTorch model...")
        try:
            model = load_trained_model(pytorch_model_path)
            
            # Convert to PyTorch tensor
            input_tensor_pt = torch.from_numpy(input_tensor).float()
            input_tensor_pt = input_tensor_pt.permute(2, 0, 1).unsqueeze(0)  # (1, 5, 256, 256)
            
            with torch.no_grad():
                output_tensor = model(input_tensor_pt)
                #output_tensor = output_tensor.squeeze(0).permute(1, 2, 0).numpy()  # (256, 256, 4)
                # Convert output to torch before visualization
                output_tensor_torch = torch.tensor(output_tensor)

                # --- NEW GOOGLE MAPS STYLE VISUALIZATION ---
                maps_style_img = render_maps_style(output_tensor_torch)

                maps_style_path = os.path.join(OUTPUT_DIR, f"{location}_maps_style.png")
                cv2.imwrite(maps_style_path, maps_style_img)
                print(f"✅ Google-Maps style map saved: {maps_style_path}\n")

                # Keep numpy version for existing visualizations
                output_tensor = output_tensor
                
            print("✓ PyTorch inference complete")
            
        except Exception as e:
            print(f"PyTorch inference failed: {e}")
    
    # Try TensorFlow/Keras
    for model_path in [tensorflow_model_path, keras_model_path]:
        if os.path.exists(model_path):
            print(f"Loading TensorFlow/Keras model from {model_path}...")
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
                
                input_batch = np.expand_dims(input_tensor, axis=0)
                output_tensor = model.predict(input_batch, verbose=0)[0]
                
                print("✓ TensorFlow/Keras inference complete")
                break
                
            except Exception as e:
                print(f"TensorFlow/Keras inference failed: {e}")
    
    # Fallback to simulated output
    if output_tensor is None:
        print("⚠ No model found, generating simulated output...")
        output_tensor = simulate_model_output(input_tensor)
    
    print(f"Output tensor: {output_tensor.shape}")
    
    # Visualization
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    # Create visualizations
    create_visualizations(input_tensor, output_tensor, location, bounds)
    
    print("\n" + "="*60)
    print("PROCESS COMPLETE! ✓")
    print("="*60)
    print(f"\nGenerated files for '{location}':")
    print(f"  📊 Input data:   {location}_input.npz")
    print(f"  🎨 Overview:     {location}_overview.png")
    print(f"  🗺️  Rebuilt map:  {location}_rebuilt_map.png")
    print(f"  📈 Output data:  {location}_output.npz")
    print(f"\nAll files saved in: {os.path.abspath(OUTPUT_DIR)}")

# Helper functions for synthetic data
def create_synthetic_slope(size):
    """Create synthetic slope data"""
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    
    # Create hilly terrain
    slope = (np.sin(X * 2) * np.cos(Y * 2) + 1) / 2
    return slope.astype(np.float32)

def create_road_network(size):
    """Create synthetic road network"""
    roads = np.zeros((size, size), dtype=np.float32)
    
    # Main roads
    roads[size//2, :] = 1.0
    roads[:, size//2] = 1.0
    
    # Secondary roads
    for i in range(1, 4):
        pos = size * i // 4
        roads[pos, :] = 0.6
        roads[:, pos] = 0.6
    
    # Add some noise
    roads += np.random.randn(size, size) * 0.1
    return np.clip(roads, 0, 1).astype(np.float32)

def create_building_layout(size):
    """Create synthetic building layout"""
    buildings = np.zeros((size, size), dtype=np.float32)
    
    # Create building blocks
    block_size = size // 8
    for i in range(2, 6):
        for j in range(2, 6):
            x_start = i * block_size
            y_start = j * block_size
            x_end = x_start + block_size - 4
            y_end = y_start + block_size - 4
            
            buildings[y_start:y_end, x_start:x_end] = 1.0
    
    # Add smaller buildings
    for _ in range(20):
        x = np.random.randint(0, size - 10)
        y = np.random.randint(0, size - 10)
        w = np.random.randint(5, 15)
        h = np.random.randint(5, 15)
        buildings[y:y+h, x:x+w] = 0.7
    
    return buildings

def create_constraints(size):
    """Create synthetic constraints (water, protected areas, etc.)"""
    constraints = np.zeros((size, size), dtype=np.float32)
    
    # Simulate a river
    x = np.arange(size)
    river_center = size // 3 + size // 6 * np.sin(x / 20)
    for i in range(size):
        center = int(river_center[i])
        constraints[center-3:center+3, i] = 0.8
    
    # Protected area in corner
    constraints[size//2:size//2+40, size//2:size//2+40] = 0.9
    
    return constraints

def simulate_model_output(input_tensor):
    """Simulate model output when no model is available"""
    size = input_tensor.shape[0]
    
    # Extract features from input
    damage = input_tensor[:, :, 1]
    buildings = input_tensor[:, :, 3]
    roads = input_tensor[:, :, 2]
    
    # Create 4 output channels based on input
    output = np.zeros((size, size, 4), dtype=np.float32)
    
    # Channel 1: Reconstruction priority (based on damage)
    output[:, :, 0] = damage
    
    # Channel 2: Green spaces (inverse of built-up areas)
    output[:, :, 1] = 1.0 - np.clip(buildings * 1.5, 0, 1)
    
    # Channel 3: Transportation network (enhance existing roads)
    output[:, :, 2] = np.clip(roads * 1.2 + 0.2, 0, 1)
    
    # Channel 4: Commercial/industrial zones (away from damage)
    safe_zones = 1.0 - damage
    commercial = np.clip(buildings * safe_zones, 0, 1)
    # Add some clustering
    from scipy.ndimage import gaussian_filter
    output[:, :, 3] = gaussian_filter(commercial, sigma=3)
    
    return output

def render_maps_style(seg_tensor):
    """
    seg_tensor: torch tensor [4, H, W] output from your model
    Returns: numpy RGB image (H, W, 3)
    """
    labels = torch.argmax(seg_tensor, dim=0).cpu().numpy()

    H, W = labels.shape
    output = np.zeros((H, W, 3), dtype=np.uint8)

    # Google Maps style colors
    COLORS = {
        0: (245, 245, 240),   # background (beige)
        1: (200, 200, 200),   # buildings (light gray)
        2: (255, 255, 255),   # roads (white)
        3: (198, 239, 206),   # green areas
    }

    # Fill by class
    for cls, color in COLORS.items():
        output[labels == cls] = color

    # Thicken roads
    roads_mask = (labels == 2).astype(np.uint8) * 255
    roads_thick = cv2.dilate(roads_mask, np.ones((3,3), np.uint8), iterations=2)
    output[roads_thick == 255] = COLORS[2]

    # Add building edges
    buildings = (labels == 1).astype(np.uint8) * 255
    edges = cv2.Canny(buildings, 50, 150)
    output[edges > 0] = (120, 120, 120)

    return output

def create_visualizations(input_tensor, output_tensor, location, bounds):
    """Create all visualizations"""
    # --- Convert tensors into NumPy ---
    if hasattr(output_tensor, "detach"):
        output_tensor = output_tensor.detach().cpu().numpy()
    if hasattr(input_tensor, "detach"):
        input_tensor = input_tensor.detach().cpu().numpy()
    
    # Ensure shape (H, W, C)
    if output_tensor.shape[0] == 4:
        output_tensor = np.transpose(output_tensor, (1, 2, 0))
    if input_tensor.shape[0] == 5:
        input_tensor = np.transpose(input_tensor, (1, 2, 0))
    
    # Color mapping for output channels
    colors = np.array([
        [1.0, 0.85, 0.3],   # Road Patches        → warm yellow/orange
        [0.9, 0.25, 0.25],  # Building Clusters   → strong red
        [0.2, 0.8, 0.35],   # Public Greenery     → natural green
        [0.45, 0.7, 1.0]    # Hydroponic Farms    → futuristic aqua-blue
    ])
    
    # Normalize output channels
    # Ensure output tensor is (H, W, 4)
    if output_tensor.ndim == 4:  # BCHW
       output_tensor = output_tensor[0]
       output_tensor = np.transpose(output_tensor, (1, 2, 0))
    elif output_tensor.ndim == 3 and output_tensor.shape[0] == 4:  # CHW
       output_tensor = np.transpose(output_tensor, (1, 2, 0))

# Normalize output channels
    normalized_channels = []
    for i in range(output_tensor.shape[2]):
        channel = output_tensor[:, :, i]
        min_val, max_val = channel.min(), channel.max()
        if max_val - min_val > 1e-6:
             channel = (channel - min_val) / (max_val - min_val)
        normalized_channels.append(channel)

    height, width, _ = output_tensor.shape
    rgb_map = np.zeros((height,width , 3), dtype=np.float32)
    for i in range(4):
        rgb_map += colors[i] * normalized_channels[i][:, :, np.newaxis]
    
    rgb_map = np.clip(rgb_map, 0, 1)
    
    # Create overview figure
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(f'GeoAI Rebuild Analysis: {location}', fontsize=16, y=0.98)
    
    # Input channels
    input_titles = ['Slope', 'Damage', 'Roads', 'Buildings', 'Constraints']
    for i in range(5):
        ax = axes[0, i]
        im = ax.imshow(input_tensor[:, :, i], cmap='viridis')
        ax.set_title(f'Input: {input_titles[i]}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Output channels
    output_titles = ['Road', 'Building', 'Green Space', 'Hydroponic Farms']
    for i in range(4):
        ax = axes[1, i]
        im = ax.imshow(output_tensor[:, :, i], cmap='viridis')
        ax.set_title(f'Output: {output_titles[i]}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Combined visualizations
    axes[1, 4].axis('off')  # Empty
    
    # Final map
    ax = axes[2, 2]
    im = ax.imshow(rgb_map)
    ax.set_title('Rebuilt Map (Combined)', fontsize=12, weight='bold')
    ax.axis('off')
    
    # Individual component contributions
    for i in range(4):
        ax = axes[2, i]
        component = colors[i] * normalized_channels[i][:, :, np.newaxis]
        ax.imshow(np.clip(component, 0, 1))
        ax.set_title(f'{output_titles[i]}', fontsize=9)
        ax.axis('off')
    
    axes[2, 4].axis('off')
    
    plt.tight_layout()
    overview_path = os.path.join(OUTPUT_DIR, f"{location}_overview.png")
    plt.savefig(overview_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✅ Overview saved: {overview_path}")
    
    # Create final map only
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_map)
    plt.axis('off')
    plt.title(f"Rebuilt Vision for {location}", fontsize=16, pad=20)
    
    final_map_path = os.path.join(OUTPUT_DIR, f"{location}_rebuilt_map.png")
    plt.savefig(final_map_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Final map saved: {final_map_path}")
    
    # Save output data
    output_npz_path = os.path.join(OUTPUT_DIR, f"{location}_output.npz")
    np.savez_compressed(output_npz_path, 
                       output=output_tensor, 
                       rgb_map=rgb_map,
                       location=location,
                       bounds=bounds)
    print(f"✅ Output data saved: {output_npz_path}")

if __name__ == "__main__":
    main()