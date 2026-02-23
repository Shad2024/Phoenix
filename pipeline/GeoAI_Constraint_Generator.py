import requests
from io import BytesIO
import numpy as np
from scipy.ndimage import zoom
import requests
import os
from pyrosm import OSM
import geopandas as gpd
import osmnx as ox
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import rasterize
from affine import Affine
import pandas as pd
import time

# --- CONFIGURATION ---
ox.settings.timeout = 300
ox.settings.log_console = True
ox.settings.use_cache = True
ox.settings.max_query_area_size = 50 * 1000 * 1000

DEM_API_KEY = "_FHWP4azAtzt1nCyjg514!JcAZDjmz@8xbGG7tyFUauvCQ8N1hYTybUd_m)bhk_y"
UNET_MODEL_PATH = "models/geoai_model_1_damage_unet.h5"

RASTER_SIZE = 256
TILE_SIZE = 32
CHANNEL_COUNT = 5

OUTPUT_DIR = "geoai_training_data_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GEOFABRIK_URLS = {
    "Luxembourg": "https://download.geofabrik.de/europe/luxembourg-latest.osm.pbf",
    "Jeddah": "https://download.geofabrik.de/asia/saudi-arabia-latest.osm.pbf",
    "Amsterdam": "https://download.geofabrik.de/europe/netherlands-latest.osm.pbf",
    "Zurich": "https://download.geofabrik.de/europe/switzerland-latest.osm.pbf",
    "Singapore": "https://download.geofabrik.de/asia/singapore-latest.osm.pbf",
    "Beirut": "https://download.geofabrik.de/asia/lebanon-latest.osm.pbf",
    "Gaza": "https://download.geofabrik.de/asia/palestine-latest.osm.pbf",
    "Kyiv": "https://download.geofabrik.de/europe/ukraine-latest.osm.pbf",
    "Palu": "https://download.geofabrik.de/asia/indonesia-latest.osm.pbf"
}

CITY_BOUNDARIES = {"Luxembourg": {
    "N": 49.65, "S": 49.58, "E": 6.20, "W": 6.10,
    "description": "Luxembourg City center: Old town, Kirchberg EU district"
},

    "Jeddah": {
    "N": 21.62, "S": 21.52, "E": 39.25, "W": 39.15,
    "description": "Jeddah central: Corniche, Al-Balad historic district"
},
    "Amsterdam": {
    "N": 52.40, "S": 52.33, "E": 4.95, "W": 4.85,
    "description": "Amsterdam central: Canal ring, historic center"
},
    "Zurich": {
    "N": 47.42, "S": 47.33, "E": 8.62, "W": 8.50,
    "description": "Zurich city: Lake Zurich, old town, financial district"
},
    "Singapore": {
    "N": 1.35, "S": 1.27, "E": 103.88, "W": 103.80,
    "description": "Singapore central: Marina Bay, Orchard Road, CBD"
},
    "Beirut": {
    "N": 33.94,
    "S": 33.84,
    "E": 35.55,
    "W": 35.46,
    "description": "Beirut extended core: Downtown, Hamra, Achrafieh, Verdun, Port"
},
"Gaza": {
    "N": 31.55, 
    "S": 31.50,  
    "E": 34.50,  
    "W": 34.44,  
    "description": "Gaza city and surrounding area: central Gaza Strip, urban neighborhoods"
}
}

CITY_SUBDIVISIONS = {}

CHANNEL_NAMES = ["1_Slope_Elevation", "2_Damage_Mask",
                 "3_Roads_Network", "4_Building_Footprints", "5_Natural_Constraints"]


def download_geofabrik_pbf(city_name):
    url = GEOFABRIK_URLS[city_name]
    os.makedirs("pbf_data", exist_ok=True)
    filepath = f"pbf_data/{city_name}.osm.pbf"

    if os.path.exists(filepath):
        print(f"✔ Using cached PBF: {filepath}")
        return filepath

    print(f"⬇ Downloading Geofabrik PBF for {city_name} ...")
    r = requests.get(url, stream=True)
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"✔ Downloaded: {filepath}")
    return filepath


def load_city_from_pbf(pbf_path, bbox):
    """
    Load roads, buildings, and natural/landuse constraints from a PBF file.
    Works for large cities by safely handling missing columns.
    """
    west, south, east, north = bbox["W"], bbox["S"], bbox["E"], bbox["N"]

    # Initialize Pyrosm with bounding box
    osm = OSM(pbf_path, bounding_box=[west, south, east, north])

    # 1) Roads: try fetching network safely
    try:
        roads = osm.get_network(network_type="all")
    except Exception as e:
        print("  ⚠️ Roads load failed:", e)
        roads = gpd.GeoDataFrame(columns=["geometry"])

    # 2) Buildings: safely
    try:
        buildings = osm.get_buildings()
    except Exception as e:
        print("  ⚠️ Buildings load failed:", e)
        buildings = gpd.GeoDataFrame(columns=["geometry"])

    # 3) Constraints: natural + landuse + leisure
    try:
        landuse = osm.get_landuse()
        # Keep only relevant features
        mask = landuse["fclass"].isin(["park", "forest", "water", "recreation_ground", "garden"]) \
            if "fclass" in landuse.columns else [False] * len(landuse)
        constraints = landuse[mask]
    except Exception as e:
        print("  ⚠️ Constraints load failed:", e)
        constraints = gpd.GeoDataFrame(columns=["geometry"])

    return roads, buildings, constraints


def download_city_osm(city_name):
    """
    Downloads PBF (if needed) and loads city data.
    For subdivisions (e.g., NYC), merges all areas into one unified dataset.
    """
    print(f"\n📦 Loading OSM data for {city_name} using Geofabrik...")
    pbf_path = download_geofabrik_pbf(city_name)

    all_roads = gpd.GeoDataFrame()
    all_buildings = gpd.GeoDataFrame()
    all_constraints = gpd.GeoDataFrame()

    # Subdivisions 
    if city_name in CITY_SUBDIVISIONS:
        for sub_name, bbox in CITY_SUBDIVISIONS[city_name].items():
            print(f" → Extracting {sub_name} ...")
            try:
                r, b, c = load_city_from_pbf(pbf_path, bbox)
                all_roads = pd.concat([all_roads, r], ignore_index=True)
                all_buildings = pd.concat(
                    [all_buildings, b], ignore_index=True)
                all_constraints = pd.concat(
                    [all_constraints, c], ignore_index=True)
            except Exception as e:
                print(f"  ❌ Error extracting {sub_name}: {e}")

        # Merge subdivisions into one city-level dataset
        all_roads = all_roads.drop_duplicates(
            subset="geometry").reset_index(drop=True)
        all_buildings = all_buildings.drop_duplicates(
            subset="geometry").reset_index(drop=True)
        all_constraints = all_constraints.drop_duplicates(
            subset="geometry").reset_index(drop=True)

    else:
        # Standard city extraction
        bbox = CITY_BOUNDARIES[city_name]
        try:
            all_roads, all_buildings, all_constraints = load_city_from_pbf(
                pbf_path, bbox)
        except Exception as e:
            print(f"  ❌ Error loading city OSM data: {e}")

    print(f"✔ Total Roads: {len(all_roads)}")
    print(f"✔ Total Buildings: {len(all_buildings)}")
    print(f"✔ Total Constraints: {len(all_constraints)}")

    return all_roads, all_buildings, all_constraints


# --------------------- Free Elevation Service (No API Key) ---------------------

def get_free_elevation_data(raster_size, bounds):
    """Use free elevation services that don't require API keys"""
    north, south, east, west = bounds['N'], bounds['S'], bounds['E'], bounds['W']

    print(f"  📡 Getting free elevation data from OpenElevation...")

    try:
        # Sample points across the area
        lats = np.linspace(south, north, 20)
        lons = np.linspace(west, east, 20)

        locations = []
        for lat in lats:
            for lon in lons:
                locations.append({"latitude": lat, "longitude": lon})

        # Use OpenElevation (free, no API key needed)
        response = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json={"locations": locations},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            elevations = [result['elevation'] for result in data['results']]
            elevation_grid = np.array(elevations).reshape(20, 20)

            # Resize to target
            from PIL import Image
            dem = np.array(Image.fromarray(elevation_grid).resize(
                (raster_size, raster_size), Image.BICUBIC))

            print(f"  ✓ Elevation data received: {dem.shape}")
            return dem.astype(np.float32)
        else:
            raise Exception(f"API returned {response.status_code}")

    except Exception as e:
        print(f"  ❌ Free elevation service failed: {e}")
        return None


# --------------------- Simplified Slope Generation ---------------------

def get_slope_channel(raster_size, bounds):
    """Get slope data - tries free services first, then simulation"""
    # Try free elevation service
    dem = get_free_elevation_data(raster_size, bounds)

    if dem is not None:
        # Compute slope from real elevation data
        return compute_slope_from_dem(dem)
    else:
        # Use simulation
        print("  ⚠️  Using simulated elevation data")
        return get_slope_simulation(raster_size, bounds)  


def compute_slope_from_dem(dem):
    """Compute slope from DEM data"""
    dy, dx = np.gradient(dem)
    slope = np.sqrt(dx ** 2 + dy ** 2)

    if slope.max() > 0:
        slope = slope / slope.max()

    return slope.astype(np.float32)


def get_slope_simulation(raster_size, bounds):
    """Realistic terrain simulation"""
    x = np.linspace(-2, 2, raster_size)
    y = np.linspace(-2, 2, raster_size)
    X, Y = np.meshgrid(x, y)

    # Create realistic terrain with multiple features
    terrain = np.zeros((raster_size, raster_size))

    # Central hill
    center_dist = np.sqrt(X ** 2 + Y ** 2)
    terrain += np.exp(-center_dist ** 2 / 2) * 0.7

    # Random hills
    for _ in range(3):
        hill_x, hill_y = np.random.uniform(-1.5, 1.5, 2)
        hill_dist = np.sqrt((X - hill_x) ** 2 + (Y - hill_y) ** 2)
        terrain += np.exp(-hill_dist ** 2 / 0.5) * np.random.uniform(0.2, 0.4)

    # Compute slope
    dy, dx = np.gradient(terrain)
    slope = np.sqrt(dx ** 2 + dy ** 2)

    if slope.max() > 0:
        slope = slope / slope.max()

    return slope.astype(np.float32)



def build_transform(north, south, east, west, width, height):
    x_res = (east - west) / float(width)
    y_res = (north - south) / float(height)
    transform = Affine(x_res, 0.0, west, 0.0, -y_res, north)
    return transform


def rasterize_gdf(
    gdf,
    raster_size,
    bounds,
    all_touched=False,
    burn_value=1.0
):
    if gdf is None or len(gdf) == 0:
        return np.zeros((raster_size, raster_size), dtype=np.float32)

    north = bounds["N"]
    south = bounds["S"]
    east  = bounds["E"]
    west  = bounds["W"]

    transform = build_transform(
        north, south, east, west,
        raster_size, raster_size
    )

    shapes = [
        (geom, float(burn_value))
        for geom in gdf.geometry
        if geom is not None and not geom.is_empty
    ]

    if not shapes:
        return np.zeros((raster_size, raster_size), dtype=np.float32)

    try:
        return rasterize(
            shapes=shapes,
            out_shape=(raster_size, raster_size),
            transform=transform,
            fill=0,
            all_touched=all_touched,
            dtype=np.float32
        )
    except Exception as e:
        print("Rasterize error:", e)
        return np.zeros((raster_size, raster_size), dtype=np.float32)



def generate_synthetic_damage(raster_size, n_clusters=3, cluster_radius=8, noise_sigma=6):
    base = np.random.rand(raster_size, raster_size)
    noise = gaussian_filter(base, sigma=noise_sigma)
    threshold = np.quantile(noise, 0.7)
    mask = (noise > threshold).astype(np.float32)

    for _ in range(n_clusters):
        cx = np.random.randint(cluster_radius, raster_size - cluster_radius)
        cy = np.random.randint(cluster_radius, raster_size - cluster_radius)
        rr = cluster_radius + np.random.randint(-3, 6)
        xg, yg = np.ogrid[:raster_size, :raster_size]
        circle = (xg - cx) ** 2 + (yg - cy) ** 2 <= rr ** 2
        mask[circle] = 1.0

    if np.random.rand() > 0.6:
        if np.random.rand() > 0.5:
            row = np.random.randint(raster_size // 4, 3 * raster_size // 4)
            mask[row - 2:row + 3, :] = 1.0
        else:
            col = np.random.randint(raster_size // 4, 3 * raster_size // 4)
            mask[:, col - 2:col + 3] = 1.0

    mask = gaussian_filter(mask, sigma=2)
    if mask.max() > 0:
        mask = mask / mask.max()

    return mask.astype(np.float32)


def split_into_tiles(tensor, tile_size=TILE_SIZE):
    H, W, C = tensor.shape
    tiles = []
    for i in range(0, H, tile_size):
        for j in range(0, W, tile_size):
            if i + tile_size <= H and j + tile_size <= W:
                tiles.append(tensor[i:i + tile_size, j:j + tile_size, :])
    return tiles


def generate_constraint_tensor(city_name, bounds, raster_size=RASTER_SIZE):
    north, south, east, west = bounds['N'], bounds['S'], bounds['E'], bounds['W']
    print(
        f"\n--- Processing City: {city_name} ({bounds.get('description', '')}) ---")

    # 1) Download OSM vectors
    print("5.1 Fetching OSM Vector Data (Channels 3,4,5) ...")
    roads_gdf, buildings_gdf, constraints_gdf = download_city_osm(city_name)

    # 2) Generate channels
    print("5.2 Generating Slope and Damage channels")
    ch1_slope = get_slope_channel(raster_size, bounds)
    ch2_damage = generate_synthetic_damage(
        raster_size, n_clusters=np.random.randint(2, 5))

    # 3) Rasterize vector layers
    print("5.3 Rasterizing vector layers to rasters")
    ch3_roads = rasterize_gdf(roads_gdf, raster_size, north,
                              south, east, west, all_touched=True, burn_value=1.0)
    ch4_buildings = rasterize_gdf(buildings_gdf, raster_size, north, south, east, west, all_touched=False,
                                  burn_value=1.0)
    ch5_constraints = rasterize_gdf(constraints_gdf, raster_size, north, south, east, west, all_touched=True,
                                    burn_value=1.0)

    # 4) Basic normalization
    def normalize_band(b):
        b = b.astype(np.float32)
        if b.max() > 0:
            return (b - b.min()) / (b.max() - b.min())
        return b

    ch1_slope = normalize_band(ch1_slope)
    ch2_damage = normalize_band(ch2_damage)
    ch3_roads = normalize_band(ch3_roads)
    ch4_buildings = normalize_band(ch4_buildings)
    ch5_constraints = normalize_band(ch5_constraints)

    # 5) Stack channels (HxWxC)
    constraint_tensor = np.stack(
        [ch1_slope, ch2_damage, ch3_roads, ch4_buildings, ch5_constraints], axis=-1)
    print(f"Final tensor shape: {constraint_tensor.shape}")

    return constraint_tensor


def main():
    all_city_tensors = {}

    for city_name, bounds in CITY_BOUNDARIES.items():
        tensor = generate_constraint_tensor(city_name, bounds, RASTER_SIZE)

        tiles = split_into_tiles(tensor, tile_size=TILE_SIZE)
        print(f" -> Generated {len(tiles)} tiles for city: {city_name}")

        city_dir = os.path.join(OUTPUT_DIR, city_name)
        os.makedirs(city_dir, exist_ok=True)

        for idx, tile in enumerate(tiles):
            outpath = os.path.join(
                city_dir, f"X_{city_name}_tile_{idx:04d}.npz")
            np.savez_compressed(outpath, X=tile.astype(np.float32))

        all_city_tensors[city_name] = tensor

    # Visualization
    first_city = next(iter(CITY_BOUNDARIES))
    preview = all_city_tensors[first_city]

    fig, axes = plt.subplots(1, CHANNEL_COUNT, figsize=(16, 4))
    fig.suptitle(f"Preview channels for {first_city}")
    for i in range(CHANNEL_COUNT):
        ax = axes[i]
        ax.imshow(preview[:, :, i], interpolation='nearest')
        ax.set_title(CHANNEL_NAMES[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    print("\nDone. Saved dataset tiles to:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
