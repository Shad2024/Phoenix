import ee
import os
import numpy as np
import rasterio
from affine import Affine 
from scipy.ndimage import gaussian_filter 

ee.Authenticate()
ee.Initialize(project="ee-damage-mapping")

# AOI bounding box
SOUTH = 31.515
NORTH = 31.535
WEST  = 34.445
EAST  = 34.470
aoi = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH])

ROWS, COLS = 3, 3
TILE_SIZE = 256
FULL_SIZE = TILE_SIZE * ROWS  # 768

BASE_DIR = "data/dem_tiles"
DAMAGE_DIR = "data/damage"
os.makedirs(DAMAGE_DIR, exist_ok=True)

def get_s1_mean(start, end):
    """Fetch Sentinel-1 VV backscatter and take mean over time range"""
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(aoi) \
        .filterDate(start, end) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
        .select('VV') \
        .mean()
    return s1

pre_sar = get_s1_mean('2023-01-01', '2023-03-01')   # adjust pre-event dates
post_sar = get_s1_mean('2023-09-01', '2023-10-01')  # adjust post-event dates


# Change = absolute difference
damage_proxy = post_sar.subtract(pre_sar).abs()
# Optional: scale to 0-1
damage_proxy = damage_proxy.unitScale(0, 0.5)
# Threshold to generate binary mask
damage_mask = damage_proxy.gt(0.1)


export_path = f"{DAMAGE_DIR}/damage_mask.tif"

task = ee.batch.Export.image.toDrive(
    image=damage_mask,
    description='AlRimal_Damage',
    folder='GEE_exports',
    fileNamePrefix='damage_mask',
    scale=10,  # meters per pixel
    region=aoi.getInfo()['coordinates'],
    fileFormat='GeoTIFF'
)
task.start()
print("Export started. Check your Google Drive folder 'GEE_exports'.")

# ⚠ Note: You need to download this TIFF manually from Drive once the export finishes

# After downloading damage_mask.tif from Google Drive:
damage_tif_path = f"{DAMAGE_DIR}/damage_mask.tif"
with rasterio.open(damage_tif_path) as src:
    damage_full = src.read(1)

# Smooth to reduce noise
damage_full = gaussian_filter(damage_full, sigma=1)
# Normalize and threshold
damage_full = (damage_full - damage_full.min()) / (damage_full.max() - damage_full.min())
damage_full = (damage_full > 0.3).astype(np.float32)
print("Full damage raster stats:",
      "min =", damage_full.min(),
      "max =", damage_full.max(),
      "non-zero =", int(damage_full.sum()))

damage_full = np.resize(damage_full, (FULL_SIZE, FULL_SIZE))
transform = Affine(
    (EAST - WEST) / FULL_SIZE, 0, WEST,
    0, -(NORTH - SOUTH) / FULL_SIZE, NORTH
)

print(" Splitting damage into tiles...")
for r in range(ROWS):
    for c in range(COLS):
        tile = damage_full[
            r * TILE_SIZE:(r + 1) * TILE_SIZE,
            c * TILE_SIZE:(c + 1) * TILE_SIZE
        ]
        tile_dir = f"{BASE_DIR}/tile_{r}_{c}"
        os.makedirs(tile_dir, exist_ok=True)
        np.save(f"{tile_dir}/damage.npy", tile)
        print(f"tile_{r}_{c} → damage pixels:", int(tile.sum()))

print("✅ GEE-based hybrid SAR damage tiling completed")

