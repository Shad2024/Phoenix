import os
import requests
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

# ===============================
# 1️⃣ CONFIG
# ===============================

API_KEY = "f4a8e8b5fb83b02585d60c0688e98e83"

# Al-Rimal bounding box (EPSG:4326)
SOUTH = 31.515
NORTH = 31.535
WEST  = 34.445
EAST  = 34.470

ROWS, COLS = 3, 3
TILE_SIZE = 256
FULL_SIZE = TILE_SIZE * ROWS  # 768

BASE_DIR = "data/dem_tiles"
DEM_PATH = "data/dem/alrimal_dem.tif"

os.makedirs("data/dem", exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)

# ===============================
# 2️⃣ DOWNLOAD DEM
# ===============================

if not os.path.exists(DEM_PATH):
    print("⬇ Downloading DEM...")

    url = (
        "https://portal.opentopography.org/API/globaldem?"
        f"demtype=COP30"
        f"&south={SOUTH}&north={NORTH}"
        f"&west={WEST}&east={EAST}"
        f"&outputFormat=GTiff"
        f"&API_Key={API_KEY}"
    )

    r = requests.get(url, stream=True)
    r.raise_for_status()

    with open(DEM_PATH, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

    print("✅ DEM downloaded")

else:
    print("✔ DEM already exists")

# ===============================
# 3️⃣ LOAD + RESAMPLE TO 768×768
# ===============================

with rasterio.open(DEM_PATH) as src:
    dem = src.read(
        1,
        out_shape=(FULL_SIZE, FULL_SIZE),
        resampling=Resampling.bilinear
    )

    dem = dem.astype(np.float32)

# Normalize (important for ML)
dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-6)

# ===============================
# 4️⃣ SPLIT INTO 3×3 TILES
# ===============================

print("✂ Splitting DEM into tiles...")

for r in range(ROWS):
    for c in range(COLS):
        tile = dem[
            r * TILE_SIZE:(r + 1) * TILE_SIZE,
            c * TILE_SIZE:(c + 1) * TILE_SIZE
        ]

        tile_dir = f"{BASE_DIR}/tile_{r}_{c}"
        os.makedirs(tile_dir, exist_ok=True)

        np.save(f"{tile_dir}/dem.npy", tile)

print("✅ DEM tiling completed")
