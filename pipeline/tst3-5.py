from pyrosm import OSM
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.features import rasterize
from affine import Affine
import os

# ===============================
# CONFIG
# ===============================
SOUTH = 31.515
NORTH = 31.535
WEST  = 34.445
EAST  = 34.470

ROWS, COLS = 3, 3
TILE_SIZE = 256
FULL_SIZE = TILE_SIZE * ROWS

BASE_DIR = "data/dem_tiles"
PBF_PATH = "data/osm/israel-and-palestine-260129.osm.pbf"
os.makedirs(BASE_DIR, exist_ok=True)

# ===============================
# LOAD OSM WITH AOI
# ===============================
aoi_bbox = [WEST, SOUTH, EAST, NORTH]  # [minx, miny, maxx, maxy]
osm = OSM(PBF_PATH, bounding_box=aoi_bbox)

# ===============================
# EXTRACT WATER (CORRECT WAY)
# ===============================
natural = osm.get_natural()


frames = []

if natural is not None:
    frames.append(natural[natural["natural"] == "water"])


if len(frames) > 0:
    water = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        crs="EPSG:4326"
    )
else:
    water = None

print("Water features in AOI:", 0 if water is None else len(water))

# ===============================
# SAVE GEOJSON (OPTIONAL)
# ===============================
if water is not None and len(water) > 0:
    os.makedirs("data/osm", exist_ok=True)
    water.to_file("data/osm/water.geojson", driver="GeoJSON")

# ===============================
# RASTERIZE
# ===============================
if water is not None and len(water) > 0:
    transform = Affine(
        (EAST - WEST) / FULL_SIZE, 0, WEST,
        0, -(NORTH - SOUTH) / FULL_SIZE, NORTH
    )

    shapes = [(geom, 1.0) for geom in water.geometry if geom.is_valid]

    water_full = rasterize(
        shapes=shapes,
        out_shape=(FULL_SIZE, FULL_SIZE),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.float32
    )

    # ===============================
    # SPLIT INTO TILES
    # ===============================
    for r in range(ROWS):
        for c in range(COLS):
            tile = water_full[
                r * TILE_SIZE:(r + 1) * TILE_SIZE,
                c * TILE_SIZE:(c + 1) * TILE_SIZE
            ]

            tile_dir = f"{BASE_DIR}/tile_{r}_{c}"
            os.makedirs(tile_dir, exist_ok=True)
            np.save(f"{tile_dir}/water.npy", tile)

print("✅ Water rasterized and tiled successfully")

