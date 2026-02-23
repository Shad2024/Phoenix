import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from skimage.draw import line, disk
from scipy.ndimage import binary_dilation, binary_fill_holes

# ---------------- CONFIG ----------------
RASTER_SIZE = 256

CLASS_BG = 0
CLASS_BUILDING = 1
CLASS_ROAD = 2
CLASS_GREEN = 3

# --------------------------------------


def generate_roads(raster_size):
    """Generate hard road network"""
    points = np.random.rand(35, 2) * raster_size
    vor = Voronoi(points)

    roads = np.zeros((raster_size, raster_size), dtype=np.uint8)

    for v1, v2 in vor.ridge_vertices:
        if v1 >= 0 and v2 >= 0:
            x1, y1 = vor.vertices[v1].astype(int)
            x2, y2 = vor.vertices[v2].astype(int)

            rr, cc = line(y1, x1, y2, x2)
            rr = np.clip(rr, 0, raster_size - 1)
            cc = np.clip(cc, 0, raster_size - 1)
            roads[rr, cc] = 1

    # Make roads thicker
    roads = binary_dilation(roads, iterations=2)

    return roads


def generate_buildings(roads):
    """Generate building blocks between roads"""
    raster_size = roads.shape[0]

    buildings = np.ones((raster_size, raster_size), dtype=np.uint8)
    buildings[roads == 1] = 0

    # Fill enclosed blocks
    buildings = binary_fill_holes(buildings)

    # Remove edges (keep city core cleaner)
    margin = 10
    buildings[:margin, :] = 0
    buildings[-margin:, :] = 0
    buildings[:, :margin] = 0
    buildings[:, -margin:] = 0

    return buildings.astype(np.uint8)


def generate_green_areas(roads, buildings):
    """Generate parks and green spaces"""
    raster_size = roads.shape[0]
    green = np.zeros_like(roads)

    num_parks = random.randint(3, 6)
    for _ in range(num_parks):
        cx = random.randint(20, raster_size - 20)
        cy = random.randint(20, raster_size - 20)
        radius = random.randint(8, 18)

        rr, cc = disk((cy, cx), radius, shape=green.shape)
        green[rr, cc] = 1

    # Remove green from roads and buildings
    green[roads == 1] = 0
    green[buildings == 1] = 0

    return green.astype(np.uint8)


def generate_city_label_map(raster_size):
    """Final HARD segmentation map"""
    roads = generate_roads(raster_size)
    buildings = generate_buildings(roads)
    green = generate_green_areas(roads, buildings)

    label_map = np.zeros((raster_size, raster_size), dtype=np.uint8)

    # Priority: roads > buildings > green
    label_map[green == 1] = CLASS_GREEN
    label_map[buildings == 1] = CLASS_BUILDING
    label_map[roads == 1] = CLASS_ROAD

    return label_map


# ---------------- MAIN ----------------
if __name__ == "__main__":

    OUTPUT_DIR = "geoai_training_data_y+"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    CITY_NAMES = [
        "City_V1",
        "City_V2",
        "City_V3",
        "City_V4",
    ]

    for city in CITY_NAMES:
        y = generate_city_label_map(RASTER_SIZE)

        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f"Y_{city}.npz"),
            label=y
        )

        print(f"✔ Saved {city}")

    # -------- VISUAL CHECK --------
    COLORS = {
        0: (245, 245, 240),  # background
        1: (200, 200, 200),  # buildings
        2: (255, 255, 255),  # roads
        3: (198, 239, 206),  # green
    }

    color_img = np.zeros((RASTER_SIZE, RASTER_SIZE, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        color_img[y == cls] = color

    plt.figure(figsize=(5, 5))
    plt.imshow(color_img)
    plt.title("Sample HARD City Layout (Training Y)")
    plt.axis("off")
    plt.show()
