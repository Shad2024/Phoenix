import torch
import numpy as np
from trrain import UNet
import rasterio
from rasterio.transform import from_bounds
import folium
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt  # for color mapping

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEST, SOUTH, EAST, NORTH = 34.445, 31.515, 34.470, 31.535  # map bounding box

# ---------------- LOAD DATA ----------------
X_after = np.load("data/al_rimalll/X_after.npy").astype(np.float32)
y_before = np.load("data/al_rimalll/y_before.npy").astype(np.float32)

# Normalize
for c in range(X_after.shape[0]):
    ch = X_after[c]
    X_after[c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

# ---------------- LOAD MODEL ----------------
in_channels = X_after.shape[0]
num_classes = y_before.shape[0]
model = UNet(in_channels, num_classes).to(device)
model.load_state_dict(torch.load("map_reconstruction_unet.pth", map_location=device))
model.eval()

# ---------------- PREDICT ----------------
X_tensor = torch.from_numpy(X_after).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(X_tensor)
    y_pred_classes = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

print("Predicted classes:", np.unique(y_pred_classes))

# ---------------- CONVERT TO GEO-RASTER ----------------
height, width = y_pred_classes.shape
transform = from_bounds(WEST, SOUTH, EAST, NORTH, width=width, height=height)

with rasterio.open(
    "predicted_map.tif",
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype=y_pred_classes.dtype,
    crs="EPSG:4326",
    transform=transform
) as dst:
    dst.write(y_pred_classes, 1)

# ---------------- CREATE INTERACTIVE MAP ----------------
with rasterio.open("predicted_map.tif") as src:
    data = src.read(1)
    bounds = src.bounds

center_lat = (bounds.top + bounds.bottom) / 2
center_lon = (bounds.left + bounds.right) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=16)

# Color mapping
norm_data = data / data.max()
rgba = (plt.cm.tab20(norm_data)[:, :, :4] * 255).astype(np.uint8)

img = Image.fromarray(rgba)
buffer = BytesIO()
img.save(buffer, format="PNG")
encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
img_url = f"data:image/png;base64,{encoded}"

folium.raster_layers.ImageOverlay(
    image=img_url,
    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
    opacity=0.7,
    name="Predicted Classes"
).add_to(m)

# Show map in notebook
m.save("predicted_map.html")
m


