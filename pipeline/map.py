import torch
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, shapes
from shapely import geometry as geom
import folium

# ---------------- Config ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE_SIZE = 256  # Not really used here, full map inference

# ---------------- UNet Definition ----------------
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_channels, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up2 = DoubleConv(128+256, 128)
        self.dconv_up1 = DoubleConv(128+64, 64)
        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(self.maxpool(conv1))
        conv3 = self.dconv_down3(self.maxpool(conv2))
        x = self.upsample(conv3)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        x = self.conv_last(x)
        return x

# ---------------- Load input data ----------------
def load_input(damage_tif):
    damage_raster = rasterio.open(damage_tif)
    damage_data = damage_raster.read(1).astype(np.float32)
    transform = damage_raster.transform

    damage_norm = (damage_data - damage_data.min()) / (
        damage_data.max() - damage_data.min() + 1e-8
    )

    # Only ONE channel like training
    X = damage_norm[np.newaxis, :, :]   # (1, H, W)

    return X, transform


# ---------------- Mask to GeoDataFrame ----------------
def mask_to_gdf(mask, transform):
    shapes_gen = shapes(mask, transform=transform)
    polygons = []
    for shp, val in shapes_gen:
        if val > 0:  # keep only positive values
            polygons.append(geom.shape(shp))
    gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
    return gdf

# ---------------- Inference ----------------
def predict_city(model_path, X, device=DEVICE):
    model = UNet(in_channels=X.shape[0], out_channels=3)  # same channels as input
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.from_numpy(X).unsqueeze(0).to(device)  # add batch dim
        output = model(X_tensor)
        Y_pred = output[0].cpu().numpy()  # shape: (C, H, W)
    return Y_pred

# ---------------- Visualization ----------------
def visualize_prediction(Y_pred, transform, output_html="predicted_city_map1.html"):

    masks = [(Y_pred[i] > 0.5).astype(np.uint8) for i in range(Y_pred.shape[0])]
    # Convert to GeoDataFrames
    gdfs = [mask_to_gdf(m, transform) for m in masks]

    buildings_gdf = gdfs[0]  # channel 0 → Buildings
    roads_gdf = gdfs[1]      # channel 1 → Roads
    water_gdf = gdfs[2]

    # Create Folium map
    center = gdfs[0].geometry.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=16, tiles=None)

    folium.GeoJson(
    roads_gdf.to_json(),
    name="Roads",
    style_function=lambda x: {"color": "blue", "weight": 1},
    marker=folium.CircleMarker(radius=0, fill=False)
    ).add_to(m)

    # BUILDINGS (red)
    folium.GeoJson(
    buildings_gdf.to_json(),
    name="Buildings",
    style_function=lambda x: {"color": "red", "weight": 1},
    marker=folium.CircleMarker(radius=0, fill=False)
    ).add_to(m)

    # WATER / PARKS (green)
    folium.GeoJson(
    water_gdf.to_json(),
    name="Water",
    style_function=lambda x: {"color": "green", "weight": 1},
    marker=folium.CircleMarker(radius=0, fill=False)
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"✅ Map saved -> {output_html}")

# ---------------- Main ----------------
if __name__ == "__main__":
    # Paths for testing (you can replace with any new input)
    damage_tif = "data/damage/damage_mask.tif"
    model_path = "unet_city_reconstruction.pth1"

    # Load input
    X, transform = load_input(damage_tif)

    # Predict reconstructed city
    Y_pred = predict_city(model_path, X)

    # Visualize prediction
    visualize_prediction(Y_pred, transform, output_html="predicted_city_map1.html")
