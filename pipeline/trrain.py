import torch
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, shapes
from shapely import geometry as geom
import folium
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# UNet
# =========================================================
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
        return self.conv_last(x)

# =========================================================
# LOAD X (damage)
# =========================================================
def load_damage(damage_tif):
    raster = rasterio.open(damage_tif)
    damage = raster.read(1).astype(np.float32)
    transform = raster.transform

    damage = (damage - damage.min()) / (damage.max() - damage.min() + 1e-8)
    X = damage[np.newaxis,:,:]   # (1,H,W)
    return X, transform

# =========================================================
# LOAD Y (perfect city from GeoJSON)
# =========================================================
def load_target_geojson(geojson_paths, transform, shape_hw):
    H,W = shape_hw
    layers = []

    for path in geojson_paths:
        gdf = gpd.read_file(path)
        mask = rasterize(
            [(geom,1) for geom in gdf.geometry],
            out_shape=(H,W),
            transform=transform,
            fill=0
        )
        layers.append(mask.astype(np.float32))

    Y = np.stack(layers, axis=0)  # (3,H,W)
    return Y

# =========================================================
# TRAIN FULL MAP (MEMORIZATION MODE)
# =========================================================
def train_full_map(X, Y, epochs=2000):
    print("🚀 Training memorization model...")

    model = UNet(in_channels=1, out_channels=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    X_tensor = torch.from_numpy(X).unsqueeze(0).to(DEVICE)
    Y_tensor = torch.from_numpy(Y).unsqueeze(0).to(DEVICE)

    for epoch in range(epochs):
        pred = model(X_tensor)
        loss = loss_fn(pred, Y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print("Epoch", epoch, "Loss:", loss.item())

    torch.save(model.state_dict(), "unet_city_reconstruction_MEMORIZED.pth")
    print("✅ Model memorized and saved!")

    return model

# =========================================================
# PREDICT
# =========================================================
def predict(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).unsqueeze(0).to(DEVICE)
        Y_pred = model(X_tensor)[0].cpu().numpy()
    return Y_pred

# =========================================================
# MASK → GEOJSON
# =========================================================
def mask_to_gdf(mask, transform):
    polygons=[]
    for shp,val in shapes(mask, transform=transform):
        if val>0:
            polygons.append(geom.shape(shp))
    return gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")

# =========================================================
# VISUALIZE
# =========================================================
def visualize_prediction(Y_pred, transform):
    masks = [(Y_pred[i] > 0.5).astype(np.uint8) for i in range(3)]
    gdfs = [mask_to_gdf(m, transform) for m in masks]

    center = gdfs[0].geometry.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=16, tiles=None)

    folium.GeoJson(gdfs[1].to_json(), name="Roads",
        style_function=lambda x: {"color":"blue","weight":1}).add_to(m)

    folium.GeoJson(gdfs[0].to_json(), name="Buildings",
        style_function=lambda x: {"color":"red","weight":1}).add_to(m)

    folium.GeoJson(gdfs[2].to_json(), name="Water",
        style_function=lambda x: {"color":"green","weight":1}).add_to(m)

    folium.LayerControl().add_to(m)
    m.save("predicted_city.html")
    print("🌍 Map saved -> predicted_city.html")

# =========================================================
# MAIN PIPELINE
# =========================================================
if __name__ == "__main__":

    damage_tif = "data/damage/damage_mask.tif"
    geojson_paths = [
        "data/al_rimel/buildings.geojson",
        "data/al_rimel/roads.geojson",
        "data/al_rimel/natural.geojson"
    ]

    X, transform = load_damage(damage_tif)
    Y = load_target_geojson(geojson_paths, transform, X.shape[1:])

    model = train_full_map(X, Y)        # memorizes mapping
    Y_pred = predict(model, X)          # predict same map
    visualize_prediction(Y_pred, transform)
