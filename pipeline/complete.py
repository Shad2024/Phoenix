import os
from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import numpy as np
import requests
import osmnx as ox
from shapely.geometry import box


class AlRimalMLPipeline:
    def __init__(self, south, north, west, east, data_dir="data/al_rimalll", dem_api_key=None):
        self.south, self.north = south, north
        self.west, self.east = west, east
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dem_api_key = dem_api_key
        self.bbox_polygon = box(self.west, self.south, self.east, self.north)
        self.resolution = 100  # pixels per side

    def fetch_osm_snapshot(self, tags, date=None):
        """
        Fetch OSM data with optional historical date.
        date = "YYYY-MM-DDTHH:MM:SSZ" format (UTC)
        """
        query_date = f'[date:"{date}"]' if date else ""
        gdfs = {}
        # Determine which osmnx API is available and call appropriately
        for name, tag_dict in tags.items():
            try:
                print(f"Fetching {name} (date={date})...")
                # Prefer modern geometries_from_polygon
                if hasattr(ox, 'geometries_from_polygon'):
                    gdf = ox.geometries_from_polygon(
                        self.bbox_polygon, tags=tag_dict, retain_all=True, overpass_settings=f"{query_date}")
                # Older osmnx versions expose features_from_polygon
                elif hasattr(ox, 'features_from_polygon'):
                    gdf = ox.features_from_polygon(self.bbox_polygon, tag_dict)
                # Fallback to bbox-based call
                else:
                    # geometries_from_bbox or features_from_bbox may exist
                    if hasattr(ox, 'geometries_from_bbox'):
                        gdf = ox.geometries_from_bbox(
                            self.north, self.south, self.east, self.west, tags=tag_dict)
                    elif hasattr(ox, 'features_from_bbox'):
                        gdf = ox.features_from_bbox(
                            self.north, self.south, self.east, self.west, tags=tag_dict)
                    else:
                        raise RuntimeError(
                            'No compatible osmnx geometry fetching function found')
                gdfs[name] = gdf
            except Exception as e:
                print(f"Failed {name}: {e}")
                gdfs[name] = gpd.GeoDataFrame()
        return gdfs

    def download_dem(self):
        """Download DEM from OpenTopography"""
        dem_path = self.data_dir / "dem.tif"
        url = "https://portal.opentopography.org/API/globaldem"
        params = {
            'demtype': 'SRTMGL1',
            'south': self.south,
            'north': self.north,
            'west': self.west,
            'east': self.east,
            'outputFormat': 'GTiff',
            'API_Key': self.dem_api_key
        }
        print("Downloading DEM...")
        r = requests.get(url, params=params, stream=True)
        if r.status_code == 200:
            with open(dem_path, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            print(f"DEM saved to {dem_path}")
        else:
            raise RuntimeError(f"DEM download failed: {r.status_code}")
        return dem_path

    def rasterize_layer(self, gdf, value_column=None):
        """Rasterize a GeoDataFrame to 2D array"""
        if gdf.empty:
            return np.zeros((self.resolution, self.resolution), dtype=np.float32)
        transform = from_bounds(self.west, self.south, self.east, self.north,
                                self.resolution, self.resolution)
        if value_column and value_column in gdf.columns:
            shapes = ((geom, val)
                      for geom, val in zip(gdf.geometry, gdf[value_column]))
        else:
            shapes = ((geom, 1) for geom in gdf.geometry)
        raster = rasterize(
            shapes,
            out_shape=(self.resolution, self.resolution),
            fill=0,
            transform=transform,
            dtype=np.float32
        )
        return raster

    def create_ml_stack(self, gdfs, dem_path):
        """Stack layers into 3D array for ML (channels, H, W)"""
        channels = []
        # Add buildings, roads, water, landuse, parks
        for layer_name in ['buildings', 'roads', 'water', 'landuse', 'parks']:
            if layer_name in gdfs:
                channels.append(self.rasterize_layer(gdfs[layer_name]))
            else:
                channels.append(
                    np.zeros((self.resolution, self.resolution), dtype=np.float32))
        # Add DEM as last channel
        with rasterio.open(dem_path) as src:
            dem = src.read(1, out_shape=(self.resolution, self.resolution))
            channels.append(dem.astype(np.float32))
        return np.stack(channels, axis=0)  # shape: (C, H, W)

    def run_pipeline(self, before_date, after_date):
        """
        Main pipeline:
        - Fetch OSM before/after
        - Download DEM
        - Rasterize to ML-ready stacks
        - Save X (after), y (before)
        """
        tags = {
            'buildings': {'building': True},
            'roads': {'highway': True},
            'water': {'waterway': True, 'natural': ['water', 'coastline']},
            'landuse': {'landuse': True},
            'parks': {'leisure': True}
        }

        # Fetch OSM snapshots
        print("\nFetching BEFORE snapshot...")
        gdfs_before = self.fetch_osm_snapshot(tags, date=before_date)

        print("\nFetching AFTER snapshot...")
        gdfs_after = self.fetch_osm_snapshot(tags, date=after_date)

        # Download DEM
        dem_path = self.download_dem()

        # Create ML stacks
        print("\nRasterizing BEFORE snapshot...")
        y = self.create_ml_stack(gdfs_before, dem_path)
        print("\nRasterizing AFTER snapshot...")
        X = self.create_ml_stack(gdfs_after, dem_path)

        # Save stacks
        X_path = self.data_dir / "X_after.npy"
        y_path = self.data_dir / "y_before.npy"
        np.save(X_path, X)
        np.save(y_path, y)
        print(f"\nSaved ML stacks:\n  • X: {X_path}\n  • y: {y_path}")
        return X, y


# ---------------- USAGE ----------------
if __name__ == "__main__":
    pipeline = AlRimalMLPipeline(
        south=31.515, north=31.535, west=34.445, east=34.470,
        dem_api_key="f4a8e8b5fb83b02585d60c0688e98e83"
    )

    before_date = "2020-01-01T00:00:00Z"
    after_date = "2024-01-01T00:00:00Z"

    X, y = pipeline.run_pipeline(before_date, after_date)
    print(f"\nShapes: X={X.shape}, y={y.shape}")
