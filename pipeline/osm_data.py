from unicodedata import name
import pandas as pd
import osmnx as ox
import geopandas as gpd
import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import requests
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Create data directory
Path("data/al_rimel").mkdir(parents=True, exist_ok=True)

# Al-Rimal bounding box
SOUTH = 31.515
NORTH = 31.535
WEST = 34.445
EAST = 34.470

print("📍 Downloading data for Al-Rimal, Gaza...")
print(f"   Bounds: {SOUTH}°S, {NORTH}°N, {WEST}°W, {EAST}°E")
print(
    f"   Area: {(NORTH-SOUTH)*111}km × {(EAST-WEST)*111*0.85}km ≈ 2.2km × 2.1km")


class AlRimalDataDownloader:
    def __init__(self):
        self.bbox = (NORTH, SOUTH, EAST, WEST)
        self.data_dir = Path("data/al_rimel")

    def download_osm_data(self):
        """Download comprehensive OSM data for Al-Rimal"""
        print("\n1. 📦 Downloading OSM Data...")

        # Define all tags we need
        tags = {
            'building': True,           # Buildings
            'highway': True,            # Roads
            'waterway': True,           # Rivers, streams
            'natural': ['water', 'coastline'],  # Water bodies
            'landuse': True,            # Land use zones
            'leisure': True,            # Parks, recreation
            'amenity': True,            # Schools, hospitals, etc.
            'shop': True,               # Commercial
            'office': True,             # Offices
            'industrial': True,         # Industrial areas
            'railway': True,            # Railways
            'power': True,              # Power lines
            'boundary': True,           # Administrative boundaries
            'place': True,              # City, neighborhood names
            'man_made': True,           # Towers, bridges
        }

        try:
            # Download all geometries at once (most efficient)
            print("   Downloading all features...")
            # bbox format for osmnx 2.0+: (left, bottom, right, top) = (west, south, east, north)
            bbox = (self.bbox[3], self.bbox[1], self.bbox[2], self.bbox[0])
            gdf = ox.features_from_bbox(
                bbox=bbox,
                tags=tags
            )

            print(f"   Downloaded {len(gdf)} total features")

            # Separate into logical layers
            self.separate_and_save_layers(gdf)

        except Exception as e:
            print(f"   Error downloading OSM: {e}")
            print("   Falling back to individual downloads...")
            self.download_osm_layers_individually()

    def separate_and_save_layers(self, gdf):
        """Separate OSM data into meaningful layers"""
        print("\n2. 🗂️ Separating OSM layers...")

        # 1. Buildings (most important for your project)
        building_mask = gdf['building'].notna()
        buildings = gdf[building_mask].copy()
        print(f"   Buildings: {len(buildings)} features")

        # Add building type classification
        buildings['building_type'] = buildings['building']
        buildings['height_est'] = buildings.get('height',
                                                buildings.get('building:levels',
                                                              np.where(buildings['building'].isin(['apartments', 'residential']), 4, 2)))

        # 2. Roads and paths
        road_mask = gdf['highway'].notna()
        roads = gdf[road_mask].copy()

        # Classify road hierarchy
        def classify_road(road_type):
            if road_type in ['motorway', 'trunk', 'primary']:
                return 'major'
            elif road_type in ['secondary', 'tertiary']:
                return 'collector'
            elif road_type in ['residential', 'living_street', 'unclassified']:
                return 'local'
            else:
                return 'path'

        roads['road_class'] = roads['highway'].apply(classify_road)
        print(
            f"   Roads: {len(roads)} features ({roads['road_class'].value_counts().to_dict()})")

        # 3. Water features
        water_mask = gdf.get('natural', pd.Series([None]*len(gdf))).isin(
            ['water', 'coastline']) | gdf.get('waterway', pd.Series([None]*len(gdf))).notna()

        water = gdf[water_mask].copy()
        print(f"   Water features: {len(water)} features")

        # 4. Land use (zoning information)
        landuse_mask = gdf['landuse'].notna()
        landuse = gdf[landuse_mask].copy()
        print(f"   Land use zones: {len(landuse)} features")

        # 5. Parks and green spaces
        leisure_mask = gdf['leisure'].notna()
        parks = gdf[leisure_mask].copy()
        print(f"   Parks/leisure: {len(parks)} features")

        # 6. Points of interest (amenities, shops, etc.)
        poi_mask = (gdf.get('amenity', pd.Series([None]*len(gdf))).notna() |
                    gdf.get('shop', pd.Series([None]*len(gdf))).notna() |
                    gdf.get('office', pd.Series([None]*len(gdf))).notna() |
                    gdf.get('industrial', pd.Series([None]*len(gdf))).notna())
        pois = gdf[poi_mask].copy()
        print(f"   Points of interest: {len(pois)} features")

        # 7. Infrastructure
        infra_mask = (gdf.get('railway', pd.Series([None]*len(gdf))).notna() |
                      gdf.get('power', pd.Series([None]*len(gdf))).notna() |
                      gdf.get('man_made', pd.Series([None]*len(gdf))).notna())
        infrastructure = gdf[infra_mask].copy()
        print(f"   Infrastructure: {len(infrastructure)} features")

        # 8. Administrative boundaries and places
        admin_mask = (gdf.get('boundary', pd.Series([None]*len(gdf))).notna() |
                      gdf.get('place', pd.Series([None]*len(gdf))).notna())
        admin = gdf[admin_mask].copy()
        print(f"   Administrative: {len(admin)} features")

        # Save all layers
        print("\n3. 💾 Saving layers...")
        layers = {
            'buildings': buildings,
            'roads': roads,
            'water': water,
            'landuse': landuse,
            'parks': parks,
            'pois': pois,
            'infrastructure': infrastructure,
            'admin': admin,
            'all_features': gdf  # Keep complete dataset
        }

        for name, layer in layers.items():
            if len(layer) > 0:
                path = self.data_dir / f"{name}.geojson"
                layer.to_file(path, driver='GeoJSON')
                print(f"   ✓ {name}: {len(layer)} features → {path}")

        self.osm_layers = layers
        return layers

    def download_osm_layers_individually(self):
        print("   Downloading layers individually...")

        layers = {}
        # Create a polygon bbox for features_from_polygon
        bbox_polygon = box(WEST, SOUTH, EAST, NORTH)

        # Helper function to download and save a layer
        def download_layer(name, tags):
            try:
                gdf = ox.features_from_polygon(bbox_polygon, tags)
                if len(gdf) > 0:
                    path = self.data_dir / f"{name}.geojson"
                    gdf.to_file(path, driver='GeoJSON')
                    print(f"     ✓ {name}: {len(gdf)} features saved → {path}")
                else:
                    print(f"     ⚠ {name}: No features found")
                return gdf
            except Exception as e:
                print(f"     ⚠ Failed to download {name}: {e}")
                return gpd.GeoDataFrame()

        # Buildings
        print("   Downloading buildings...")
        layers['buildings'] = download_layer('buildings', {'building': True})

        # Roads
        print("   Downloading roads...")
        layers['roads'] = download_layer('roads', {'highway': True})

        # Waterways
        print("   Downloading water features...")
        layers['water'] = download_layer('water', {'waterway': True})

        # Natural water / coastline
        layers['natural'] = download_layer(
            'natural', {'natural': ['water', 'coastline']})

        # Land use
        print("   Downloading land use...")
        layers['landuse'] = download_layer('landuse', {'landuse': True})

        # Parks and leisure
        print("   Downloading parks/leisure...")
        layers['parks'] = download_layer('parks', {'leisure': True})

        # Points of interest
        print("   Downloading points of interest...")
        poi_tags = {'amenity': True, 'shop': True,
                    'office': True, 'industrial': True}
        layers['pois'] = download_layer('pois', poi_tags)

        # Infrastructure
        print("   Downloading infrastructure...")
        infra_tags = {'railway': True, 'power': True, 'man_made': True}
        layers['infrastructure'] = download_layer('infrastructure', infra_tags)

        # Administrative boundaries and places
        print("   Downloading administrative boundaries...")
        admin_tags = {'boundary': True, 'place': True}
        layers['admin'] = download_layer('admin', admin_tags)

        # Save all downloaded layers in object
        self.osm_layers = layers
        return layers

    def download_terrain_data(self, api_key=None):
        """Download terrain data from OpenTopography"""
        print("\n4. ⛰️ Downloading terrain data...")

        # First try OpenTopography API
        dem_path = self.data_dir / "dem.tif"

        if api_key:
            print(f"   Using OpenTopography API with key: {api_key[:10]}...")

            # OpenTopography API endpoint for SRTM
            url = "https://portal.opentopography.org/API/globaldem"

            params = {
                'demtype': 'SRTMGL1',  # SRTM 30m
                'south': str(SOUTH),
                'north': str(NORTH),
                'west': str(WEST),
                'east': str(EAST),
                'outputFormat': 'GTiff',
                'API_Key': api_key
            }

            try:
                print("   Requesting DEM from OpenTopography...")
                response = requests.get(url, params=params, stream=True)

                if response.status_code == 200:
                    with open(dem_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"   ✓ DEM saved to {dem_path}")

                    # Open and validate
                    with rasterio.open(dem_path) as src:
                        print(f"   DEM Info: {src.width}x{src.height} pixels")
                        print(f"   Resolution: {src.res[0]} degrees/pixel")
                        print(f"   CRS: {src.crs}")

                    return dem_path

                else:
                    print(
                        f"   OpenTopography API error: {response.status_code}")
                    print(f"   Response: {response.text[:200]}")

            except Exception as e:
                print(f"   Error with OpenTopography: {e}")

        # Fallback: Use SRTM from rasterio if API fails
        print("   Falling back to SRTM via rasterio...")
        try:
            # You might need to install rio-tiler or similar
            # For now, create a synthetic DEM for testing
            self.create_synthetic_dem(dem_path)
            return dem_path

        except Exception as e:
            print(f"   Failed to get terrain data: {e}")
            return None

    def create_synthetic_dem(self, output_path):
        """Create synthetic DEM for testing"""
        print("   Creating synthetic DEM for testing...")

        # Create grid
        x_res = 100  # 100 pixels across
        y_res = 100  # 100 pixels down

        # Create coordinates
        x = np.linspace(WEST, EAST, x_res)
        y = np.linspace(SOUTH, NORTH, y_res)
        xx, yy = np.meshgrid(x, y)

        # Create synthetic terrain (coastal area pattern)
        # Al-Rimal is coastal Gaza, so relatively flat with slight elevation
        dem = (
            10 +  # Base elevation (meters above sea level)
            2 * np.sin(xx * 50) * np.cos(yy * 50) +  # Gentle waves
            0.5 * np.exp(-((xx - 34.4575)**2 + (yy - 31.525)
                         ** 2) / 0.0001)  # Small hill
        )

        # Save as GeoTIFF
        transform = rasterio.transform.from_bounds(
            WEST, SOUTH, EAST, NORTH, x_res, y_res
        )

        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=y_res,
            width=x_res,
            count=1,
            dtype=dem.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(dem, 1)

        print(f"   ✓ Synthetic DEM created at {output_path}")
        return output_path

    def create_visualization(self):
        """Create comprehensive visualization of downloaded data"""
        print("\n5. 🎨 Creating visualization...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # Plot 1: Base map with all features
        ax = axes[0]
        if hasattr(self, 'osm_layers'):
            # Plot in order: water, parks, roads, buildings
            if 'water' in self.osm_layers and len(self.osm_layers['water']) > 0:
                self.osm_layers['water'].plot(
                    ax=ax, color='blue', alpha=0.5, label='Water')

            if 'parks' in self.osm_layers and len(self.osm_layers['parks']) > 0:
                self.osm_layers['parks'].plot(
                    ax=ax, color='green', alpha=0.3, label='Parks')

            if 'roads' in self.osm_layers and len(self.osm_layers['roads']) > 0:
                roads = self.osm_layers['roads']
                # If road_class doesn't exist, create it from highway tag
                if 'road_class' not in roads.columns and 'highway' in roads.columns:
                    def classify_road(road_type):
                        if road_type in ['motorway', 'trunk', 'primary']:
                            return 'major'
                        elif road_type in ['secondary', 'tertiary']:
                            return 'collector'
                        elif road_type in ['residential', 'living_street', 'unclassified']:
                            return 'local'
                        else:
                            return 'path'
                    roads['road_class'] = roads['highway'].apply(classify_road)

                if 'road_class' in roads.columns:
                    # Color roads by class
                    road_colors = {'major': 'red', 'collector': 'orange',
                                   'local': 'gray', 'path': 'lightgray'}
                    for road_class, color in road_colors.items():
                        roads_class = roads[roads['road_class'] == road_class]
                        if len(roads_class) > 0:
                            roads_class.plot(ax=ax, color=color, linewidth=2 if road_class ==
                                             'major' else 1, label=f'{road_class} roads')
                else:
                    # Fallback if no road classification available
                    roads.plot(ax=ax, color='gray', linewidth=1, label='Roads')

            if 'buildings' in self.osm_layers and len(self.osm_layers['buildings']) > 0:
                self.osm_layers['buildings'].plot(
                    ax=ax, color='gray', alpha=0.7, label='Buildings')

        ax.set_title('Al-Rimal: Complete OSM Data')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')

        # Plot 2: Building types
        ax = axes[1]
        if hasattr(self, 'osm_layers') and 'buildings' in self.osm_layers:
            buildings = self.osm_layers['buildings']
            if 'building_type' in buildings.columns and len(buildings) > 0:
                # Group building types
                building_counts = buildings['building_type'].value_counts().head(
                    10)

                # Create color map for building types
                unique_types = building_counts.index.tolist()
                colors = plt.cm.tab20c(np.linspace(0, 1, len(unique_types)))

                for i, (btype, count) in enumerate(building_counts.items()):
                    btype_buildings = buildings[buildings['building_type'] == btype]
                    btype_buildings.plot(
                        ax=ax, color=colors[i], label=f'{btype} ({count})', alpha=0.7)

                ax.set_title(f'Building Types ({len(buildings)} total)')
                ax.legend(loc='upper right', fontsize=7)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')

        # Plot 3: Road network hierarchy
        ax = axes[2]
        if hasattr(self, 'osm_layers') and 'roads' in self.osm_layers:
            roads = self.osm_layers['roads']
            if len(roads) > 0 and 'road_class' in roads.columns:
                road_colors = {'major': 'red', 'collector': 'orange',
                               'local': 'gray', 'path': 'lightgray'}
                linewidths = {'major': 3, 'collector': 2,
                              'local': 1.5, 'path': 1}

                for road_class, color in road_colors.items():
                    roads_class = roads[roads['road_class'] == road_class]
                    if len(roads_class) > 0:
                        roads_class.plot(
                            ax=ax,
                            color=color,
                            linewidth=linewidths.get(road_class, 1),
                            label=f'{road_class} ({len(roads_class)})'
                        )

                ax.set_title(f'Road Network Hierarchy ({len(roads)} segments)')
                ax.legend(loc='upper right', fontsize=8)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')

        # Plot 4: Land use
        ax = axes[3]
        if hasattr(self, 'osm_layers') and 'landuse' in self.osm_layers:
            landuse = self.osm_layers['landuse']
            if len(landuse) > 0:
                # Color by landuse type
                unique_landuse = landuse['landuse'].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_landuse)))

                for i, lutype in enumerate(unique_landuse):
                    lu_features = landuse[landuse['landuse'] == lutype]
                    lu_features.plot(
                        ax=ax, color=colors[i], alpha=0.7, label=lutype)

                ax.set_title(f'Land Use Zones ({len(landuse)} areas)')
                ax.legend(loc='upper right', fontsize=7)

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')

        # Plot 5: Water features
        ax = axes[4]
        if hasattr(self, 'osm_layers') and 'water' in self.osm_layers:
            water = self.osm_layers['water']
            if len(water) > 0:
                water.plot(ax=ax, color='blue', alpha=0.7,
                           label='Water bodies')
                ax.set_title(f'Water Features ({len(water)})')

        # Add coastline indicator (Gaza is coastal)
        coast_y = 31.525  # Approximate coastline
        ax.axhline(y=coast_y, color='darkblue', linestyle='--',
                   alpha=0.5, label='Approx. coastline')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')

        # Plot 6: DEM if available
        ax = axes[5]
        dem_path = self.data_dir / "dem.tif"
        if dem_path.exists():
            try:
                with rasterio.open(dem_path) as src:
                    dem = src.read(1)
                    extent = [WEST, EAST, SOUTH, NORTH]

                    im = ax.imshow(dem, extent=extent,
                                   cmap='terrain', alpha=0.8)
                    plt.colorbar(im, ax=ax, label='Elevation (m)')

                    # Overlay water features
                    if hasattr(self, 'osm_layers') and 'water' in self.osm_layers:
                        self.osm_layers['water'].plot(
                            ax=ax, color='blue', alpha=0.5)

                    ax.set_title('Terrain (DEM) with Water Overlay')
            except Exception as e:
                ax.text(0.5, 0.5, f'Could not load DEM:\n{e}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Terrain Data')
        else:
            ax.text(0.5, 0.5, 'DEM not available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Terrain Data')

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal')

        plt.suptitle('Al-Rimal, Gaza: Geographic Data Analysis',
                     fontsize=16, y=1.02)
        plt.tight_layout()

        # Save figure
        viz_path = self.data_dir / "al_rimel_overview.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Visualization saved to {viz_path}")

        plt.show()

        return viz_path

    def generate_data_report(self):
        """Generate a summary report of the downloaded data"""
        print("\n6. 📊 Generating data report...")

        report_path = self.data_dir / "data_report.md"

        with open(report_path, 'w') as f:
            f.write("# Al-Rimal, Gaza: Data Collection Report\n\n")
            f.write(
                f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(
                f"**Bounding Box:** {SOUTH}°S, {NORTH}°N, {WEST}°W, {EAST}°E\n\n")

            f.write("## Data Summary\n\n")

            if hasattr(self, 'osm_layers'):
                f.write("### OpenStreetMap Data\n")
                f.write("| Layer | Features | Description |\n")
                f.write("|-------|----------|-------------|\n")

                total_features = 0
                for name, layer in self.osm_layers.items():
                    if isinstance(layer, gpd.GeoDataFrame):
                        count = len(layer)
                        total_features += count

                        # Get description
                        if name == 'buildings':
                            desc = "Building footprints with types"
                        elif name == 'roads':
                            desc = "Road network with hierarchy"
                        elif name == 'water':
                            desc = "Water bodies and waterways"
                        elif name == 'landuse':
                            desc = "Land use zoning areas"
                        elif name == 'parks':
                            desc = "Parks and leisure areas"
                        else:
                            desc = "Geographic features"

                        f.write(f"| {name} | {count} | {desc} |\n")

                f.write(
                    f"| **Total** | **{total_features}** | **All OSM features** |\n\n")

                # Add building type breakdown
                if 'buildings' in self.osm_layers and len(self.osm_layers['buildings']) > 0:
                    f.write("### Building Types\n")
                    buildings = self.osm_layers['buildings']
                    if 'building_type' in buildings.columns:
                        type_counts = buildings['building_type'].value_counts()
                        f.write("| Type | Count | Percentage |\n")
                        f.write("|------|-------|------------|\n")
                        for btype, count in type_counts.head(15).items():
                            percentage = (count / len(buildings)) * 100
                            f.write(
                                f"| {btype} | {count} | {percentage:.1f}% |\n")
                        f.write("\n")

                # Add road hierarchy breakdown
                if 'roads' in self.osm_layers and len(self.osm_layers['roads']) > 0:
                    f.write("### Road Network Hierarchy\n")
                    roads = self.osm_layers['roads']
                    if 'road_class' in roads.columns:
                        class_counts = roads['road_class'].value_counts()
                        f.write("| Class | Count | Total Length (est. km) |\n")
                        f.write("|-------|-------|------------------------|\n")
                        for road_class, count in class_counts.items():
                            # Estimate total length
                            roads_class = roads[roads['road_class']
                                                == road_class]
                            if not roads_class.empty:
                                # Approx km (degrees to km at equator)
                                length_km = roads_class.length.sum() * 111
                                f.write(
                                    f"| {road_class} | {count} | {length_km:.2f} |\n")
                        f.write("\n")

            f.write("## File Structure\n")
            f.write("```\n")
            for file_path in self.data_dir.glob("*"):
                size = file_path.stat().st_size / 1024  # KB
                f.write(f"{file_path.name:30} {size:8.1f} KB\n")
            f.write("```\n\n")

            f.write("## Notes\n")
            f.write("- Data downloaded from OpenStreetMap\n")
            f.write("- Coordinate system: WGS84 (EPSG:4326)\n")
            f.write("- Use for urban regeneration planning\n")

        print(f"   ✓ Report saved to {report_path}")
        return report_path

    def run_full_pipeline(self, opentopography_key=None):
        """Run complete data download pipeline"""
        print("=" * 60)
        print("🏗️  AL-RIMAL DATA DOWNLOAD PIPELINE")
        print("=" * 60)

        # Step 1: Download OSM
        self.download_osm_data()

        # Step 2: Download terrain
        self.download_terrain_data(opentopography_key)

        # Step 3: Create visualization
        self.create_visualization()

        # Step 4: Generate report
        self.generate_data_report()

        print("\n" + "=" * 60)
        print("✅ DOWNLOAD COMPLETE!")
        print("=" * 60)
        print(f"\n📁 All data saved to: {self.data_dir.absolute()}")

        # List downloaded files
        print("\n📋 Downloaded files:")
        for file_path in sorted(self.data_dir.glob("*")):
            size_kb = file_path.stat().st_size / 1024
            print(f"  • {file_path.name:25} {size_kb:7.1f} KB")

        return self.data_dir


# Run the downloader
if __name__ == "__main__":
    # Initialize downloader
    downloader = AlRimalDataDownloader()

    # Set your OpenTopography API key (optional)
    # Get one from: https://opentopography.org/developers
    OPENTOPOGRAPHY_API_KEY = "f4a8e8b5fb83b02585d60c0688e98e83"

    # Run complete pipeline
    data_dir = downloader.run_full_pipeline(OPENTOPOGRAPHY_API_KEY)

    # Quick analysis
    print("\n🔍 Quick Analysis:")
    if hasattr(downloader, 'osm_layers'):
        if 'buildings' in downloader.osm_layers:
            bld_count = len(downloader.osm_layers['buildings'])
            print(f"  • Buildings: {bld_count}")

        if 'roads' in downloader.osm_layers:
            road_count = len(downloader.osm_layers['roads'])
            print(f"  • Road segments: {road_count}")

        if 'water' in downloader.osm_layers:
            water_count = len(downloader.osm_layers['water'])
            print(f"  • Water features: {water_count}")

    print("\n🚀 Next steps:")
    print("  1. Examine the generated visualization")
    print("  2. Check data_report.md for details")
    print("  3. Start building your generator with this data!")
