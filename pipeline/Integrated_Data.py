# damage_integrator.py
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import mapping, box, Polygon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class DamageIntegrator:
    def __init__(self, osm_data_dir="data/osm", damage_tif_path="data/damage/damage_mask.tif"):
        self.osm_dir = Path(osm_data_dir)
        self.damage_path = Path(damage_tif_path)
        self.output_dir = Path("data/al_rimel_integrated")
        self.output_dir.mkdir(exist_ok=True)

    def load_all_data(self):
        """Load both OSM and damage data"""
        print("📂 Loading all data...")

        # Load OSM layers
        osm_layers = {}
        for geojson_file in self.osm_dir.glob("*.geojson"):
            layer_name = geojson_file.stem
            osm_layers[layer_name] = gpd.read_file(geojson_file)
            print(f"  ✓ {layer_name}: {len(osm_layers[layer_name])} features")

        # Load damage raster
        with rasterio.open(self.damage_path) as src:
            damage_raster = src.read(1)
            damage_transform = src.transform
            damage_crs = src.crs

        print(f"  ✓ Damage raster: {damage_raster.shape} pixels")
        print(f"     Damage CRS: {damage_crs}")

        return osm_layers, damage_raster, damage_transform, damage_crs

    def align_coordinate_systems(self, osm_layers, damage_crs):
        """Ensure all data in same CRS"""
        print("\n🔄 Aligning coordinate systems...")

        aligned_osm = {}

        # OSM is typically in EPSG:4326 (WGS84)
        # Damage raster from GEE is likely in UTM or Web Mercator

        for name, gdf in osm_layers.items():
            if not gdf.empty:
                # Check current CRS
                if gdf.crs is None:
                    gdf.crs = "EPSG:4326"  # Assume WGS84 for OSM

                # Transform to match damage raster CRS
                aligned = gdf.to_crs(damage_crs)
                aligned_osm[name] = aligned
                print(f"  ✓ {name}: {gdf.crs} → {damage_crs}")

        return aligned_osm

    def create_damage_vector_from_raster(self, damage_raster, transform, crs, threshold=0.5):
        """Convert damage raster to vector polygons"""
        print("\n🔄 Creating vector damage polygons...")

        # Create binary mask
        binary_damage = (damage_raster > threshold).astype(np.uint8)

        # Find contours of damaged areas
        from skimage import measure

        contours = measure.find_contours(binary_damage, 0.5)

        # Convert contours to polygons
        damage_polygons = []
        for contour in contours:
            # Transform pixel coordinates to real-world coordinates
            world_coords = [
                rasterio.transform.xy(transform, row, col)
                for row, col in contour
            ]

            # Create polygon (close the loop)
            if len(world_coords) > 3:
                # Add first point at end to close polygon
                world_coords.append(world_coords[0])
                poly = Polygon(world_coords)
                damage_polygons.append(poly)

        # Create GeoDataFrame
        damage_gdf = gpd.GeoDataFrame(
            geometry=damage_polygons,
            crs=crs
        )

        # Add damage intensity (raster value average)
        damage_gdf['damage_intensity'] = self.calculate_polygon_intensity(
            damage_gdf, damage_raster, transform
        )

        # Classify damage severity
        damage_gdf['damage_class'] = self.classify_damage_severity(
            damage_gdf['damage_intensity']
        )

        print(f"  ✓ Created {len(damage_gdf)} damage polygons")
        print(
            f"     Classes: {damage_gdf['damage_class'].value_counts().to_dict()}")

        return damage_gdf

    def assign_damage_to_buildings(self, buildings_gdf, damage_gdf):
        """Assign damage status to each building based on overlap"""
        print("\n🏢 Assigning damage to buildings...")

        # Create a copy to avoid modifying original
        buildings = buildings_gdf.copy()

        # Initialize damage columns
        buildings['damaged'] = False
        buildings['damage_intensity'] = 0.0
        buildings['damage_class'] = 'intact'

        # Spatial join: find buildings that intersect damage polygons
        damaged_buildings = gpd.sjoin(
            buildings,
            damage_gdf[['geometry', 'damage_intensity', 'damage_class']],
            how='inner',
            predicate='intersects'
        )

        # Update building damage status
        for idx in damaged_buildings.index:
            # Get the maximum damage intensity for this building
            building_damage = damaged_buildings.loc[idx]
            if isinstance(building_damage, pd.DataFrame):
                # Multiple damage polygons intersect
                # Note: sjoin adds '_right' suffix to damage_gdf columns
                damage_col = 'damage_intensity_right' if 'damage_intensity_right' in building_damage.columns else 'damage_intensity'
                class_col = 'damage_class_right' if 'damage_class_right' in building_damage.columns else 'damage_class'

                max_intensity = building_damage[damage_col].max()
                worst_class = building_damage.iloc[building_damage[damage_col].argmax(
                )][class_col]
            else:
                # Single damage polygon intersects
                damage_col = 'damage_intensity_right' if 'damage_intensity_right' in building_damage.index else 'damage_intensity'
                class_col = 'damage_class_right' if 'damage_class_right' in building_damage.index else 'damage_class'

                max_intensity = building_damage[damage_col]
                worst_class = building_damage[class_col]

            buildings.loc[idx, 'damaged'] = True
            buildings.loc[idx, 'damage_intensity'] = max_intensity
            buildings.loc[idx, 'damage_class'] = worst_class

        # Statistics
        total_buildings = len(buildings)
        damaged_count = buildings['damaged'].sum()
        damage_percentage = (damaged_count / total_buildings) * 100

        print(f"  ✓ Total buildings: {total_buildings}")
        print(
            f"  ✓ Damaged buildings: {damaged_count} ({damage_percentage:.1f}%)")
        print(f"  ✓ Damage class distribution:")
        print(buildings['damage_class'].value_counts().to_dict())

        return buildings

    def assign_damage_to_roads(self, roads_gdf, damage_gdf):
        """Assign damage status to roads"""
        print("\n🛣️ Assigning damage to roads...")

        roads = roads_gdf.copy()
        roads['damaged'] = False
        roads['damage_class'] = 'intact'

        # Buffer roads slightly for intersection
        roads_buffered = roads.copy()
        roads_buffered.geometry = roads_buffered.geometry.buffer(5)  # 5 meters

        # Find intersections
        damaged_roads = gpd.sjoin(
            roads_buffered,
            damage_gdf[['geometry', 'damage_class']],
            how='inner',
            predicate='intersects'
        )

        # Update road damage
        for idx in damaged_roads.index.unique():
            roads.loc[idx, 'damaged'] = True
            # Get the worst damage class intersecting this road
            road_damage = damaged_roads.loc[idx]
            if isinstance(road_damage, pd.DataFrame):
                # Note: sjoin adds '_right' suffix to damage_gdf columns
                class_col = 'damage_class_right' if 'damage_class_right' in road_damage.columns else 'damage_class'
                worst_class = self.get_worst_damage_class(
                    road_damage[class_col])
            else:
                class_col = 'damage_class_right' if 'damage_class_right' in road_damage.index else 'damage_class'
                worst_class = road_damage[class_col]
            roads.loc[idx, 'damage_class'] = worst_class

        print(f"  ✓ Total road segments: {len(roads)}")
        print(f"  ✓ Damaged road segments: {roads['damaged'].sum()}")

        return roads

    def get_worst_damage_class(self, damage_classes):
        """Get the worst (most severe) damage class from a series"""
        severity_order = {'destroyed': 4, 'severe': 3,
                          'moderate': 2, 'minor': 1, 'intact': 0}

        worst = 'intact'
        worst_severity = 0

        for damage_class in damage_classes:
            severity = severity_order.get(damage_class, 0)
            if severity > worst_severity:
                worst = damage_class
                worst_severity = severity

        return worst

    def calculate_reconstruction_priority(self, buildings_gdf):
        """Calculate priority for reconstruction"""
        print("\n🎯 Calculating reconstruction priorities...")

        buildings = buildings_gdf.copy()

        # Priority factors (higher = more urgent)
        priorities = []

        for idx, building in buildings.iterrows():
            priority_score = 0

            # 1. Damage severity (most important)
            damage_weights = {
                'destroyed': 10,
                'severe': 7,
                'moderate': 4,
                'minor': 2,
                'intact': 0
            }
            priority_score += damage_weights.get(building['damage_class'], 0)

            # 2. Building type importance
            type_weights = {
                'hospital': 5, 'school': 4, 'government': 4,
                'apartments': 3, 'residential': 2,
                'commercial': 2, 'industrial': 1
            }
            building_type = building.get('building_type', 'residential')
            priority_score += type_weights.get(building_type, 1)

            # 3. Proximity to intact infrastructure (lower = better)
            # Buildings near intact roads get higher priority
            if 'distance_to_intact_road' in building:
                if building['distance_to_intact_road'] < 50:  # meters
                    priority_score += 2

            priorities.append(priority_score)

        buildings['reconstruction_priority'] = priorities

        # Normalize to 0-100
        max_priority = buildings['reconstruction_priority'].max()
        if max_priority > 0:
            buildings['reconstruction_priority'] = (
                buildings['reconstruction_priority'] / max_priority * 100
            ).astype(int)

        # Classify priority levels
        def classify_priority(score):
            if score >= 80:
                return 'critical'
            elif score >= 60:
                return 'high'
            elif score >= 40:
                return 'medium'
            elif score >= 20:
                return 'low'
            else:
                return 'preserve'

        buildings['priority_class'] = buildings['reconstruction_priority'].apply(
            classify_priority
        )

        print("  ✓ Priority distribution:")
        print(buildings['priority_class'].value_counts().to_dict())

        return buildings

    def calculate_polygon_intensity(self, polygons_gdf, raster_data, transform):
        """Calculate average raster intensity for each polygon"""
        intensities = []

        for idx, polygon in polygons_gdf.iterrows():
            geom = polygon.geometry
            try:
                # Get raster values within the polygon using the already loaded raster data
                from rasterio.windows import from_bounds

                x_min, y_min, x_max, y_max = geom.bounds
                window = from_bounds(x_min, y_min, x_max, y_max, transform)

                # Sample raster values
                row_start = int(max(0, window.row_off))
                row_end = int(
                    min(raster_data.shape[1], window.row_off + window.height))
                col_start = int(max(0, window.col_off))
                col_end = int(
                    min(raster_data.shape[2], window.col_off + window.width))

                clipped_data = raster_data[:,
                                           row_start:row_end, col_start:col_end]

                # Calculate mean intensity
                valid_values = clipped_data[clipped_data > 0]
                if len(valid_values) > 0:
                    intensity = float(valid_values.mean())
                else:
                    intensity = 0.0
            except Exception:
                intensity = 0.0

            intensities.append(intensity)

        return intensities

    def classify_damage_severity(self, intensity_series):
        """Classify damage severity based on intensity values"""
        def classify(intensity):
            if intensity >= 0.8:
                return 'destroyed'
            elif intensity >= 0.6:
                return 'severe'
            elif intensity >= 0.4:
                return 'moderate'
            elif intensity > 0:
                return 'minor'
            else:
                return 'intact'

        return intensity_series.apply(classify)

    def create_integrated_visualization(self, osm_layers, damage_gdf, damaged_buildings):
        """Create comprehensive visualization"""
        print("\n🎨 Creating integrated visualization...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Damage raster overlay
        ax = axes[0, 0]
        if 'damage_mask.tif' in str(self.damage_path):
            with rasterio.open(self.damage_path) as src:
                damage_data = src.read(1)
                show(damage_data, ax=ax, cmap='Reds', alpha=0.6)
                ax.set_title('SAR Damage Intensity')

        # Plot 2: Damage polygons
        ax = axes[0, 1]
        if not damage_gdf.empty:
            damage_gdf.plot(
                column='damage_class',
                categorical=True,
                legend=True,
                ax=ax,
                cmap='Reds',
                alpha=0.5
            )
            ax.set_title('Damage Polygons by Class')

        # Plot 3: Building damage
        ax = axes[0, 2]
        if 'buildings' in osm_layers and not damaged_buildings.empty:
            # Color by damage class
            color_map = {
                'destroyed': 'darkred',
                'severe': 'red',
                'moderate': 'orange',
                'minor': 'yellow',
                'intact': 'gray'
            }

            for damage_class, color in color_map.items():
                class_buildings = damaged_buildings[damaged_buildings['damage_class'] == damage_class]
                if not class_buildings.empty:
                    class_buildings.plot(
                        ax=ax, color=color, label=damage_class, alpha=0.7)

            ax.legend(title='Damage Class')
            ax.set_title(
                f'Building Damage ({len(damaged_buildings)} buildings)')

        # Plot 4: Road damage
        ax = axes[1, 0]
        if 'roads' in osm_layers:
            roads = osm_layers['roads']
            if 'damage_class' in roads.columns:
                # Color roads by damage
                road_colors = {'destroyed': 'red', 'severe': 'orange',
                               'moderate': 'yellow', 'intact': 'gray'}

                for damage_class, color in road_colors.items():
                    class_roads = roads[roads['damage_class'] == damage_class]
                    if not class_roads.empty:
                        class_roads.plot(ax=ax, color=color,
                                         linewidth=2, label=damage_class)

                ax.legend(title='Road Damage')
                ax.set_title('Road Network Damage')

        # Plot 5: Reconstruction priorities
        ax = axes[1, 1]
        if 'reconstruction_priority' in damaged_buildings.columns:
            damaged_buildings.plot(
                column='reconstruction_priority',
                cmap='RdYlGn_r',  # Red (high) to Green (low)
                legend=True,
                ax=ax,
                alpha=0.8,
                legend_kwds={'label': 'Priority Score'}
            )
            ax.set_title('Reconstruction Priorities')

        # Plot 6: Intact vs Damaged clusters
        ax = axes[1, 2]
        if not damage_gdf.empty and 'buildings' in osm_layers:
            # Show damage polygons
            damage_gdf.plot(ax=ax, color='red', alpha=0.3,
                            label='Damaged areas')

            # Show intact buildings
            intact_buildings = damaged_buildings[damaged_buildings['damage_class'] == 'intact']
            if not intact_buildings.empty:
                intact_buildings.plot(
                    ax=ax, color='green', alpha=0.7, label='Intact buildings')

            # Show damaged buildings
            damaged_only = damaged_buildings[damaged_buildings['damage_class'] != 'intact']
            if not damaged_only.empty:
                damaged_only.plot(ax=ax, color='orange',
                                  alpha=0.7, label='Damaged buildings')

            ax.legend()
            ax.set_title('Intact Clusters vs Damage Clusters')

        plt.suptitle('Al-Rimal: Integrated Damage Assessment', fontsize=16)
        plt.tight_layout()

        # Save visualization
        viz_path = self.output_dir / "integrated_damage_assessment.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Visualization saved to {viz_path}")

        plt.show()

    def run_integration_pipeline(self):
        """Run complete integration pipeline"""
        print("=" * 60)
        print("🔄 INTEGRATING OSM + SAR DAMAGE DATA")
        print("=" * 60)

        # Step 1: Load all data
        osm_layers, damage_raster, damage_transform, damage_crs = self.load_all_data()

        # Step 2: Align coordinate systems
        aligned_osm = self.align_coordinate_systems(osm_layers, damage_crs)

        # Step 3: Create vector damage polygons from raster
        damage_gdf = self.create_damage_vector_from_raster(
            damage_raster, damage_transform, damage_crs
        )

        # Step 4: Assign damage to buildings
        damaged_buildings = None
        if 'buildings' in aligned_osm:
            damaged_buildings = self.assign_damage_to_buildings(
                aligned_osm['buildings'], damage_gdf
            )
            aligned_osm['buildings'] = damaged_buildings

        # Step 5: Assign damage to roads
        if 'roads' in aligned_osm:
            damaged_roads = self.assign_damage_to_roads(
                aligned_osm['roads'], damage_gdf
            )
            aligned_osm['roads'] = damaged_roads

        # Step 6: Calculate reconstruction priorities
        if 'buildings' in aligned_osm:
            prioritized_buildings = self.calculate_reconstruction_priority(
                aligned_osm['buildings']
            )
            aligned_osm['buildings'] = prioritized_buildings

        # Step 7: Save integrated data
        print("\n💾 Saving integrated data...")
        for name, gdf in aligned_osm.items():
            if not gdf.empty:
                output_path = self.output_dir / f"{name}_integrated.geojson"
                gdf.to_file(output_path, driver='GeoJSON')
                print(f"  ✓ {name} → {output_path}")

        # Save damage polygons
        damage_gdf.to_file(self.output_dir /
                           "damage_polygons.geojson", driver='GeoJSON')

        # Step 8: Create visualization
        if damaged_buildings is not None:
            self.create_integrated_visualization(
                aligned_osm, damage_gdf, damaged_buildings)

        # Step 9: Generate summary
        self.generate_integration_summary(aligned_osm, damage_gdf)

        print("\n" + "=" * 60)
        print("✅ INTEGRATION COMPLETE!")
        print("=" * 60)

        return aligned_osm, damage_gdf

    def generate_integration_summary(self, osm_layers, damage_gdf):
        """Generate summary report"""
        summary_path = self.output_dir / "integration_summary.md"

        with open(summary_path, 'w') as f:
            f.write("# Al-Rimal Damage Integration Summary\n\n")

            # Building damage summary
            if 'buildings' in osm_layers:
                buildings = osm_layers['buildings']
                total_buildings = len(buildings)
                damaged = buildings['damaged'].sum()

                f.write("## Building Damage\n")
                f.write(f"- **Total buildings:** {total_buildings}\n")
                f.write(
                    f"- **Damaged buildings:** {damaged} ({damaged/total_buildings*100:.1f}%)\n")

                if 'damage_class' in buildings.columns:
                    f.write("\n### Damage Class Distribution\n")
                    for cls, count in buildings['damage_class'].value_counts().items():
                        f.write(
                            f"- {cls}: {count} ({count/total_buildings*100:.1f}%)\n")

                # Reconstruction priorities
                if 'reconstruction_priority' in buildings.columns:
                    f.write("\n## Reconstruction Priorities\n")
                    for cls, count in buildings['priority_class'].value_counts().items():
                        f.write(f"- {cls}: {count}\n")

            # Damage area
            if not damage_gdf.empty:
                total_area = damage_gdf.geometry.area.sum()
                f.write(f"\n## Damage Area\n")
                f.write(f"- **Total damaged area:** {total_area:.0f} m²\n")
                f.write(
                    f"- **Number of damage clusters:** {len(damage_gdf)}\n")

        print(f"  ✓ Summary saved to {summary_path}")


# Run the integration
if __name__ == "__main__":
    integrator = DamageIntegrator()
    osm_layers, damage_gdf = integrator.run_integration_pipeline()

    # Quick stats
    print("\n📊 Quick Statistics:")
    if 'buildings' in osm_layers:
        bld = osm_layers['buildings']
        print(f"• Buildings: {len(bld)} total")
        print(
            f"• Damaged: {bld['damaged'].sum()} ({bld['damaged'].sum()/len(bld)*100:.1f}%)")

    print(f"• Damage clusters: {len(damage_gdf)}")
    print(f"• Output directory: {integrator.output_dir}")
