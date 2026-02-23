#rule based 
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from pathlib import Path

class UrbanRegenerator:
    def __init__(self, constraints_dir="data/constraints"):
        print("📁 Loading constraints from:", constraints_dir)
        self.constraints = self.load_constraints(constraints_dir)
        print("✅ Loaded constraints:", list(self.constraints.keys()))
        
        # Debug: Show what's in each constraint
        for name, gdf in self.constraints.items():
            print(f"   {name}: {len(gdf)} features")
            if len(gdf) > 0:
                print(f"     CRS: {gdf.crs}, Bounds: {gdf.total_bounds}")
        
        self.output_dir = Path("output/regeneration_v1")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_constraints(self, constraints_dir):
        """Load all constraint layers"""
        constraints = {}
        for file in Path(constraints_dir).glob("*.geojson"):
            name = file.stem.replace("_constraints", "").replace("_zones", "")
            gdf = gpd.read_file(file)
            constraints[name] = gdf
            print(f"   Loaded {file.name} → '{name}' with {len(gdf)} features")
        return constraints
    
    def generate_regeneration_plan(self):
        """Generate complete regeneration plan"""
        print("\n🏗️ GENERATING REGENERATION PLAN")
        print("=" * 50)
        
        plan = {}
        
        # 1. PRESERVE intact zones
        if 'preservation' in self.constraints:
            preserved = self.preserve_intact_areas()
            plan['preserved'] = preserved
            print(f"1. Preserved {len(preserved)} intact areas")
        
        # 2. REBUILD damaged areas
        if 'rebuild' in self.constraints:
            print(f"\n📊 Rebuild zones found: {len(self.constraints['rebuild'])}")
            for i, zone in self.constraints['rebuild'].iterrows():
                print(f"   Zone {i}: Area ≈ {zone.geometry.area:.8f} sq deg")
            
            rebuilt = self.rebuild_damaged_areas()
            plan['rebuilt'] = rebuilt
            print(f"2. Rebuilt {len(rebuilt)} areas")
            
            # Count total buildings
            total_buildings = sum(len(zone['buildings']) for zone in rebuilt)
            print(f"   Total buildings generated: {total_buildings}")
        
        # 3. REGENERATE road network
        roads = self.regenerate_road_network()
        plan['roads'] = roads
        print(f"3. Created {len(roads)} road segments")
        
        # 4. APPLY water constraints
        if 'water' in self.constraints:
            water_constraints = self.constraints['water']
            plan['water_constraints'] = water_constraints
            print(f"4. Applied {len(water_constraints)} water constraints")
        
        return plan
    
    def preserve_intact_areas(self):
        """Preserve intact buildings and areas"""
        return self.constraints['preservation']
    
    def rebuild_damaged_areas(self):
        """Generate new urban fabric in damaged areas"""
        rebuilt_areas = []
        
        for idx, damage_zone in self.constraints['rebuild'].iterrows():
            print(f"\n🔨 Processing rebuild zone {idx}")
            print(f"   Bounds: {damage_zone.geometry.bounds}")
            
            # Create new buildings in this damaged area
            new_buildings = self.generate_buildings_in_zone(damage_zone.geometry)
            print(f"   Generated {len(new_buildings)} buildings")
            
            # DEBUG: Show sample building locations
            if len(new_buildings) > 0:
                sample = new_buildings.geometry.iloc[0]
                print(f"   Sample building centroid: {sample.centroid.coords[0]}")
            
            # Create new roads
            new_roads = self.generate_roads_in_zone(damage_zone.geometry, new_buildings)
            print(f"   Generated {len(new_roads)} road segments")
            
            rebuilt_areas.append({
                'zone_id': idx,
                'geometry': damage_zone.geometry,
                'buildings': new_buildings,
                'roads': new_roads
            })
        
        return rebuilt_areas
    
    def generate_buildings_in_zone(self, zone_geometry):
        """Generate building footprints within a zone"""
        buildings = []
        
        # Get zone bounds
        minx, miny, maxx, maxy = zone_geometry.bounds
        print(f"     Zone bounds: ({minx:.6f}, {miny:.6f}) to ({maxx:.6f}, {maxy:.6f})")
        
        # Create a grid of buildings - INCREASED SIZE FOR VISIBILITY
        grid_size = 0.00001  # Increased from 0.00015 (~25m spacing for visibility)
        
        x = minx + grid_size/2  # Start from center of first cell
        building_count = 0
        
        while x < maxx - grid_size/2:
            y = miny + grid_size/2
            while y < maxy - grid_size/2:
                point = Point(x, y)
                
                # Check if point is inside zone and not in water
                if (zone_geometry.contains(point) and 
                    self.is_buildable_location(point)):
                    
                    # Create LARGER building footprint for visibility
                    building = point.buffer(0.00008)  # Increased from 0.000045 (~8m radius)
                    buildings.append(building)
                    building_count += 1
                
                y += grid_size * 1.8  # Increased spacing
            x += grid_size * 1.8  # Increased spacing
        
        print(f"     Placed {building_count} buildings in grid")
        
        # Create GeoDataFrame with proper CRS
        if buildings:
            crs = zone_geometry.crs if hasattr(zone_geometry, 'crs') else "EPSG:4326"
            return gpd.GeoDataFrame(geometry=buildings, crs=crs)
        else:
            # Return empty GeoDataFrame with proper CRS
            crs = zone_geometry.crs if hasattr(zone_geometry, 'crs') else "EPSG:4326"
            return gpd.GeoDataFrame(geometry=[], crs=crs)
    
    def is_buildable_location(self, point):
        """Check if location is buildable (not in water, etc.)"""
        if 'water' in self.constraints:
            for water_zone in self.constraints['water'].geometry:
                if water_zone.contains(point):
                    return False
        return True
    
    def generate_roads_in_zone(self, zone_geometry, buildings):
        """Generate road network connecting buildings"""
        roads = []
        
        if len(buildings) == 0:
            return gpd.GeoDataFrame(geometry=[], crs=zone_geometry.crs)
        
        # Create a simple grid road pattern
        minx, miny, maxx, maxy = zone_geometry.bounds
        
        # East-West roads
        y_positions = np.linspace(miny, maxy, 4)
        for y in y_positions[1:-1]:  # Skip edges
            road = LineString([(minx, y), (maxx, y)])
            if zone_geometry.intersects(road):
                roads.append(road)
        
        # North-South roads  
        x_positions = np.linspace(minx, maxx, 4)
        for x in x_positions[1:-1]:
            road = LineString([(x, miny), (x, maxy)])
            if zone_geometry.intersects(road):
                roads.append(road)
        
        if roads:
            crs = zone_geometry.crs if hasattr(zone_geometry, 'crs') else "EPSG:4326"
            return gpd.GeoDataFrame(geometry=roads, crs=crs)
        else:
            return gpd.GeoDataFrame(geometry=[], crs=zone_geometry.crs)
    
    def regenerate_road_network(self):
        """Regenerate complete road network"""
        # Al-Rimal bounding box
        bbox = Polygon([
            (34.445, 31.515),
            (34.470, 31.515),
            (34.470, 31.535),
            (34.445, 31.535)
        ])
        
        # Create grid roads
        roads = []
        
        # East-West arterials
        for y in [31.520, 31.530]:
            road = LineString([(34.445, y), (34.470, y)])
            roads.append(road)
        
        # North-South arterials
        for x in [34.450, 34.460]:
            road = LineString([(x, 31.515), (x, 31.535)])
            roads.append(road)
        
        # Local streets
        for y in np.linspace(31.515, 31.535, 8)[1:-1]:
            road = LineString([(34.445, y), (34.470, y)])
            roads.append(road)
        
        for x in np.linspace(34.445, 34.470, 8)[1:-1]:
            road = LineString([(x, 31.515), (x, 31.535)])
            roads.append(road)
        
        return gpd.GeoDataFrame(geometry=roads, crs="EPSG:4326")
    
    def save_plan(self, plan):
        """Save regeneration plan"""
        print("\n💾 Saving regeneration plan...")
        
        # Save preserved areas
        if 'preserved' in plan and len(plan['preserved']) > 0:
            plan['preserved'].to_file(self.output_dir / "preserved_areas.geojson")
            print(f"   Saved preserved areas: {len(plan['preserved'])} features")
        
        # Save rebuilt areas
        if 'rebuilt' in plan and len(plan['rebuilt']) > 0:
            # Combine all rebuilt buildings
            all_rebuilt_buildings = []
            for zone in plan['rebuilt']:
                if len(zone['buildings']) > 0:
                    all_rebuilt_buildings.extend(zone['buildings'].geometry.tolist())
            
            if all_rebuilt_buildings:
                rebuilt_buildings_gdf = gpd.GeoDataFrame(
                    geometry=all_rebuilt_buildings,
                    crs=plan['rebuilt'][0]['buildings'].crs
                )
                rebuilt_buildings_gdf.to_file(self.output_dir / "rebuilt_buildings.geojson")
                print(f"   Saved rebuilt buildings: {len(all_rebuilt_buildings)} features")
        
        # Save roads
        if 'roads' in plan and len(plan['roads']) > 0:
            plan['roads'].to_file(self.output_dir / "regenerated_roads.geojson")
            print(f"   Saved roads: {len(plan['roads'])} features")
        
        # Save water constraints
        if 'water_constraints' in plan and len(plan['water_constraints']) > 0:
            plan['water_constraints'].to_file(self.output_dir / "water_constraints.geojson")
            print(f"   Saved water constraints: {len(plan['water_constraints'])} features")
        
        print(f"✅ Plan saved to {self.output_dir}")
    
    def visualize_plan(self, plan):
        """Visualize the regeneration plan"""
        print("\n🎨 Creating visualization...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot 1: Before (constraints)
        ax = axes[0]
        
        # Plot water constraints
        if 'water_constraints' in plan and len(plan['water_constraints']) > 0:
            plan['water_constraints'].plot(ax=ax, color='blue', alpha=0.3, label='Water (no-build)')
        
        # Plot damage zones
        if 'rebuild' in self.constraints and len(self.constraints['rebuild']) > 0:
            self.constraints['rebuild'].plot(ax=ax, color='red', alpha=0.3, label='Damage zones')
        
        # Plot preservation zones
        if 'preserved' in plan and len(plan['preserved']) > 0:
            plan['preserved'].plot(ax=ax, color='green', alpha=0.3, label='Preservation zones')
        
        ax.set_xlim(34.445, 34.470)
        ax.set_ylim(31.515, 31.535)
        ax.set_title('Before: Constraints & Damage Zones')
        ax.legend()
        
        # Plot 2: After (regenerated)
        ax = axes[1]
        
        # FIRST plot preserved areas as background
        if 'preserved' in plan and len(plan['preserved']) > 0:
            plan['preserved'].plot(ax=ax, color='lightgreen', alpha=0.5, label='Preserved areas')
        
        # THEN plot rebuilt buildings (on top)
        if 'rebuilt' in plan and len(plan['rebuilt']) > 0:
            all_buildings = []
            for zone in plan['rebuilt']:
                if len(zone['buildings']) > 0:
                    all_buildings.extend(zone['buildings'].geometry.tolist())
            
            if all_buildings:
                buildings_gdf = gpd.GeoDataFrame(
                    geometry=all_buildings,
                    crs=plan['rebuilt'][0]['buildings'].crs if plan['rebuilt'][0]['buildings'].crs else "EPSG:4326"
                )
                print(f"   Plotting {len(buildings_gdf)} buildings")
                buildings_gdf.plot(ax=ax, color='orange', alpha=0.8, label='New buildings', markersize=50)
            else:
                print("   WARNING: No buildings to plot!")
        
        # Plot roads
        if 'roads' in plan and len(plan['roads']) > 0:
            plan['roads'].plot(ax=ax, color='black', linewidth=2, label='Roads')
        
        # Plot water (on top)
        if 'water_constraints' in plan and len(plan['water_constraints']) > 0:
            plan['water_constraints'].plot(ax=ax, color='blue', alpha=0.5, label='Water')
        
        ax.set_xlim(34.445, 34.470)
        ax.set_ylim(31.515, 31.535)
        ax.set_title('After: Regenerated Urban Fabric')
        ax.legend()
        
        plt.suptitle('Al-Rimal Urban Regeneration: Before vs After', fontsize=16)
        plt.tight_layout()
        
        # Save
        viz_path = self.output_dir / "regeneration_plan.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to {viz_path}")
        
        plt.show()

# Run the generator
if __name__ == "__main__":
    print("🚀 STARTING URBAN REGENERATION GENERATOR (DEBUG VERSION)")
    print("=" * 60)
    
    # Create generator
    generator = UrbanRegenerator("data/constraints")
    
    # Generate plan
    plan = generator.generate_regeneration_plan()
    
    # Save plan
    generator.save_plan(plan)
    
    # Visualize
    generator.visualize_plan(plan)
    
    print("\n" + "=" * 60)
    print("✅ GENERATION COMPLETE!")
    print("=" * 60)