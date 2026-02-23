import geopandas as gpd
from shapely.geometry import MultiPolygon
from pathlib import Path

print("⚡ CREATING CONSTRAINT MAP")
print("=" * 50)

# Load data
data_dir = Path("data/al_rimel_integrated")
buildings = gpd.read_file(data_dir / "buildings_integrated.geojson")
water = gpd.read_file(data_dir / "water_integrated.geojson")

# 1. Water constraints (no-build zones)
print("1. Water constraints...")
# Buffer around water (e.g., 30m setback)
water_buffered = water.copy()
water_buffered['geometry'] = water_buffered.buffer(0.00027)  # ~30m at equator
water_buffered['constraint_type'] = 'water_buffer'

# 2. Identify intact clusters (preservation zones)
print("2. Preservation zones...")
if 'damage_class' in buildings.columns:
    intact_buildings = buildings[buildings['damage_class'] == 'intact']
    
    # Cluster intact buildings
    from sklearn.cluster import DBSCAN
    import numpy as np
    
    centroids = np.array([
        [geom.centroid.x, geom.centroid.y] 
        for geom in intact_buildings.geometry.centroid
    ])
    
    # Cluster with 50m threshold
    clustering = DBSCAN(eps=0.00045, min_samples=3).fit(centroids)
    intact_buildings['cluster'] = clustering.labels_
    
    # Create preservation zones around clusters
    preservation_zones = []
    for cluster_id in intact_buildings['cluster'].unique():
        if cluster_id != -1:  # Skip noise
            cluster = intact_buildings[intact_buildings['cluster'] == cluster_id]
            # Create convex hull around cluster
            cluster_geom = MultiPolygon([geom for geom in cluster.geometry])
            zone = cluster_geom.convex_hull.buffer(0.00027)  # 30m buffer
            preservation_zones.append(zone)
    
    preservation_gdf = gpd.GeoDataFrame(
        geometry=preservation_zones,
        crs=buildings.crs
    )
    preservation_gdf['constraint_type'] = 'preservation_zone'
    
    print(f"   ✓ Created {len(preservation_zones)} preservation zones")

# 3. Damaged areas (rebuild zones)
print("3. Rebuild zones...")
damage = gpd.read_file(data_dir / "damage_polygons.geojson")
damage['constraint_type'] = 'rebuild_zone'

# 4. Save constraints
constraints_dir = Path("data/constraints")
constraints_dir.mkdir(exist_ok=True)

water_buffered.to_file(constraints_dir / "water_constraints.geojson", driver='GeoJSON')
preservation_gdf.to_file(constraints_dir / "preservation_zones.geojson", driver='GeoJSON')
damage.to_file(constraints_dir / "rebuild_zones.geojson", driver='GeoJSON')

print(f"\n✅ Constraints saved to {constraints_dir}")