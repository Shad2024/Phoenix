import geopandas as gpd

roads = gpd.read_file("data/al_rimel/roads.geojson")
buildings = gpd.read_file("data/al_rimel/buildings.geojson")
parks = gpd.read_file("data/al_rimel/parks.geojson")  
damage = gpd.read_file("data/al_rimel_integrated/damage_polygons.geojson")       


print("roads:", len(roads))
print("buildings:", len(buildings))

roads = roads.to_crs(epsg=4326)
buildings = buildings.to_crs(epsg=4326)
parks = parks.to_crs(epsg=4326)
damage = damage.to_crs(epsg=4326)

import folium

# find map center automatically
center = buildings.geometry.unary_union.centroid
m = folium.Map(location=[center.y, center.x], zoom_start=15,tiles=None,locate_control=False)

# Filter to keep only Polygons and Lines
buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
roads = roads[roads.geometry.type.isin(['LineString', 'MultiLineString'])]


# ROADS (blue)
folium.GeoJson(
    roads.to_json(),
    name="Roads",
    style_function=lambda x: {"color": "blue", "weight": 2},
    marker=folium.CircleMarker(radius=0, fill=False)
).add_to(m)

# BUILDINGS )
folium.GeoJson(
    buildings.to_json(),
    name="Buildings",
    style_function=lambda x: {"color": "black", "weight": 1},
    marker=folium.CircleMarker(radius=0, fill=False)
).add_to(m)

#  damag BUILDINGS 
folium.GeoJson(
    damage.to_json(),
    name="damage",
    style_function=lambda x: {"color": "red", "weight": 1},
    marker=folium.CircleMarker(radius=0, fill=False)
).add_to(m)
folium.GeoJson(
    buildings.to_json(),
    name="Buildings",
    style_function=lambda x: {"color": "black", "weight": 1},
    marker=folium.CircleMarker(radius=0, fill=False)
).add_to(m)

# WATER 
folium.GeoJson(
    parks.to_json(),
    name="Parks",
    style_function=lambda x: {"color": "green"},
    marker=folium.CircleMarker(radius=0, fill=False)
).add_to(m)

folium.LayerControl().add_to(m)

m.save("my_geojson_map22.html")
print("Map saved -> my_geojson_map22.html")


 