import geopandas as gpd

file = r"data\SubDistricts_2011.geojsonl"

gdf = gpd.read_file(file)

print("=" * 70)
print("TOTAL RECORDS:", len(gdf))

print("\nCOLUMNS:")
print(gdf.columns.tolist())

print("\nFIRST 5 ROWS:")
print(gdf.head())

print("\nCRS:")
print(gdf.crs)
