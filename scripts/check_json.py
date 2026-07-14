import geopandas as gpd

json_file = r"C:\Users\rasoo_me13iy3\OneDrive\Desktop\YOUR_FOLDER\YOUR_FILE.json"

gdf = gpd.read_file(json_file)

print("=" * 60)
print("Columns:")
print(gdf.columns.tolist())

print("\nFirst 5 rows:")
print(gdf.head())

print("\nTotal records:", len(gdf))