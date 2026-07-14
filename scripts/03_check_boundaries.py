import geopandas as gpd

print("=" * 60)
print("STEP 3 : CHECK SUBDISTRICT BOUNDARIES")
print("=" * 60)

FILE = r"C:\Users\rasoo_me13iy3\OneDrive\Desktop\Tehsil_GPS_Projects\data\SubDistricts_2011.geojsonl"

print("\nReading GeoJSONL file...")

gdf = gpd.read_file(FILE)

print("\nDone!")

print("\nNumber of tehsils:", len(gdf))

print("\nColumns:")
print(gdf.columns.tolist())

print("\nFirst 5 rows:")
print(gdf.head())