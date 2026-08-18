import pandas as pd
import geopandas as gpd
from rapidfuzz import process, fuzz

print("=" * 60)
print("STEP 6 : FUZZY MATCHING")
print("=" * 60)

# --------------------------------------------------
# File paths
# --------------------------------------------------
crop_file = r"C:\Users\rasoo_me13iy3\OneDrive\Desktop\Tehsil_GPS_Projects\output\Tehsil_GPS_Improved.xlsx"
boundary_file = r"C:\Users\rasoo_me13iy3\OneDrive\Desktop\Tehsil_GPS_Projects\data\SubDistricts_2011.geojsonl"

print("Reading crop dataset...")
crop = pd.read_excel(crop_file)

print("Reading boundary dataset...")
gdf = gpd.read_file(boundary_file)

# --------------------------------------------------
# Compute centroid coordinates
# --------------------------------------------------
print("Calculating centroids...")

gdf = gdf.to_crs(epsg=4326)

gdf["Latitude"] = gdf.geometry.centroid.y
gdf["Longitude"] = gdf.geometry.centroid.x

# --------------------------------------------------
# Remove blank tehsil names
# --------------------------------------------------
gdf["sdtname"] = gdf["sdtname"].astype(str).str.strip()
gdf = gdf[gdf["sdtname"] != ""]

# --------------------------------------------------
# Detect crop columns
# --------------------------------------------------
tehsil_col = None

for col in crop.columns:
    c = col.lower()
    if "tehsil" in c:
        tehsil_col = col
        break

if tehsil_col is None:
    raise Exception("Tehsil column not found.")

lat_col = "Latitude"
lon_col = "Longitude"

print("Tehsil column :", tehsil_col)
print("Latitude column :", lat_col)
print("Longitude column:", lon_col)

boundary_names = gdf["sdtname"].tolist()

matched = 0

print("\nRunning fuzzy matching...\n")

for idx in crop.index:

    if pd.notna(crop.at[idx, lat_col]):
        continue

    query = str(crop.at[idx, tehsil_col]).strip()

    if query == "":
        continue

    result = process.extractOne(
        query,
        boundary_names,
        scorer=fuzz.token_sort_ratio
    )

    if result is None:
        continue

    best_name, score, _ = result

    if score >= 90:

        row = gdf[gdf["sdtname"] == best_name].iloc[0]

        crop.at[idx, lat_col] = row["Latitude"]
        crop.at[idx, lon_col] = row["Longitude"]

        matched += 1

remaining = crop[lat_col].isna().sum()

print("=" * 60)
print("New fuzzy matches :", matched)
print("Still unmatched   :", remaining)
print("=" * 60)

output = r"C:\Users\rasoo_me13iy3\OneDrive\Desktop\Tehsil_GPS_Projects\output\Tehsil_GPS_Final.xlsx"

crop.to_excel(output, index=False)

print("\nSaved to:")
print(output)
