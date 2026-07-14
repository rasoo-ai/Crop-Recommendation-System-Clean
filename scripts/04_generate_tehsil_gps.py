import pandas as pd
import geopandas as gpd
import re

print("="*60)
print("STEP 4 : GENERATE TEHSIL GPS")
print("="*60)

# ----------------------------
# Files
# ----------------------------

EXCEL = r"C:\Users\rasoo_me13iy3\OneDrive\Desktop\Tehsil_GPS_Projects\output\cleaned_data.xlsx"

BOUNDARY = r"C:\Users\rasoo_me13iy3\OneDrive\Desktop\Tehsil_GPS_Projects\data\SubDistricts_2011.geojsonl"

OUTPUT = r"C:\Users\rasoo_me13iy3\OneDrive\Desktop\Tehsil_GPS_Projects\output\Tehsil_GPS.xlsx"

# ----------------------------
# Cleaning function
# ----------------------------

def clean(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = re.sub(
        r'\b(tehsil|taluk|taluka|mandal|block|subdivision|sub-division|sub district)\b',
        '',
        text
    )

    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# ----------------------------
# Read data
# ----------------------------

print("\nReading crop dataset...")
df = pd.read_excel(EXCEL)

print("Reading boundaries...")
gdf = gpd.read_file(BOUNDARY)

# ----------------------------
# Calculate centroids
# ----------------------------

print("\nCalculating centroids...")

gdf = gdf.to_crs(32644)
gdf["centroid"] = gdf.geometry.centroid
gdf = gdf.set_geometry("centroid")
gdf = gdf.to_crs(4326)

gdf["Latitude"] = gdf.geometry.y
gdf["Longitude"] = gdf.geometry.x

# ----------------------------
# Clean names
# ----------------------------

df["State_clean"] = df["State_Name"].apply(clean)
df["District_clean"] = df["District_Name"].apply(clean)
df["Tehsil_clean"] = df["Tehsil_Name"].apply(clean)

gdf["State_clean"] = gdf["stname"].apply(clean)
gdf["District_clean"] = gdf["dtname"].apply(clean)
gdf["Tehsil_clean"] = gdf["sdtname"].apply(clean)

# ----------------------------
# Merge
# ----------------------------

merged = df.merge(
    gdf[
        [
            "State_clean",
            "District_clean",
            "Tehsil_clean",
            "Latitude",
            "Longitude"
        ]
    ],
    on=[
        "State_clean",
        "District_clean",
        "Tehsil_clean"
    ],
    how="left"
)

print("\nMatched :", merged["Latitude"].notna().sum())
print("Unmatched:", merged["Latitude"].isna().sum())

merged.to_excel(OUTPUT, index=False)

print("\nSaved:")
print(OUTPUT)