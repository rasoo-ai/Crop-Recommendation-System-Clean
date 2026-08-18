import pandas as pd
import geopandas as gpd
import re

print("="*60)
print("STEP 5 : IMPROVED TEHSIL MATCHING")
print("="*60)

# ----------------------------------------------------
# FILES
# ----------------------------------------------------

DATA = r"C:\Users\rasoo_me13iy3\OneDrive\Desktop\Tehsil_GPS_Projects\output\cleaned_data.xlsx"

BOUNDARY = r"C:\Users\rasoo_me13iy3\OneDrive\Desktop\Tehsil_GPS_Projects\data\SubDistricts_2011.geojsonl"

OUTPUT = r"C:\Users\rasoo_me13iy3\OneDrive\Desktop\Tehsil_GPS_Projects\output\Tehsil_GPS_Improved.xlsx"

# ----------------------------------------------------
# CLEAN FUNCTION
# ----------------------------------------------------

def clean(x):

    if pd.isna(x):
        return ""

    x = str(x).lower().strip()

    x = x.replace(".", " ")

    x = re.sub(
        r'\b(tehsil|taluk|taluka|mandal|block|subdivision|sub district)\b',
        '',
        x
    )

    x = re.sub(r'[^a-z0-9 ]', '', x)
    x = re.sub(r'\s+', ' ', x)

    return x.strip()

# ----------------------------------------------------
# READ FILES
# ----------------------------------------------------

print("Reading crop dataset...")
df = pd.read_excel(DATA)

print("Reading boundaries...")
gdf = gpd.read_file(BOUNDARY)

# ----------------------------------------------------
# CENTROIDS
# ----------------------------------------------------

gdf = gdf.to_crs(32644)

gdf["centroid"] = gdf.geometry.centroid

gdf = gdf.set_geometry("centroid")

gdf = gdf.to_crs(4326)

gdf["Latitude"] = gdf.geometry.y
gdf["Longitude"] = gdf.geometry.x

# ----------------------------------------------------
# CLEAN NAMES
# ----------------------------------------------------

df["State_clean"] = df["State_Name"].apply(clean)
df["Tehsil_clean"] = df["Tehsil_Name"].apply(clean)

gdf["State_clean"] = gdf["stname"].apply(clean)
gdf["Tehsil_clean"] = gdf["sdtname"].apply(clean)

# ----------------------------------------------------
# REMOVE DUPLICATES
# ----------------------------------------------------

gps = gdf[
    ["State_clean", "Tehsil_clean", "Latitude", "Longitude"]
].drop_duplicates()

# ----------------------------------------------------
# MERGE ONLY BY STATE + TEHSIL
# ----------------------------------------------------

merged = df.merge(
    gps,
    on=["State_clean", "Tehsil_clean"],
    how="left"
)

print()

print("Matched :", merged["Latitude"].notna().sum())

print("Unmatched :", merged["Latitude"].isna().sum())

merged.to_excel(OUTPUT, index=False)

print()

print("Saved to")

print(OUTPUT)
