import pandas as pd
import geopandas as gpd
import os

print("=" * 60)
print("STEP 09 : NORMALIZATION (CLEAN)")
print("=" * 60)

crop = pd.read_excel("output/cleaned_data.xlsx")
boundary = gpd.read_file("data/SubDistricts_2011.geojsonl")

def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).upper().strip()
    for c in "&.,-/()'":
        text = text.replace(c, " ")
    text = text.replace(" DISTRICT", "")
    text = text.replace(" TEHSIL", "")
    text = " ".join(text.split())
    return text

# crop
crop["STATE_NORM"] = crop["State_Name"].apply(normalize)
crop["DISTRICT_NORM"] = crop["District_Name"].apply(normalize)
crop["TEHSIL_NORM"] = crop["Tehsil_Name"].apply(normalize)

# boundary
boundary["STATE_NORM"] = boundary["stname"].apply(normalize)
boundary["DISTRICT_NORM"] = boundary["dtname"].apply(normalize)
boundary["TEHSIL_NORM"] = boundary["sdtname"].apply(normalize)

os.makedirs("output", exist_ok=True)

crop.to_excel("output/crop_normalized.xlsx", index=False)
boundary.to_file("output/boundary_normalized.geojson", driver="GeoJSON")

print("STEP 9 DONE")
