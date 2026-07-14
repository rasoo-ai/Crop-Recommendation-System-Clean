import pandas as pd
import geopandas as gpd

print("=" * 60)
print("STEP 12 : FINAL DATASET CREATION")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
input_file = "output/Tehsil_GPS_Improved.xlsx"  # change if needed

df = pd.read_excel(input_file)

print("Data loaded successfully")
print("Shape:", df.shape)
print("Columns:", list(df.columns))

import geopandas as gpd

print("=" * 60)
print("STEP 12 : FINAL EXPORT ONLY")
print("=" * 60)

# Load already-clean GIS output from Step 10
gdf = gpd.read_file("output/Tehsil_GPS_Exact.geojson")

# Save final outputs
gdf.to_file("output/FINAL_Tehsil_GPS.geojson", driver="GeoJSON")
gdf.to_excel("output/FINAL_Tehsil_GPS.xlsx", index=False)

print("\n============================================================")
print("FINAL EXPORT COMPLETED SUCCESSFULLY")
print("Excel  : output/FINAL_Tehsil_GPS.xlsx")
print("GeoJSON: output/FINAL_Tehsil_GPS.geojson")
print("Rows   :", len(gdf))
print("============================================================")