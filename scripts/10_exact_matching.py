import pandas as pd
import geopandas as gpd
import os

print("=" * 60)
print("STEP 10 : EXACT MATCHING")
print("=" * 60)

# -------------------------------------------------------
# Read datasets
# -------------------------------------------------------
print("Reading normalized datasets...")

crop = pd.read_excel("output/crop_normalized.xlsx")
boundary = gpd.read_file("output/boundary_normalized.geojson")

# -------------------------------------------------------
# Prepare boundary dataset
# -------------------------------------------------------
print("Preparing boundary dataset...")

boundary = boundary[
    ["STATE_NORM", "DISTRICT_NORM", "TEHSIL_NORM", "geometry"]
].copy()

# centroid coordinates
# Reproject to metric CRS for accurate centroid
boundary_projected = boundary.to_crs(epsg=3857)
 
centroids = boundary_projected.geometry.centroid

# Convert back to lat/lon
boundary["Longitude"] = centroids.to_crs(epsg=4326).x
boundary["Latitude"] = centroids.to_crs(epsg=4326).y

# -------------------------------------------------------
# Exact matching (ONLY ONCE)
# -------------------------------------------------------
print("Performing exact matching...")

merged = crop.merge(
    boundary,
    on=["STATE_NORM", "DISTRICT_NORM", "TEHSIL_NORM"],
    how="left"
)

# -------------------------------------------------------
# Match statistics
# -------------------------------------------------------
matched = merged["Latitude"].notna().sum()
unmatched = merged["Latitude"].isna().sum()

print()
print("Matched :", matched)
print("Unmatched :", unmatched)

# -------------------------------------------------------
# Create GeoDataFrame (FINAL STEP)
# -------------------------------------------------------
print("Creating GeoDataFrame...")

gdf = gpd.GeoDataFrame(
    merged,
    geometry=gpd.points_from_xy(merged["Longitude"], merged["Latitude"]),
    crs="EPSG:4326"
)

# -------------------------------------------------------
# Save outputs
# -------------------------------------------------------
os.makedirs("output", exist_ok=True)

print("Saving final GIS-ready dataset...")

gdf.to_excel("output/Tehsil_GPS_Exact.xlsx", index=False)
gdf.to_file("output/Tehsil_GPS_Exact.geojson", driver="GeoJSON")

# -------------------------------------------------------
# Save unmatched separately
# -------------------------------------------------------
unmatched_df = merged[merged["Latitude"].isna()]
unmatched_df.to_excel("output/Still_Unmatched.xlsx", index=False)

print()
print("Files saved successfully:")
print("output/Tehsil_GPS_Exact.xlsx")
print("output/Tehsil_GPS_Exact.geojson")
print("output/Still_Unmatched.xlsx")
