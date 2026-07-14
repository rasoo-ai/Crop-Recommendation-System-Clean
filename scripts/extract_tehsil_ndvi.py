import geopandas as gpd
import rioxarray
from rasterstats import zonal_stats
import pandas as pd

print("=" * 60)
print("STEP 14E : EXTRACT TEHSIL NDVI")
print("=" * 60)

# Load Telangana tehsils
gdf = gpd.read_file("output/TELANGANA_tehsils.geojson")

# Load NDVI raster
ndvi = rioxarray.open_rasterio("output/TELANGANA_NDVI.tif").squeeze()

# Calculate mean NDVI for each tehsil
stats = zonal_stats(
    gdf,
    ndvi.values,
    affine=ndvi.rio.transform(),
    stats=["mean"],
    nodata=ndvi.rio.nodata
)

# Add NDVI values to GeoDataFrame
gdf["Mean_NDVI"] = [s["mean"] for s in stats]

# Save CSV
output_csv = "output/TELANGANA_tehsil_ndvi.csv"
gdf.drop(columns="geometry").to_csv(output_csv, index=False)

print(f"CSV saved successfully: {output_csv}")
print("=" * 60)
print("STEP 14E COMPLETED")
print("=" * 60)