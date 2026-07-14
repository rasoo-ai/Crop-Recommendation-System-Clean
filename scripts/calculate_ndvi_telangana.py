import geopandas as gpd
from pystac_client import Client
import planetary_computer
from odc.stac import load
import rioxarray

print("=" * 60)
print("STEP 14D : CALCULATE NDVI")
print("=" * 60)

# Load Telangana boundary
gdf = gpd.read_file("output/TELANGANA_tehsils.geojson")

# Get bounding box
bbox = gdf.total_bounds.tolist()

# Connect to Planetary Computer
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Search for Sentinel-2 imagery
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-03-01/2025-03-31",
    query={"eo:cloud_cover": {"lt": 10}},
)

items = list(search.items())

if len(items) == 0:
    raise Exception("No Sentinel-2 images found!")

scene = items[0]

print("Loading B04 and B08...")

data = load(
    [scene],
    bands=["B04", "B08"],
    bbox=bbox,
    resolution=30,   # Lower resolution to reduce memory usage
)

# Calculate NDVI
ndvi = (data.B08 - data.B04) / (data.B08 + data.B04)

# Save output
output_file = "output/TELANGANA_NDVI.tif"
ndvi.rio.to_raster(output_file)

print(f"NDVI saved successfully: {output_file}")
print("=" * 60)
print("STEP 14D COMPLETED")
print("=" * 60)