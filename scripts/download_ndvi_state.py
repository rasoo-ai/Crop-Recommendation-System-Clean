import geopandas as gpd
from pystac_client import Client
import planetary_computer

print("=" * 60)
print("STEP 14C : DOWNLOAD SENTINEL-2 FOR ONE STATE")
print("=" * 60)

# Load Telangana boundary
gdf = gpd.read_file("output/TELANGANA_tehsils.geojson")

# Get bounding box
bbox = gdf.total_bounds.tolist()

print("Bounding Box:")
print(bbox)

# Connect to Planetary Computer
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

print("\nSearching Sentinel-2...")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-03-01/2025-03-31",
    query={"eo:cloud_cover": {"lt": 10}},
)

items = list(search.items())

print(f"Scenes Found: {len(items)}")

if len(items) == 0:
    print("No Sentinel-2 images found.")
    exit()

scene = items[0]

print("\nSelected Scene")
print("-" * 40)
print("ID:", scene.id)
print("Date:", scene.datetime)
print("Cloud Cover:", scene.properties["eo:cloud_cover"])

print("\nAvailable Bands:")
for band in scene.assets.keys():
    print("-", band)

print("=" * 60)
print("STEP 14C COMPLETED")
print("=" * 60)