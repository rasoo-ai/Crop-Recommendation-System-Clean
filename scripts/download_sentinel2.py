from pystac_client import Client
import planetary_computer as pc
import geopandas as gpd

print("=" * 60)
print("STEP 13 : DOWNLOAD SENTINEL-2")
print("=" * 60)

# -------------------------------------------------------
# Load study area boundary
# -------------------------------------------------------
print("Loading boundary...")

BOUNDARY_FILE = PROJECT_ROOT / "output" / "boundary_normalized.geojson"

# Get bounding box
bbox = boundary.total_bounds.tolist()

print("Bounding Box:")
print(bbox)

# -------------------------------------------------------
# Connect to Microsoft Planetary Computer
# -------------------------------------------------------
print("\nConnecting to Microsoft Planetary Computer...")

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# -------------------------------------------------------
# Search Sentinel-2 imagery
# -------------------------------------------------------
print("Searching Sentinel-2 scenes...")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2025-01-01/2025-03-31",
    query={"eo:cloud_cover": {"lt": 10}},
)

items = list(search.items())

print(f"\nScenes Found: {len(items)}")

if len(items) == 0:
    raise Exception("No Sentinel-2 images found for the selected area and dates.")

# -------------------------------------------------------
# Select first scene
# -------------------------------------------------------
scene = pc.sign(items[0])

print("\nSelected Scene")
print("-" * 40)
print("ID          :", scene.id)
print("Date        :", scene.datetime)
print("Cloud Cover :", scene.properties.get("eo:cloud_cover"))

print("\nAvailable Bands:")

for band in scene.assets.keys():
    print(" -", band)

print("\n" + "=" * 60)
print("STEP 13 COMPLETED SUCCESSFULLY")
print("=" * 60)
