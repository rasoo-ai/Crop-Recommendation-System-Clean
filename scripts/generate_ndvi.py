"""
generate_all_states_ndvi.py
Memory Optimized NDVI Generator
Microsoft Planetary Computer + odc.stac

Author : Your Name
"""

import gc
import logging
from pathlib import Path

import geopandas as gpd
import planetary_computer
import pystac_client

import odc.stac
import rioxarray

from config import *

# -------------------------------------------------------
# Logging
# -------------------------------------------------------

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

print("=" * 60)
print("MICROSOFT PLANETARY COMPUTER NDVI GENERATOR")
print("=" * 60)

# -------------------------------------------------------
# STAC Connection
# -------------------------------------------------------

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

print("✓ Connected to Planetary Computer")

# -------------------------------------------------------
# Find all State GeoJSON files
# -------------------------------------------------------

STATE_FILES = sorted(
    OUTPUT_DIR.glob("*_tehsils.geojson")
)

print(f"\nFound {len(STATE_FILES)} State Files\n")

# -------------------------------------------------------
# Loop over States
# -------------------------------------------------------

for state_file in STATE_FILES:

    state_name = (
        state_file.stem
        .replace("_tehsils", "")
    )

    print("-" * 60)
    print(state_name)

    output_tif = NDVI_DIR / f"{state_name}_ndvi.tif"

    if output_tif.exists():

        print("Already Generated")

        continue

    logging.info(f"Processing {state_name}")

    gdf = gpd.read_file(state_file)

    gdf = gdf.to_crs(4326)

    bbox = tuple(gdf.total_bounds)

    print("Searching Sentinel-2 ...")

    search = catalog.search(

        collections=[COLLECTION],

        bbox=bbox,

        datetime=f"{START_DATE}/{END_DATE}",

        query={
            "eo:cloud_cover": {
                "lt": MAX_CLOUD_COVER
            }
        }

    )

    items = list(search.items())

    print(f"Images Found : {len(items)}")

    if len(items) == 0:

        print("No Images")

        continue

# ==========================================================
# Load Sentinel-2 using odc.stac
# ==========================================================

print("Loading Sentinel-2...")

try:

    ds = odc.stac.load(

        items,

        bands=["red", "nir"],

        bbox=bbox,

        resolution=30,

        chunks={
            "x": CHUNK_SIZE,
            "y": CHUNK_SIZE
        },

        groupby="solar_day"

    )

except Exception as e:

    print(e)

    continue

print("Sentinel Loaded")

# ==========================================================
# Median Composite
# ==========================================================

print("Creating Median Composite...")

ds = ds.median(dim="time", skipna=True)

# ==========================================================
# NDVI
# ==========================================================

print("Calculating NDVI...")

ndvi = (
    (ds.nir - ds.red)
    /
    (ds.nir + ds.red)
)

ndvi = ndvi.rename("NDVI")

print("NDVI Created")
# ==========================================================
# Clip to State Boundary
# ==========================================================

print("Clipping...")

ndvi = ndvi.rio.write_crs("EPSG:4326")

ndvi = ndvi.rio.clip(
    gdf.geometry,
    gdf.crs,
    drop=True
)

print("Clip Complete")
# ==========================================================
# Save GeoTIFF
# ==========================================================

print("Saving...")

ndvi.rio.to_raster(

    output_tif,

    compress=COMPRESSION,

    tiled=True

)

print(f"Saved : {output_tif}")
# ==========================================================
# Cleanup
# ==========================================================

del ds
del ndvi
del items
del gdf

gc.collect()

print("Memory Cleared\n")