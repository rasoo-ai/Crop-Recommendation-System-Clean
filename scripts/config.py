"""
Project Configuration
Tehsils_GPS_Projects
"""

from pathlib import Path

# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# Input Data
# ============================================================

BOUNDARY_FILE = PROJECT_ROOT / "boundary_normalized.geojson"

# ============================================================
# Output Directories
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "output"

NDVI_DIR = OUTPUT_DIR / "ndvi"

LOG_DIR = PROJECT_ROOT / "logs"

TEMP_DIR = PROJECT_ROOT / "temp"

# Create directories automatically
for directory in [OUTPUT_DIR, NDVI_DIR, LOG_DIR, TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# Sentinel-2 Configuration
# ============================================================

COLLECTION = "sentinel-2-l2a"

START_DATE = "2025-01-01"

END_DATE = "2025-12-31"

MAX_CLOUD_COVER = 20

RESOLUTION = 30

# ============================================================
# Dask Configuration
# ============================================================

CHUNK_SIZE = 1024

# ============================================================
# Tile Configuration
# ============================================================

TILE_SIZE = 0.5

# ============================================================
# Retry Configuration
# ============================================================

MAX_RETRIES = 3

RETRY_WAIT = 10

# ============================================================
# Memory
# ============================================================

MEMORY_LIMIT_GB = 8

GC_INTERVAL = 1

# ============================================================
# Compression
# ============================================================

COMPRESSION = "LZW"

BIGTIFF = "IF_SAFER"

# ============================================================
# Logging
# ============================================================

LOG_FILE = LOG_DIR / "generate_ndvi.log"

LOG_LEVEL = "INFO"
print("Config file loaded successfully")
