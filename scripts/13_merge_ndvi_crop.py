import pandas as pd
from pathlib import Path

print("=" * 60)
print("MERGING CROP DATASET WITH NDVI")
print("=" * 60)

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

crop_file = PROJECT_ROOT / "output" / "Crop_Normalized.xlsx"
ndvi_file = PROJECT_ROOT / "output" / "ALL_INDIA_TEHSIL_NDVI.csv"
output_file = PROJECT_ROOT / "output" / "FINAL_CROP_DATASET_WITH_NDVI.xlsx"

# ==========================================================
# Read Data
# ==========================================================

crop = pd.read_excel(crop_file)
ndvi = pd.read_csv(ndvi_file)

print(f"Crop Rows : {len(crop)}")
print(f"NDVI Rows : {len(ndvi)}")

# ==========================================================
# Keep only required NDVI columns
# ==========================================================

ndvi = ndvi[
    [
        "STATE_NORM",
        "DISTRICT_NORM",
        "TEHSIL_NORM",
        "MEAN_NDVI",
    ]
]

# Remove duplicate NDVI records
ndvi = ndvi.drop_duplicates(
    subset=["STATE_NORM", "DISTRICT_NORM", "TEHSIL_NORM"]
)

print(f"NDVI Rows After Removing Duplicates : {len(ndvi)}")

# ==========================================================
# Merge
# ==========================================================

merged = crop.merge(
    ndvi,
    on=["STATE_NORM", "DISTRICT_NORM", "TEHSIL_NORM"],
    how="left"
)

# ==========================================================
# Report
# ==========================================================

missing = merged["MEAN_NDVI"].isna().sum()

print("\nMerge Completed")
print(f"Rows           : {len(merged)}")
print(f"Columns        : {len(merged.columns)}")
print(f"Missing NDVI   : {missing}")

# ==========================================================
# Save
# ==========================================================

merged.to_excel(output_file, index=False)

print(f"\nSaved Successfully:\n{output_file}")
