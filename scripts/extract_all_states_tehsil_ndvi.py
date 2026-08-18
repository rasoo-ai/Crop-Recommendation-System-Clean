import pandas as pd
from pathlib import Path

print("=" * 60)
print("MERGING ALL STATE NDVI FILES")
print("=" * 60)

OUTPUT_DIR = Path("output")

csv_files = sorted(OUTPUT_DIR.glob("*_tehsil_ndvi.csv"))

all_data = []

for csv_file in csv_files:

    # Skip previously merged files
    if csv_file.name.upper().startswith("ALL_INDIA"):
        continue

    print(f"Reading : {csv_file.name}")

    df = pd.read_csv(csv_file)

    all_data.append(df)

# Merge all state CSVs
final = pd.concat(all_data, ignore_index=True)

# Remove duplicate tehsils
final = final.drop_duplicates(
    subset=["STATE_NORM", "DISTRICT_NORM", "TEHSIL_NORM"]
)

print("\nTotal Rows :", len(final))

# Save merged file
final.to_csv(
    OUTPUT_DIR / "ALL_INDIA_TEHSIL_NDVI.csv",
    index=False
)

print("\nSaved : output/ALL_INDIA_TEHSIL_NDVI.csv")
print("=" * 60)
