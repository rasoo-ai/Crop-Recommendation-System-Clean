import pandas as pd

crop = pd.read_excel("output/Crop_Normalized.xlsx")
ndvi = pd.read_csv("output/ALL_INDIA_TEHSIL_NDVI.csv")

# Clean all key columns
for df in [crop, ndvi]:
    for col in ["STATE_NORM", "DISTRICT_NORM", "TEHSIL_NORM"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.upper()
            .str.strip()
        )

crop_keys = set(zip(crop["STATE_NORM"], crop["DISTRICT_NORM"], crop["TEHSIL_NORM"]))
ndvi_keys = set(zip(ndvi["STATE_NORM"], ndvi["DISTRICT_NORM"], ndvi["TEHSIL_NORM"]))

matched = crop_keys & ndvi_keys
missing = crop_keys - ndvi_keys

print("="*60)
print("Crop unique keys :", len(crop_keys))
print("NDVI unique keys :", len(ndvi_keys))
print("Matched keys     :", len(matched))
print("Missing keys     :", len(missing))

print("\nFirst 30 missing keys:\n")

for x in list(missing)[:30]:
    print(x)
