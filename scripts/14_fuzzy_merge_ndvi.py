import pandas as pd
from rapidfuzz import process, fuzz

# Read data
crop = pd.read_excel("output/Crop_Normalized.xlsx")
ndvi = pd.read_csv("output/ALL_INDIA_TEHSIL_NDVI.csv")

# Clean names
for df in [crop, ndvi]:
    for col in ["STATE_NORM", "DISTRICT_NORM", "TEHSIL_NORM"]:
        df[col] = df[col].astype(str).str.upper().str.strip()

# Exact merge first
merged = crop.merge(
    ndvi[["STATE_NORM","DISTRICT_NORM","TEHSIL_NORM","MEAN_NDVI"]],
    on=["STATE_NORM","DISTRICT_NORM","TEHSIL_NORM"],
    how="left"
)

print("Exact Matches :", merged["MEAN_NDVI"].notna().sum())

# Remaining unmatched
missing = merged[merged["MEAN_NDVI"].isna()].copy()

print("Need Fuzzy :", len(missing))

matches = []

for idx,row in missing.iterrows():

    subset = ndvi[
        ndvi["STATE_NORM"]==row["STATE_NORM"]
    ]

    if len(subset)==0:
        continue

    choices = subset["TEHSIL_NORM"].tolist()

    result = process.extractOne(
        row["TEHSIL_NORM"],
        choices,
        scorer=fuzz.ratio
    )

    if result and result[1] >= 90:

        matched_name = result[0]

        ndvi_row = subset[
            subset["TEHSIL_NORM"]==matched_name
        ].iloc[0]

        merged.loc[idx,"MEAN_NDVI"] = ndvi_row["MEAN_NDVI"]

print("Matched After Fuzzy :", merged["MEAN_NDVI"].notna().sum())

merged.to_excel(
    "output/FINAL_CROP_DATASET_WITH_NDVI.xlsx",
    index=False
)

print("Done.")