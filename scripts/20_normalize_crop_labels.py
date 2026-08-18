import pandas as pd

INPUT_FILE = "output/Crop_Normalized_BEFORE_LABEL_CLEANUP.xlsx"
OUTPUT_FILE = "output/Crop_Normalized.xlsx"

df = pd.read_excel(INPUT_FILE)

# Convert crop column to string while preserving missing values
df["Crop"] = df["Crop"].astype("string").str.strip()

# Treat text versions of missing values as missing
missing_values = {
    "",
    "nan",
    "NaN",
    "None",
    "NULL",
    "null",
}

df["Crop"] = df["Crop"].replace(
    list(missing_values),
    pd.NA
)

# Normalize duplicate crop names
crop_mapping = {
    "Paddy (Rice)": "Rice",
    "Oilseeds (Mustard)": "Mustard",
    "Pulses (Arhar)": "Pulses",
    "pulses": "Pulses",
    "Sugar Cane": "Sugarcane",
}

df["Crop"] = df["Crop"].replace(
    crop_mapping
)

# Remove rows without a crop label
df = df[
    df["Crop"].notna()
].copy()

# Remove clearly invalid numeric-looking crop label
df = df[
    df["Crop"] != "86.09"
].copy()

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("=" * 60)
print("CROP LABEL CLEANUP COMPLETED")
print("=" * 60)

print("\nClean crop distribution:")
print(
    df["Crop"]
    .value_counts()
    .to_string()
)

print(
    "\nMissing crop labels:",
    df["Crop"].isna().sum()
)

print(
    "Total rows:",
    len(df)
)
