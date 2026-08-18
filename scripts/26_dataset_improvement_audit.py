import pandas as pd


FILE = "output/Crop_Normalized.xlsx"

df = pd.read_excel(FILE)


# ==========================================================
# BASIC INFORMATION
# ==========================================================

print("=" * 80)
print("DATASET IMPROVEMENT AUDIT")
print("=" * 80)

print(f"\nTotal rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")


# ==========================================================
# CROP COUNTS
# ==========================================================

print("\n" + "=" * 80)
print("CROP COUNTS")
print("=" * 80)

crop_counts = (
    df["Crop"]
    .value_counts(dropna=False)
)

print(crop_counts.to_string())


# ==========================================================
# STATE + CROP COUNTS
# ==========================================================

print("\n" + "=" * 80)
print("STATE + CROP COUNTS")
print("=" * 80)

state_crop = (
    df.groupby(
        ["State_Name", "Crop"],
        dropna=False
    )
    .size()
    .reset_index(name="Records")
    .sort_values(
        ["State_Name", "Records"],
        ascending=[True, False]
    )
)

print(state_crop.to_string(index=False))


# ==========================================================
# TEHSIL COVERAGE
# ==========================================================

print("\n" + "=" * 80)
print("TEHSIL COVERAGE BY CROP")
print("=" * 80)

tehsil_coverage = (
    df.groupby("Crop")["Tehsil_Name"]
    .nunique()
    .sort_values()
)

print(tehsil_coverage.to_string())


# ==========================================================
# MISSING VALUES
# ==========================================================

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

missing = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
)

print(missing.to_string())


# ==========================================================
# DUPLICATES
# ==========================================================

print("\n" + "=" * 80)
print("DUPLICATES")
print("=" * 80)

print(
    "Duplicate complete rows:",
    df.duplicated().sum()
)


# ==========================================================
# NUMERIC RANGES
# ==========================================================

numeric_cols = [
    "pH_Value",
    "Nitrogen_Value (N)",
    "Phosphorus_Value (P)",
    "Potassium_Value (K)",
    "Electrical_Conductivity (EC)",
    "Organic_Carbon (%)",
    "Soil_Moisture (%)",
    "Zinc (%)",
    "Iron (%)",
    "Manganese (%)",
    "Copper (%)",
    "Boron (%)",
    "Sulphur (%)",
    "Rainfall_cm",
    "temperature_celsius",
    "humidity_percentage",
]


print("\n" + "=" * 80)
print("NUMERIC RANGES")
print("=" * 80)

print(
    df[numeric_cols]
    .describe()
    .T[
        ["min", "max", "mean", "std"]
    ]
    .to_string()
)


# ==========================================================
# TARGETED WEAK AREAS
# ==========================================================

target_states = [
    "Jharkhand",
    "Jammu And Kashmir",
]

target_crops = [
    "Maize",
    "Mustard",
    "Pulses",
    "Rice",
    "Wheat",
    "Apple",
    "Walnut",
    "Vegetables",
]


print("\n" + "=" * 80)
print("TARGETED WEAK AREAS")
print("=" * 80)

target_df = df[
    df["State_Name"].isin(target_states)
    & df["Crop"].isin(target_crops)
].copy()

target_summary = (
    target_df
    .groupby(
        ["State_Name", "Crop"]
    )
    .size()
    .reset_index(name="Records")
    .sort_values(
        ["State_Name", "Records"]
    )
)

print(
    target_summary.to_string(
        index=False
    )
)


# ==========================================================
# SMALL STATE/CROP GROUPS
# ==========================================================

print("\n" + "=" * 80)
print("STATE + CROP GROUPS WITH FEWER THAN 10 RECORDS")
print("=" * 80)

small_groups = state_crop[
    state_crop["Records"] < 10
]

if small_groups.empty:
    print("None")
else:
    print(
        small_groups.to_string(
            index=False
        )
    )


# ==========================================================
# SAVE REPORTS
# ==========================================================

state_crop.to_excel(
    "output/State_Crop_Record_Counts.xlsx",
    index=False
)

tehsil_coverage.to_excel(
    "output/Crop_Tehsil_Coverage.xlsx"
)

missing.to_excel(
    "output/Dataset_Missing_Values.xlsx"
)

print("\n" + "=" * 80)
print("AUDIT COMPLETED")
print("=" * 80)

print(
    "\nSaved:"
    "\n- output/State_Crop_Record_Counts.xlsx"
    "\n- output/Crop_Tehsil_Coverage.xlsx"
    "\n- output/Dataset_Missing_Values.xlsx"
)
