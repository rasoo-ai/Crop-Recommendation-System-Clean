import pandas as pd

FILE = "output/Crop_Normalized.xlsx"

df = pd.read_excel(FILE)

print("=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)


# ==========================================================
# 1. MISSING SULPHUR
# ==========================================================

print("\n" + "=" * 80)
print("MISSING SULPHUR RECORDS")
print("=" * 80)

sulphur_missing = df[
    df["Sulphur (%)"].isna()
].copy()

print(
    "Missing Sulphur records:",
    len(sulphur_missing)
)

cols = [
    "State_Name",
    "District_Name",
    "Tehsil_Name",
    "Crop",
    "Soil_Type",
    "Rainfall_cm",
    "temperature_celsius",
    "humidity_percentage",
    "Sulphur (%)",
]

available_cols = [
    c for c in cols
    if c in sulphur_missing.columns
]

print(
    sulphur_missing[
        available_cols
    ].to_string(index=False)
)


# ==========================================================
# 2. RAINFALL ABOVE 1000 CM
# ==========================================================

print("\n" + "=" * 80)
print("HIGH RAINFALL RECORDS (>1000 CM)")
print("=" * 80)

high_rain = df[
    pd.to_numeric(
        df["Rainfall_cm"],
        errors="coerce"
    ) > 1000
].copy()

print(
    "Records above 1000 cm:",
    len(high_rain)
)

rain_cols = [
    "State_Name",
    "District_Name",
    "Tehsil_Name",
    "Crop",
    "Rainfall_cm",
    "temperature_celsius",
    "humidity_percentage",
]

available_rain_cols = [
    c for c in rain_cols
    if c in high_rain.columns
]

print(
    high_rain[
        available_rain_cols
    ]
    .sort_values(
        "Rainfall_cm",
        ascending=False
    )
    .to_string(index=False)
)


# ==========================================================
# 3. RAINFALL ABOVE 2000 CM
# ==========================================================

print("\n" + "=" * 80)
print("VERY HIGH RAINFALL RECORDS (>2000 CM)")
print("=" * 80)

very_high_rain = df[
    pd.to_numeric(
        df["Rainfall_cm"],
        errors="coerce"
    ) > 2000
].copy()

print(
    "Records above 2000 cm:",
    len(very_high_rain)
)

if not very_high_rain.empty:

    print(
        very_high_rain[
            available_rain_cols
        ]
        .sort_values(
            "Rainfall_cm",
            ascending=False
        )
        .to_string(index=False)
    )


# ==========================================================
# 4. RAINFALL STATISTICS
# ==========================================================

print("\n" + "=" * 80)
print("RAINFALL SUMMARY")
print("=" * 80)

rainfall = pd.to_numeric(
    df["Rainfall_cm"],
    errors="coerce"
)

print(
    rainfall.describe().to_string()
)

print(
    "\nTop 20 rainfall values:"
)

print(
    rainfall
    .sort_values(
        ascending=False
    )
    .head(20)
    .to_string(
        index=False
    )
)


# ==========================================================
# 5. SULPHUR BY STATE / CROP
# ==========================================================

print("\n" + "=" * 80)
print("MISSING SULPHUR BY STATE")
print("=" * 80)

print(
    sulphur_missing[
        "State_Name"
    ]
    .value_counts()
    .to_string()
)

print("\nMissing Sulphur by crop:")

print(
    sulphur_missing[
        "Crop"
    ]
    .value_counts()
    .to_string()
)


print("\n" + "=" * 80)
print("CHECK COMPLETED")
print("=" * 80)
