import os
import sys
import pandas as pd
import numpy as np

print("=" * 78)
print("SMART KISAN - BENCHMARK CONSISTENCY AUDIT")
print("=" * 78)

print("""
SAFE AUDIT
- No .pkl model will be created.
- Existing model will NOT be changed.
- Streamlit will NOT be changed.
- No application files will be changed.
- This audit compares the preprocessing assumptions used by
  Steps 47-50.
""")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "output", "Crop_Normalized.xlsx")

print("=" * 78)
print("DATASET")
print("=" * 78)

print("Dataset path:", DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(DATA_PATH)

df = pd.read_excel(DATA_PATH)

print("Raw records:", len(df))
print("Raw columns:", len(df.columns))

print("\nRAW COLUMN LIST")
for i, col in enumerate(df.columns, 1):
    print(f"{i:02d}. {col}")

print("\n" + "=" * 78)
print("COLUMN TYPES")
print("=" * 78)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [
    c for c in df.columns
    if c not in numeric_cols
]

print("Numeric columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))

print("\nNUMERIC")
for c in numeric_cols:
    print(" ", c)

print("\nCATEGORICAL / NON-NUMERIC")
for c in categorical_cols:
    print(" ", c)

print("\n" + "=" * 78)
print("TARGET CANDIDATES")
print("=" * 78)

for c in df.columns:
    unique = df[c].nunique(dropna=True)
    if unique <= 20:
        print(f"{c:35} unique={unique}")

print("\n" + "=" * 78)
print("MISSING VALUES")
print("=" * 78)

missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("No missing values.")
else:
    print(missing.to_string())

print("\n" + "=" * 78)
print("POSSIBLE TARGET COLUMNS")
print("=" * 78)

target_candidates = []

for c in df.columns:
    name = str(c).lower()
    if any(x in name for x in [
        "crop",
        "target",
        "label",
        "class",
        "recommend"
    ]):
        target_candidates.append(c)

if target_candidates:
    for c in target_candidates:
        print(
            f"{c}: "
            f"dtype={df[c].dtype}, "
            f"unique={df[c].nunique(dropna=True)}"
        )
else:
    print("No obvious target column found.")

print("\n" + "=" * 78)
print("BENCHMARK-LIKE CROP FILTER")
print("=" * 78)

KNOWN_CROPS = [
    "Apple",
    "Cotton",
    "Maize",
    "Mustard",
    "Pulses",
    "Rice",
    "Vegetables",
    "Walnut",
    "Wheat",
]

for c in target_candidates:
    values = set(
        df[c].dropna().astype(str).str.strip().unique()
    )

    matches = sorted(values.intersection(KNOWN_CROPS))

    if matches:
        print("Candidate target:", c)
        print("Recognized crops:", matches)

        filtered = df[
            df[c].astype(str).str.strip().isin(KNOWN_CROPS)
        ].copy()

        print("Filtered records:", len(filtered))

        print("\nCrop distribution:")
        print(
            filtered[c]
            .astype(str)
            .str.strip()
            .value_counts()
            .sort_index()
            .to_string()
        )

print("\n" + "=" * 78)
print("CRITICAL COMPARISON")
print("=" * 78)

print("""
Expected Step 47-49 benchmark:
  Records                 : 5,644
  Features                : 18
  Numeric features        : 16
  Categorical features    : 2
  Processed features      : 216
  Training records        : 4,515
  Testing records         : 1,129
  Balanced training       : 19,593

Step 50 reported:
  Records                 : 5,636
  Features                : 30
  Numeric features        : 18
  Categorical features    : 12
  Processed features      : 16,578
  Training records        : 4,508
  Testing records         : 1,128
  Balanced training       : 19,557
""")

print("=" * 78)
print("AUDIT CONCLUSION")
print("=" * 78)

print("""
If the current dataset contains columns that were not used by
Steps 47-49, Step 50 must NOT be used as a replacement benchmark.

The next development step should restore one canonical preprocessing
pipeline and make Steps 51+ use exactly the same:
  - target column
  - feature columns
  - crop filter
  - missing-value handling
  - train/test split
  - encoder
  - processed feature count
  - oversampling procedure
  - Random Forest configuration

No deployment decision should be made until this consistency issue
is resolved.
""")

print("=" * 78)
print("AUDIT COMPLETE")
print("=" * 78)