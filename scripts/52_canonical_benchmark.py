# ============================================================
# SMART KISAN - CANONICAL BENCHMARK PIPELINE
# ============================================================
#
# SAFE BENCHMARK
# - Does NOT create a .pkl model
# - Does NOT modify the Streamlit application
# - Does NOT overwrite an existing model
# - Establishes ONE canonical preprocessing pipeline
# - This pipeline should be reused by Steps 53+
#
# IMPORTANT:
# Soil_Moisture (%) may be stored as object/string in Excel.
# We explicitly convert all canonical numeric features with
# pd.to_numeric(errors="coerce") BEFORE missing-value handling.
# ============================================================

import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "output",
    "Crop_Normalized.xlsx"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "Canonical_Benchmark.xlsx"
)

TARGET_COLUMN = "Crop"

# ------------------------------------------------------------
# EXACT BENCHMARK CROP SET
# ------------------------------------------------------------

RECOGNIZED_CROPS = [
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

# ------------------------------------------------------------
# EXACT CANONICAL FEATURES
# ------------------------------------------------------------

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "State_Name",
    "Soil_Type",
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fail(message):
    print()
    print("!" * 78)
    print("BENCHMARK VALIDATION FAILED")
    print("!" * 78)
    print(message)
    raise RuntimeError(message)


def normalize_numeric_series(series):
    """
    Convert Excel/object/string numeric values safely.

    Handles:
      12
      "12"
      "12.5"
      "12%"
      " 12.5 "
      empty strings
      invalid strings

    Invalid values become NaN and are handled by the canonical
    missing-value removal step.
    """

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series
        .astype(str)
        .str.strip()
        .replace({
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "NULL": np.nan,
            "null": np.nan,
        })
    )

    # Remove percentage signs if present.
    cleaned = cleaned.str.replace("%", "", regex=False)

    # Remove commas from values such as "1,234.5".
    cleaned = cleaned.str.replace(",", "", regex=False)

    return pd.to_numeric(cleaned, errors="coerce")


def oversample_full(X_train, y_train, seed=42):
    """
    Full random oversampling to the maximum class size.

    Each class is sampled WITH replacement until every class
    reaches the same number of records.

    This reproduces the benchmark balancing strategy.
    """

    rng = np.random.RandomState(seed)

    X_train = X_train.copy()
    y_train = pd.Series(y_train).reset_index(drop=True)

    class_counts = y_train.value_counts()

    max_count = int(class_counts.max())

    X_parts = []
    y_parts = []

    for class_name in sorted(class_counts.index):

        class_indices = np.where(
            y_train.values == class_name
        )[0]

        if len(class_indices) == 0:
            continue

        if len(class_indices) < max_count:
            extra_indices = rng.choice(
                class_indices,
                size=max_count - len(class_indices),
                replace=True,
            )

            selected_indices = np.concatenate(
                [class_indices, extra_indices]
            )
        else:
            selected_indices = class_indices.copy()

        X_class = X_train.iloc[selected_indices].copy()
        y_class = y_train.iloc[selected_indices].copy()

        X_parts.append(X_class)
        y_parts.append(y_class)

    X_balanced = pd.concat(
        X_parts,
        axis=0,
        ignore_index=True
    )

    y_balanced = pd.concat(
        y_parts,
        axis=0,
        ignore_index=True
    )

    # Deterministic shuffle.
    shuffle_indices = rng.permutation(len(y_balanced))

    X_balanced = X_balanced.iloc[
        shuffle_indices
    ].reset_index(drop=True)

    y_balanced = y_balanced.iloc[
        shuffle_indices
    ].reset_index(drop=True)

    return X_balanced, y_balanced


# ============================================================
# START
# ============================================================

banner("SMART KISAN - CANONICAL BENCHMARK PIPELINE")

print(
    """
SAFE BENCHMARK
- Existing .pkl model will NOT be changed.
- No .pkl model will be created.
- Streamlit will NOT be changed.
- No application files will be changed.
- Establishes the canonical benchmark used by Steps 53+.
"""
)

# ============================================================
# LOAD DATA
# ============================================================

banner("LOADING DATASET")

print(f"Dataset path: {DATASET_PATH}")

if not os.path.exists(DATASET_PATH):
    fail(
        "Dataset not found:\n"
        f"{DATASET_PATH}"
    )

df = pd.read_excel(DATASET_PATH)

print(f"Raw records: {len(df):,}")
print(f"Raw columns: {len(df.columns)}")

# ============================================================
# REQUIRED COLUMN CHECK
# ============================================================

banner("REQUIRED COLUMN CHECK")

required_columns = (
    [TARGET_COLUMN]
    + FEATURE_COLUMNS
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    fail(
        "Required columns are missing:\n"
        + "\n".join(
            f" - {col}"
            for col in missing_columns
        )
    )

print("All canonical columns are present.")

# ============================================================
# TARGET CHECK
# ============================================================

banner("TARGET CHECK")

print(f"Target column: {TARGET_COLUMN}")
print(f"Target dtype : {df[TARGET_COLUMN].dtype}")
print(
    f"Target unique values: "
    f"{df[TARGET_COLUMN].nunique()}"
)

# ============================================================
# CROP FILTER
# ============================================================

banner("BENCHMARK CROP FILTER")

before_filter = len(df)

df = df[
    df[TARGET_COLUMN].isin(RECOGNIZED_CROPS)
].copy()

after_filter = len(df)

print(
    f"Before crop filtering : {before_filter:,}"
)
print(
    f"After crop filtering  : {after_filter:,}"
)
print(
    f"Rows removed          : "
    f"{before_filter - after_filter:,}"
)

print()
print("Crop distribution:")
print(
    df[TARGET_COLUMN]
    .value_counts()
    .sort_index()
)

# ============================================================
# CANONICAL FEATURE DEFINITION
# ============================================================

banner("CANONICAL FEATURE DEFINITION")

print("Numeric features:")

for i, feature in enumerate(
    NUMERIC_FEATURES,
    start=1
):
    print(f"{i:02d}. {feature}")

print()
print("Categorical features:")

for i, feature in enumerate(
    CATEGORICAL_FEATURES,
    start=1
):
    print(f"{i:02d}. {feature}")

print()
print(
    f"Numeric feature count     : "
    f"{len(NUMERIC_FEATURES)}"
)

print(
    f"Categorical feature count : "
    f"{len(CATEGORICAL_FEATURES)}"
)

print(
    f"Total feature count       : "
    f"{len(FEATURE_COLUMNS)}"
)

# ============================================================
# NUMERIC TYPE NORMALIZATION
# ============================================================

banner("NUMERIC TYPE NORMALIZATION")

print(
    "Converting all 16 canonical numeric features "
    "with pd.to_numeric(..., errors='coerce')."
)

conversion_report = []

for feature in NUMERIC_FEATURES:

    before_dtype = str(df[feature].dtype)

    before_non_null = int(
        df[feature].notna().sum()
    )

    df[feature] = normalize_numeric_series(
        df[feature]
    )

    after_dtype = str(df[feature].dtype)

    after_non_null = int(
        df[feature].notna().sum()
    )

    newly_invalid = (
        before_non_null - after_non_null
    )

    conversion_report.append({
        "feature": feature,
        "before_dtype": before_dtype,
        "after_dtype": after_dtype,
        "new_invalid_values": newly_invalid,
    })

    print(
        f"{feature:30s} "
        f"{before_dtype:12s} -> "
        f"{after_dtype:12s} "
        f"new NaN: {newly_invalid}"
    )

# ============================================================
# CATEGORICAL NORMALIZATION
# ============================================================

for feature in CATEGORICAL_FEATURES:

    df[feature] = (
        df[feature]
        .astype("string")
        .str.strip()
    )

# ============================================================
# MISSING VALUE HANDLING
# ============================================================

banner("MISSING VALUE HANDLING")

missing_before = (
    df[FEATURE_COLUMNS]
    .isna()
    .sum()
)

print("Missing values before cleaning:")

any_missing = False

for feature, count in missing_before.items():

    if count > 0:
        any_missing = True
        print(
            f"{feature:30s}: {int(count)}"
        )

if not any_missing:
    print("No missing feature values found.")

rows_before_cleaning = len(df)

df = df.dropna(
    subset=FEATURE_COLUMNS + [TARGET_COLUMN]
).copy()

rows_removed = (
    rows_before_cleaning - len(df)
)

print()
print(
    f"Rows before missing-value removal : "
    f"{rows_before_cleaning:,}"
)

print(
    f"Rows removed                       : "
    f"{rows_removed:,}"
)

print(
    f"Benchmark records                  : "
    f"{len(df):,}"
)

# ============================================================
# FEATURE DATA TYPE VALIDATION
# ============================================================

banner("FEATURE DATA TYPE VALIDATION")

print("Checking numeric features...")

for feature in NUMERIC_FEATURES:

    if not pd.api.types.is_numeric_dtype(
        df[feature]
    ):
        fail(
            "Expected numeric feature is not numeric:\n"
            f"{feature}"
        )

print(
    "All 16 numeric features are numeric."
)

print()
print("Checking categorical features...")

for feature in CATEGORICAL_FEATURES:

    if (
        not pd.api.types.is_object_dtype(df[feature])
        and not pd.api.types.is_string_dtype(df[feature])
        and not pd.api.types.is_categorical_dtype(df[feature])
    ):
        fail(
            "Expected categorical feature is not "
            "categorical/object/string:\n"
            f"{feature}"
        )

print(
    "Both categorical features are valid."
)

# ============================================================
# FINAL FEATURE MATRIX
# ============================================================

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()

# ============================================================
# TARGET VALIDATION
# ============================================================

banner("TARGET VALIDATION")

target_values = sorted(
    y.unique().tolist()
)

print(
    f"Target classes: {len(target_values)}"
)

print()
for crop in target_values:
    print(
        f"{crop:15s} "
        f"{int((y == crop).sum()):,}"
    )

if set(target_values) != set(RECOGNIZED_CROPS):
    fail(
        "Final target classes do not match "
        "the canonical crop list."
    )

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

banner("TRAIN / TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

print(
    f"Training records : {len(X_train):,}"
)

print(
    f"Testing records  : {len(X_test):,}"
)

train_indices = set(X_train.index)
test_indices = set(X_test.index)

overlap = train_indices.intersection(
    test_indices
)

print(
    f"Index overlap    : {len(overlap)}"
)

if len(overlap) != 0:
    fail(
        "Train/test leakage detected."
    )

print("Leakage check: PASSED")

# ============================================================
# PREPROCESSING
# ============================================================

banner("PREPROCESSING")

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    # Compatibility with older sklearn.
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            encoder,
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
)

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

X_train_processed = np.asarray(
    X_train_processed,
    dtype=np.float64
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float64
)

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape :",
    X_test_processed.shape
)

processed_feature_count = (
    X_train_processed.shape[1]
)

print()
print(
    f"Processed feature count: "
    f"{processed_feature_count}"
)

# ============================================================
# EXPECTED BENCHMARK CHECK
# ============================================================

EXPECTED_RECORDS = 5644
EXPECTED_TRAIN = 4515
EXPECTED_TEST = 1129
EXPECTED_PROCESSED_FEATURES = 216

banner("BENCHMARK EXPECTATION CHECK")

checks = [
    (
        "Benchmark records",
        len(df),
        EXPECTED_RECORDS
    ),
    (
        "Training records",
        len(X_train),
        EXPECTED_TRAIN
    ),
    (
        "Testing records",
        len(X_test),
        EXPECTED_TEST
    ),
    (
        "Processed features",
        processed_feature_count,
        EXPECTED_PROCESSED_FEATURES
    ),
]

all_passed = True

for name, actual, expected in checks:

    status = "PASS" if actual == expected else "FAIL"

    print(
        f"{name:25s} "
        f"Actual={actual:,} "
        f"Expected={expected:,} "
        f"[{status}]"
    )

    if actual != expected:
        all_passed = False

if not all_passed:

    print()
    print(
        "WARNING: The canonical feature typing is now "
        "correct, but one or more benchmark counts differ."
    )

    print(
        "This is a benchmark-data consistency issue, "
        "not a Python dtype failure."
    )

# ============================================================
# FULL OVERSAMPLING
# ============================================================

banner("FULL OVERSAMPLING")

X_train_df = X_train.reset_index(drop=True)
y_train_series = y_train.reset_index(drop=True)

print(
    f"Original training records : "
    f"{len(X_train_df):,}"
)

X_balanced, y_balanced = oversample_full(
    X_train_df,
    y_train_series,
    seed=SEED,
)

print(
    f"Balanced training records : "
    f"{len(X_balanced):,}"
)

print()
print("Balanced distribution:")
print(
    y_balanced
    .value_counts()
    .sort_index()
)

# ============================================================
# OPTIONAL RF SANITY CHECK
# ============================================================

banner("RANDOM FOREST SANITY CHECK")

print(
    "Training a temporary benchmark RF only to verify "
    "the canonical pipeline."
)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=SEED,
    n_jobs=-1,
)

X_balanced_processed = preprocessor.transform(
    X_balanced
)

X_balanced_processed = np.asarray(
    X_balanced_processed,
    dtype=np.float64
)

rf.fit(
    X_balanced_processed,
    y_balanced
)

predictions = rf.predict(
    X_test_processed
)

print(
    f"Temporary RF predictions: "
    f"{len(predictions):,}"
)

print(
    "Temporary RF training: PASSED"
)

# ============================================================
# SAVE REPORT
# ============================================================

banner("SAVING CANONICAL BENCHMARK REPORT")

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

summary = pd.DataFrame([
    {
        "item": "Raw records",
        "value": len(pd.read_excel(DATASET_PATH))
    },
    {
        "item": "Filtered records",
        "value": after_filter
    },
    {
        "item": "Benchmark records",
        "value": len(df)
    },
    {
        "item": "Numeric features",
        "value": len(NUMERIC_FEATURES)
    },
    {
        "item": "Categorical features",
        "value": len(CATEGORICAL_FEATURES)
    },
    {
        "item": "Total raw features",
        "value": len(FEATURE_COLUMNS)
    },
    {
        "item": "Processed features",
        "value": processed_feature_count
    },
    {
        "item": "Training records",
        "value": len(X_train)
    },
    {
        "item": "Testing records",
        "value": len(X_test)
    },
    {
        "item": "Balanced training records",
        "value": len(X_balanced)
    },
    {
        "item": "Random seed",
        "value": SEED
    },
])

distribution = (
    y.value_counts()
    .sort_index()
    .rename_axis("Crop")
    .reset_index(name="Count")
)

balanced_distribution = (
    y_balanced.value_counts()
    .sort_index()
    .rename_axis("Crop")
    .reset_index(name="Count")
)

conversion_df = pd.DataFrame(
    conversion_report
)

with pd.ExcelWriter(
    REPORT_PATH,
    engine="openpyxl"
) as writer:

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    distribution.to_excel(
        writer,
        sheet_name="Crop_Distribution",
        index=False
    )

    balanced_distribution.to_excel(
        writer,
        sheet_name="Balanced_Distribution",
        index=False
    )

    conversion_df.to_excel(
        writer,
        sheet_name="Numeric_Conversion",
        index=False
    )

    pd.DataFrame({
        "Numeric_Feature": NUMERIC_FEATURES
    }).to_excel(
        writer,
        sheet_name="Numeric_Features",
        index=False
    )

    pd.DataFrame({
        "Categorical_Feature": CATEGORICAL_FEATURES
    }).to_excel(
        writer,
        sheet_name="Categorical_Features",
        index=False
    )

print(
    f"Report saved to:\n{REPORT_PATH}"
)

# ============================================================
# FINAL STATUS
# ============================================================

banner("CANONICAL BENCHMARK COMPLETE")

print(
    """
CANONICAL PIPELINE:

Target:
    Crop

Crop filter:
    Apple
    Cotton
    Maize
    Mustard
    Pulses
    Rice
    Vegetables
    Walnut
    Wheat

Features:
    16 numeric
    2 categorical
    18 total

Numeric conversion:
    pd.to_numeric(errors="coerce")

Missing values:
    Removed after numeric conversion

Split:
    80/20 stratified
    random_state = 42

Preprocessing:
    Numeric = passthrough
    Categorical = OneHotEncoder(handle_unknown="ignore")

Oversampling:
    Full random oversampling
    random_state = 42

No .pkl model was created.
No existing .pkl model was changed.
No Streamlit file was changed.

IMPORTANT:
Steps 53+ should copy/reuse THIS canonical preprocessing
definition instead of defining their own feature columns.
"""
)

if all_passed:
    print(
        "BENCHMARK COUNTS: PASSED"
    )
else:
    print(
        "BENCHMARK COUNTS: REVIEW REQUIRED"
    )

print("=" * 78)