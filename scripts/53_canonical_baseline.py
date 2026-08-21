# ============================================================
# SMART KISAN - CANONICAL BASELINE EXPERIMENT
# ============================================================
#
# SAFE EXPERIMENT
# - No .pkl model created
# - Existing .pkl model untouched
# - Streamlit untouched
# - Uses EXACT Step 52 canonical benchmark
# - Establishes the official baseline for Steps 54+
# ============================================================

import os
import random
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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
    "Canonical_Baseline_Experiment.xlsx"
)

TARGET = "Crop"

CROPS = [
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

FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

# ============================================================
# HELPERS
# ============================================================

def numeric_clean(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce"
        )

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
        .str.replace(
            "%",
            "",
            regex=False
        )
        .str.replace(
            ",",
            "",
            regex=False
        )
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce"
    )


def oversample_full(
    X,
    y,
    seed=42
):
    rng = np.random.RandomState(seed)

    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    counts = y.value_counts()

    max_count = int(
        counts.max()
    )

    X_parts = []
    y_parts = []

    for crop in sorted(counts.index):

        indices = np.where(
            y.values == crop
        )[0]

        extra = max_count - len(indices)

        if extra > 0:

            sampled = rng.choice(
                indices,
                size=extra,
                replace=True
            )

            indices = np.concatenate(
                [indices, sampled]
            )

        X_parts.append(
            X.iloc[indices]
        )

        y_parts.append(
            y.iloc[indices]
        )

    X_balanced = pd.concat(
        X_parts,
        ignore_index=True
    )

    y_balanced = pd.concat(
        y_parts,
        ignore_index=True
    )

    shuffle = rng.permutation(
        len(y_balanced)
    )

    X_balanced = (
        X_balanced
        .iloc[shuffle]
        .reset_index(drop=True)
    )

    y_balanced = (
        y_balanced
        .iloc[shuffle]
        .reset_index(drop=True)
    )

    return (
        X_balanced,
        y_balanced
    )


def print_metrics(
    name,
    y_true,
    predictions
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    per_class = f1_score(
        y_true,
        predictions,
        labels=CROPS,
        average=None,
        zero_division=0
    )

    f1_by_crop = dict(
        zip(CROPS, per_class)
    )

    weak_crops = [
        "Apple",
        "Mustard",
        "Vegetables",
        "Walnut",
    ]

    weak_f1 = np.mean([
        f1_by_crop[crop]
        for crop in weak_crops
    ])

    print()
    print(
        f"{name}"
    )

    print(
        f"Accuracy    : "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro F1    : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : "
        f"{weighted_f1:.4f}"
    )

    print(
        f"Weak Crop F1: "
        f"{weak_f1:.4f}"
    )

    print()
    print("Per-crop F1:")

    for crop in CROPS:

        print(
            f"{crop:15s}: "
            f"{f1_by_crop[crop]:.4f}"
        )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Weak Crop F1": weak_f1,
        **{
            f"F1_{crop}":
            f1_by_crop[crop]
            for crop in CROPS
        }
    }


# ============================================================
# START
# ============================================================

print("=" * 78)
print("SMART KISAN - CANONICAL BASELINE EXPERIMENT")
print("=" * 78)

print(
    """
SAFE EXPERIMENT

- Step 52 canonical preprocessing is used.
- No .pkl model will be created.
- Existing .pkl model will NOT be changed.
- Streamlit will NOT be changed.
- No application files will be changed.
- This becomes the official baseline for Steps 54+.
"""
)

# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 78)
print("LOADING DATASET")
print("=" * 78)

print(
    f"Dataset path: {DATASET_PATH}"
)

df = pd.read_excel(
    DATASET_PATH
)

print(
    f"Raw records: {len(df):,}"
)

# ============================================================
# CROP FILTER
# ============================================================

df = df[
    df[TARGET].isin(CROPS)
].copy()

print(
    f"After benchmark crop filtering: "
    f"{len(df):,}"
)

# ============================================================
# EXACT CANONICAL NUMERIC CONVERSION
# ============================================================

for feature in NUMERIC_FEATURES:

    df[feature] = numeric_clean(
        df[feature]
    )

# ============================================================
# EXACT CANONICAL CATEGORICAL HANDLING
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

before_clean = len(df)

df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()

print(
    f"Rows removed for missing values: "
    f"{before_clean - len(df):,}"
)

print(
    f"Benchmark records: "
    f"{len(df):,}"
)

# ============================================================
# DATA CHECK
# ============================================================

if len(df) != 5644:
    raise RuntimeError(
        f"Canonical record count mismatch: "
        f"{len(df)} instead of 5644"
    )

# ============================================================
# X / Y
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ============================================================
# SPLIT
# ============================================================

print()
print("=" * 78)
print("TRAIN / TEST SPLIT")
print("=" * 78)

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

if set(X_train.index).intersection(
    set(X_test.index)
):
    raise RuntimeError(
        "Train/test leakage detected."
    )

print(
    "Leakage check: PASSED"
)

# ============================================================
# PREPROCESSING
# ============================================================

print()
print("=" * 78)
print("PREPROCESSING")
print("=" * 78)

try:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

except TypeError:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            encoder,
            CATEGORICAL_FEATURES
        ),
    ],
    remainder="drop"
)

X_train_processed = (
    preprocessor
    .fit_transform(X_train)
)

X_test_processed = (
    preprocessor
    .transform(X_test)
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
    f"Training processed shape: "
    f"{X_train_processed.shape}"
)

print(
    f"Testing processed shape : "
    f"{X_test_processed.shape}"
)

if X_train_processed.shape[1] != 216:
    raise RuntimeError(
        "Processed feature count is not 216."
    )

# ============================================================
# FULL OVERSAMPLING
# ============================================================

print()
print("=" * 78)
print("FULL OVERSAMPLING")
print("=" * 78)

X_train_balanced, y_train_balanced = (
    oversample_full(
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        seed=SEED
    )
)

print(
    f"Original training records : "
    f"{len(X_train):,}"
)

print(
    f"Balanced training records : "
    f"{len(X_train_balanced):,}"
)

print()
print(
    y_train_balanced
    .value_counts()
    .sort_index()
)

# Process balanced data using
# the SAME fitted canonical preprocessor.
X_balanced_processed = (
    preprocessor.transform(
        X_train_balanced
    )
)

X_balanced_processed = np.asarray(
    X_balanced_processed,
    dtype=np.float64
)

if X_balanced_processed.shape[1] != 216:
    raise RuntimeError(
        "Balanced processed feature count "
        "is not 216."
    )

# ============================================================
# RANDOM FOREST
# ============================================================

print()
print("=" * 78)
print("TRAINING CANONICAL RANDOM FOREST")
print("=" * 78)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=SEED,
    n_jobs=-1
)

print("Training...")

model.fit(
    X_balanced_processed,
    y_train_balanced
)

print(
    "Training complete."
)

# ============================================================
# BASELINE ARGMAX
# ============================================================

print()
print("=" * 78)
print("CANONICAL BASELINE ARGMAX")
print("=" * 78)

probabilities = model.predict_proba(
    X_test_processed
)

classes = model.classes_

predicted_indices = np.argmax(
    probabilities,
    axis=1
)

predictions = classes[
    predicted_indices
]

baseline_result = print_metrics(
    "Canonical Baseline",
    y_test,
    predictions
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 78)
print("CONFUSION MATRIX")
print("=" * 78)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=CROPS
)

cm_df = pd.DataFrame(
    cm,
    index=CROPS,
    columns=CROPS
)

print(cm_df)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 78)
print("CLASSIFICATION REPORT")
print("=" * 78)

print(
    classification_report(
        y_test,
        predictions,
        labels=CROPS,
        zero_division=0
    )
)

# ============================================================
# SAVE REPORT
# ============================================================

print()
print("=" * 78)
print("SAVING REPORT")
print("=" * 78)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

summary_df = pd.DataFrame([
    baseline_result
])

crop_distribution = (
    y.value_counts()
    .sort_index()
    .rename_axis("Crop")
    .reset_index(name="Count")
)

balanced_distribution = (
    y_train_balanced
    .value_counts()
    .sort_index()
    .rename_axis("Crop")
    .reset_index(name="Count")
)

with pd.ExcelWriter(
    REPORT_PATH,
    engine="openpyxl"
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="Baseline",
        index=False
    )

    cm_df.to_excel(
        writer,
        sheet_name="Confusion_Matrix"
    )

    crop_distribution.to_excel(
        writer,
        sheet_name="Crop_Distribution",
        index=False
    )

    balanced_distribution.to_excel(
        writer,
        sheet_name="Balanced_Distribution",
        index=False
    )

print(
    f"Report saved to:\n"
    f"{REPORT_PATH}"
)

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 78)
print("CANONICAL BASELINE COMPLETE")
print("=" * 78)

print(
    f"""
Official canonical benchmark:

Records          : {len(df):,}
Train            : {len(X_train):,}
Test             : {len(X_test):,}
Processed        : {X_train_processed.shape[1]}
Balanced train   : {len(X_train_balanced):,}

Accuracy         : {baseline_result["Accuracy"]:.4f}
Macro F1         : {baseline_result["Macro F1"]:.4f}
Weighted F1      : {baseline_result["Weighted F1"]:.4f}
Weak Crop F1     : {baseline_result["Weak Crop F1"]:.4f}

NO .pkl MODEL CREATED.
NO EXISTING MODEL OVERWRITTEN.
NO STREAMLIT FILE CHANGED.

This result is now the canonical baseline.
"""
)

print("=" * 78)