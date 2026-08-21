"""
SMART KISAN - STEP 53
CANONICAL BASELINE EVALUATION

SAFE EVALUATION
- Existing Streamlit app will NOT be changed.
- Existing .pkl model will NOT be changed.
- No model will be saved.
- No application files will be changed.
- Exact Step 52 canonical benchmark is reused.
- Exact feature definition is reused.
- Exact crop filter is reused.
- Exact missing-value handling is reused.
- Exact 80/20 stratified split is reused.
- Exact random_state=42 is reused.
- Exact full random oversampling is reused.
- Random Forest configuration is kept consistent with the benchmark.

Purpose:
    Establish the authoritative baseline metrics for Steps 53+.
"""

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = (
    r"C:\Users\rasoo_me13iy3\OneDrive\Desktop"
    r"\Crop-Recommendation-System-Clean\output\Crop_Normalized.xlsx"
)

OUTPUT_PATH = (
    r"C:\Users\rasoo_me13iy3\OneDrive\Desktop"
    r"\Crop-Recommendation-System-Clean\output"
    r"\Canonical_Baseline_Evaluation.xlsx"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "Crop"

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

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================================
# HELPERS
# ============================================================================

def heading(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fail(message):
    print()
    print("!" * 78)
    print("CANONICAL BASELINE EVALUATION FAILED")
    print("!" * 78)
    raise RuntimeError(message)


# ============================================================================
# START
# ============================================================================

heading("SMART KISAN - STEP 53 CANONICAL BASELINE EVALUATION")

print("""
SAFE EVALUATION
- Existing Streamlit app will NOT be changed.
- Existing .pkl model will NOT be changed.
- No model will be saved.
- No application files will be changed.
- Exact Step 52 canonical benchmark is reused.
- Establishes the authoritative baseline for Steps 53+.
""")


# ============================================================================
# LOAD DATA
# ============================================================================

heading("LOADING DATASET")

if not os.path.exists(DATASET_PATH):
    fail(f"Dataset not found:\n{DATASET_PATH}")

print(f"Dataset path: {DATASET_PATH}")

df = pd.read_excel(DATASET_PATH)

print(f"Raw records: {len(df):,}")
print(f"Raw columns: {len(df.columns)}")


# ============================================================================
# REQUIRED COLUMN CHECK
# ============================================================================

heading("REQUIRED COLUMN CHECK")

required_columns = [TARGET] + FEATURES

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    fail(
        "Missing canonical columns:\n"
        + "\n".join(f"  - {x}" for x in missing_columns)
    )

print("All canonical columns are present.")


# ============================================================================
# TARGET CHECK
# ============================================================================

heading("TARGET CHECK")

if TARGET not in df.columns:
    fail(f"Target column not found: {TARGET}")

print(f"Target column: {TARGET}")
print(f"Target dtype : {df[TARGET].dtype}")
print(f"Target unique values: {df[TARGET].nunique()}")


# ============================================================================
# CROP FILTER
# ============================================================================

heading("BENCHMARK CROP FILTER")

before_filter = len(df)

df = df[df[TARGET].isin(RECOGNIZED_CROPS)].copy()

after_filter = len(df)

print(f"Before crop filtering : {before_filter:,}")
print(f"After crop filtering  : {after_filter:,}")
print(f"Rows removed          : {before_filter - after_filter:,}")

print()
print("Crop distribution:")
print(df[TARGET].value_counts().sort_index())


# ============================================================================
# NUMERIC NORMALIZATION
# ============================================================================

heading("NUMERIC TYPE NORMALIZATION")

print(
    'Converting all 16 canonical numeric features with '
    'pd.to_numeric(..., errors="coerce").'
)

for col in NUMERIC_FEATURES:
    before_dtype = df[col].dtype
    before_na = df[col].isna().sum()

    df[col] = pd.to_numeric(df[col], errors="coerce")

    after_dtype = df[col].dtype
    after_na = df[col].isna().sum()

    print(
        f"{col:<30} "
        f"{str(before_dtype):<12} -> "
        f"{str(after_dtype):<12} "
        f"new NaN: {after_na - before_na}"
    )


# ============================================================================
# FEATURE TYPE VALIDATION
# ============================================================================

heading("FEATURE DATA TYPE VALIDATION")

print("Checking numeric features...")

for col in NUMERIC_FEATURES:
    if not pd.api.types.is_numeric_dtype(df[col]):
        fail(f"Canonical numeric feature is not numeric: {col}")

print("All 16 numeric features are numeric.")

print()
print("Checking categorical features...")

for col in CATEGORICAL_FEATURES:
    if pd.api.types.is_numeric_dtype(df[col]):
        fail(f"Canonical categorical feature is numeric: {col}")

print("Both categorical features are valid.")


# ============================================================================
# MISSING VALUE HANDLING
# ============================================================================

heading("MISSING VALUE HANDLING")

missing_counts = df[FEATURES].isna().sum()

print("Missing values before cleaning:")

any_missing = False

for col, count in missing_counts.items():
    if count > 0:
        print(f"{col:<30}: {count}")
        any_missing = True

if not any_missing:
    print("None")

before_cleaning = len(df)

df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()

after_cleaning = len(df)

print()
print(f"Rows before missing-value removal : {before_cleaning:,}")
print(f"Rows removed                       : {before_cleaning - after_cleaning:,}")
print(f"Benchmark records                  : {after_cleaning:,}")


# ============================================================================
# BENCHMARK COUNT CHECK
# ============================================================================

heading("CANONICAL BENCHMARK COUNT CHECK")

EXPECTED_RECORDS = 5644

print(
    f"Benchmark records: "
    f"Actual={len(df):,} Expected={EXPECTED_RECORDS:,}"
)

if len(df) != EXPECTED_RECORDS:
    fail(
        f"Canonical benchmark record count mismatch. "
        f"Expected {EXPECTED_RECORDS}, got {len(df)}."
    )

print("Benchmark record count: PASS")


# ============================================================================
# TARGET VALIDATION
# ============================================================================

heading("TARGET VALIDATION")

classes = sorted(df[TARGET].unique())

print(f"Target classes: {len(classes)}")
print()

print(df[TARGET].value_counts().sort_index())

if classes != sorted(RECOGNIZED_CROPS):
    fail(
        "Target class set does not match canonical benchmark.\n"
        f"Expected: {sorted(RECOGNIZED_CROPS)}\n"
        f"Actual  : {classes}"
    )


# ============================================================================
# TRAIN / TEST SPLIT
# ============================================================================

heading("TRAIN / TEST SPLIT")

X = df[FEATURES].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training records : {len(X_train):,}")
print(f"Testing records  : {len(X_test):,}")


# ============================================================================
# LEAKAGE CHECK
# ============================================================================

train_indices = set(X_train.index)
test_indices = set(X_test.index)

overlap = train_indices.intersection(test_indices)

print(f"Index overlap    : {len(overlap)}")

if overlap:
    fail("Train/test index overlap detected.")

print("Leakage check: PASSED")


# ============================================================================
# SPLIT SIZE CHECK
# ============================================================================

EXPECTED_TRAIN = 4515
EXPECTED_TEST = 1129

if len(X_train) != EXPECTED_TRAIN:
    fail(
        f"Training size mismatch: "
        f"expected {EXPECTED_TRAIN}, got {len(X_train)}"
    )

if len(X_test) != EXPECTED_TEST:
    fail(
        f"Testing size mismatch: "
        f"expected {EXPECTED_TEST}, got {len(X_test)}"
    )

print("Canonical split sizes: PASS")


# ============================================================================
# PREPROCESSING
# ============================================================================

heading("PREPROCESSING")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            NUMERIC_FEATURES,
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(
    f"Processed training shape: "
    f"{X_train_processed.shape}"
)

print(
    f"Processed testing shape : "
    f"{X_test_processed.shape}"
)

processed_features = X_train_processed.shape[1]

print()
print(f"Processed feature count: {processed_features}")

if processed_features != 216:
    fail(
        f"Processed feature count mismatch: "
        f"expected 216, got {processed_features}"
    )

print("Processed feature count: PASS")


# ============================================================================
# FULL RANDOM OVERSAMPLING
# ============================================================================

heading("FULL OVERSAMPLING")

train_df = pd.DataFrame(
    X_train_processed.toarray()
    if hasattr(X_train_processed, "toarray")
    else X_train_processed
)

train_df["_TARGET_"] = y_train.to_numpy()

original_training_records = len(train_df)

max_class_count = train_df["_TARGET_"].value_counts().max()

balanced_parts = []

for class_name in sorted(train_df["_TARGET_"].unique()):
    class_df = train_df[
        train_df["_TARGET_"] == class_name
    ]

    sampled = class_df.sample(
        n=max_class_count,
        replace=True,
        random_state=RANDOM_STATE,
    )

    balanced_parts.append(sampled)

balanced_df = pd.concat(
    balanced_parts,
    ignore_index=True,
)

balanced_training_records = len(balanced_df)

X_train_balanced = balanced_df.drop(
    columns=["_TARGET_"]
)

y_train_balanced = balanced_df["_TARGET_"]

print(
    f"Original training records : "
    f"{original_training_records:,}"
)

print(
    f"Balanced training records : "
    f"{balanced_training_records:,}"
)

print()
print("Balanced distribution:")
print(y_train_balanced.value_counts().sort_index())


if balanced_training_records != 19593:
    fail(
        f"Balanced training size mismatch: "
        f"expected 19593, got {balanced_training_records}"
    )

print()
print("Balanced training size: PASS")


# ============================================================================
# RANDOM FOREST
# ============================================================================

heading("TRAINING CANONICAL BASELINE RANDOM FOREST")

print("Training temporary RF...")
print("No model will be saved.")

# Keep this configuration aligned with the benchmark.
# If your existing benchmark scripts use a different RF configuration,
# replace ONLY this estimator block with that exact configuration.

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf.fit(
    X_train_balanced,
    y_train_balanced,
)

print("Training complete.")


# ============================================================================
# PREDICTION
# ============================================================================

heading("BASELINE PREDICTION")

y_pred = rf.predict(X_test_processed)

print(f"Predictions generated: {len(y_pred):,}")

if len(y_pred) != len(y_test):
    fail("Prediction count does not match test records.")


# ============================================================================
# METRICS
# ============================================================================

heading("CANONICAL BASELINE METRICS")

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

report_dict = classification_report(
    y_test,
    y_pred,
    labels=RECOGNIZED_CROPS,
    output_dict=True,
    zero_division=0,
)

print(f"Accuracy    : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")


# ============================================================================
# PER-CROP METRICS
# ============================================================================

heading("PER-CROP METRICS")

per_crop_rows = []

for crop in RECOGNIZED_CROPS:
    if crop in report_dict:
        row = report_dict[crop]

        per_crop_rows.append(
            {
                "Crop": crop,
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1": row["f1-score"],
                "Support": int(row["support"]),
            }
        )

per_crop_df = pd.DataFrame(per_crop_rows)

print(
    per_crop_df.to_string(
        index=False,
        formatters={
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format,
        },
    )
)


# ============================================================================
# WEAK CROP F1
# ============================================================================

weak_crops = [
    "Apple",
    "Cotton",
    "Mustard",
    "Pulses",
    "Vegetables",
    "Walnut",
]

weak_crop_f1_values = []

for crop in weak_crops:
    if crop in report_dict:
        weak_crop_f1_values.append(
            report_dict[crop]["f1-score"]
        )

weak_crop_f1 = float(
    np.mean(weak_crop_f1_values)
)

print()
print(
    f"Weak Crop F1: "
    f"{weak_crop_f1:.4f}"
)


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

heading("CONFUSION MATRIX")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=RECOGNIZED_CROPS,
)

cm_df = pd.DataFrame(
    cm,
    index=RECOGNIZED_CROPS,
    columns=RECOGNIZED_CROPS,
)

print(cm_df.to_string())


# ============================================================================
# MOST COMMON CONFUSIONS
# ============================================================================

heading("TOP CONFUSION PAIRS")

confusion_rows = []

for i, actual in enumerate(RECOGNIZED_CROPS):
    for j, predicted in enumerate(RECOGNIZED_CROPS):

        if i == j:
            continue

        count = cm[i, j]

        if count > 0:
            confusion_rows.append(
                {
                    "Actual": actual,
                    "Predicted": predicted,
                    "Count": int(count),
                }
            )

confusion_df = pd.DataFrame(confusion_rows)

if not confusion_df.empty:
    confusion_df = confusion_df.sort_values(
        "Count",
        ascending=False,
    ).reset_index(drop=True)

    print(
        confusion_df.head(20).to_string(
            index=False
        )
    )
else:
    print("No off-diagonal confusion detected.")


# ============================================================================
# SAVE REPORT
# ============================================================================

heading("SAVING BASELINE EVALUATION REPORT")

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

summary_df = pd.DataFrame(
    [
        {
            "Model": "Canonical Baseline RF",
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "Weak Crop F1": weak_crop_f1,
            "Benchmark Records": len(df),
            "Training Records": len(X_train),
            "Testing Records": len(X_test),
            "Processed Features": processed_features,
            "Balanced Training Records": balanced_training_records,
        }
    ]
)

with pd.ExcelWriter(
    OUTPUT_PATH,
    engine="openpyxl",
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="Summary",
        index=False,
    )

    per_crop_df.to_excel(
        writer,
        sheet_name="Per_Crop_Metrics",
        index=False,
    )

    cm_df.to_excel(
        writer,
        sheet_name="Confusion_Matrix",
    )

    if not confusion_df.empty:
        confusion_df.to_excel(
            writer,
            sheet_name="Top_Confusions",
            index=False,
        )

print(f"Report saved to:\n{OUTPUT_PATH}")


# ============================================================================
# FINAL
# ============================================================================

heading("STEP 53 COMPLETE")

print(
    f"""
CANONICAL BASELINE

Accuracy    : {accuracy * 100:.2f}%
Macro F1    : {macro_f1:.4f}
Weighted F1 : {weighted_f1:.4f}
Weak Crop F1: {weak_crop_f1:.4f}

Benchmark records         : {len(df):,}
Training records          : {len(X_train):,}
Testing records           : {len(X_test):,}
Processed features        : {processed_features:,}
Balanced training records: {balanced_training_records:,}

No .pkl model was created.
No existing .pkl model was changed.
No Streamlit file was changed.

IMPORTANT:
This report is now the canonical baseline for Step 55+
confusion-pair experiments.
"""
)

print("=" * 78)