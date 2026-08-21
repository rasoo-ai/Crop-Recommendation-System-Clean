from pathlib import Path
import sys
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


# =============================================================================
# SMART KISAN - CANONICAL CONFUSION ANALYSIS
# =============================================================================

print("=" * 78)
print("SMART KISAN - CANONICAL CONFUSION ANALYSIS")
print("=" * 78)

print("""
SAFE ANALYSIS

- Uses the canonical pipeline established by Step 52.
- Uses the canonical baseline definition established by Step 53.
- No .pkl model will be created.
- Existing .pkl model will NOT be changed.
- Streamlit will NOT be changed.
- No application files will be changed.
- No decision rule will be deployed.
- This step only identifies actual confusion pairs.
""")


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\rasoo_me13iy3\OneDrive\Desktop\Crop-Recommendation-System-Clean"
)

DATASET_PATH = PROJECT_ROOT / "output" / "Crop_Normalized.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "output" / "Canonical_Confusion_Analysis.xlsx"


# =============================================================================
# CANONICAL DEFINITIONS
# =============================================================================

TARGET_COLUMN = "Crop"

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

RANDOM_STATE = 42
TEST_SIZE = 0.20


# =============================================================================
# HELPERS
# =============================================================================

def fail(message):
    print()
    print("!" * 78)
    print("CANONICAL CONFUSION ANALYSIS FAILED")
    print("!" * 78)
    print(message)
    raise RuntimeError(message)


def check_columns(df):
    required = [TARGET_COLUMN] + FEATURES

    missing = [c for c in required if c not in df.columns]

    if missing:
        fail(
            "Required canonical columns are missing:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


# =============================================================================
# DATASET CHECK
# =============================================================================

print()
print("=" * 78)
print("LOADING DATASET")
print("=" * 78)

print(f"Dataset path: {DATASET_PATH}")

if not DATASET_PATH.exists():
    fail(
        "Dataset not found:\n"
        f"{DATASET_PATH}\n\n"
        "Expected canonical project path:\n"
        f"{PROJECT_ROOT}"
    )

df = pd.read_excel(DATASET_PATH)

print(f"Raw records: {len(df):,}")
print(f"Raw columns: {len(df.columns)}")


# =============================================================================
# REQUIRED COLUMN CHECK
# =============================================================================

print()
print("=" * 78)
print("REQUIRED COLUMN CHECK")
print("=" * 78)

check_columns(df)

print("All canonical columns are present.")


# =============================================================================
# TARGET FILTER
# =============================================================================

print()
print("=" * 78)
print("BENCHMARK CROP FILTER")
print("=" * 78)

before_filter = len(df)

df = df[df[TARGET_COLUMN].isin(RECOGNIZED_CROPS)].copy()

after_filter = len(df)

print(f"Before crop filtering : {before_filter:,}")
print(f"After crop filtering  : {after_filter:,}")
print(f"Rows removed          : {before_filter - after_filter:,}")

print()
print(df[TARGET_COLUMN].value_counts().sort_index())


# =============================================================================
# NUMERIC TYPE NORMALIZATION
# =============================================================================

print()
print("=" * 78)
print("NUMERIC TYPE NORMALIZATION")
print("=" * 78)

for column in NUMERIC_FEATURES:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

print("All canonical numeric features converted with:")
print('pd.to_numeric(..., errors="coerce")')


# =============================================================================
# MISSING VALUE HANDLING
# =============================================================================

print()
print("=" * 78)
print("MISSING VALUE HANDLING")
print("=" * 78)

missing_before = df[FEATURES + [TARGET_COLUMN]].isna().sum()

print("Missing values before cleaning:")

any_missing = False

for column, count in missing_before.items():
    if count > 0:
        print(f"{column:<30}: {count}")
        any_missing = True

if not any_missing:
    print("None")

rows_before_cleaning = len(df)

df = df.dropna(
    subset=FEATURES + [TARGET_COLUMN]
).copy()

rows_after_cleaning = len(df)

print()
print(
    f"Rows before missing-value removal : "
    f"{rows_before_cleaning:,}"
)

print(
    f"Rows removed                       : "
    f"{rows_before_cleaning - rows_after_cleaning:,}"
)

print(
    f"Benchmark records                  : "
    f"{rows_after_cleaning:,}"
)


# =============================================================================
# CANONICAL VALIDATION
# =============================================================================

print()
print("=" * 78)
print("CANONICAL BENCHMARK VALIDATION")
print("=" * 78)

expected_records = 5644

if len(df) != expected_records:
    fail(
        f"Canonical record count mismatch.\n"
        f"Expected: {expected_records:,}\n"
        f"Actual  : {len(df):,}"
    )

print(f"Benchmark records: {len(df):,} [PASS]")

for column in NUMERIC_FEATURES:
    if not pd.api.types.is_numeric_dtype(df[column]):
        fail(
            f"Numeric feature is not numeric after normalization:\n"
            f"{column}"
        )

print("Numeric feature validation: PASS")

for column in CATEGORICAL_FEATURES:
    if column not in df.columns:
        fail(f"Missing categorical feature: {column}")

print("Categorical feature validation: PASS")


# =============================================================================
# TARGET DISTRIBUTION
# =============================================================================

print()
print("=" * 78)
print("TARGET DISTRIBUTION")
print("=" * 78)

target_counts = df[TARGET_COLUMN].value_counts().sort_index()

print(target_counts)


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

print()
print("=" * 78)
print("TRAIN / TEST SPLIT")
print("=" * 78)

X = df[FEATURES].copy()
y = df[TARGET_COLUMN].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Training records : {len(X_train):,}")
print(f"Testing records  : {len(X_test):,}")

train_indices = set(X_train.index)
test_indices = set(X_test.index)

overlap = train_indices.intersection(test_indices)

print(f"Index overlap     : {len(overlap)}")

if overlap:
    fail("Train/test leakage detected.")

print("Leakage check: PASSED")


# =============================================================================
# PREPROCESSING
# =============================================================================

print()
print("=" * 78)
print("PREPROCESSING")
print("=" * 78)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
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

print(
    f"Processed feature count : "
    f"{processed_features}"
)

if processed_features != 216:
    fail(
        f"Unexpected processed feature count.\n"
        f"Expected: 216\n"
        f"Actual  : {processed_features}"
    )

print("Processed feature count check: PASS")


# =============================================================================
# FULL RANDOM OVERSAMPLING
# =============================================================================

print()
print("=" * 78)
print("FULL OVERSAMPLING")
print("=" * 78)

train_processed_df = pd.DataFrame(
    X_train_processed.toarray()
    if hasattr(X_train_processed, "toarray")
    else X_train_processed
)

train_processed_df["_TARGET_"] = y_train.to_numpy()

max_count = train_processed_df["_TARGET_"].value_counts().max()

balanced_parts = []

rng = np.random.RandomState(RANDOM_STATE)

for crop in RECOGNIZED_CROPS:

    crop_df = train_processed_df[
        train_processed_df["_TARGET_"] == crop
    ]

    if len(crop_df) == 0:
        fail(f"No training records for crop: {crop}")

    sampled = crop_df.sample(
        n=max_count,
        replace=True,
        random_state=RANDOM_STATE,
    )

    balanced_parts.append(sampled)

balanced_train = pd.concat(
    balanced_parts,
    ignore_index=True,
)

X_balanced = balanced_train.drop(
    columns=["_TARGET_"]
).to_numpy()

y_balanced = balanced_train["_TARGET_"].to_numpy()

print(
    f"Original training records : "
    f"{len(X_train):,}"
)

print(
    f"Balanced training records : "
    f"{len(X_balanced):,}"
)

print()
print(
    pd.Series(y_balanced).value_counts().sort_index()
)


# =============================================================================
# RANDOM FOREST
# =============================================================================

print()
print("=" * 78)
print("TRAINING CANONICAL RANDOM FOREST")
print("=" * 78)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("Training...")

model.fit(
    X_balanced,
    y_balanced,
)

print("Training complete.")


# =============================================================================
# PREDICTIONS
# =============================================================================

y_pred = model.predict(X_test_processed)

if hasattr(X_test_processed, "toarray"):
    X_test_for_probability = X_test_processed
else:
    X_test_for_probability = X_test_processed

probabilities = model.predict_proba(
    X_test_for_probability
)

classes = model.classes_

print()
print("=" * 78)
print("BASELINE PREDICTIONS")
print("=" * 78)

accuracy = accuracy_score(
    y_test,
    y_pred,
)

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

print(f"Accuracy    : {accuracy:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

print()
print("=" * 78)
print("CONFUSION MATRIX")
print("=" * 78)

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


# =============================================================================
# PER-CROP F1
# =============================================================================

print()
print("=" * 78)
print("PER-CROP F1")
print("=" * 78)

report_dict = classification_report(
    y_test,
    y_pred,
    labels=RECOGNIZED_CROPS,
    output_dict=True,
    zero_division=0,
)

per_crop_rows = []

for crop in RECOGNIZED_CROPS:

    per_crop_rows.append(
        {
            "Crop": crop,
            "Precision": report_dict[crop]["precision"],
            "Recall": report_dict[crop]["recall"],
            "F1": report_dict[crop]["f1-score"],
            "Support": report_dict[crop]["support"],
        }
    )

per_crop_df = pd.DataFrame(per_crop_rows)

print(
    per_crop_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# =============================================================================
# ACTUAL CONFUSION PAIRS
# =============================================================================

print()
print("=" * 78)
print("ACTUAL CONFUSION PAIRS")
print("=" * 78)

confusion_rows = []

for true_index, true_crop in enumerate(RECOGNIZED_CROPS):

    for pred_index, predicted_crop in enumerate(RECOGNIZED_CROPS):

        if true_crop == predicted_crop:
            continue

        count = cm[true_index, pred_index]

        if count > 0:

            confusion_rows.append(
                {
                    "True Crop": true_crop,
                    "Predicted Crop": predicted_crop,
                    "Errors": int(count),
                }
            )

confusion_df = pd.DataFrame(
    confusion_rows
)

if len(confusion_df) == 0:

    print("No off-diagonal confusion pairs found.")

else:

    confusion_df = confusion_df.sort_values(
        by="Errors",
        ascending=False,
    ).reset_index(drop=True)

    print(confusion_df.to_string(index=False))


# =============================================================================
# TOP CONFUSION PAIRS
# =============================================================================

print()
print("=" * 78)
print("TOP CONFUSION PAIRS")
print("=" * 78)

if len(confusion_df) > 0:

    top_pairs = confusion_df.head(15).copy()

    for i, row in top_pairs.iterrows():

        print(
            f"{i + 1:02d}. "
            f"{row['True Crop']} -> "
            f"{row['Predicted Crop']} : "
            f"{row['Errors']} error(s)"
        )

else:

    print("No confusion pairs available.")


# =============================================================================
# TARGETED PAIR SUMMARY
# =============================================================================

print()
print("=" * 78)
print("TARGETED PAIR CANDIDATES")
print("=" * 78)

candidate_pairs = [
    ("Maize", "Mustard"),
    ("Rice", "Mustard"),
    ("Pulses", "Mustard"),
    ("Maize", "Rice"),
    ("Maize", "Walnut"),
    ("Rice", "Apple"),
    ("Maize", "Wheat"),
    ("Rice", "Pulses"),
    ("Wheat", "Vegetables"),
    ("Vegetables", "Wheat"),
]

candidate_rows = []

for true_crop, predicted_crop in candidate_pairs:

    true_index = RECOGNIZED_CROPS.index(true_crop)
    predicted_index = RECOGNIZED_CROPS.index(predicted_crop)

    errors = int(
        cm[true_index, predicted_index]
    )

    reverse_errors = int(
        cm[
            RECOGNIZED_CROPS.index(predicted_crop),
            RECOGNIZED_CROPS.index(true_crop),
        ]
    )

    candidate_rows.append(
        {
            "True Crop": true_crop,
            "Predicted Crop": predicted_crop,
            "Forward Errors": errors,
            "Reverse Errors": reverse_errors,
            "Total Pair Errors": errors + reverse_errors,
        }
    )

candidate_df = pd.DataFrame(candidate_rows)

candidate_df = candidate_df.sort_values(
    by="Total Pair Errors",
    ascending=False,
).reset_index(drop=True)

print(candidate_df.to_string(index=False))


# =============================================================================
# IMPORTANT INTERPRETATION
# =============================================================================

print()
print("=" * 78)
print("INTERPRETATION")
print("=" * 78)

print("""
This step does NOT test or deploy any decision rule.

It only establishes:

1. The canonical baseline confusion matrix.
2. Which true crops are being confused.
3. Which predicted crops receive those errors.
4. The number of errors for each pair.
5. Which pairs are large enough to justify a later controlled experiment.

A confusion pair should NOT automatically be boosted.

Step 55+ should only test pairs supported by this canonical
confusion analysis and should compare every rule against this
exact baseline.
""")


# =============================================================================
# SAVE REPORT
# =============================================================================

print()
print("=" * 78)
print("SAVING REPORT")
print("=" * 78)

metadata_df = pd.DataFrame(
    [
        ["Dataset records after filtering", after_filter],
        ["Benchmark records", len(df)],
        ["Training records", len(X_train)],
        ["Testing records", len(X_test)],
        ["Processed features", processed_features],
        ["Balanced training records", len(X_balanced)],
        ["Accuracy", accuracy],
        ["Macro F1", macro_f1],
        ["Weighted F1", weighted_f1],
        ["Random state", RANDOM_STATE],
    ],
    columns=["Metric", "Value"],
)

with pd.ExcelWriter(
    OUTPUT_PATH,
    engine="openpyxl",
) as writer:

    metadata_df.to_excel(
        writer,
        sheet_name="Benchmark Summary",
        index=False,
    )

    cm_df.to_excel(
        writer,
        sheet_name="Confusion Matrix",
    )

    per_crop_df.to_excel(
        writer,
        sheet_name="Per Crop F1",
        index=False,
    )

    confusion_df.to_excel(
        writer,
        sheet_name="All Confusion Pairs",
        index=False,
    )

    candidate_df.to_excel(
        writer,
        sheet_name="Targeted Candidates",
        index=False,
    )

print(f"Report saved to:")
print(OUTPUT_PATH)


# =============================================================================
# COMPLETE
# =============================================================================

print()
print("=" * 78)
print("CANONICAL CONFUSION ANALYSIS COMPLETE")
print("=" * 78)

print()
print("Canonical baseline:")
print(f"    Accuracy    : {accuracy:.4f}")
print(f"    Macro F1    : {macro_f1:.4f}")
print(f"    Weighted F1 : {weighted_f1:.4f}")

print()
print("NO .pkl MODEL CREATED.")
print("NO EXISTING MODEL OVERWRITTEN.")
print("NO STREAMLIT FILE CHANGED.")
print()
print("DO NOT DEPLOY ANY DECISION RULE YET.")
print("=" * 78)