import os
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample


print("=" * 78)
print("SMART KISAN - FINAL CANDIDATE VALIDATION")
print("=" * 78)

print("""
SAFE VALIDATION

- Existing Streamlit app will NOT be changed.
- Existing production .pkl will NOT be changed.
- No model will be saved.
- No application files will be changed.
- Candidate will only be trained and evaluated.
""")


# ==========================================================
# LOAD DATA
# ==========================================================

DATA_FILE = "output/Crop_Normalized.xlsx"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

df = pd.read_excel(DATA_FILE)

print(f"Raw dataset records: {len(df):,}")


# ==========================================================
# FEATURES
# Agro_Climatic Zone intentionally removed
# ==========================================================

features = [
    "Soil_Type",
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
    "State_Name",
]

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

categorical_cols = [
    "Soil_Type",
    "State_Name",
]

target = "Crop"


# ==========================================================
# VERIFY FEATURES
# ==========================================================

required_columns = features + [target]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

print()
print("Feature verification: PASSED")
print(f"Feature count: {len(features)}")
print("Agro_Climatic Zone included:", "Agro_Climatic Zone" in features)


# ==========================================================
# CLEAN
# ==========================================================

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

before_clean = len(df)

df = df.dropna(
    subset=features + [target]
).copy()

df = df.drop_duplicates()

crop_mapping = {
    "Sugar Cane": "Sugarcane",
    "Paddy (Rice)": "Rice",
    "Oilseeds (Mustard)": "Mustard",
    "Pulses (Arhar)": "Pulses",
}

df[target] = df[target].replace(
    crop_mapping
)

counts = df[target].value_counts()

valid_classes = counts[
    counts >= 20
].index

df = df[
    df[target].isin(valid_classes)
].copy()

print()
print("DATA CLEANING")
print("-" * 78)
print(f"Before cleaning : {before_clean:,}")
print(f"After cleaning  : {len(df):,}")
print(f"Crop classes    : {len(valid_classes)}")

print()
print("Crop distribution:")
print(df[target].value_counts())


# ==========================================================
# SPLIT
# ==========================================================

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print()
print("=" * 78)
print("TRAIN / TEST VALIDATION")
print("=" * 78)

print(f"Training records : {len(X_train):,}")
print(f"Testing records  : {len(X_test):,}")

print()
print("Training distribution:")
print(y_train.value_counts())

print()
print("Testing distribution:")
print(y_test.value_counts())


# ==========================================================
# LEAKAGE CHECK
# ==========================================================

train_indices = set(X_train.index)
test_indices = set(X_test.index)

overlap = train_indices.intersection(test_indices)

print()
print("=" * 78)
print("DATA LEAKAGE CHECK")
print("=" * 78)

print(f"Train/test index overlap: {len(overlap)}")

if overlap:
    raise RuntimeError(
        "DATA LEAKAGE DETECTED!"
    )

print("Leakage check: PASSED")


# ==========================================================
# FULL OVERSAMPLING
# TRAINING DATA ONLY
# ==========================================================

train_df = X_train.copy()
train_df[target] = y_train.values

max_count = train_df[target].value_counts().max()

balanced_parts = []

for crop, group in train_df.groupby(target):

    if len(group) < max_count:

        group = resample(
            group,
            replace=True,
            n_samples=max_count,
            random_state=42,
        )

    balanced_parts.append(group)

balanced_train = pd.concat(
    balanced_parts,
    ignore_index=True
)

balanced_train = balanced_train.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_balanced = balanced_train[features]
y_balanced = balanced_train[target]

print()
print("=" * 78)
print("BALANCING VALIDATION")
print("=" * 78)

print(f"Original training records : {len(X_train):,}")
print(f"Balanced training records : {len(X_balanced):,}")

print()
print("Balanced distribution:")
print(y_balanced.value_counts())


# ==========================================================
# PREPROCESSOR
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_cols,
        ),
        (
            "num",
            "passthrough",
            numeric_cols,
        ),
    ]
)


# ==========================================================
# FINAL CANDIDATE
# ==========================================================

candidate = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=700,
                max_depth=30,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
                class_weight=None,
            ),
        ),
    ]
)


# ==========================================================
# TRAIN
# ==========================================================

print()
print("=" * 78)
print("TRAINING FINAL CANDIDATE")
print("=" * 78)

candidate.fit(
    X_balanced,
    y_balanced
)

print("Training complete.")


# ==========================================================
# ENCODED FEATURE COUNT
# ==========================================================

preprocessor_fitted = candidate.named_steps["preprocessor"]

encoded_count = (
    preprocessor_fitted
    .transform(X_test.iloc[:1])
    .shape[1]
)

print()
print(f"Encoded feature count: {encoded_count}")


# ==========================================================
# PREDICT
# ==========================================================

predictions = candidate.predict(
    X_test
)


# ==========================================================
# METRICS
# ==========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

print()
print("=" * 78)
print("FINAL VALIDATION RESULTS")
print("=" * 78)

print(f"Accuracy    : {accuracy * 100:.2f}%")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")


# ==========================================================
# PER-CROP REPORT
# ==========================================================

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0,
)

print()
print("=" * 78)
print("PER-CROP PERFORMANCE")
print("=" * 78)

rows = []

for crop in sorted(valid_classes):

    if crop not in report:
        continue

    rows.append({
        "Crop": crop,
        "Precision": report[crop]["precision"],
        "Recall": report[crop]["recall"],
        "F1": report[crop]["f1-score"],
        "Samples": int(report[crop]["support"]),
    })

crop_df = pd.DataFrame(rows)

print(
    crop_df.to_string(
        index=False,
        formatters={
            "Precision": "{:.3f}".format,
            "Recall": "{:.3f}".format,
            "F1": "{:.3f}".format,
        },
    )
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

labels = sorted(valid_classes)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels,
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels,
)

print()
print("=" * 78)
print("CONFUSION MATRIX")
print("=" * 78)

print(
    cm_df.to_string()
)


# ==========================================================
# WEAK CROP SUMMARY
# ==========================================================

print()
print("=" * 78)
print("WEAK-CROP SUMMARY")
print("=" * 78)

weak = crop_df[
    crop_df["F1"] < 0.50
]

strong = crop_df[
    crop_df["F1"] >= 0.70
]

print()
print(
    f"Strong crops (F1 >= 0.70): "
    f"{len(strong)}"
)

if len(strong):
    print(
        ", ".join(strong["Crop"])
    )

print()
print(
    f"Weak crops (F1 < 0.50): "
    f"{len(weak)}"
)

if len(weak):
    print(
        ", ".join(weak["Crop"])
    )


# ==========================================================
# EXPECTED RESULT CHECK
# ==========================================================

print()
print("=" * 78)
print("REPRODUCTION CHECK")
print("=" * 78)

EXPECTED_MACRO_F1 = 0.6780

difference = abs(
    macro_f1 - EXPECTED_MACRO_F1
)

print(
    f"Expected Macro F1 : {EXPECTED_MACRO_F1:.4f}"
)

print(
    f"Actual Macro F1   : {macro_f1:.4f}"
)

print(
    f"Difference        : {difference:.4f}"
)

if difference <= 0.005:
    print()
    print("REPRODUCTION CHECK: PASSED")
else:
    print()
    print("REPRODUCTION CHECK: FAILED")
    print(
        "The candidate did not reproduce the "
        "previous experiment closely enough."
    )


# ==========================================================
# SAFETY
# ==========================================================

print()
print("=" * 78)
print("VALIDATION COMPLETE")
print("=" * 78)

print()
print("NO .pkl MODEL WAS CREATED.")
print("NO EXISTING MODEL WAS OVERWRITTEN.")
print("NO STREAMLIT FILE WAS CHANGED.")
print()
print("DO NOT DEPLOY YET.")
print("=" * 78)