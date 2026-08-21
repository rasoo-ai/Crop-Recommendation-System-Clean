import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import resample


print("=" * 78)
print("SMART KISAN - REPRODUCIBILITY / BENCHMARK CONSISTENCY CHECK")
print("=" * 78)

print("""
SAFE CHECK
- No .pkl model will be created.
- Existing model will NOT be changed.
- Streamlit will NOT be changed.
- Multiple identical runs will be compared.
- Baseline and Global Boost 1.20 will be compared.
""")

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output",
    "Crop_Normalized.xlsx"
)

RANDOM_STATE = 42

TARGET = "Crop"

VALID_CROPS = [
    "Rice",
    "Maize",
    "Wheat",
    "Pulses",
    "Cotton",
    "Mustard",
    "Vegetables",
    "Apple",
    "Walnut",
]

NUMERIC_FEATURES = [
    "pH_Value",
    "Nitrogen_Value (N)",
    "Phosphorus_Value (P)",
    "Potassium_Value (K)",
    "Electrical_Conductivity (EC)",
    "Organic_Carbon (%)",
    "Zinc (%)",
    "Iron (%)",
    "Manganese (%)",
    "Copper (%)",
    "Boron (%)",
    "Sulphur (%)",
    "Rainfall_cm",
    "temperature_celsius",
    "humidity_percentage",
    "Soil_Moisture (%)",
]

CATEGORICAL_FEATURES = [
    "State_Name",
    "Soil_Type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

WEAK_CROPS = [
    "Apple",
    "Mustard",
    "Vegetables",
    "Walnut",
]

print()
print("=" * 78)
print("LOADING DATASET")
print("=" * 78)

df = pd.read_excel(DATA_PATH)

print(f"Raw records: {len(df):,}")

# Exact benchmark crop filtering
df = df[df[TARGET].isin(VALID_CROPS)].copy()

print(f"After benchmark crop filtering: {len(df):,}")

# Keep only benchmark columns
required_columns = FEATURES + [TARGET]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n" +
        "\n".join(missing_columns)
    )

df = df[required_columns].copy()

before_missing = len(df)

df = df.dropna(subset=required_columns).copy()

removed_missing = before_missing - len(df)

print(f"Rows removed for missing values: {removed_missing:,}")
print(f"Benchmark records: {len(df):,}")

print()
print("=" * 78)
print("BENCHMARK DATASET CHECK")
print("=" * 78)

print(f"Records: {len(df):,}")
print(f"Features: {len(FEATURES)}")

print()
print("Feature count:")
print(f"Numeric     : {len(NUMERIC_FEATURES)}")
print(f"Categorical : {len(CATEGORICAL_FEATURES)}")
print(f"Total       : {len(FEATURES)}")

# ------------------------------------------------------------------
# SPLIT
# ------------------------------------------------------------------

X = df[FEATURES].copy()
y = df[TARGET].copy()

indices = np.arange(len(df))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X,
    y,
    indices,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print()
print("=" * 78)
print("TRAIN / TEST SPLIT")
print("=" * 78)

print(f"Training records : {len(X_train):,}")
print(f"Testing records  : {len(X_test):,}")

overlap = len(set(idx_train).intersection(set(idx_test)))

print(f"Index overlap    : {overlap}")

if overlap != 0:
    raise RuntimeError("DATA LEAKAGE DETECTED")

print("Leakage check: PASSED")

# ------------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            NUMERIC_FEATURES
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            CATEGORICAL_FEATURES
        ),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print()
print("=" * 78)
print("PREPROCESSING")
print("=" * 78)

print(
    f"Processed training features: "
    f"{X_train_processed.shape[1]}"
)

print(
    f"Processed testing features : "
    f"{X_test_processed.shape[1]}"
)

# ------------------------------------------------------------------
# OVERSAMPLING
# ------------------------------------------------------------------

train_processed = pd.DataFrame(
    X_train_processed
)

train_processed[TARGET] = y_train.reset_index(drop=True)

max_count = train_processed[TARGET].value_counts().max()

balanced_parts = []

for crop in VALID_CROPS:

    crop_data = train_processed[
        train_processed[TARGET] == crop
    ]

    if len(crop_data) == 0:
        continue

    if len(crop_data) < max_count:

        crop_data = resample(
            crop_data,
            replace=True,
            n_samples=max_count,
            random_state=RANDOM_STATE
        )

    balanced_parts.append(crop_data)

balanced = pd.concat(
    balanced_parts,
    ignore_index=True
)

X_balanced = balanced.drop(columns=[TARGET]).values
y_balanced = balanced[TARGET].values

print()
print("=" * 78)
print("FULL OVERSAMPLING")
print("=" * 78)

print(f"Original training records : {len(X_train):,}")
print(f"Balanced training records : {len(X_balanced):,}")

print()
print("Balanced distribution:")
print(
    pd.Series(y_balanced)
    .value_counts()
    .sort_index()
)

# ------------------------------------------------------------------
# TRAIN / EVALUATE
# ------------------------------------------------------------------

def train_and_evaluate(run_number, decision_boost=1.0):

    print()
    print("-" * 78)
    print(
        f"RUN {run_number} | "
        f"Decision boost = {decision_boost:.2f}"
    )
    print("-" * 78)

    model = RandomForestClassifier(
        n_estimators=700,
        max_depth=50,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_balanced,
        y_balanced
    )

    probabilities = model.predict_proba(
        X_test_processed
    )

    classes = model.classes_

    if decision_boost == 1.0:

        predictions = classes[
            np.argmax(probabilities, axis=1)
        ]

    else:

        adjusted = probabilities.copy()

        for crop in WEAK_CROPS:

            if crop in classes:

                crop_index = list(classes).index(crop)

                adjusted[:, crop_index] *= decision_boost

        predictions = classes[
            np.argmax(adjusted, axis=1)
        ]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    weak_f1_values = []

    for crop in WEAK_CROPS:

        true_mask = y_test.values == crop

        if true_mask.sum() == 0:
            continue

        from sklearn.metrics import f1_score as single_f1

        crop_f1 = single_f1(
            y_test,
            predictions,
            labels=[crop],
            average="macro",
            zero_division=0
        )

        weak_f1_values.append(crop_f1)

    weak_f1 = (
        float(np.mean(weak_f1_values))
        if weak_f1_values
        else 0.0
    )

    print(f"Accuracy    : {accuracy * 100:.2f}%")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print(f"Weak Crop F1: {weak_f1:.4f}")

    return {
        "run": run_number,
        "boost": decision_boost,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "weak_f1": weak_f1,
    }


# ------------------------------------------------------------------
# REPEATED BASELINE
# ------------------------------------------------------------------

print()
print("=" * 78)
print("REPEATED BASELINE TEST")
print("=" * 78)

results = []

for run in range(1, 4):

    results.append(
        train_and_evaluate(
            run,
            decision_boost=1.0
        )
    )

# ------------------------------------------------------------------
# REPEATED BOOST
# ------------------------------------------------------------------

print()
print("=" * 78)
print("REPEATED GLOBAL BOOST 1.20 TEST")
print("=" * 78)

for run in range(1, 4):

    results.append(
        train_and_evaluate(
            run,
            decision_boost=1.20
        )
    )

# ------------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------------

results_df = pd.DataFrame(results)

print()
print("=" * 78)
print("REPRODUCIBILITY SUMMARY")
print("=" * 78)

print(
    results_df.to_string(
        index=False,
        formatters={
            "accuracy": "{:.4f}".format,
            "macro_f1": "{:.4f}".format,
            "weighted_f1": "{:.4f}".format,
            "weak_f1": "{:.4f}".format,
            "boost": "{:.2f}".format,
        }
    )
)

print()
print("=" * 78)
print("BASELINE STABILITY")
print("=" * 78)

baseline = results_df[
    results_df["boost"] == 1.0
]

print(
    f"Accuracy mean : "
    f"{baseline['accuracy'].mean():.4f}"
)

print(
    f"Accuracy std  : "
    f"{baseline['accuracy'].std(ddof=0):.6f}"
)

print(
    f"Macro F1 mean : "
    f"{baseline['macro_f1'].mean():.4f}"
)

print(
    f"Macro F1 std  : "
    f"{baseline['macro_f1'].std(ddof=0):.6f}"
)

print(
    f"Weak F1 mean  : "
    f"{baseline['weak_f1'].mean():.4f}"
)

print(
    f"Weak F1 std   : "
    f"{baseline['weak_f1'].std(ddof=0):.6f}"
)

print()
print("=" * 78)
print("GLOBAL BOOST 1.20 STABILITY")
print("=" * 78)

boost = results_df[
    results_df["boost"] == 1.20
]

print(
    f"Accuracy mean : "
    f"{boost['accuracy'].mean():.4f}"
)

print(
    f"Accuracy std  : "
    f"{boost['accuracy'].std(ddof=0):.6f}"
)

print(
    f"Macro F1 mean : "
    f"{boost['macro_f1'].mean():.4f}"
)

print(
    f"Macro F1 std  : "
    f"{boost['macro_f1'].std(ddof=0):.6f}"
)

print(
    f"Weak F1 mean  : "
    f"{boost['weak_f1'].mean():.4f}"
)

print(
    f"Weak F1 std   : "
    f"{boost['weak_f1'].std(ddof=0):.6f}"
)

# ------------------------------------------------------------------
# COMPARISON
# ------------------------------------------------------------------

print()
print("=" * 78)
print("BOOST VS BASELINE")
print("=" * 78)

accuracy_change = (
    boost["accuracy"].mean()
    - baseline["accuracy"].mean()
)

macro_change = (
    boost["macro_f1"].mean()
    - baseline["macro_f1"].mean()
)

weak_change = (
    boost["weak_f1"].mean()
    - baseline["weak_f1"].mean()
)

print(
    f"Accuracy change : "
    f"{accuracy_change * 100:+.3f} percentage points"
)

print(
    f"Macro F1 change : "
    f"{macro_change:+.4f}"
)

print(
    f"Weak F1 change  : "
    f"{weak_change:+.4f}"
)

# ------------------------------------------------------------------
# SAVE REPORT
# ------------------------------------------------------------------

output_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "output",
    "Reproducibility_Check.xlsx"
)

results_df.to_excel(
    output_path,
    index=False
)

print()
print("=" * 78)
print("REPRODUCIBILITY CHECK COMPLETE")
print("=" * 78)

print()
print(f"Report saved to:")
print(output_path)

print()
print("NO .pkl MODEL WAS CREATED.")
print("NO EXISTING MODEL WAS OVERWRITTEN.")
print("NO STREAMLIT FILE WAS CHANGED.")
print()
print("DO NOT DEPLOY YET.")
print("=" * 78)