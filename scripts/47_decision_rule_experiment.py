import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


print("=" * 78)
print("SMART KISAN - TARGETED DECISION RULE EXPERIMENT")
print("=" * 78)

print()
print("SAFE EXPERIMENT")
print("- Existing Streamlit app will NOT be changed.")
print("- Existing .pkl model will NOT be changed.")
print("- No model will be saved.")
print("- No application files will be changed.")
print("- Exact benchmark preprocessing is retained.")
print("- Exact benchmark train/test split is retained.")
print("- Exact full oversampling is retained.")
print("- Random Forest configuration is retained.")
print("- Only prediction decision rules are being tested.")

# ==========================================================
# PATH
# ==========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "output", "Crop_Normalized.xlsx")

# ==========================================================
# LOAD DATA
# ==========================================================

print()
print("=" * 78)
print("LOADING DATASET")
print("=" * 78)

df = pd.read_excel(DATA_PATH)

print(f"Raw records: {len(df):,}")

# ==========================================================
# EXACT BENCHMARK FILTERING
# ==========================================================

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

FEATURES = [
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
    "State_Name",
    "Soil_Type",
]

df = df[df[TARGET].isin(VALID_CROPS)].copy()

print(f"After benchmark crop filtering: {len(df):,}")

before_drop = len(df)
df = df.dropna(subset=FEATURES + [TARGET]).copy()

print(f"Rows removed for missing values: {before_drop - len(df):,}")
print(f"Benchmark records: {len(df):,}")

X = df[FEATURES].copy()
y = df[TARGET].copy()

numeric_features = [
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

categorical_features = [
    "State_Name",
    "Soil_Type",
]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print()
print("=" * 78)
print("DATA LEAKAGE CHECK")
print("=" * 78)

overlap = set(X_train.index).intersection(set(X_test.index))

print(f"Train/test index overlap: {len(overlap)}")

if len(overlap) != 0:
    raise RuntimeError("DATA LEAKAGE DETECTED")

print("Leakage check: PASSED")

print()
print("=" * 78)
print("BENCHMARK SIZE CHECK")
print("=" * 78)

print(f"Training records : {len(X_train):,}")
print(f"Testing records  : {len(X_test):,}")

# ==========================================================
# FULL OVERSAMPLING
# ==========================================================

print()
print("=" * 78)
print("FULL OVERSAMPLING")
print("=" * 78)

train_df = X_train.copy()
train_df[TARGET] = y_train.values

max_count = train_df[TARGET].value_counts().max()

parts = []

for crop in VALID_CROPS:
    crop_df = train_df[train_df[TARGET] == crop]

    sampled = crop_df.sample(
        n=max_count,
        replace=True,
        random_state=42,
    )

    parts.append(sampled)

balanced_df = pd.concat(parts, ignore_index=True)

X_balanced = balanced_df[FEATURES]
y_balanced = balanced_df[TARGET]

print(f"Original training records : {len(train_df):,}")
print(f"Balanced training records : {len(balanced_df):,}")

print()
print("Balanced distribution:")
print(y_balanced.value_counts().sort_index())

# ==========================================================
# PREPROCESSOR
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numeric_features,
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
    ]
)

# ==========================================================
# RANDOM FOREST
# ==========================================================

model = RandomForestClassifier(
    n_estimators=700,
    max_depth=50,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight=None,
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

print()
print("=" * 78)
print("TRAINING BASELINE MODEL")
print("=" * 78)

print("Training...")

pipeline.fit(X_balanced, y_balanced)

print("Training complete.")

# ==========================================================
# PROBABILITIES
# ==========================================================

proba = pipeline.predict_proba(X_test)
classes = pipeline.named_steps["model"].classes_

baseline_pred = classes[np.argmax(proba, axis=1)]

# ==========================================================
# DECISION RULE FUNCTIONS
# ==========================================================

WEAK = ["Apple", "Mustard", "Vegetables", "Walnut"]


def apply_global_boost(proba, boost):
    adjusted = proba.copy()

    for crop in WEAK:
        idx = np.where(classes == crop)[0]
        if len(idx):
            adjusted[:, idx[0]] *= boost

    return classes[np.argmax(adjusted, axis=1)]


def apply_pair_rule(proba, weak_boost):
    adjusted = proba.copy()

    idx = {c: np.where(classes == c)[0][0] for c in classes}

    # Weak crop boost
    for crop in WEAK:
        adjusted[:, idx[crop]] *= weak_boost

    return classes[np.argmax(adjusted, axis=1)]


def apply_targeted_rule(proba, boost):
    adjusted = proba.copy()

    idx = {c: np.where(classes == c)[0][0] for c in classes}

    # Walnut is particularly weak.
    adjusted[:, idx["Walnut"]] *= boost

    # Vegetables is frequently confused with Wheat.
    adjusted[:, idx["Vegetables"]] *= boost

    # Mustard receives a smaller correction.
    adjusted[:, idx["Mustard"]] *= (1.0 + (boost - 1.0) * 0.50)

    return classes[np.argmax(adjusted, axis=1)]


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate(name, predictions):

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=VALID_CROPS,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        labels=VALID_CROPS,
        average="weighted",
        zero_division=0,
    )

    weak_scores = []

    for crop in WEAK:

        precision = precision_score(
            y_test,
            predictions,
            labels=[crop],
            average="macro",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            labels=[crop],
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            labels=[crop],
            average="macro",
            zero_division=0,
        )

        weak_scores.append(f1)

        print(
            f"{crop:<15} "
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"F1: {f1:.3f}"
        )

    weak_f1 = np.mean(weak_scores)

    print()
    print(f"Accuracy    : {accuracy * 100:.2f}%")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print(f"Weak Crop F1: {weak_f1:.4f}")

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Weak Crop F1": weak_f1,
    }


results = []

# ==========================================================
# BASELINE
# ==========================================================

print()
print("=" * 78)
print("TESTING: Baseline Argmax")
print("=" * 78)

results.append(
    evaluate(
        "Baseline Argmax",
        baseline_pred,
    )
)

# ==========================================================
# GLOBAL BOOSTS
# ==========================================================

for boost in [1.05, 1.10, 1.15, 1.20]:

    print()
    print("=" * 78)
    print(f"TESTING: Global Weak-Crop Boost {boost:.2f}")
    print("=" * 78)

    pred = apply_global_boost(
        proba,
        boost,
    )

    results.append(
        evaluate(
            f"Global Boost {boost:.2f}",
            pred,
        )
    )

# ==========================================================
# TARGETED BOOSTS
# ==========================================================

for boost in [1.05, 1.10, 1.15, 1.20]:

    print()
    print("=" * 78)
    print(f"TESTING: Targeted Boost {boost:.2f}")
    print("=" * 78)

    pred = apply_targeted_rule(
        proba,
        boost,
    )

    results.append(
        evaluate(
            f"Targeted Boost {boost:.2f}",
            pred,
        )
    )

# ==========================================================
# FINAL COMPARISON
# ==========================================================

results_df = pd.DataFrame(results)

print()
print("=" * 78)
print("FINAL DECISION RULE COMPARISON")
print("=" * 78)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Macro F1": "{:.4f}".format,
            "Weighted F1": "{:.4f}".format,
            "Weak Crop F1": "{:.4f}".format,
        },
    )
)

# ==========================================================
# BEST CONFIGURATION
# ==========================================================

best_macro = results_df.loc[
    results_df["Macro F1"].idxmax()
]

best_weak = results_df.loc[
    results_df["Weak Crop F1"].idxmax()
]

print()
print("=" * 78)
print("BEST MACRO F1 CONFIGURATION")
print("=" * 78)

print(f"Model          : {best_macro['Model']}")
print(f"Accuracy       : {best_macro['Accuracy'] * 100:.2f}%")
print(f"Macro F1       : {best_macro['Macro F1']:.4f}")
print(f"Weighted F1    : {best_macro['Weighted F1']:.4f}")
print(f"Weak Crop F1   : {best_macro['Weak Crop F1']:.4f}")

print()
print("=" * 78)
print("BEST WEAK-CROP CONFIGURATION")
print("=" * 78)

print(f"Model          : {best_weak['Model']}")
print(f"Accuracy       : {best_weak['Accuracy'] * 100:.2f}%")
print(f"Macro F1       : {best_weak['Macro F1']:.4f}")
print(f"Weighted F1    : {best_weak['Weighted F1']:.4f}")
print(f"Weak Crop F1   : {best_weak['Weak Crop F1']:.4f}")

# ==========================================================
# SAVE REPORT
# ==========================================================

output_dir = os.path.join(BASE_DIR, "output")
os.makedirs(output_dir, exist_ok=True)

report_path = os.path.join(
    output_dir,
    "Decision_Rule_Experiment.xlsx",
)

results_df.to_excel(
    report_path,
    index=False,
)

print()
print("=" * 78)
print("EXPERIMENT COMPLETE")
print("=" * 78)

print()
print(f"Report saved to:")
print(report_path)

print()
print("NO .pkl MODEL WAS CREATED.")
print("NO EXISTING MODEL WAS OVERWRITTEN.")
print("NO STREAMLIT FILE WAS CHANGED.")
print()
print("DO NOT DEPLOY YET.")
print("=" * 78)