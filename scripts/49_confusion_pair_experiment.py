import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from imblearn.over_sampling import RandomOverSampler


print("=" * 78)
print("SMART KISAN - CONFUSION PAIR EXPERIMENT")
print("=" * 78)

print("""
SAFE EXPERIMENT
- Existing Streamlit app will NOT be changed.
- Existing .pkl model will NOT be changed.
- No model will be saved.
- No application files will be changed.
- Exact benchmark preprocessing is retained.
- Exact benchmark train/test split is retained.
- Exact full oversampling is retained.
- Random Forest configuration is retained.
- Only targeted confusion-pair decision rules are tested.
""")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "output", "Crop_Normalized.xlsx")
OUTPUT_PATH = os.path.join(
    BASE_DIR, "output", "Confusion_Pair_Experiment.xlsx"
)

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

ALLOWED_CROPS = [
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

TARGET = "Crop"

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

# Confusion pairs identified by Step 45.
CONFUSION_PAIRS = [
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

# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------

print("=" * 78)
print("LOADING DATASET")
print("=" * 78)

print(f"Dataset path: {DATA_PATH}")

df = pd.read_excel(DATA_PATH)

print(f"Raw records: {len(df):,}")

# Exact benchmark crop filtering.
df = df[df[TARGET].isin(ALLOWED_CROPS)].copy()

print(f"After benchmark crop filtering: {len(df):,}")

# Exact benchmark feature selection.
required_columns = FEATURES + [TARGET]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required benchmark columns:\n"
        + "\n".join(missing_columns)
    )

df = df[required_columns].copy()

before_missing = len(df)

df = df.dropna().copy()

removed_missing = before_missing - len(df)

print(f"Rows removed for missing values: {removed_missing:,}")
print(f"Benchmark records: {len(df):,}")

# ---------------------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("DATA LEAKAGE CHECK")
print("=" * 78)

X = df[FEATURES].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("Train/test index overlap:", len(set(X_train.index) & set(X_test.index)))

if len(set(X_train.index) & set(X_test.index)) != 0:
    raise RuntimeError("DATA LEAKAGE DETECTED")

print("Leakage check: PASSED")

print()
print("=" * 78)
print("BENCHMARK SIZE CHECK")
print("=" * 78)

print(f"Training records : {len(X_train):,}")
print(f"Testing records  : {len(X_test):,}")

# ---------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("PREPROCESSING")
print("=" * 78)

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
                handle_unknown="ignore",
                sparse_output=False,
            ),
            CATEGORICAL_FEATURES,
        ),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(
    f"Processed training features: "
    f"{X_train_processed.shape[1]}"
)

print(
    f"Processed testing features : "
    f"{X_test_processed.shape[1]}"
)

# ---------------------------------------------------------------------
# FULL OVERSAMPLING
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("FULL OVERSAMPLING")
print("=" * 78)

oversampler = RandomOverSampler(
    random_state=RANDOM_STATE
)

X_balanced, y_balanced = oversampler.fit_resample(
    X_train_processed,
    y_train,
)

print(f"Original training records : {len(X_train):,}")
print(f"Balanced training records : {len(X_balanced):,}")

print()
print("Balanced distribution:")

print(
    pd.Series(y_balanced)
    .value_counts()
    .sort_index()
)

# ---------------------------------------------------------------------
# RANDOM FOREST
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("TRAINING BASELINE MODEL")
print("=" * 78)

model = RandomForestClassifier(
    n_estimators=700,
    max_depth=50,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight=None,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("Training...")

model.fit(
    X_balanced,
    y_balanced,
)

print("Training complete.")

# ---------------------------------------------------------------------
# PROBABILITIES
# ---------------------------------------------------------------------

probabilities = model.predict_proba(X_test_processed)

# IMPORTANT:
# Convert classes to NumPy array so array indexing is safe.
classes = np.asarray(model.classes_)

baseline_indices = np.argmax(probabilities, axis=1)

baseline_predictions = classes[baseline_indices]

# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def calculate_metrics(y_true, predictions):
    accuracy = accuracy_score(y_true, predictions)

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )

    weak_scores = []

    for crop in WEAK_CROPS:
        mask = y_true == crop

        if mask.sum() == 0:
            continue

        score = f1_score(
            y_true[mask],
            predictions[mask],
            labels=[crop],
            average="macro",
            zero_division=0,
        )

        weak_scores.append(score)

    weak_f1 = (
        float(np.mean(weak_scores))
        if weak_scores
        else 0.0
    )

    return accuracy, macro_f1, weighted_f1, weak_f1


def apply_pair_rule(
    probabilities,
    classes,
    source_crop,
    target_crop,
    boost,
):
    """
    If the target crop is predicted and the source crop has
    sufficiently competitive probability, boost the source crop.

    This is a controlled decision-rule experiment only.
    """

    adjusted = probabilities.copy()

    try:
        source_idx = np.where(classes == source_crop)[0][0]
        target_idx = np.where(classes == target_crop)[0][0]
    except IndexError:
        return classes[np.argmax(adjusted, axis=1)]

    target_prob = adjusted[:, target_idx]
    source_prob = adjusted[:, source_idx]

    # Boost source only when target currently dominates,
    # and source has a meaningful probability.
    mask = (
        (target_prob > source_prob)
        & (source_prob * boost >= target_prob)
    )

    adjusted[mask, source_idx] = (
        adjusted[mask, source_idx] * boost
    )

    predictions = classes[
        np.argmax(adjusted, axis=1)
    ]

    return predictions


def apply_multiple_rules(
    probabilities,
    classes,
    rules,
):
    adjusted = probabilities.copy()

    for source_crop, target_crop, boost in rules:

        try:
            source_idx = np.where(
                classes == source_crop
            )[0][0]

            target_idx = np.where(
                classes == target_crop
            )[0][0]
        except IndexError:
            continue

        target_prob = adjusted[:, target_idx]
        source_prob = adjusted[:, source_idx]

        mask = (
            (target_prob > source_prob)
            & (source_prob * boost >= target_prob)
        )

        adjusted[mask, source_idx] = (
            adjusted[mask, source_idx] * boost
        )

    return classes[
        np.argmax(adjusted, axis=1)
    ]


# ---------------------------------------------------------------------
# BASELINE
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("BASELINE")
print("=" * 78)

accuracy, macro_f1, weighted_f1, weak_f1 = calculate_metrics(
    y_test.to_numpy(),
    baseline_predictions,
)

print(f"Accuracy    : {accuracy * 100:.2f}%")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")
print(f"Weak Crop F1: {weak_f1:.4f}")

results = []

results.append(
    {
        "Model": "Baseline",
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Weak Crop F1": weak_f1,
    }
)

# ---------------------------------------------------------------------
# INDIVIDUAL CONFUSION PAIR TESTS
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("INDIVIDUAL CONFUSION-PAIR TESTS")
print("=" * 78)

boost_values = [1.05, 1.10, 1.15, 1.20]

for source_crop, target_crop in CONFUSION_PAIRS:

    print()
    print("-" * 78)
    print(
        f"PAIR: {source_crop} -> {target_crop}"
    )
    print("-" * 78)

    for boost in boost_values:

        predictions = apply_pair_rule(
            probabilities,
            classes,
            source_crop,
            target_crop,
            boost,
        )

        accuracy, macro_f1, weighted_f1, weak_f1 = (
            calculate_metrics(
                y_test.to_numpy(),
                predictions,
            )
        )

        name = (
            f"{source_crop} -> {target_crop} "
            f"Boost {boost:.2f}"
        )

        print(
            f"{name:<38} "
            f"Accuracy: {accuracy * 100:6.2f}% | "
            f"Macro F1: {macro_f1:.4f} | "
            f"Weak F1: {weak_f1:.4f}"
        )

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Macro F1": macro_f1,
                "Weighted F1": weighted_f1,
                "Weak Crop F1": weak_f1,
            }
        )

# ---------------------------------------------------------------------
# COMBINED TARGETED RULES
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("COMBINED CONFUSION-PAIR RULES")
print("=" * 78)

rule_sets = {
    "Weak Pair Rules": [
        ("Mustard", "Apple", 1.10),
        ("Vegetables", "Apple", 1.10),
        ("Walnut", "Vegetables", 1.10),
        ("Vegetables", "Wheat", 1.10),
        ("Apple", "Walnut", 1.10),
    ],
    "Strong Pair Rules": [
        ("Mustard", "Apple", 1.20),
        ("Vegetables", "Apple", 1.20),
        ("Walnut", "Vegetables", 1.20),
        ("Vegetables", "Wheat", 1.20),
        ("Apple", "Walnut", 1.20),
    ],
}

for rule_name, rules in rule_sets.items():

    predictions = apply_multiple_rules(
        probabilities,
        classes,
        rules,
    )

    accuracy, macro_f1, weighted_f1, weak_f1 = (
        calculate_metrics(
            y_test.to_numpy(),
            predictions,
        )
    )

    print()
    print(
        f"{rule_name}"
    )
    print(
        f"Accuracy    : {accuracy * 100:.2f}%"
    )
    print(
        f"Macro F1    : {macro_f1:.4f}"
    )
    print(
        f"Weighted F1 : {weighted_f1:.4f}"
    )
    print(
        f"Weak Crop F1: {weak_f1:.4f}"
    )

    results.append(
        {
            "Model": rule_name,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "Weak Crop F1": weak_f1,
        }
    )

# ---------------------------------------------------------------------
# FINAL COMPARISON
# ---------------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["Macro F1", "Weak Crop F1", "Accuracy"],
    ascending=False,
).reset_index(drop=True)

print()
print("=" * 78)
print("FINAL CONFUSION-PAIR COMPARISON")
print("=" * 78)

display_df = results_df.copy()

for column in [
    "Accuracy",
    "Macro F1",
    "Weighted F1",
    "Weak Crop F1",
]:
    display_df[column] = display_df[column].map(
        lambda x: f"{x:.4f}"
    )

print(display_df.to_string(index=False))

# ---------------------------------------------------------------------
# BEST CONFIGURATION
# ---------------------------------------------------------------------

best = results_df.iloc[0]

print()
print("=" * 78)
print("BEST CONFIGURATION")
print("=" * 78)

print(f"Model          : {best['Model']}")
print(f"Accuracy       : {best['Accuracy'] * 100:.2f}%")
print(f"Macro F1       : {best['Macro F1']:.4f}")
print(f"Weighted F1    : {best['Weighted F1']:.4f}")
print(f"Weak Crop F1   : {best['Weak Crop F1']:.4f}")

# ---------------------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------------------

print()
print("=" * 78)
print("SAVING EXPERIMENT REPORT")
print("=" * 78)

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

with pd.ExcelWriter(
    OUTPUT_PATH,
    engine="openpyxl",
) as writer:

    results_df.to_excel(
        writer,
        sheet_name="Comparison",
        index=False,
    )

    pd.DataFrame(
        {
            "Actual": y_test.to_numpy(),
            "Baseline": baseline_predictions,
        }
    ).to_excel(
        writer,
        sheet_name="Baseline_Predictions",
        index=False,
    )

print(f"Report saved to:")
print(OUTPUT_PATH)

print()
print("=" * 78)
print("CONFUSION-PAIR EXPERIMENT COMPLETE")
print("=" * 78)

print()
print("NO .pkl MODEL WAS CREATED.")
print("NO EXISTING MODEL WAS OVERWRITTEN.")
print("NO STREAMLIT FILE WAS CHANGED.")
print()
print("DO NOT DEPLOY YET.")
print("=" * 78)