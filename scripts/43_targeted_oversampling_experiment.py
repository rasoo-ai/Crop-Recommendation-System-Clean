import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample


print("=" * 78)
print("SMART KISAN - TARGETED OVERSAMPLING EXPERIMENT")
print("=" * 78)

print("""
SAFE EXPERIMENT
- Existing Streamlit app will NOT be changed.
- Existing .pkl model will NOT be changed.
- No model will be saved.
- No application files will be changed.
- Same preprocessing is retained.
- Same train/test split is retained.
- Agro_Climatic Zone is excluded.
- All 18 features are retained.
- Random Forest configuration is fixed.
- Only oversampling strategy is being tested.
""")

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_excel(
    "output/Crop_Normalized.xlsx"
)

print(f"Dataset loaded: {len(df):,} records")

# ==========================================================
# FEATURES
# ==========================================================

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

features = categorical_cols + numeric_cols

target = "Crop"

# ==========================================================
# CLEAN
# ==========================================================

for col in numeric_cols:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

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

# ==========================================================
# LEAKAGE CHECK
# ==========================================================

print()
print("=" * 78)
print("DATA LEAKAGE CHECK")
print("=" * 78)

overlap = len(
    set(X_train.index).intersection(
        set(X_test.index)
    )
)

print(
    f"Train/test index overlap: {overlap}"
)

if overlap == 0:
    print("Leakage check: PASSED")
else:
    print("Leakage check: FAILED")
    raise RuntimeError(
        "Train/test leakage detected."
    )

# ==========================================================
# ORIGINAL DISTRIBUTION
# ==========================================================

train_counts = y_train.value_counts()

print()
print("=" * 78)
print("ORIGINAL TRAINING DISTRIBUTION")
print("=" * 78)

print(
    train_counts.to_string()
)

max_count = train_counts.max()

# ==========================================================
# OVERSAMPLING STRATEGIES
# ==========================================================

strategies = {
    "Current Full Balance": {
        "target_mode": "full",
    },

    "75 Percent Balance": {
        "target_mode": "75",
    },

    "50 Percent Balance": {
        "target_mode": "50",
    },

    "Targeted Weak Crop": {
        "target_mode": "targeted",
    },
}

weak_crops = [
    "Apple",
    "Mustard",
    "Vegetables",
    "Walnut",
]

# ==========================================================
# BUILD BALANCED DATA
# ==========================================================

def build_training_set(
    strategy_name,
    strategy_config
):

    train_df = X_train.copy()

    train_df[target] = y_train.values

    original_counts = (
        train_df[target].value_counts()
    )

    maximum = original_counts.max()

    parts = []

    for crop, group in train_df.groupby(
        target
    ):

        current_count = len(group)

        if strategy_config["target_mode"] == "full":

            desired_count = maximum

        elif strategy_config["target_mode"] == "75":

            desired_count = int(
                maximum * 0.75
            )

            desired_count = max(
                desired_count,
                current_count
            )

        elif strategy_config["target_mode"] == "50":

            desired_count = int(
                maximum * 0.50
            )

            desired_count = max(
                desired_count,
                current_count
            )

        elif strategy_config["target_mode"] == "targeted":

            if crop in weak_crops:

                desired_count = int(
                    maximum * 0.75
                )

            else:

                desired_count = current_count

        else:

            raise ValueError(
                "Unknown strategy"
            )

        if current_count < desired_count:

            group = resample(
                group,
                replace=True,
                n_samples=desired_count,
                random_state=42,
            )

        parts.append(group)

    balanced = pd.concat(
        parts,
        ignore_index=True
    )

    balanced = balanced.sample(
        frac=1,
        random_state=42
    ).reset_index(
        drop=True
    )

    return balanced


# ==========================================================
# DATASET INFORMATION
# ==========================================================

print()
print("=" * 78)
print("DATASET INFORMATION")
print("=" * 78)

print(
    f"Total records       : {len(df):,}"
)

print(
    f"Training records    : {len(X_train):,}"
)

print(
    f"Testing records     : {len(X_test):,}"
)

# ==========================================================
# TRAINING
# ==========================================================

results = []

for strategy_name, strategy_config in strategies.items():

    print()
    print("=" * 78)
    print(
        f"TESTING: {strategy_name}"
    )
    print("=" * 78)

    balanced_train = build_training_set(
        strategy_name,
        strategy_config
    )

    X_balanced = balanced_train[
        features
    ]

    y_balanced = balanced_train[
        target
    ]

    print()
    print(
        f"Balanced records: "
        f"{len(balanced_train):,}"
    )

    print()
    print("Training distribution:")
    print(
        y_balanced.value_counts().to_string()
    )

    # ------------------------------------------------------
    # PREPROCESSOR
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=700,
                    max_depth=50,
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

    print()
    print("Random Forest parameters:")
    print("  n_estimators      : 700")
    print("  max_depth         : 50")
    print("  min_samples_split : 5")
    print("  min_samples_leaf  : 2")
    print("  max_features      : sqrt")
    print("  class_weight      : None")

    print()
    print("Training...")

    model.fit(
        X_balanced,
        y_balanced
    )

    print("Training complete.")

    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------

    predictions = model.predict(
        X_test
    )

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

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    weak_values = []

    print()
    print(
        f"Accuracy    : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1    : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : "
        f"{weighted_f1:.4f}"
    )

    print()
    print("Weak-Crop Performance")
    print("-" * 78)

    for crop in weak_crops:

        if crop not in report:
            continue

        crop_f1 = report[crop][
            "f1-score"
        ]

        weak_values.append(
            crop_f1
        )

        print(
            f"{crop:<15}"
            f" Precision: "
            f"{report[crop]['precision']:.3f}"
            f" | Recall: "
            f"{report[crop]['recall']:.3f}"
            f" | F1: "
            f"{crop_f1:.3f}"
        )

    weak_f1 = (
        sum(weak_values)
        / len(weak_values)
    )

    print()
    print(
        f"Average Weak-Crop F1: "
        f"{weak_f1:.4f}"
    )

    results.append({
        "Model": strategy_name,
        "Training Records": len(
            balanced_train
        ),
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Weak Crop F1": weak_f1,
    })

# ==========================================================
# FINAL COMPARISON
# ==========================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Macro F1",
    ascending=False
)

print()
print("=" * 78)
print("FINAL OVERSAMPLING COMPARISON")
print("=" * 78)

display_df = results_df.copy()

display_df["Accuracy"] = (
    display_df["Accuracy"] * 100
).round(2)

display_df["Macro F1"] = (
    display_df["Macro F1"]
).round(4)

display_df["Weighted F1"] = (
    display_df["Weighted F1"]
).round(4)

display_df["Weak Crop F1"] = (
    display_df["Weak Crop F1"]
).round(4)

print(
    display_df.to_string(
        index=False
    )
)

# ==========================================================
# BEST
# ==========================================================

best = results_df.iloc[0]

print()
print("=" * 78)
print("BEST OVERSAMPLING CONFIGURATION")
print("=" * 78)

print(
    f"Model            : "
    f"{best['Model']}"
)

print(
    f"Training Records : "
    f"{int(best['Training Records']):,}"
)

print(
    f"Accuracy         : "
    f"{best['Accuracy'] * 100:.2f}%"
)

print(
    f"Macro F1         : "
    f"{best['Macro F1']:.4f}"
)

print(
    f"Weighted F1      : "
    f"{best['Weighted F1']:.4f}"
)

print(
    f"Weak Crop F1     : "
    f"{best['Weak Crop F1']:.4f}"
)

# ==========================================================
# CURRENT BENCHMARK
# ==========================================================

current_accuracy = 0.9300
current_macro_f1 = 0.6879
current_weighted_f1 = 0.9403
current_weak_f1 = 0.3762

print()
print("=" * 78)
print("CURRENT BENCHMARK COMPARISON")
print("=" * 78)

print(
    f"Current Accuracy : "
    f"{current_accuracy * 100:.2f}%"
)

print(
    f"New Accuracy     : "
    f"{best['Accuracy'] * 100:.2f}%"
)

print(
    f"Accuracy change  : "
    f"{(best['Accuracy'] - current_accuracy) * 100:+.2f} percentage points"
)

print()

print(
    f"Current Macro F1 : "
    f"{current_macro_f1:.4f}"
)

print(
    f"New Macro F1     : "
    f"{best['Macro F1']:.4f}"
)

print(
    f"Macro F1 change  : "
    f"{best['Macro F1'] - current_macro_f1:+.4f}"
)

print()

print(
    f"Current Weighted : "
    f"{current_weighted_f1:.4f}"
)

print(
    f"New Weighted     : "
    f"{best['Weighted F1']:.4f}"
)

print(
    f"Weighted change  : "
    f"{best['Weighted F1'] - current_weighted_f1:+.4f}"
)

print()

print(
    f"Current Weak F1  : "
    f"{current_weak_f1:.4f}"
)

print(
    f"New Weak F1      : "
    f"{best['Weak Crop F1']:.4f}"
)

print(
    f"Weak F1 change   : "
    f"{best['Weak Crop F1'] - current_weak_f1:+.4f}"
)

# ==========================================================
# SAFETY
# ==========================================================

print()
print("=" * 78)
print("EXPERIMENT COMPLETE")
print("=" * 78)

print()
print("NO .pkl MODEL WAS CREATED.")
print("NO EXISTING MODEL WAS OVERWRITTEN.")
print("NO STREAMLIT FILE WAS CHANGED.")
print()
print("DO NOT DEPLOY YET.")
print("=" * 78)
