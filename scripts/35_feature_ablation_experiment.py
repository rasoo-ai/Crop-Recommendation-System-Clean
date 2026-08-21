import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample


print("=" * 78)
print("SMART KISAN - FEATURE ABLATION EXPERIMENT")
print("=" * 78)

print("""
SAFE EXPERIMENT
- Existing Streamlit app will NOT be changed.
- Existing .pkl model will NOT be changed.
- No model will be saved.
- Same dataset/preprocessing as script 19.
""")


# ==========================================================
# LOAD SAME DATA AS SCRIPT 19
# ==========================================================

df = pd.read_excel(
    "output/Crop_Normalized.xlsx"
)

print(f"Dataset loaded: {len(df):,} records")


# ==========================================================
# FEATURES
# ==========================================================

all_features = [
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
    "Agro_Climatic Zone",
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
    "Agro_Climatic Zone",
]

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
    subset=all_features + [target]
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

X = df[all_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ==========================================================
# BALANCE TRAINING DATA
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

X_balanced = balanced_train[all_features]
y_balanced = balanced_train[target]


# ==========================================================
# INFORMATION
# ==========================================================

print()
print("=" * 78)
print("DATASET INFORMATION")
print("=" * 78)

print(f"Total records       : {len(df):,}")
print(f"Training records    : {len(X_train):,}")
print(f"Testing records     : {len(X_test):,}")
print(f"Balanced records    : {len(X_balanced):,}")
print(f"Crop classes        : {len(valid_classes)}")


# ==========================================================
# EXPERIMENTS
# ==========================================================

experiments = {

    "All Features": {
        "features": all_features,
        "categorical": categorical_cols,
        "numeric": numeric_cols,
    },

    "Without Agro Zone": {
        "features": [
            c for c in all_features
            if c != "Agro_Climatic Zone"
        ],
        "categorical": [
            c for c in categorical_cols
            if c != "Agro_Climatic Zone"
        ],
        "numeric": numeric_cols,
    },

    "Without State": {
        "features": [
            c for c in all_features
            if c != "State_Name"
        ],
        "categorical": [
            c for c in categorical_cols
            if c != "State_Name"
        ],
        "numeric": numeric_cols,
    },

    "Without Soil Type": {
        "features": [
            c for c in all_features
            if c != "Soil_Type"
        ],
        "categorical": [
            c for c in categorical_cols
            if c != "Soil_Type"
        ],
        "numeric": numeric_cols,
    },

    "Numeric Only": {
        "features": numeric_cols,
        "categorical": [],
        "numeric": numeric_cols,
    },
}


# ==========================================================
# RUN EXPERIMENTS
# ==========================================================

results = []

for name, config in experiments.items():

    print()
    print("=" * 78)
    print(f"TESTING: {name}")
    print("=" * 78)

    features = config["features"]
    cats = config["categorical"]
    nums = config["numeric"]

    print(f"Feature count: {len(features)}")
    print("Training...")

    transformers = []

    if cats:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                cats,
            )
        )

    if nums:
        transformers.append(
            (
                "num",
                "passthrough",
                nums,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=30,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    model.fit(
        X_balanced[features],
        y_balanced
    )

    predictions = model.predict(
        X_test[features]
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

    print()
    print(f"Accuracy    : {accuracy * 100:.2f}%")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    print()
    print("Per-Crop Performance")
    print("-" * 78)

    for crop in sorted(valid_classes):

        if crop not in report:
            continue

        print(
            f"{crop:<20}"
            f" Precision: {report[crop]['precision']:.3f}"
            f" | Recall: {report[crop]['recall']:.3f}"
            f" | F1: {report[crop]['f1-score']:.3f}"
            f" | Samples: {int(report[crop]['support'])}"
        )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
    })


# ==========================================================
# FINAL RESULTS
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Macro F1",
    ascending=False
)

print()
print("=" * 78)
print("FINAL FEATURE COMPARISON")
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
print("BEST FEATURE CONFIGURATION")
print("=" * 78)

print(f"Model       : {best['Model']}")
print(f"Accuracy    : {best['Accuracy'] * 100:.2f}%")
print(f"Macro F1    : {best['Macro F1']:.4f}")
print(f"Weighted F1 : {best['Weighted F1']:.4f}")


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
print("Use Macro F1 and weak-crop performance")
print("to choose the next model.")
print("=" * 78)