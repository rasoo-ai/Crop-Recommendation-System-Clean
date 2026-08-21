import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample


print("=" * 78)
print("SMART KISAN - FEATURE IMPORTANCE EXPERIMENT")
print("=" * 78)

print("""
SAFE EXPERIMENT
- Existing Streamlit app will NOT be changed.
- Existing .pkl model will NOT be changed.
- No model will be saved.
- No application files will be changed.
- Same preprocessing as current benchmark.
- Agro_Climatic Zone is excluded.
- Full oversampling is retained.
""")

# ==========================================================
# LOAD
# ==========================================================

df = pd.read_excel(
    "output/Crop_Normalized.xlsx"
)

print(f"Dataset loaded: {len(df):,}")

# ==========================================================
# FEATURES
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

if overlap != 0:
    raise RuntimeError(
        "DATA LEAKAGE DETECTED"
    )

print("Leakage check: PASSED")

# ==========================================================
# OVERSAMPLE
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

print(
    f"Balanced records    : {len(X_balanced):,}"
)

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
# BASELINE MODEL
# ==========================================================

print()
print("=" * 78)
print("TRAINING BASELINE")
print("=" * 78)

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
                class_weight=None,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

model.fit(
    X_balanced,
    y_balanced
)

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

print()
print(
    f"Accuracy    : {accuracy * 100:.2f}%"
)

print(
    f"Macro F1    : {macro_f1:.4f}"
)

print(
    f"Weighted F1 : {weighted_f1:.4f}"
)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

print()
print("=" * 78)
print("FEATURE IMPORTANCE")
print("=" * 78)

fitted_preprocessor = model.named_steps[
    "preprocessor"
]

rf = model.named_steps[
    "model"
]

encoded_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

importances = rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": encoded_names,
    "Importance": importances,
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print()
print("TOP 30 ENCODED FEATURES")
print("-" * 78)

print(
    importance_df.head(30).to_string(
        index=False
    )
)

# ==========================================================
# AGGREGATE ORIGINAL FEATURES
# ==========================================================

def original_feature(encoded_name):

    if encoded_name.startswith("cat__"):
        name = encoded_name.replace(
            "cat__",
            "",
            1
        )

        for col in categorical_cols:
            if name.startswith(col + "_"):
                return col

        return name

    if encoded_name.startswith("num__"):
        return encoded_name.replace(
            "num__",
            "",
            1
        )

    return encoded_name


importance_df[
    "Original Feature"
] = importance_df[
    "Feature"
].apply(original_feature)

aggregate = (
    importance_df
    .groupby("Original Feature")[
        "Importance"
    ]
    .sum()
    .sort_values(
        ascending=False
    )
)

print()
print("=" * 78)
print("AGGREGATED ORIGINAL FEATURE IMPORTANCE")
print("=" * 78)

print(
    aggregate.to_string()
)

# ==========================================================
# WEAK CROP PERFORMANCE
# ==========================================================

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0,
)

weak_crops = [
    "Apple",
    "Mustard",
    "Vegetables",
    "Walnut",
]

weak_scores = []

print()
print("=" * 78)
print("WEAK-CROP BASELINE")
print("=" * 78)

for crop in weak_crops:

    if crop not in report:
        continue

    f1 = report[crop]["f1-score"]

    weak_scores.append(f1)

    print(
        f"{crop:<15}"
        f" Precision: "
        f"{report[crop]['precision']:.3f}"
        f" | Recall: "
        f"{report[crop]['recall']:.3f}"
        f" | F1: "
        f"{f1:.3f}"
    )

weak_f1 = (
    sum(weak_scores) /
    len(weak_scores)
)

print()
print(
    f"Average Weak-Crop F1: "
    f"{weak_f1:.4f}"
)

# ==========================================================
# LOW IMPORTANCE FEATURES
# ==========================================================

print()
print("=" * 78)
print("LOW IMPORTANCE FEATURES")
print("=" * 78)

low_features = aggregate[
    aggregate < aggregate.median()
]

print(
    low_features.to_string()
)

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
