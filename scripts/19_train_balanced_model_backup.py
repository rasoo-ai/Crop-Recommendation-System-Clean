import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_excel(
    "output/Crop_Normalized.xlsx"
)

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
    subset=features + [target]
)

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
# TRAIN / TEST SPLIT
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
# OVERSAMPLE TRAINING DATA ONLY
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
# RANDOM FOREST
# ==========================================================

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
                max_depth=40,
                min_samples_split=5,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# ==========================================================
# TRAIN
# ==========================================================

print("=" * 70)
print("BALANCED CROP MODEL")
print("=" * 70)

print("\nOriginal training distribution:")
print(
    y_train.value_counts()
)

print("\nBalanced training distribution:")
print(
    y_balanced.value_counts()
)

print("\nTraining balanced model...")

model.fit(
    X_balanced,
    y_balanced
)


# ==========================================================
# EVALUATE ON UNTOUCHED TEST SET
# ==========================================================

pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    pred
)

report = classification_report(
    y_test,
    pred,
    zero_division=0
)

report_dict = classification_report(
    y_test,
    pred,
    output_dict=True,
    zero_division=0
)

macro_f1 = report_dict[
    "macro avg"
]["f1-score"]

weighted_f1 = report_dict[
    "weighted avg"
]["f1-score"]


print("\n" + "=" * 70)
print("BALANCED MODEL RESULTS")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

print(
    f"Macro F1: {macro_f1:.2f}"
)

print(
    f"Weighted F1: {weighted_f1:.2f}"
)

print("\nClassification Report:")
print(report)


# ==========================================================
# SAVE EXPERIMENTAL MODEL
# ==========================================================

joblib.dump(
    model,
    "output/crop_prediction_model_balanced.pkl"
)

print(
    "\nExperimental model saved to:"
    " output/crop_prediction_model_balanced.pkl"
)

print("=" * 70)
print("BALANCED MODEL TEST COMPLETED")
print("=" * 70)
