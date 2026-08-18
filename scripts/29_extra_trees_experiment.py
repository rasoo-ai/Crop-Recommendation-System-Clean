import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==========================================================
# LOAD
# ==========================================================

df = pd.read_excel(
    "output/Crop_Normalized.xlsx"
)


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


# ==========================================================
# CLEAN - SAME AS CURRENT PIPELINE
# ==========================================================

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

df = df.dropna(
    subset=features + ["Crop"]
)

df = df.drop_duplicates()

crop_mapping = {
    "Sugar Cane": "Sugarcane",
    "Paddy (Rice)": "Rice",
    "Oilseeds (Mustard)": "Mustard",
    "Pulses (Arhar)": "Pulses",
}

df["Crop"] = df["Crop"].replace(
    crop_mapping
)

counts = df["Crop"].value_counts()

valid_classes = counts[
    counts >= 20
].index

df = df[
    df["Crop"].isin(valid_classes)
].copy()

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================================
# SAME TEST SPLIT
# ==========================================================

X = df[features]
y = df["Crop"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ==========================================================
# PREPROCESSING
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
# EXTRA TREES
# ==========================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
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

print("=" * 75)
print("EXTRA TREES EXPERIMENT")
print("=" * 75)

print(
    f"\nTraining records: {len(X_train)}"
)

print(
    f"Test records: {len(X_test)}"
)

print("\nTraining...")

model.fit(
    X_train,
    y_train
)


# ==========================================================
# EVALUATION
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
    output_dict=True,
    zero_division=0
)

print("\n" + "=" * 75)
print("RESULT")
print("=" * 75)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{report['macro avg']['f1-score']:.3f}"
)

print(
    f"Weighted F1: "
    f"{report['weighted avg']['f1-score']:.3f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        pred,
        zero_division=0
    )
)


# ==========================================================
# SAVE EXPERIMENTAL MODEL
# ==========================================================

joblib.dump(
    model,
    "output/crop_prediction_model_extra_trees.pkl"
)

print(
    "\nExperimental model saved to:"
    " output/crop_prediction_model_extra_trees.pkl"
)

print("=" * 75)
print("EXPERIMENT COMPLETED")
print("=" * 75)
