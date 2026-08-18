import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_excel("output/Crop_Normalized.xlsx")


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
    "District_Name",
    "Tehsil_Name",
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
    "District_Name",
    "Tehsil_Name",
    "Agro_Climatic Zone",
]

target = "Crop"


# ==========================================================
# CLEAN
# ==========================================================

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
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

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


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
# MODEL
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
                class_weight="balanced_subsample",
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
print("GEOGRAPHIC FEATURE EXPERIMENT")
print("=" * 70)

print(
    "\nFeatures include:"
    "\nState + District + Tehsil + Agro-climatic zone"
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train,
)


# ==========================================================
# EVALUATE
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

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

print(
    f"Macro F1: {report['macro avg']['f1-score']:.2f}"
)

print(
    f"Weighted F1: {report['weighted avg']['f1-score']:.2f}"
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
# STATE-WISE EVALUATION
# ==========================================================

results = X_test[
    ["State_Name"]
].copy()

results["Actual"] = y_test.values
results["Predicted"] = pred

results["Correct"] = (
    results["Actual"]
    == results["Predicted"]
)

state_summary = (
    results
    .groupby("State_Name")
    .agg(
        Test_Samples=("Correct", "size"),
        Correct=("Correct", "sum"),
        Wrong=("Correct", lambda x: (~x).sum()),
    )
)

state_summary["Accuracy"] = (
    state_summary["Correct"]
    / state_summary["Test_Samples"]
    * 100
)

print("\n" + "=" * 70)
print("STATE-WISE ACCURACY")
print("=" * 70)

print(
    state_summary
    .sort_values("Accuracy")
    .to_string()
)


# ==========================================================
# SAVE EXPERIMENTAL MODEL
# ==========================================================

import joblib

joblib.dump(
    model,
    "output/crop_prediction_model_geo.pkl"
)

state_summary.to_excel(
    "output/Geo_Model_State_Accuracy.xlsx"
)

print(
    "\nExperimental model saved to:"
    " output/crop_prediction_model_geo.pkl"
)

print(
    "State results saved to:"
    " output/Geo_Model_State_Accuracy.xlsx"
)

print("=" * 70)
print("EXPERIMENT COMPLETED")
print("=" * 70)
