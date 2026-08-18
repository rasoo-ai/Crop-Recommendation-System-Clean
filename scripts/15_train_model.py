# 15_train_model.py

import time
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    top_k_accuracy_score,
)
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

start = time.time()

print("=" * 70)
print("CROP PREDICTION MODEL TRAINING")
print("=" * 70)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

df = pd.read_excel("output/Crop_Normalized.xlsx")

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

target = "Crop"

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

df = df[features + [target]]

# -------------------------------------------------
# Data Cleaning
# -------------------------------------------------

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()
df = df.drop_duplicates()

crop_mapping = {
    "Sugar Cane": "Sugarcane",
    "Paddy (Rice)": "Rice",
    "Oilseeds (Mustard)": "Mustard",
    "Pulses (Arhar)": "Pulses",
}

df[target] = df[target].replace(crop_mapping)

# Keep classes having at least 20 samples
counts = df[target].value_counts()
valid_classes = counts[counts >= 20].index

df = df[df[target].isin(valid_classes)]

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset:", df.shape)
print(df[target].value_counts())

# -------------------------------------------------
# Train Test Split
# -------------------------------------------------

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            [
                "Soil_Type",
                "State_Name",
                "Agro_Climatic Zone",
            ],
        ),
        (
            "num",
            "passthrough",
            numeric_cols,
        ),
    ]
)

# -------------------------------------------------
# Random Forest
# -------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                class_weight="balanced_subsample",
                bootstrap=True,
                n_jobs=-1,
            ),
        ),
    ]
)

params = {
    "model__n_estimators": [300, 500, 700],
    "model__max_depth": [20, 30, 40, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
}
# -------------------------------------------------
# Hyperparameter Search
# -------------------------------------------------

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=params,
    n_iter=40,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("\nTraining model...")

search.fit(X_train, y_train)

best = search.best_estimator_

# -------------------------------------------------
# Best Parameters
# -------------------------------------------------

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Best Parameters:")
print(search.best_params_)

print(
    "Best CV Accuracy:",
    round(search.best_score_ * 100, 2),
    "%",
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------

pred = best.predict(X_test)
prob = best.predict_proba(X_test)

accuracy = accuracy_score(y_test, pred)

top3 = top_k_accuracy_score(
    y_test,
    prob,
    labels=best.classes_,
    k=3,
)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
print("Top-3 Accuracy:", round(top3 * 100, 2), "%")

print("\nClassification Report\n")
print(classification_report(
    y_test,
    pred,
    zero_division=0,
))

# -------------------------------------------------
# Feature Importance
# -------------------------------------------------

rf = best.named_steps["model"]

feature_names = (
    best.named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf.feature_importances_,
})

importance = importance.sort_values(
    "Importance",
    ascending=False,
)

print("\nTop 20 Important Features")
print("=" * 70)
print(importance.head(20))

# -------------------------------------------------
# Save Model
# -------------------------------------------------

joblib.dump(
    best,
    "output/crop_prediction_model_balanced.pkl",
)

print("\nModel saved successfully!")

print(
    "Execution Time:",
    round(time.time() - start, 2),
    "seconds",
)

print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)
