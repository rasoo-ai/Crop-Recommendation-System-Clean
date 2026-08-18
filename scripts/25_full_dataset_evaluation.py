import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==========================================================
# LOAD DATA AND MODEL
# ==========================================================

df = pd.read_excel(
    "output/Crop_Normalized.xlsx"
)

model = joblib.load(
    "output/crop_prediction_model_balanced.pkl"
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


# ==========================================================
# CLEAN EXACTLY LIKE CURRENT EVALUATION
# ==========================================================

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
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
# DATA
# ==========================================================

X = df[features]
y = df["Crop"]


# ==========================================================
# SAME TRAIN/TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# PREDICTIONS
# ==========================================================

train_pred = model.predict(
    X_train
)

test_pred = model.predict(
    X_test
)

all_pred = model.predict(
    X
)


# ==========================================================
# ACCURACY
# ==========================================================

train_accuracy = accuracy_score(
    y_train,
    train_pred
)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

all_accuracy = accuracy_score(
    y,
    all_pred
)


# ==========================================================
# RESULTS
# ==========================================================

print("=" * 75)
print("FULL DATASET MODEL EVALUATION")
print("=" * 75)

print(
    f"\nTotal eligible records: {len(df)}"
)

print(
    f"Training records:       {len(X_train)}"
)

print(
    f"Test records:           {len(X_test)}"
)

print("\n" + "-" * 75)

print(
    f"Training accuracy:      {train_accuracy * 100:.2f}%"
)

print(
    f"Held-out test accuracy: {test_accuracy * 100:.2f}%"
)

print(
    f"All-record accuracy:    {all_accuracy * 100:.2f}%"
)

print("-" * 75)

print(
    "\nNote:"
)

print(
    "The held-out test accuracy is the official "
    "generalization metric."
)

print(
    "The all-record accuracy is diagnostic only "
    "because most of those records were used during training."
)

print("=" * 75)
