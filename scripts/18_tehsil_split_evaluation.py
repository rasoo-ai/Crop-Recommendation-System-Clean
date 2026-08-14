import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

from sklearn.model_selection import GroupShuffleSplit

# --------------------------------------------------
# Load dataset and model
# --------------------------------------------------

df = pd.read_excel("output/Crop_Normalized.xlsx")
model = joblib.load("output/crop_prediction_model.pkl")

# --------------------------------------------------
# Same feature definition as training
# --------------------------------------------------

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

target = "Crop"

# --------------------------------------------------
# Clean data
# --------------------------------------------------

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=features + [target, "State_Name", "District_Name", "Tehsil_Name"]
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

# Keep same class rule as training
counts = df[target].value_counts()
valid_classes = counts[counts >= 20].index
df = df[df[target].isin(valid_classes)].copy()

# --------------------------------------------------
# Group by Tehsil
# --------------------------------------------------

groups = (
    df["State_Name"].astype(str)
    + " | "
    + df["District_Name"].astype(str)
    + " | "
    + df["Tehsil_Name"].astype(str)
)

X = df[features]
y = df[target]

# --------------------------------------------------
# Split by TEHSIL
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups,
    )
)

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# --------------------------------------------------
# Evaluate
# --------------------------------------------------

pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pred
)

print("=" * 70)
print("UNSEEN TEHSIL EVALUATION")
print("=" * 70)

print(
    f"Test records: {len(X_test)}"
)

print(
    f"Test tehsils: {groups.iloc[test_idx].nunique()}"
)

print(
    f"\nTehsil-held-out Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred,
        zero_division=0,
    )
)