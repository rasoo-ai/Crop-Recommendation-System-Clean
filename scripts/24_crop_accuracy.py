import pandas as pd
import joblib

from sklearn.model_selection import train_test_split


# -----------------------------
# Load data and model
# -----------------------------

df = pd.read_excel(
    "output/Crop_Normalized.xlsx"
)

model = joblib.load(
    "output/crop_prediction_model_balanced.pkl"
)


# -----------------------------
# Same features as evaluation
# -----------------------------

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


# -----------------------------
# Clean exactly like evaluation
# -----------------------------

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


# -----------------------------
# Same test split
# -----------------------------

X = df[features]
y = df["Crop"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

pred = model.predict(X_test)


# -----------------------------
# Crop accuracy
# -----------------------------

result = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": pred
})

result["Correct"] = (
    result["Actual"]
    == result["Predicted"]
)

summary = (
    result
    .groupby("Actual")
    .agg(
        Test_Samples=("Correct", "size"),
        Correct=("Correct", "sum"),
        Wrong=("Correct", lambda x: (~x).sum()),
    )
)

summary["Accuracy"] = (
    summary["Correct"]
    / summary["Test_Samples"]
    * 100
)

summary = summary.sort_values(
    "Accuracy"
)

print("=" * 75)
print("CROP-WISE TEST ACCURACY")
print("=" * 75)

print(summary.to_string(
    formatters={
        "Accuracy": "{:.2f}%".format
    }
))

print("\nTotal test records:",
      len(result))

print("Total correct:",
      int(result["Correct"].sum()))

print("Total wrong:",
      int((~result["Correct"]).sum()))

summary.to_excel(
    "output/Crop_Wise_Accuracy.xlsx"
)

print(
    "\nSaved to: output/Crop_Wise_Accuracy.xlsx"
)
