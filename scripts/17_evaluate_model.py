import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Load
# --------------------------------------------------

df = pd.read_excel("output/Crop_Normalized.xlsx")
model = joblib.load("output/crop_prediction_model.pkl")

# --------------------------------------------------
# Features
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

# --------------------------------------------------
# Clean data
# --------------------------------------------------

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

counts = df[target].value_counts()
valid_classes = counts[counts >= 20].index
df = df[df[target].isin(valid_classes)]

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# --------------------------------------------------
# Recreate same test split
# --------------------------------------------------

X = df[features]
y = df[target]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# --------------------------------------------------
# Predict
# --------------------------------------------------

pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    pred
)

print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        pred,
        zero_division=0
    )
)

# --------------------------------------------------
# Wrong predictions
# --------------------------------------------------

wrong = X_test.copy()

wrong["Actual_Crop"] = y_test.values
wrong["Predicted_Crop"] = pred

wrong = wrong[
    wrong["Actual_Crop"]
    != wrong["Predicted_Crop"]
]

print("\n" + "=" * 70)
print("WRONG PREDICTIONS")
print("=" * 70)

print(
    f"\nWrong predictions: {len(wrong)}"
)

print(
    f"Total test records: {len(X_test)}"
)

if len(X_test) > 0:

    error_rate = (
        len(wrong) / len(X_test) * 100
    )

    print(
        f"Error rate: {error_rate:.2f}%"
    )

# --------------------------------------------------
# Most common mistakes
# --------------------------------------------------

if not wrong.empty:

    print("\nMost common incorrect predictions:")

    mistakes = (
        wrong
        .groupby(
            ["Actual_Crop", "Predicted_Crop"]
        )
        .size()
        .reset_index(
            name="Count"
        )
        .sort_values(
            "Count",
            ascending=False
        )
    )

    print(
        mistakes.to_string(index=False)
    )

# --------------------------------------------------
# Save wrong predictions
# --------------------------------------------------

wrong.to_excel(
    "output/Wrong_Predictions.xlsx",
    index=False
)

print(
    "\nWrong predictions saved to:"
    " output/Wrong_Predictions.xlsx"
)

# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

labels = sorted(
    y_test.unique()
)

cm = confusion_matrix(
    y_test,
    pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.to_excel(
    "output/Confusion_Matrix.xlsx"
)

print(
    "Confusion matrix saved to:"
    " output/Confusion_Matrix.xlsx"
)