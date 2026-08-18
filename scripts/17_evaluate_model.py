import json
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ==========================================================
# LOAD DATASET AND MODEL
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

target = "Crop"


# ==========================================================
# DATA CLEANING
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
# PREDICTIONS
# ==========================================================

pred = model.predict(
    X_test
)


# ==========================================================
# ACCURACY
# ==========================================================

accuracy = accuracy_score(
    y_test,
    pred
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

report_text = classification_report(
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


# ==========================================================
# WRONG PREDICTIONS
# ==========================================================

wrong = X_test.copy()

wrong["Actual_Crop"] = y_test.values

wrong["Predicted_Crop"] = pred

wrong = wrong[
    wrong["Actual_Crop"]
    != wrong["Predicted_Crop"]
]


# ==========================================================
# COMMON MISTAKES
# ==========================================================

mistakes = (
    wrong
    .groupby(
        [
            "Actual_Crop",
            "Predicted_Crop",
        ]
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


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

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


# ==========================================================
# SUMMARY METRICS
# ==========================================================

macro_f1 = report_dict[
    "macro avg"
]["f1-score"]

weighted_f1 = report_dict[
    "weighted avg"
]["f1-score"]

error_rate = (
    len(wrong)
    / len(X_test)
)

metrics = {
    "test_accuracy": round(
        float(accuracy),
        6
    ),
    "macro_f1": round(
        float(macro_f1),
        6
    ),
    "weighted_f1": round(
        float(weighted_f1),
        6
    ),
    "test_records": int(
        len(X_test)
    ),
    "wrong_predictions": int(
        len(wrong)
    ),
    "error_rate": round(
        float(error_rate),
        6
    ),
}


# ==========================================================
# SAVE RESULTS
# ==========================================================

wrong.to_excel(
    "output/Wrong_Predictions.xlsx",
    index=False
)

cm_df.to_excel(
    "output/Confusion_Matrix.xlsx"
)

with open(
    "output/model_metrics.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(
    f"\nTest Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{macro_f1:.2f}"
)

print(
    f"Weighted F1: "
    f"{weighted_f1:.2f}"
)

print(
    f"Wrong Predictions: "
    f"{len(wrong)}"
)

print(
    f"Total Test Records: "
    f"{len(X_test)}"
)

print(
    f"Error Rate: "
    f"{error_rate * 100:.2f}%"
)

print("\nClassification Report:")
print(report_text)

print("\n" + "=" * 70)
print("MOST COMMON INCORRECT PREDICTIONS")
print("=" * 70)

if mistakes.empty:

    print("No incorrect predictions.")

else:

    print(
        mistakes.to_string(
            index=False
        )
    )

print(
    "\nWrong predictions saved to:"
    " output/Wrong_Predictions.xlsx"
)

print(
    "Confusion matrix saved to:"
    " output/Confusion_Matrix.xlsx"
)

print(
    "Metrics saved to:"
    " output/model_metrics.json"
)

print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)
