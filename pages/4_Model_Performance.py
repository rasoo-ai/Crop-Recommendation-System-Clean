import json
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Performance")

# --------------------------------------------------
# Load data, model, and canonical metrics
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_excel("output/Crop_Normalized.xlsx")


@st.cache_resource
def load_model():
    return joblib.load("output/crop_prediction_model.pkl")


@st.cache_data
def load_metrics():
    with open(
        "output/model_metrics.json",
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


df = load_data()
model = load_model()
metrics = load_metrics()

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
# Prepare evaluation data exactly like 17_evaluate_model.py
# --------------------------------------------------

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
    random_state=42,
).reset_index(drop=True)

X = df[features]
y = df[target]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

pred = model.predict(X_test)

report = classification_report(
    y_test,
    pred,
    output_dict=True,
    zero_division=0,
)

# --------------------------------------------------
# Overall metrics
# --------------------------------------------------

st.subheader("Overall Performance")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Test Accuracy",
        f"{metrics['test_accuracy'] * 100:.2f}%"
    )

with c2:
    st.metric(
        "Macro F1",
        f"{metrics['macro_f1']:.2f}"
    )

with c3:
    st.metric(
        "Weighted F1",
        f"{metrics['weighted_f1']:.2f}"
    )

# --------------------------------------------------
# Additional information
# --------------------------------------------------

st.caption(
    f"Test records: {metrics['test_records']} | "
    f"Wrong predictions: {metrics['wrong_predictions']} | "
    f"Error rate: {metrics['error_rate'] * 100:.2f}%"
)

# --------------------------------------------------
# Per-crop performance
# --------------------------------------------------

st.markdown("---")
st.subheader("Per-Crop Performance")

rows = []

for crop in sorted(valid_classes):

    if crop in report:

        rows.append({
            "Crop": crop,
            "Precision": round(
                report[crop]["precision"], 3
            ),
            "Recall": round(
                report[crop]["recall"], 3
            ),
            "F1 Score": round(
                report[crop]["f1-score"], 3
            ),
            "Test Samples": int(
                report[crop]["support"]
            ),
        })

crop_df = pd.DataFrame(rows)

st.dataframe(
    crop_df,
    width="stretch",
    hide_index=True,
)

# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

st.markdown("---")
st.subheader("Confusion Matrix")

labels = sorted(
    y_test.unique()
)

cm = confusion_matrix(
    y_test,
    pred,
    labels=labels,
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

image = ax.imshow(cm)

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))

ax.set_xticklabels(
    labels,
    rotation=45,
    ha="right",
)

ax.set_yticklabels(labels)

ax.set_xlabel("Predicted Crop")
ax.set_ylabel("Actual Crop")

for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
        )

fig.colorbar(image, ax=ax)

plt.tight_layout()

st.pyplot(fig)

# --------------------------------------------------
# Interpretation
# --------------------------------------------------

st.markdown("---")
st.subheader("How to Interpret the Results")

st.write(
    f"The current model has a test accuracy of "
    f"{metrics['test_accuracy'] * 100:.2f}% "
    f"and a macro F1-score of "
    f"{metrics['macro_f1']:.2f}."
)

st.info(
    "Accuracy is calculated on a held-out test split. "
    "Macro F1 is useful because the crop classes are imbalanced."
)