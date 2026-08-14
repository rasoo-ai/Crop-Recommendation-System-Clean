import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Performance")

st.write(
    "Evaluation of the Random Forest crop recommendation model "
    "using the project dataset."
)

# --------------------------------------------------
# Load model and data
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("output/crop_prediction_model.pkl")


@st.cache_data
def load_data():
    return pd.read_excel("output/Crop_Normalized.xlsx")


model = load_model()
df = load_data()

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

# --------------------------------------------------
# Same random test split used by training
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

pred = model.predict(X_test)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, pred)

report = classification_report(
    y_test,
    pred,
    output_dict=True,
    zero_division=0,
)

macro_f1 = report["macro avg"]["f1-score"]
weighted_f1 = report["weighted avg"]["f1-score"]

# --------------------------------------------------
# Summary metrics
# --------------------------------------------------

st.subheader("Overall Performance")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Test Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with c2:
    st.metric(
        "Macro F1",
        f"{macro_f1:.2f}"
    )

with c3:
    st.metric(
        "Weighted F1",
        f"{weighted_f1:.2f}"
    )

# --------------------------------------------------
# Per-crop metrics
# --------------------------------------------------

st.markdown("---")
st.subheader("Per-Crop Performance")

crop_rows = []

for crop in sorted(valid_classes):

    if crop in report:

        crop_rows.append({
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

crop_df = pd.DataFrame(crop_rows)

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
    labels=labels
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

image = ax.imshow(cm)

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))

ax.set_xticklabels(labels, rotation=45, ha="right")
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
# Explanation
# --------------------------------------------------

st.markdown("---")
st.subheader("How to Interpret the Results")

st.write(
    f"The model achieved a test accuracy of "
    f"{accuracy * 100:.2f}% on the current held-out test split."
)

st.write(
    f"The macro F1-score is {macro_f1:.2f}, which gives "
    "equal importance to each crop class and is therefore "
    "useful when the dataset contains imbalanced crop counts."
)

st.info(
    "The displayed accuracy is based on the project's random "
    "train/test evaluation. Geographic holdout evaluation should "
    "be reported separately when measuring performance on genuinely "
    "unseen tehsils."
)