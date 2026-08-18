import json
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Model Performance - Smart Kisan",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background-color: #f7faf7; }
.block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Model Performance")
st.write("Evaluation metrics for the Smart Kisan Random Forest classifier on held-out test data.")

# ==========================================================
# LOAD
# ==========================================================

@st.cache_data
def load_data():
    return pd.read_excel("output/Crop_Normalized.xlsx")

@st.cache_resource
def load_model():
    return joblib.load("output/crop_prediction_model_balanced.pkl")

@st.cache_data
def load_metrics():
    with open("output/model_metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)

df      = load_data()
model   = load_model()
metrics = load_metrics()

# ==========================================================
# FEATURES
# ==========================================================

features = [
    "Soil_Type", "pH_Value", "Nitrogen_Value (N)",
    "Phosphorus_Value (P)", "Potassium_Value (K)",
    "Electrical_Conductivity (EC)", "Organic_Carbon (%)",
    "Soil_Moisture (%)", "Zinc (%)", "Iron (%)",
    "Manganese (%)", "Copper (%)", "Boron (%)", "Sulphur (%)",
    "Rainfall_cm", "temperature_celsius", "humidity_percentage",
    "State_Name", "Agro_Climatic Zone",
]

numeric_cols = [
    "pH_Value", "Nitrogen_Value (N)", "Phosphorus_Value (P)",
    "Potassium_Value (K)", "Electrical_Conductivity (EC)",
    "Organic_Carbon (%)", "Soil_Moisture (%)", "Zinc (%)",
    "Iron (%)", "Manganese (%)", "Copper (%)", "Boron (%)",
    "Sulphur (%)", "Rainfall_cm", "temperature_celsius",
    "humidity_percentage",
]

target = "Crop"

# ==========================================================
# PREPARE DATA
# ==========================================================

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=features + [target]).drop_duplicates()

crop_mapping = {
    "Sugar Cane":         "Sugarcane",
    "Paddy (Rice)":       "Rice",
    "Oilseeds (Mustard)": "Mustard",
    "Pulses (Arhar)":     "Pulses",
}
df[target] = df[target].replace(crop_mapping)

valid_classes = df[target].value_counts()
valid_classes = valid_classes[valid_classes >= 20].index

df = df[df[target].isin(valid_classes)].copy()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df[features]
y = df[target]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pred   = model.predict(X_test)
report = classification_report(y_test, pred, output_dict=True, zero_division=0)

# ==========================================================
# OVERALL METRICS
# ==========================================================

st.subheader("Overall Performance")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Test Accuracy",    f"{metrics['test_accuracy']*100:.2f}%")
m2.metric("Macro F1",         f"{metrics['macro_f1']:.3f}")
m3.metric("Weighted F1",      f"{metrics['weighted_f1']:.3f}")
m4.metric("Test Records",     f"{metrics['test_records']:,}")
m5.metric("Wrong Predictions",f"{metrics['wrong_predictions']:,}")

st.caption(
    f"Error rate: {metrics['error_rate']*100:.2f}% | "
    f"Model: Random Forest (balanced) | "
    f"Split: 80/20 stratified"
)

# ==========================================================
# TABS
# ==========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Per-Crop F1",
    "🔥 Confusion Matrix",
    "🌟 Feature Importance",
    "📋 Full Report",
])

# ----------------------------------------------------------
# TAB 1 — PER CROP F1
# ----------------------------------------------------------

with tab1:
    st.subheader("Per-Crop F1 Score")
    st.caption("F1 = harmonic mean of precision and recall. Higher is better.")

    rows = []
    for crop in sorted(valid_classes):
        if crop in report:
            rows.append({
                "Crop":         crop,
                "Precision":    round(report[crop]["precision"], 3),
                "Recall":       round(report[crop]["recall"],    3),
                "F1 Score":     round(report[crop]["f1-score"],  3),
                "Test Samples": int(report[crop]["support"]),
            })

    crop_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)

    # Horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, max(4, len(crop_df)*0.5)))
    colors = ["#22c55e" if f >= 0.7 else "#f59e0b" if f >= 0.5 else "#ef4444"
              for f in crop_df["F1 Score"]]
    bars = ax.barh(crop_df["Crop"], crop_df["F1 Score"], color=colors, height=0.6)

    for bar, val in zip(bars, crop_df["F1 Score"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=10)

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 Score")
    ax.axvline(x=0.7, color="#22c55e", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=0.5, color="#f59e0b", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_facecolor("#f7faf7")
    fig.patch.set_facecolor("#f7faf7")

    legend_patches = [
        mpatches.Patch(color="#22c55e", label="High (≥0.70)"),
        mpatches.Patch(color="#f59e0b", label="Moderate (0.50–0.70)"),
        mpatches.Patch(color="#ef4444", label="Low (<0.50)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Table
    st.dataframe(crop_df, use_container_width=True, hide_index=True)

    # Insight
    high   = crop_df[crop_df["F1 Score"] >= 0.7]["Crop"].tolist()
    low    = crop_df[crop_df["F1 Score"] <  0.5]["Crop"].tolist()
    if high:
        st.success(f"Strong performers: **{', '.join(high)}**")
    if low:
        st.warning(
            f"Weak performers: **{', '.join(low)}** — "
            f"too few training samples. Collecting more data for these crops will improve F1."
        )

# ----------------------------------------------------------
# TAB 2 — CONFUSION MATRIX
# ----------------------------------------------------------

with tab2:
    st.subheader("Confusion Matrix")
    st.caption("Rows = actual crop. Columns = predicted crop. Diagonal = correct predictions.")

    labels = sorted(y_test.unique())
    cm     = confusion_matrix(y_test, pred, labels=labels)

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(cm, cmap="YlOrRd")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted Crop", fontsize=12)
    ax.set_ylabel("Actual Crop",    fontsize=12)

    # Annotate cells
    thresh = cm.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm[i,j] > thresh else "black"
            ax.text(j, i, cm[i,j], ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold" if i==j else "normal")

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Misclassification summary
    st.subheader("Top Misclassifications")
    misses = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i != j and cm[i,j] > 0:
                misses.append({
                    "Actual":    actual,
                    "Predicted": predicted,
                    "Count":     cm[i,j],
                })
    if misses:
        miss_df = pd.DataFrame(misses).sort_values("Count", ascending=False).head(10)
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# TAB 3 — FEATURE IMPORTANCE
# ----------------------------------------------------------

with tab3:
    st.subheader("Feature Importance")
    st.caption("How much each feature contributed to the model's decisions (Mean Decrease in Impurity).")

    try:
        # Get feature importances from pipeline or direct model
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_names  = features
        elif hasattr(model, "named_steps"):
            last_step   = list(model.named_steps.values())[-1]
            importances = last_step.feature_importances_
            try:
                feat_names = model.named_steps["preprocessor"].get_feature_names_out().tolist()
            except Exception:
                feat_names = features
        else:
            importances = None

        if importances is not None:
            imp_series = pd.Series(importances, index=feat_names[:len(importances)])
            imp_series = imp_series.sort_values(ascending=False).head(15)

            # Friendly labels
            LABELS = {
                "Soil_Moisture (%)":            "Soil Moisture",
                "Nitrogen_Value (N)":           "Nitrogen (N)",
                "Rainfall_cm":                  "Rainfall",
                "temperature_celsius":          "Temperature",
                "pH_Value":                     "Soil pH",
                "Potassium_Value (K)":          "Potassium (K)",
                "Phosphorus_Value (P)":         "Phosphorus (P)",
                "Organic_Carbon (%)":           "Organic Carbon",
                "humidity_percentage":          "Humidity",
                "Electrical_Conductivity (EC)": "EC",
                "Zinc (%)":                     "Zinc",
                "Iron (%)":                     "Iron",
                "Manganese (%)":                "Manganese",
                "Copper (%)":                   "Copper",
                "Boron (%)":                    "Boron",
                "Sulphur (%)":                  "Sulphur",
                "State_Name":                   "State",
                "Agro_Climatic Zone":           "Agro-Climatic Zone",
                "Soil_Type":                    "Soil Type",
            }

            imp_df = pd.DataFrame({
                "Feature":    [LABELS.get(f, f) for f in imp_series.index],
                "Importance": imp_series.values,
            })

            fig, ax = plt.subplots(figsize=(10, 6))
            colors  = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(imp_df)))[::-1]
            bars    = ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1],
                              color=colors[::-1], height=0.6)

            for bar, val in zip(bars, imp_df["Importance"][::-1]):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va="center", fontsize=9)

            ax.set_xlabel("Importance Score")
            ax.set_facecolor("#f7faf7")
            fig.patch.set_facecolor("#f7faf7")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Top 3 insight
            top3 = imp_df.head(3)
            st.info(
                f"Top 3 most important features: "
                f"**{top3.iloc[0]['Feature']}** ({top3.iloc[0]['Importance']:.3f}), "
                f"**{top3.iloc[1]['Feature']}** ({top3.iloc[1]['Importance']:.3f}), "
                f"**{top3.iloc[2]['Feature']}** ({top3.iloc[2]['Importance']:.3f}). "
                f"These are the measurements that matter most for crop recommendations."
            )

            st.dataframe(imp_df, use_container_width=True, hide_index=True)

        else:
            st.info("Feature importance not available for this model type.")

    except Exception as e:
        st.error(f"Could not compute feature importance: {e}")

# ----------------------------------------------------------
# TAB 4 — FULL REPORT
# ----------------------------------------------------------

with tab4:
    st.subheader("Full Classification Report")

    full_rows = []
    for crop in sorted(valid_classes):
        if crop in report:
            full_rows.append({
                "Crop":         crop,
                "Precision":    round(report[crop]["precision"], 3),
                "Recall":       round(report[crop]["recall"],    3),
                "F1 Score":     round(report[crop]["f1-score"],  3),
                "Support":      int(report[crop]["support"]),
            })

    full_df = pd.DataFrame(full_rows)
    st.dataframe(full_df, use_container_width=True, hide_index=True)

    # Averages
    st.markdown("#### Averages")
    avg_rows = []
    for avg_key in ["macro avg", "weighted avg"]:
        if avg_key in report:
            avg_rows.append({
                "Average Type": avg_key.title(),
                "Precision":    round(report[avg_key]["precision"], 3),
                "Recall":       round(report[avg_key]["recall"],    3),
                "F1 Score":     round(report[avg_key]["f1-score"],  3),
                "Support":      int(report[avg_key]["support"]),
            })
    st.dataframe(pd.DataFrame(avg_rows), use_container_width=True, hide_index=True)

    st.info(
        "Accuracy is calculated on a held-out 20% test split. "
        "Macro F1 treats all classes equally — useful for imbalanced datasets. "
        "Weighted F1 weights by class support."
    )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()
st.caption('Smart Kisan Model Performance | Random Forest Classifier')
