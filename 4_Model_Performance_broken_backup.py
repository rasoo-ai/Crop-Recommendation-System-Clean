import json
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import io

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import label_binarize

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
/* ---- Base ---- */
.stApp { background-color: #f0f4f0; }
.block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem; }
footer { visibility: hidden; }

/* ---- Metric cards ---- */
[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-left: 4px solid #22c55e;
}
[data-testid="metric-container"]:nth-child(5) {
    border-left-color: #ef4444;
}
[data-testid="metric-container"]:nth-child(6) {
    border-left-color: #ef4444;
}

/* ---- Alert boxes ---- */
.weak-alert {
    background: #fff7ed;
    border: 1px solid #fb923c;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-top: 0.5rem;
}
.strong-alert {
    background: #f0fdf4;
    border: 1px solid #22c55e;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-top: 0.5rem;
}

/* ---- Section headers ---- */
h2, h3 { color: #166534; }

/* ---- Download button ---- */
.stDownloadButton > button {
    background-color: #166534;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.4rem 1rem;
}
.stDownloadButton > button:hover {
    background-color: #15803d;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Model Performance")
st.write("Evaluation metrics for the **Smart Kisan** Random Forest classifier on held-out test data.")

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pred        = model.predict(X_test)
pred_proba  = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
report      = classification_report(y_test, pred, output_dict=True, zero_division=0)
labels      = sorted(y_test.unique())

# Baseline accuracy (most frequent class)
baseline_acc = y_test.value_counts(normalize=True).max()

# Wrong prediction records
wrong_mask = pred != y_test.values
wrong_df   = X_test[wrong_mask].copy()
wrong_df["Actual"]    = y_test.values[wrong_mask]
wrong_df["Predicted"] = pred[wrong_mask]
wrong_df = wrong_df.reset_index(drop=True)

# ==========================================================
# OVERALL METRICS
# ==========================================================

st.subheader("Overall Performance")

col_acc, col_macro, col_weighted, col_baseline, col_records, col_wrong = st.columns(6)
col_acc.metric("Test Accuracy",     f"{metrics['test_accuracy']*100:.2f}%",
               delta=f"+{(metrics['test_accuracy'] - baseline_acc)*100:.1f}% vs baseline")
col_macro.metric("Macro F1",        f"{metrics['macro_f1']:.3f}")
col_weighted.metric("Weighted F1",  f"{metrics['weighted_f1']:.3f}")
col_baseline.metric("Baseline Acc", f"{baseline_acc*100:.1f}%",
                    help="Random-guess accuracy (most frequent class)")
col_records.metric("Test Records",  f"{metrics['test_records']:,}")
col_wrong.metric("Wrong Predictions", f"{metrics['wrong_predictions']:,}")

st.caption(
    f"Error rate: **{metrics['error_rate']*100:.2f}%** | "
    f"Model: Random Forest (balanced) | "
    f"Split: 80/20 stratified | "
    f"Baseline: {baseline_acc*100:.1f}%"
)

# ==========================================================
# TABS
# ==========================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Per-Crop F1",
    "🔍 Confusion Matrix",
    "🌱 Feature Importance",
    "❌ Error Analysis",
    "📉 Learning Curve",
    "📐 ROC / PR Curves",
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

    fig, ax = plt.subplots(figsize=(10, max(4, len(crop_df) * 0.52)))
    colors = ["#16a34a" if f >= 0.7 else "#f59e0b" if f >= 0.5 else "#ef4444"
              for f in crop_df["F1 Score"]]
    bars = ax.barh(crop_df["Crop"], crop_df["F1 Score"], color=colors, height=0.6)

    for bar, val in zip(bars, crop_df["F1 Score"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

    macro_f1 = metrics["macro_f1"]
    ax.axvline(x=macro_f1, color="#6366f1", linestyle="--", linewidth=1.5,
               label=f"Macro F1 ({macro_f1:.3f})")
    ax.axvline(x=0.7, color="#16a34a", linestyle=":", alpha=0.6, linewidth=1)
    ax.axvline(x=0.5, color="#f59e0b", linestyle=":", alpha=0.6, linewidth=1)

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 Score")
    ax.set_facecolor("#f7faf7")
    fig.patch.set_facecolor("#f7faf7")

    legend_patches = [
        mpatches.Patch(color="#16a34a", label="High (≥0.70)"),
        mpatches.Patch(color="#f59e0b", label="Moderate (0.50–0.70)"),
        mpatches.Patch(color="#ef4444", label="Low (<0.50)"),
        mpatches.Patch(color="#6366f1", label=f"Macro F1 ({macro_f1:.3f})"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    high = crop_df[crop_df["F1 Score"] >= 0.7]["Crop"].tolist()
    low  = crop_df[crop_df["F1 Score"] <  0.5]["Crop"].tolist()
    mid  = crop_df[(crop_df["F1 Score"] >= 0.5) & (crop_df["F1 Score"] < 0.7)]["Crop"].tolist()

    if high:
        st.success(f"✅ **Strong (F1 ≥ 0.70):** {', '.join(high)}")
    if mid:
        st.warning(f"⚠️ **Moderate (0.50–0.70):** {', '.join(mid)} — room for improvement.")
    if low:
        st.error(
            f"🚨 **Weak (F1 < 0.50):** {', '.join(low)} — "
            f"likely too few training samples. Collect more data for these crops."
        )

    st.dataframe(crop_df, use_container_width=True, hide_index=True)

    # Download
    csv_bytes = crop_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download F1 Report (CSV)", csv_bytes,
                       file_name="per_crop_f1.csv", mime="text/csv")

# ----------------------------------------------------------
# TAB 2 — CONFUSION MATRIX
# ----------------------------------------------------------

with tab2:
    st.subheader("Confusion Matrix")

    cm_mode = st.radio("View mode", ["Raw counts", "Normalized (%)"], horizontal=True)

    cm = confusion_matrix(y_test, pred, labels=labels)
    cm_display = cm.astype(float)
    if cm_mode == "Normalized (%)":
        cm_display = cm_display / cm_display.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(cm_display, cmap="YlOrRd")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted Crop", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Crop",    fontsize=12)

    thresh = cm_display.max() / 2
    fmt = ".1f" if cm_mode == "Normalized (%)" else "d"
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_display[i, j]
            text = f"{val:{fmt}}{'%' if cm_mode == 'Normalized (%)' else ''}"
            color = "white" if val > thresh else "black"
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=8,
                    fontweight="bold" if i == j else "normal")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("%" if cm_mode == "Normalized (%)" else "Count", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("Rows = actual crop. Columns = predicted crop. Diagonal = correct predictions.")

    # Top misclassifications
    st.subheader("🔴 Top Misclassified Pairs")
    misses = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i != j and cm[i, j] > 0:
                misses.append({"Actual": actual, "Predicted": predicted, "Count": int(cm[i, j])})
    if misses:
        miss_df = pd.DataFrame(misses).sort_values("Count", ascending=False).head(10)
        miss_df["% of Actual Class"] = miss_df.apply(
            lambda r: f"{r['Count'] / cm[labels.index(r['Actual'])].sum() * 100:.1f}%", axis=1
        )
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

        # Download confusion matrix
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        buf = io.BytesIO()
        cm_df.to_csv(buf)
        st.download_button("⬇️ Download Confusion Matrix (CSV)", buf.getvalue(),
                           file_name="confusion_matrix.csv", mime="text/csv")

# ----------------------------------------------------------
# TAB 3 — FEATURE IMPORTANCE
# ----------------------------------------------------------

with tab3:
    st.subheader("Feature Importance")
    st.caption("Mean Decrease in Impurity — how much each feature reduces prediction uncertainty.")

    try:
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
            imp_series = imp_series.sort_values(ascending=False)

            top_n = st.slider("Show top N features", min_value=5,
                               max_value=min(20, len(imp_series)), value=15)
            imp_top = imp_series.head(top_n)

            imp_df = pd.DataFrame({
                "Feature":    [LABELS.get(f, f) for f in imp_top.index],
                "Importance": imp_top.values,
            })

            fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
            palette = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(imp_df)))[::-1]
            bars = ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1],
                           color=palette[::-1], height=0.6)

            for bar, val in zip(bars, imp_df["Importance"][::-1]):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=9)

            ax.set_xlabel("Importance Score")
            ax.set_facecolor("#f7faf7")
            fig.patch.set_facecolor("#f7faf7")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            top3 = imp_df.head(3)
            st.info(
                f"🔑 **Top 3 drivers:** "
                f"**{top3.iloc[0]['Feature']}** ({top3.iloc[0]['Importance']:.4f}), "
                f"**{top3.iloc[1]['Feature']}** ({top3.iloc[1]['Importance']:.4f}), "
                f"**{top3.iloc[2]['Feature']}** ({top3.iloc[2]['Importance']:.4f}). "
                f"These are the measurements that matter most for crop recommendations."
            )

            st.dataframe(imp_df, use_container_width=True, hide_index=True)

            csv_imp = imp_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Feature Importance (CSV)", csv_imp,
                               file_name="feature_importance.csv", mime="text/csv")

        else:
            st.info("Feature importance not available for this model type.")

    except Exception as e:
        st.error(f"Could not compute feature importance: {e}")

# ----------------------------------------------------------
# TAB 4 — ERROR ANALYSIS
# ----------------------------------------------------------

with tab4:
    st.subheader("❌ Error Analysis")
    st.caption(f"Showing all {len(wrong_df)} wrong predictions from the test set.")

    # Filter by crop
    all_crops = ["All"] + sorted(wrong_df["Actual"].unique().tolist())
    selected_crop = st.selectbox("Filter by Actual Crop", all_crops)

    filtered = wrong_df if selected_crop == "All" else wrong_df[wrong_df["Actual"] == selected_crop]

    st.markdown(f"**{len(filtered)} wrong predictions** shown.")

    display_cols = ["Actual", "Predicted"] + [
        c for c in ["pH_Value", "Nitrogen_Value (N)", "Rainfall_cm",
                    "temperature_celsius", "humidity_percentage", "Soil_Type"]
        if c in filtered.columns
    ]
    st.dataframe(filtered[display_cols].reset_index(drop=True),
                 use_container_width=True, hide_index=True)

    # Error distribution chart
    st.markdown("#### Error Count by Crop")
    err_counts = wrong_df["Actual"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(err_counts.index, err_counts.values, color="#ef4444", alpha=0.8, width=0.6)
    ax.set_xlabel("Actual Crop")
    ax.set_ylabel("# Wrong Predictions")
    ax.set_xticklabels(err_counts.index, rotation=45, ha="right", fontsize=9)
    ax.set_facecolor("#f7faf7")
    fig.patch.set_facecolor("#f7faf7")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Download errors
    csv_err = wrong_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Wrong Predictions (CSV)", csv_err,
                       file_name="wrong_predictions.csv", mime="text/csv")

# ----------------------------------------------------------
# TAB 5 — LEARNING CURVE
# ----------------------------------------------------------

with tab5:
    st.subheader("📉 Learning Curve")
    st.caption(
        "Shows how model accuracy changes as more training data is used. "
        "A large gap between train and test = overfitting. Converging lines = more data helps."
    )

    @st.cache_data(show_spinner="Computing learning curve (this may take a minute)...")
    def compute_learning_curve(_model, _X, _y):
        from sklearn.model_selection import learning_curve
        sizes = np.linspace(0.1, 1.0, 8)
        train_sizes, train_scores, test_scores = learning_curve(
            _model, _X, _y,
            train_sizes=sizes,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
        )
        return train_sizes, train_scores, test_scores

    try:
        train_sizes, train_scores, test_scores = compute_learning_curve(model, X, y)

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        test_mean  = test_scores.mean(axis=1)
        test_std   = test_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_sizes, train_mean, "o-", color="#16a34a", label="Training Accuracy", lw=2)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color="#16a34a")
        ax.plot(train_sizes, test_mean, "o-", color="#6366f1", label="CV Validation Accuracy", lw=2)
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                        alpha=0.15, color="#6366f1")

        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.set_facecolor("#f7faf7")
        fig.patch.set_facecolor("#f7faf7")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        gap = train_mean[-1] - test_mean[-1]
        if gap > 0.1:
            st.warning(f"⚠️ Train–Val gap: **{gap:.2%}** — model may be overfitting. Consider regularization or more data.")
        else:
            st.success(f"✅ Train–Val gap: **{gap:.2%}** — model generalises well.")

        # Cross-val summary
        st.markdown("#### 5-Fold Cross-Validation Summary")
        @st.cache_data(show_spinner="Running 5-fold CV...")
        def compute_cv(_model, _X, _y):
            return cross_val_score(_model, _X, _y, cv=5, scoring="accuracy", n_jobs=-1)

        cv_scores = compute_cv(model, X, y)
        cv_df = pd.DataFrame({
            "Fold":     [f"Fold {i+1}" for i in range(len(cv_scores))],
            "Accuracy": [f"{s*100:.2f}%" for s in cv_scores],
        })
        cv_df.loc[len(cv_df)] = ["Mean ± Std",
                                  f"{cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%"]
        st.dataframe(cv_df, use_container_width=True, hide_index=True)
        st.info(
            f"5-Fold CV Mean Accuracy: **{cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%** — "
            f"this is more reliable than the single test-split result."
        )

    except Exception as e:
        st.error(f"Could not compute learning curve: {e}")

# ----------------------------------------------------------
# TAB 6 — ROC / PR CURVES
# ----------------------------------------------------------

with tab6:
    st.subheader("📐 ROC & Precision-Recall Curves")
    st.caption("Per-class curves. OvR (One-vs-Rest) strategy. AUC closer to 1.0 = better.")

    if pred_proba is not None:
        classes = model.classes_

        # ROC curves
        st.markdown("#### ROC Curves (One-vs-Rest)")
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = plt.cm.tab20(np.linspace(0, 1, len(classes)))

        macro_auc_vals = []
        for i, cls in enumerate(classes):
            y_bin = (y_test == cls).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            macro_auc_vals.append(roc_auc)
            ax.plot(fpr, tpr, lw=1.5, color=palette[i], label=f"{cls} (AUC={roc_auc:.2f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.50)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        ax.set_facecolor("#f7faf7")
        fig.patch.set_facecolor("#f7faf7")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        mean_auc = np.mean(macro_auc_vals)
        st.success(f"📊 Mean ROC-AUC across all crops: **{mean_auc:.3f}**")

        # PR curves
        st.markdown("#### Precision-Recall Curves (One-vs-Rest)")
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cls in enumerate(classes):
            y_bin = (y_test == cls).astype(int)
            prec, rec, _ = precision_recall_curve(y_bin, pred_proba[:, i])
            ap = average_precision_score(y_bin, pred_proba[:, i])
            ax.plot(rec, prec, lw=1.5, color=palette[i], label=f"{cls} (AP={ap:.2f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.set_facecolor("#f7faf7")
        fig.patch.set_facecolor("#f7faf7")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Calibration (Brier Score per class)
        st.markdown("#### Calibration — Brier Score per Crop")
        st.caption("Brier Score = mean squared error of probability predictions. Lower is better (0 = perfect).")
        brier_rows = []
        for i, cls in enumerate(classes):
            y_bin = (y_test == cls).astype(int)
            bs = brier_score_loss(y_bin, pred_proba[:, i])
            brier_rows.append({"Crop": cls, "Brier Score": round(bs, 4)})
        brier_df = pd.DataFrame(brier_rows).sort_values("Brier Score")
        st.dataframe(brier_df, use_container_width=True, hide_index=True)

    else:
        st.info("Probability estimates not available for this model. Set `probability=True` if using SVM.")

# ----------------------------------------------------------
# TAB 7 — FULL REPORT
# ----------------------------------------------------------

with tab7:
    st.subheader("📋 Full Classification Report")

    full_rows = []
    for crop in sorted(valid_classes):
        if crop in report:
            full_rows.append({
                "Crop":      crop,
                "Precision": round(report[crop]["precision"], 3),
                "Recall":    round(report[crop]["recall"],    3),
                "F1 Score":  round(report[crop]["f1-score"],  3),
                "Support":   int(report[crop]["support"]),
            })

    full_df = pd.DataFrame(full_rows)
    st.dataframe(full_df, use_container_width=True, hide_index=True)

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
        "**Accuracy** is calculated on a held-out 20% test split. "
        "**Macro F1** treats all classes equally — useful for imbalanced datasets. "
        "**Weighted F1** weights by class support."
    )

    # Download full report
    csv_full = full_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Full Report (CSV)", csv_full,
                       file_name="full_classification_report.csv", mime="text/csv")

# ==========================================================
# FOOTER
# ==========================================================

st.divider()
st.caption(
    "📊 Smart Kisan Model Performance | "
    "Random Forest Classifier (balanced) | "
    f"Test accuracy: {metrics['test_accuracy']*100:.2f}% | "
    f"Macro F1: {metrics['macro_f1']:.3f} | "
    f"Baseline: {baseline_acc*100:.1f}%"
)
