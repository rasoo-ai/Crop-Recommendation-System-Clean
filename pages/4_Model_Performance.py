import json
import io
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, brier_score_loss,
)
from sklearn.model_selection import train_test_split, cross_val_score

st.set_page_config(
    page_title="Model Performance - Smart Kisan",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #f0f7ee 0%, #e8f5e2 100%); }
.block-container { max-width: 1300px; padding-top: 1.5rem; padding-bottom: 3rem; }
footer { visibility: hidden; }
.page-header {
    background: linear-gradient(135deg, #1a5c2a, #2d8a45);
    border-radius: 16px; padding: 2rem 2.5rem; color: white;
    margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(26,92,42,0.25);
}
.page-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.page-header p  { margin: 0.5rem 0 0; opacity: 0.85; }
.metric-grid {
    display: grid; grid-template-columns: repeat(6, 1fr);
    gap: 0.8rem; margin-bottom: 1.5rem;
}
.metric-box {
    background: white; border-radius: 14px; padding: 1.2rem 1rem;
    text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    border-top: 4px solid #2d8a45;
}
.metric-box.red  { border-top-color: #ef4444; }
.metric-box.blue { border-top-color: #6366f1; }
.metric-box .val { font-size: 1.8rem; font-weight: 700; color: #1a5c2a; line-height: 1; }
.metric-box.red  .val { color: #dc2626; }
.metric-box.blue .val { color: #4f46e5; }
.metric-box .lbl { font-size: 0.72rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.3rem; }
.metric-box .delta { font-size: 0.75rem; color: #16a34a; font-weight: 500; margin-top: 0.2rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a5c2a 0%, #2d8a45 100%); }
[data-testid="stSidebar"] * { color: white !important; }
.stDownloadButton > button { background: #1a5c2a !important; color: white !important; border-radius: 8px !important; border: none !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌾 Smart Kisan")
    st.markdown("---")
    st.page_link("app.py",                         label="🏠 Home")
    st.page_link("pages/2_Crop_Recommendation.py",  label="🌱 Crop Recommendation")
    st.page_link("pages/3_Tehsil_Analysis.py",      label="📍 Tehsil Analysis")
    st.page_link("pages/4_Model_Performance.py",    label="📊 Model Performance")
    st.markdown("---")
    st.markdown("**Model V1** | 🟢 93.53%")

st.markdown("""
<div class="page-header">
    <h1>📊 Model Performance</h1>
    <p>Deep evaluation — per-crop F1, confusion matrix, feature importance, error analysis, learning curves and ROC curves.</p>
</div>
""", unsafe_allow_html=True)

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
# FEATURES & PREP
# ==========================================================

features = [
    "Soil_Type", "pH_Value", "Nitrogen_Value (N)", "Phosphorus_Value (P)",
    "Potassium_Value (K)", "Electrical_Conductivity (EC)", "Organic_Carbon (%)",
    "Soil_Moisture (%)", "Zinc (%)", "Iron (%)", "Manganese (%)", "Copper (%)",
    "Boron (%)", "Sulphur (%)", "Rainfall_cm", "temperature_celsius",
    "humidity_percentage", "State_Name", "Agro_Climatic Zone",
]
numeric_cols = [f for f in features if f not in ["Soil_Type","State_Name","Agro_Climatic Zone"]]

LABELS = {
    "Soil_Moisture (%)":"Soil Moisture","Nitrogen_Value (N)":"Nitrogen (N)",
    "Rainfall_cm":"Rainfall","temperature_celsius":"Temperature",
    "pH_Value":"Soil pH","Potassium_Value (K)":"Potassium (K)",
    "Phosphorus_Value (P)":"Phosphorus (P)","Organic_Carbon (%)":"Organic Carbon",
    "humidity_percentage":"Humidity","Electrical_Conductivity (EC)":"EC",
    "Zinc (%)":"Zinc","Iron (%)":"Iron","Manganese (%)":"Manganese",
    "Copper (%)":"Copper","Boron (%)":"Boron","Sulphur (%)":"Sulphur",
    "State_Name":"State","Agro_Climatic Zone":"Agro-Climatic Zone","Soil_Type":"Soil Type",
}

target = "Crop"
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=features + [target]).drop_duplicates()
df[target] = df[target].replace({
    "Sugar Cane":"Sugarcane","Paddy (Rice)":"Rice",
    "Oilseeds (Mustard)":"Mustard","Pulses (Arhar)":"Pulses"
})

valid_classes = df[target].value_counts()
valid_classes = valid_classes[valid_classes >= 20].index
df = df[df[target].isin(valid_classes)].copy()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
pred       = model.predict(X_test)
pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
report     = classification_report(y_test, pred, output_dict=True, zero_division=0)
labels     = sorted(y_test.unique())
baseline   = y_test.value_counts(normalize=True).max()

wrong_mask = pred != y_test.values
wrong_df   = X_test[wrong_mask].copy()
wrong_df["Actual"]    = y_test.values[wrong_mask]
wrong_df["Predicted"] = pred[wrong_mask]
wrong_df = wrong_df.reset_index(drop=True)

# ==========================================================
# METRIC CARDS
# ==========================================================

delta = (metrics["test_accuracy"] - baseline) * 100
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-box">
        <div class="val">{metrics['test_accuracy']*100:.2f}%</div>
        <div class="lbl">Test Accuracy</div>
        <div class="delta">+{delta:.1f}% vs baseline</div>
    </div>
    <div class="metric-box">
        <div class="val">{metrics['macro_f1']:.3f}</div>
        <div class="lbl">Macro F1</div>
    </div>
    <div class="metric-box">
        <div class="val">{metrics['weighted_f1']:.3f}</div>
        <div class="lbl">Weighted F1</div>
    </div>
    <div class="metric-box blue">
        <div class="val">{baseline*100:.1f}%</div>
        <div class="lbl">Baseline Acc</div>
    </div>
    <div class="metric-box">
        <div class="val">{metrics['test_records']:,}</div>
        <div class="lbl">Test Records</div>
    </div>
    <div class="metric-box red">
        <div class="val">{metrics['wrong_predictions']:,}</div>
        <div class="lbl">Wrong Predictions</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.caption(
    f"Error rate: **{metrics['error_rate']*100:.2f}%** | "
    f"Model: Random Forest (balanced) | Split: 80/20 stratified | "
    f"Baseline: {baseline*100:.1f}%"
)

# ==========================================================
# PLOT HELPER
# ==========================================================

def style_fig(fig, ax):
    ax.set_facecolor("#f7faf7")
    fig.patch.set_facecolor("#f7faf7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
# TAB 1 — PER-CROP F1
# ----------------------------------------------------------

with tab1:
    st.subheader("Per-Crop F1 Score")
    st.caption("F1 = harmonic mean of precision & recall. Green >= 0.70 | Yellow 0.50-0.70 | Red < 0.50")

    rows = [
        {
            "Crop": c,
            "Precision": round(report[c]["precision"], 3),
            "Recall": round(report[c]["recall"], 3),
            "F1 Score": round(report[c]["f1-score"], 3),
            "Test Samples": int(report[c]["support"]),
        }
        for c in sorted(valid_classes) if c in report
    ]
    crop_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(crop_df) * 0.52)))
    colors = ["#16a34a" if f >= 0.7 else "#f59e0b" if f >= 0.5 else "#ef4444"
              for f in crop_df["F1 Score"]]
    bars = ax.barh(crop_df["Crop"], crop_df["F1 Score"], color=colors, height=0.6)
    for bar, val in zip(bars, crop_df["F1 Score"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    macro = metrics["macro_f1"]
    ax.axvline(x=macro, color="#6366f1", linestyle="--", linewidth=1.8, label=f"Macro F1 ({macro:.3f})")
    ax.axvline(x=0.7,   color="#16a34a", linestyle=":", alpha=0.6, linewidth=1)
    ax.axvline(x=0.5,   color="#f59e0b", linestyle=":", alpha=0.6, linewidth=1)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 Score")
    legend_patches = [
        mpatches.Patch(color="#16a34a", label="High (>=0.70)"),
        mpatches.Patch(color="#f59e0b", label="Moderate (0.50-0.70)"),
        mpatches.Patch(color="#ef4444", label="Low (<0.50)"),
        mpatches.Patch(color="#6366f1", label=f"Macro F1 ({macro:.3f})"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    style_fig(fig, ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    high = crop_df[crop_df["F1 Score"] >= 0.7]["Crop"].tolist()
    mid  = crop_df[(crop_df["F1 Score"] >= 0.5) & (crop_df["F1 Score"] < 0.7)]["Crop"].tolist()
    low  = crop_df[crop_df["F1 Score"] < 0.5]["Crop"].tolist()
    if high: st.success(f"Strong (>=0.70): {', '.join(high)}")
    if mid:  st.warning(f"Moderate (0.50-0.70): {', '.join(mid)}")
    if low:  st.error(f"Weak (<0.50): {', '.join(low)} — collect more training data.")

    st.dataframe(crop_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download F1 Report (CSV)",
        crop_df.to_csv(index=False).encode(),
        file_name="per_crop_f1.csv", mime="text/csv"
    )

# ----------------------------------------------------------
# TAB 2 — CONFUSION MATRIX
# ----------------------------------------------------------

with tab2:
    st.subheader("Confusion Matrix")
    cm_mode = st.radio("View", ["Raw counts","Normalized (%)"], horizontal=True)
    cm = confusion_matrix(y_test, pred, labels=labels)
    cm_disp = cm.astype(float)
    if cm_mode == "Normalized (%)":
        cm_disp = cm_disp / cm_disp.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(cm_disp, cmap="YlOrRd")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted Crop", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Crop", fontsize=12)
    thresh = cm_disp.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_disp[i, j]
            txt = f"{v:.1f}%" if cm_mode == "Normalized (%)" else str(int(v))
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if v > thresh else "black",
                    fontsize=8, fontweight="bold" if i == j else "normal")
    fig.colorbar(im, ax=ax, shrink=0.8)
    style_fig(fig, ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("Rows = actual crop. Columns = predicted crop. Diagonal = correct predictions.")

    st.subheader("Top Misclassified Pairs")
    misses = [
        {"Actual": labels[i], "Predicted": labels[j],
         "Count": int(cm[i, j]), "% of Actual": f"{cm[i,j]/cm[i].sum()*100:.1f}%"}
        for i in range(len(labels)) for j in range(len(labels))
        if i != j and cm[i, j] > 0
    ]
    if misses:
        miss_df = pd.DataFrame(misses).sort_values("Count", ascending=False).head(10)
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

    buf = io.BytesIO()
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(buf)
    st.download_button("Download Confusion Matrix (CSV)", buf.getvalue(),
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
            last = list(model.named_steps.values())[-1]
            importances = last.feature_importances_
            try:
                feat_names = model.named_steps["preprocessor"].get_feature_names_out().tolist()
            except Exception:
                feat_names = features
        else:
            importances = None

        if importances is not None:
            imp = pd.Series(importances, index=feat_names[:len(importances)]).sort_values(ascending=False)
            top_n = st.slider("Show top N features", 5, min(20, len(imp)), 15)
            imp_top = imp.head(top_n)
            imp_df = pd.DataFrame({
                "Feature": [LABELS.get(f, f) for f in imp_top.index],
                "Importance": imp_top.values,
            })

            fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.42)))
            palette = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(imp_df)))[::-1]
            bars = ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1],
                           color=palette[::-1], height=0.6)
            for bar, val in zip(bars, imp_df["Importance"][::-1]):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=9)
            ax.set_xlabel("Importance Score")
            style_fig(fig, ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            t3 = imp_df.head(3)
            st.info(
                f"Top 3 drivers: "
                f"**{t3.iloc[0]['Feature']}** ({t3.iloc[0]['Importance']:.4f}), "
                f"**{t3.iloc[1]['Feature']}** ({t3.iloc[1]['Importance']:.4f}), "
                f"**{t3.iloc[2]['Feature']}** ({t3.iloc[2]['Importance']:.4f})"
            )
            st.dataframe(imp_df, use_container_width=True, hide_index=True)
            st.download_button("Download Feature Importance (CSV)",
                               imp_df.to_csv(index=False).encode(),
                               file_name="feature_importance.csv", mime="text/csv")
        else:
            st.info("Feature importance not available for this model type.")

    except Exception as e:
        st.error(f"Could not compute feature importance: {e}")

# ----------------------------------------------------------
# TAB 4 — ERROR ANALYSIS
# ----------------------------------------------------------

with tab4:
    st.subheader("Error Analysis")
    st.caption(f"All {len(wrong_df)} wrong predictions from the test set.")

    all_crops = ["All"] + sorted(wrong_df["Actual"].unique().tolist())
    sel = st.selectbox("Filter by Actual Crop", all_crops)
    filtered = wrong_df if sel == "All" else wrong_df[wrong_df["Actual"] == sel]
    st.markdown(f"**{len(filtered)} wrong predictions** shown.")

    disp_cols = ["Actual","Predicted"] + [
        c for c in ["pH_Value","Nitrogen_Value (N)","Rainfall_cm",
                    "temperature_celsius","humidity_percentage","Soil_Type"]
        if c in filtered.columns
    ]
    st.dataframe(filtered[disp_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown("#### Error Count by Crop")
    err_counts = wrong_df["Actual"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(err_counts.index, err_counts.values, color="#ef4444", alpha=0.8, width=0.6)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(int(bar.get_height())), ha="center", fontsize=9)
    ax.set_xlabel("Actual Crop")
    ax.set_ylabel("Wrong Predictions")
    ax.set_xticklabels(err_counts.index, rotation=45, ha="right", fontsize=9)
    style_fig(fig, ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.download_button("Download Wrong Predictions (CSV)",
                       wrong_df.to_csv(index=False).encode(),
                       file_name="wrong_predictions.csv", mime="text/csv")

# ----------------------------------------------------------
# TAB 5 — LEARNING CURVE
# ----------------------------------------------------------

with tab5:
    st.subheader("Learning Curve")
    st.caption("How accuracy changes with more training data. Large train-val gap = overfitting.")

    @st.cache_data(show_spinner="Computing learning curve...")
    def compute_lc(_model, _X, _y):
        from sklearn.model_selection import learning_curve
        sizes = np.linspace(0.1, 1.0, 8)
        ts, tr_s, te_s = learning_curve(
            _model, _X, _y, train_sizes=sizes,
            cv=3, scoring="accuracy", n_jobs=-1, random_state=42
        )
        return ts, tr_s, te_s

    try:
        ts, tr_s, te_s = compute_lc(model, X, y)
        tr_m, tr_std = tr_s.mean(axis=1), tr_s.std(axis=1)
        te_m, te_std = te_s.mean(axis=1), te_s.std(axis=1)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ts, tr_m, "o-", color="#16a34a", lw=2, label="Training Accuracy")
        ax.fill_between(ts, tr_m - tr_std, tr_m + tr_std, alpha=0.15, color="#16a34a")
        ax.plot(ts, te_m, "o-", color="#6366f1", lw=2, label="CV Validation Accuracy")
        ax.fill_between(ts, te_m - te_std, te_m + te_std, alpha=0.15, color="#6366f1")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.5, 1.02)
        ax.legend(fontsize=10)
        style_fig(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        gap = tr_m[-1] - te_m[-1]
        if gap > 0.1: st.warning(f"Train-Val gap: **{gap:.2%}** — consider regularization or more data.")
        else:         st.success(f"Train-Val gap: **{gap:.2%}** — model generalises well.")

        st.markdown("#### 5-Fold Cross-Validation")

        @st.cache_data(show_spinner="Running 5-fold CV...")
        def compute_cv(_model, _X, _y):
            return cross_val_score(_model, _X, _y, cv=5, scoring="accuracy", n_jobs=-1)

        cv = compute_cv(model, X, y)
        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(cv))],
            "Accuracy": [f"{s*100:.2f}%" for s in cv],
        })
        cv_df.loc[len(cv_df)] = ["Mean +/- Std", f"{cv.mean()*100:.2f}% +/- {cv.std()*100:.2f}%"]
        st.dataframe(cv_df, use_container_width=True, hide_index=True)
        st.info(f"5-Fold CV: **{cv.mean()*100:.2f}% +/- {cv.std()*100:.2f}%** — more reliable than a single test split.")

    except Exception as e:
        st.error(f"Could not compute learning curve: {e}")

# ----------------------------------------------------------
# TAB 6 — ROC / PR CURVES
# ----------------------------------------------------------

with tab6:
    st.subheader("ROC and Precision-Recall Curves")
    if pred_proba is not None:
        classes = model.classes_
        palette = plt.cm.tab20(np.linspace(0, 1, len(classes)))

        st.markdown("#### ROC Curves (One-vs-Rest)")
        fig, ax = plt.subplots(figsize=(10, 6))
        aucs = []
        for i, cls in enumerate(classes):
            yb = (y_test == cls).astype(int)
            fpr, tpr, _ = roc_curve(yb, pred_proba[:, i])
            ra = auc(fpr, tpr); aucs.append(ra)
            ax.plot(fpr, tpr, lw=1.5, color=palette[i], label=f"{cls} ({ra:.2f})")
        ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0,1); ax.set_ylim(0,1.02)
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        style_fig(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.success(f"Mean ROC-AUC: **{np.mean(aucs):.3f}**")

        st.markdown("#### Precision-Recall Curves (One-vs-Rest)")
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cls in enumerate(classes):
            yb = (y_test == cls).astype(int)
            prec, rec, _ = precision_recall_curve(yb, pred_proba[:, i])
            ap = average_precision_score(yb, pred_proba[:, i])
            ax.plot(rec, prec, lw=1.5, color=palette[i], label=f"{cls} (AP={ap:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0,1); ax.set_ylim(0,1.02)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        style_fig(fig, ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Brier Score per Crop")
        st.caption("Lower is better — measures probability calibration quality.")
        brier = [
            {"Crop": cls, "Brier Score": round(brier_score_loss((y_test==cls).astype(int), pred_proba[:,i]), 4)}
            for i, cls in enumerate(classes)
        ]
        st.dataframe(pd.DataFrame(brier).sort_values("Brier Score"), use_container_width=True, hide_index=True)
    else:
        st.info("Probability estimates not available for this model.")

# ----------------------------------------------------------
# TAB 7 — FULL REPORT
# ----------------------------------------------------------

with tab7:
    st.subheader("Full Classification Report")
    full_rows = [
        {
            "Crop": c,
            "Precision": round(report[c]["precision"], 3),
            "Recall": round(report[c]["recall"], 3),
            "F1 Score": round(report[c]["f1-score"], 3),
            "Support": int(report[c]["support"]),
        }
        for c in sorted(valid_classes) if c in report
    ]
    full_df = pd.DataFrame(full_rows)
    st.dataframe(full_df, use_container_width=True, hide_index=True)

    st.markdown("#### Averages")
    avg_rows = [
        {
            "Average": k.title(),
            "Precision": round(report[k]["precision"], 3),
            "Recall": round(report[k]["recall"], 3),
            "F1": round(report[k]["f1-score"], 3),
            "Support": int(report[k]["support"]),
        }
        for k in ["macro avg","weighted avg"] if k in report
    ]
    st.dataframe(pd.DataFrame(avg_rows), use_container_width=True, hide_index=True)
    st.info(
        "Macro F1 treats all classes equally. "
        "Weighted F1 weights by class support. "
        "Accuracy is on a held-out 20% test split."
    )
    st.download_button("Download Full Report (CSV)",
                       full_df.to_csv(index=False).encode(),
                       file_name="full_classification_report.csv", mime="text/csv")

# ==========================================================
# FOOTER
# ==========================================================

st.divider()
st.caption(
    f"Smart Kisan Model Performance | Random Forest (balanced) | "
    f"Test accuracy: {metrics['test_accuracy']*100:.2f}% | "
    f"Macro F1: {metrics['macro_f1']:.3f} | "
    f"Baseline: {baseline*100:.1f}%"
)