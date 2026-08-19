import streamlit as st

st.set_page_config(
    page_title="Model Performance | Smart Kisan",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f6f8f4;
    }

    [data-testid="stSidebar"] {
        background-color: white;
    }

    h1, h2, h3 {
        color: #123d2d !important;
    }

    .hero {
        background: linear-gradient(135deg, #164d37, #1b6b4b);
        padding: 40px;
        border-radius: 22px;
        margin-bottom: 25px;
    }

    .hero-label {
        color: #cce8d5;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    .hero-title {
        color: white;
        font-size: 42px;
        font-weight: 800;
        line-height: 1.1;
    }

    .hero-text {
        color: #edf7ef;
        font-size: 17px;
        line-height: 1.6;
        max-width: 750px;
        margin-top: 12px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e1e8e2;
        border-radius: 16px;
        padding: 22px;
        min-height: 130px;
    }

    .metric-label {
        color: #64766c;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #123d2d;
        font-size: 30px;
        font-weight: 800;
        margin-top: 7px;
    }

    .metric-note {
        color: #89958e;
        font-size: 12px;
        margin-top: 5px;
    }

    .info-card {
        background: white;
        border: 1px solid #e1e8e2;
        border-radius: 16px;
        padding: 24px;
        min-height: 170px;
    }

    .info-icon {
        font-size: 30px;
    }

    .info-title {
        color: #123d2d;
        font-size: 19px;
        font-weight: 750;
        margin-top: 8px;
    }

    .info-text {
        color: #64766c;
        line-height: 1.6;
        margin-top: 8px;
    }

    .about-box {
        background: #eef5ee;
        border-radius: 18px;
        padding: 30px;
        margin-top: 15px;
    }

    .about-title {
        color: #123d2d;
        font-size: 30px;
        font-weight: 800;
    }

    .about-subtitle {
        color: #123d2d;
        font-size: 18px;
        font-weight: 650;
        margin-top: 8px;
    }

    .about-text {
        color: #64766c;
        line-height: 1.7;
        margin-top: 14px;
    }

    .warning-box {
        background: white;
        border-left: 4px solid #9fd56a;
        border-radius: 0 10px 10px 0;
        padding: 15px;
        margin-top: 20px;
        color: #64766c;
        line-height: 1.6;
    }

    .footer {
        text-align: center;
        color: #7a887f;
        padding: 30px 0;
        margin-top: 40px;
        border-top: 1px solid #dfe7e0;
    }

    .footer strong {
        color: #1b6b4b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.title("🌾 Smart Kisan")
    st.caption("AI-assisted crop decision support")

    st.divider()

    st.subheader("Navigation")

    st.page_link("app.py", label="🏠 Home")
    st.page_link(
        "pages/2_Crop_Recommendation.py",
        label="🌱 Crop Recommendation",
    )
    st.page_link(
        "pages/3_Tehsil_Analysis.py",
        label="📍 Tehsil Analysis",
    )
    st.page_link(
        "pages/4_Model_Performance.py",
        label="📊 Model Performance",
    )

    st.divider()

    st.caption("PRODUCTION MODEL")
    st.markdown("**Balanced Random Forest**")
    st.caption("Held-out accuracy")
    st.success("93.53%")
    st.caption("1,129 test records · 80/20 stratified split")


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-label">MODEL INTELLIGENCE</div>
        <div class="hero-title">Model Performance</div>
        <div class="hero-text">
            A transparent view of how the Smart Kisan
            Balanced Random Forest performs on held-out
            evaluation data.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "These metrics describe offline model evaluation. "
    "They are not a guarantee that every individual farm "
    "prediction will be correct."
)


# ---------------------------------------------------------
# PERFORMANCE OVERVIEW
# ---------------------------------------------------------

st.caption("PERFORMANCE OVERVIEW")
st.header("Key evaluation indicators")

st.write(
    "Production model evaluation using the project's "
    "saved held-out evaluation results."
)

metrics = [
    ("Test accuracy", "93.53%", "Held-out evaluation"),
    ("Macro F1", "0.682", "Class-balanced performance"),
    ("Weighted F1", "0.943", "Support-weighted performance"),
    ("Test records", "1,129", "80/20 stratified split"),
]

columns = st.columns(4)

for column, item in zip(columns, metrics):

    label, value, note = item

    with column:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# EVALUATION DETAILS
# ---------------------------------------------------------

st.caption("EVALUATION DETAILS")
st.subheader("Additional evaluation indicators")

details = [
    ("Top-3 accuracy", "98.14%", "Top three predictions"),
    ("Wrong predictions", "73", "Held-out records"),
    ("Error rate", "6.47%", "1 − accuracy"),
]

columns = st.columns(3)

for column, item in zip(columns, details):

    label, value, note = item

    with column:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# MODEL SUMMARY
# ---------------------------------------------------------

st.caption("MODEL SUMMARY")
st.header("How the model was evaluated")

c1, c2 = st.columns(2)

with c1:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-icon">🌲</div>
            <div class="info-title">
                Balanced Random Forest
            </div>
            <div class="info-text">
                Production-style classification model used
                by the Smart Kisan crop recommendation workflow.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:

    st.markdown(
        """
        <div class="info-card">
            <div class="info-icon">🧪</div>
            <div class="info-title">
                80 / 20 Stratified Split
            </div>
            <div class="info-text">
                A held-out evaluation set is used to measure
                general classification performance separately
                from model training.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# CROP LEVEL
# ---------------------------------------------------------

st.caption("CROP-LEVEL PERFORMANCE")
st.header("Crop evaluation")

st.write(
    "Precision, recall, and F1 performance across the crop "
    "classes represented in the evaluation data."
)

st.warning(
    "Crop-level evaluation is not displayed because the "
    "original held-out predictions and labels are not "
    "currently available in the project files."
)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

st.caption("CONFUSION MATRIX")
st.header("Prediction patterns")

st.write(
    "Shows where the model's predicted crop differs from "
    "the actual crop in the held-out evaluation set."
)

st.warning(
    "The original held-out true labels and predictions are "
    "not currently stored in the project, so a confusion "
    "matrix cannot be recreated reliably."
)


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

st.caption("FEATURE IMPORTANCE")
st.header("What the model uses")

st.write(
    "Relative importance of the agricultural input features "
    "used by the Random Forest."
)

st.warning(
    "Feature-importance values are not available from the "
    "currently saved model/evaluation files."
)


# ---------------------------------------------------------
# RESPONSIBLE USE
# ---------------------------------------------------------

st.caption("RESPONSIBLE USE")

st.markdown(
    """
    <div class="about-box">

        <div class="about-title">
            Model limitations
        </div>

        <div class="about-subtitle">
            Accuracy is useful, but it is not the whole story.
        </div>

        <div class="about-text">
            Model performance is measured on historical project
            data and may not represent every soil type, climate,
            season, farm practice, or future growing condition.
        </div>

        <div class="about-text">
            A high overall accuracy does not mean every crop
            class has identical performance.
        </div>

        <div class="warning-box">
            Recommendations should be considered alongside local
            seasonality, irrigation availability, market conditions,
            field observations, and qualified agricultural advice.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        <strong>Smart Kisan</strong>
        · AI-powered crop decision support
        · Validate recommendations with local agricultural expertise.
    </div>
    """,
    unsafe_allow_html=True,
)
