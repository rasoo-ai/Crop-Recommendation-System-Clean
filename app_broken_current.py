import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Kisan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f6f8f4;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e9e3;
    }

    h1, h2, h3 {
        color: #123d2d !important;
    }

    .hero {
        background: linear-gradient(135deg, #164d37, #1b6b4b);
        padding: 45px;
        border-radius: 22px;
        margin-bottom: 30px;
    }

    .hero-label {
        color: #cce8d5;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .hero-title {
        color: white;
        font-size: 44px;
        font-weight: 800;
        line-height: 1.1;
        margin-top: 12px;
    }

    .hero-text {
        color: #edf7ef;
        font-size: 17px;
        line-height: 1.6;
        max-width: 760px;
        margin-top: 15px;
    }

    .card {
        background: #ffffff;
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #e1e8e2;
        min-height: 180px;
    }

    .card-icon {
        font-size: 32px;
    }

    .card-title {
        color: #123d2d;
        font-size: 20px;
        font-weight: 750;
        margin-top: 10px;
    }

    .card-text {
        color: #64766c;
        line-height: 1.6;
        margin-top: 8px;
    }

    .metric-card {
        background: #ffffff;
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #e1e8e2;
        min-height: 125px;
    }

    .metric-label {
        color: #64766c;
        font-size: 14px;
    }

    .metric-value {
        color: #123d2d;
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-note {
        color: #89958e;
        font-size: 12px;
        margin-top: 5px;
    }

    .footer {
        text-align: center;
        color: #7a887f;
        padding: 30px 0;
        margin-top: 45px;
        border-top: 1px solid #dfe7e0;
    }

    .footer strong {
        color: #1b6b4b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🌾 Smart Kisan")
    st.caption("AI-assisted crop decision support")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🌱 Crop Recommendation",
            "📍 Tehsil Analysis",
            "📊 Model Performance",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.caption("PRODUCTION MODEL")

    st.markdown("**Balanced Random Forest**")

    st.caption("Held-out accuracy")

    st.success("93.53%")

    st.caption("1,129 test records · 80/20 stratified split")


# =========================================================
# METRIC CARD
# =========================================================

def metric_card(label, value, note):
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


# =========================================================
# HOME
# =========================================================

def show_home():

    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">
                SMART AGRICULTURE PLATFORM
            </div>

            <div class="hero-title">
                Farm decisions,<br>
                made clearer.
            </div>

            <div class="hero-text">
                AI-powered crop recommendations for smarter farming —
                bringing soil, weather, and regional context into one
                practical decision workspace.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Explore Smart Kisan")

    st.write(
        "Your agricultural decision workspace. "
        "Choose a tool to explore farm recommendations "
        "or understand the intelligence behind the platform."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="card">
                <div class="card-icon">🌱</div>
                <div class="card-title">
                    Crop Recommendation
                </div>
                <div class="card-text">
                    Enter soil, weather, location, and nutrient
                    conditions to generate model-based crop
                    recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <div class="card-icon">📊</div>
                <div class="card-title">
                    Model Analytics
                </div>
                <div class="card-text">
                    Explore model accuracy, evaluation indicators,
                    and responsible-use limitations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.header("How Smart Kisan works")

    st.write(
        "A practical workflow from farm conditions "
        "to an understandable recommendation."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("01 · Enter conditions")
        st.write(
            "Provide soil, nutrient, weather, and "
            "location information."
        )

    with col2:
        st.subheader("02 · AI analysis")
        st.write(
            "The crop recommendation model evaluates "
            "the agricultural input conditions."
        )

    with col3:
        st.subheader("03 · Review")
        st.write(
            "Review the recommendation and supporting "
            "information before making a decision."
        )

    st.markdown("---")

    st.header("Designed for agriculture")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌱 Soil-aware")
        st.write(
            "Uses soil and nutrient information to "
            "understand farm conditions."
        )

    with col2:
        st.subheader("🌦️ Contextual")
        st.write(
            "Considers weather and regional agricultural "
            "conditions."
        )

    with col3:
        st.subheader("🔎 Transparent")
        st.write(
            "Keeps model limitations visible and "
            "encourages responsible use."
        )

    st.info(
        "Smart Kisan provides decision-support insights. "
        "Crop selection should also consider local seasonality, "
        "irrigation availability, market conditions, field "
        "observations, and qualified agricultural guidance."
    )


# =========================================================
# CROP RECOMMENDATION
# =========================================================

def show_crop_recommendation():

    st.title("🌱 Crop Recommendation")

    st.write(
        "Enter soil, weather, location, and nutrient "
        "conditions to generate a model-based crop "
        "recommendation."
    )

    st.subheader("Farm conditions")

    col1, col2, col3 = st.columns(3)

    with col1:
        nitrogen = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            value=50.0,
        )

    with col2:
        phosphorus = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            value=50.0,
        )

    with col3:
        potassium = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            value=50.0,
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.number_input(
            "Temperature (°C)",
            value=25.0,
        )

    with col2:
        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
        )

    with col3:
        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=100.0,
        )

    if st.button(
        "🌱 Generate Recommendation",
        type="primary",
        width="stretch",
    ):
        st.info(
            "The recommendation interface is ready. "
            "Connect the saved Balanced Random Forest model "
            "to generate the actual crop prediction."
        )


# =========================================================
# TEHSIL ANALYSIS
# =========================================================

def show_tehsil_analysis():

    st.title("📍 Tehsil Analysis")

    st.write(
        "Explore agricultural patterns and records "
        "available in the Smart Kisan project dataset."
    )

    st.info(
        "This page presents descriptive insights from "
        "the offline project dataset. It is not a live "
        "GIS service or real-time agricultural advisory."
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

def show_model_performance():

    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">
                MODEL INTELLIGENCE
            </div>

            <div class="hero-title">
                Model Performance
            </div>

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
        "They are not a guarantee that every individual "
        "farm prediction will be correct."
    )

    st.caption("PERFORMANCE OVERVIEW")
    st.header("Key evaluation indicators")

    st.write(
        "Production model evaluation using the project's "
        "saved held-out evaluation results."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card(
            "Test accuracy",
            "93.53%",
            "Held-out evaluation",
        )

    with col2:
        metric_card(
            "Macro F1",
            "0.682",
            "Class-balanced performance",
        )

    with col3:
        metric_card(
            "Weighted F1",
            "0.943",
            "Support-weighted performance",
        )

    with col4:
        metric_card(
            "Test records",
            "1,129",
            "80/20 stratified split",
        )

    st.markdown("---")

    st.caption("EVALUATION DETAILS")
    st.header("Additional evaluation indicators")

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card(
            "Top-3 accuracy",
            "98.14%",
            "Top three predictions",
        )

    with col2:
        metric_card(
            "Wrong predictions",
            "73",
            "Held-out records",
        )

    with col3:
        metric_card(
            "Error rate",
            "6.47%",
            "1 − accuracy",
        )

    st.markdown("---")

    st.caption("MODEL SUMMARY")
    st.header("How the model was evaluated")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌲 Balanced Random Forest")
        st.write(
            "Production-style classification model used "
            "by the Smart Kisan crop recommendation workflow."
        )

    with col2:
        st.subheader("🧪 80 / 20 Stratified Split")
        st.write(
            "A held-out evaluation set is used to measure "
            "general classification performance separately "
            "from model training."
        )

    st.markdown("---")

    st.caption("CROP-LEVEL PERFORMANCE")
    st.header("Crop evaluation")

    st.write(
        "Precision, recall, and F1 performance across the "
        "crop classes represented in the evaluation data."
    )

    st.warning(
        "Crop-level evaluation is not displayed because "
        "the original held-out predictions and labels are "
        "not currently available in the project files."
    )

    st.caption("CONFUSION MATRIX")
    st.header("Prediction patterns")

    st.write(
        "Shows where the model's predicted crop differs "
        "from the actual crop in the held-out evaluation set."
    )

    st.warning(
        "The original held-out true labels and predictions "
        "are not currently stored in the project, so a "
        "confusion matrix cannot be recreated reliably."
    )

    st.caption("FEATURE IMPORTANCE")
    st.header("What the model uses")

    st.write(
        "Relative importance of the agricultural input "
        "features used by the Random Forest."
    )

    st.warning(
        "Feature-importance values are not available from "
        "the currently saved model/evaluation files."
    )

    st.markdown("---")

    st.caption("RESPONSIBLE USE")
    st.header("Model limitations")

    st.subheader(
        "Accuracy is useful, but it is not the whole story."
    )

    st.write(
        "Model performance is measured on historical project "
        "data and may not represent every soil type, climate, "
        "season, farm practice, or future growing condition."
    )

    st.write(
        "A high overall accuracy does not mean every crop "
        "class has identical performance."
    )

    st.warning(
        "Recommendations should be considered alongside "
        "local seasonality, irrigation availability, market "
        "conditions, field observations, and qualified "
        "agricultural advice."
    )


# =========================================================
# ROUTING
# =========================================================

if page == "🏠 Home":
    show_home()

elif page == "🌱 Crop Recommendation":
    show_crop_recommendation()

elif page == "📍 Tehsil Analysis":
    show_tehsil_analysis()

elif page == "📊 Model Performance":
    show_model_performance()


# =========================================================
# FOOTER
# =========================================================

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