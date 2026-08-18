import streamlit as st


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Smart Kisan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# LIGHT CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f7faf7;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero-title {
            color: #166534;
            font-size: 46px;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .hero-subtitle {
            color: #15803d;
            font-size: 21px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .hero-text {
            color: #475569;
            font-size: 16px;
            line-height: 1.6;
            max-width: 900px;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("🌾 Smart Kisan")
    st.caption("Smart farming assistant")

    st.divider()

    st.subheader("🧭 Navigation")

    st.page_link(
        "app.py",
        label="🏠 Home",
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

    st.success(
        "Model V1\n\n"
        "Random Forest\n\n"
        "Test Accuracy: 93.53%\n\n"
        "Test Records: 1,129"
    )


# ==========================================================
# HERO
# ==========================================================

st.markdown(
    '<div class="hero-title">🌾 Smart Kisan</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-subtitle">AI-powered farming decisions made simple.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-text">'
    "Smart Kisan combines agricultural data and machine learning "
    "to support crop recommendations and regional agricultural analysis."
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


# ==========================================================
# WELCOME
# ==========================================================

st.header("👨‍🌾 Welcome")

st.write(
    "Smart Kisan is being developed as a farmer-focused agriculture "
    "assistant. The platform will combine crop recommendation, "
    "regional analysis, weather information and market information."
)


# ==========================================================
# MAIN FEATURES
# ==========================================================

st.header("🌱 Main Features")


# -------------------------
# Crop Recommendation
# -------------------------

with st.container(border=True):

    st.subheader("🌱 Crop Recommendation")

    st.write(
        "Get a machine-learning-based crop recommendation "
        "using soil, weather and location information."
    )

    st.info(
        "Main AI feature — will be developed on Day 2."
    )


# -------------------------
# Tehsil Analysis
# -------------------------

with st.container(border=True):

    st.subheader("📍 Tehsil Analysis")

    st.write(
        "Explore crop information and representative "
        "recommendations for different regions."
    )

    st.page_link(
        "pages/3_Tehsil_Analysis.py",
        label="📍 Open Tehsil Analysis",
    )


# -------------------------
# Model Performance
# -------------------------

with st.container(border=True):

    st.subheader("📊 Model Performance")

    st.write(
        "View the verified evaluation metrics of the "
        "current crop recommendation model."
    )

    st.page_link(
        "pages/4_Model_Performance.py",
        label="📊 Open Model Performance",
    )


# -------------------------
# Future Features
# -------------------------

col1, col2, col3 = st.columns(3)


with col1:

    with st.container(border=True):

        st.subheader("🌦️ Weather")

        st.write(
            "Weather information for irrigation, "
            "spraying and farm planning."
        )

        st.warning(
            "Coming soon"
        )


with col2:

    with st.container(border=True):

        st.subheader("💰 Mandi Prices")

        st.write(
            "Market information to help farmers "
            "compare crop prices."
        )

        st.warning(
            "Coming soon"
        )


with col3:

    with st.container(border=True):

        st.subheader("👨‍🌾 My Farm")

        st.write(
            "Store farm details, crop history and "
            "recommendations."
        )

        st.warning(
            "Coming soon"
        )


# ==========================================================
# CURRENT ML BASELINE
# ==========================================================

st.divider()

st.header("✅ Current ML Baseline")

with st.container(border=True):

    st.subheader(
        "Random Forest Crop Recommendation Model"
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Test Accuracy",
        "93.53%",
    )

    m2.metric(
        "Top-3 Accuracy",
        "98.14%",
    )

    m3.metric(
        "Macro F1",
        "0.68",
    )

    m4.metric(
        "Test Records",
        "1,129",
    )

    st.write(
        "This is the current Model V1 baseline. "
        "It will remain unchanged while the application "
        "is being developed."
    )


# ==========================================================
# DEVELOPMENT STATUS
# ==========================================================

st.header("🚀 Development Status")

with st.container(border=True):

    st.subheader(
        "Day 1 — Foundation Complete ✅"
    )

    st.write(
        "The Smart Kisan Home page and navigation foundation "
        "are complete."
    )

    st.info(
        "Next: build the farmer-friendly Crop Recommendation page "
        "using the existing production ML model."
    )


# ==========================================================
# DISCLAIMER
# ==========================================================

st.divider()

st.caption(
    "🌾 Smart Kisan provides data-driven agricultural guidance. "
    "Recommendations are not guarantees of crop yield or profit. "
    "Farmers should consider local conditions and qualified "
    "agricultural advice before making planting decisions."
)
