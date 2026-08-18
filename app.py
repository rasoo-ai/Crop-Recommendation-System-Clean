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
# SIMPLE CSS
# ==========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7faf7;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
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
# HEADER
# ==========================================================

st.title("🌾 Smart Kisan")

st.subheader(
    "AI-powered farming decisions made simple."
)

st.write(
    "Smart Kisan uses agricultural data and machine learning "
    "to support crop recommendations and regional agricultural analysis."
)

st.divider()


# ==========================================================
# WELCOME
# ==========================================================

st.header("👨‍🌾 Welcome")

st.write(
    "Smart Kisan is being developed as a farmer-focused "
    "agriculture assistant. The platform combines machine "
    "learning with regional agricultural information and "
    "future weather and market services."
)


# ==========================================================
# MAIN FEATURES
# ==========================================================

st.header("🌱 Main Features")


c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.subheader("🌱 Crop Recommendation")
        st.write(
            "Get a machine-learning-based crop recommendation "
            "using soil, weather and location information."
        )
        st.info("Main AI feature — Day 2")


with c2:
    with st.container(border=True):
        st.subheader("📍 Tehsil Analysis")
        st.write(
            "Explore crop information and representative "
            "recommendations for different regions."
        )
        st.page_link(
            "pages/3_Tehsil_Analysis.py",
            label="Open Tehsil Analysis",
        )


with c3:
    with st.container(border=True):
        st.subheader("📊 Model Performance")
        st.write(
            "View the verified evaluation metrics and "
            "understand the current ML model."
        )
        st.page_link(
            "pages/4_Model_Performance.py",
            label="Open Model Performance",
        )


c4, c5, c6 = st.columns(3)


with c4:
    with st.container(border=True):
        st.subheader("🌦️ Weather")
        st.write(
            "Weather information will support irrigation, "
            "spraying and farm planning."
        )
        st.warning("Coming in a later development phase.")


with c5:
    with st.container(border=True):
        st.subheader("💰 Mandi Prices")
        st.write(
            "Market information can help farmers compare "
            "crop prices and selling opportunities."
        )
        st.warning("Coming in a later development phase.")


with c6:
    with st.container(border=True):
        st.subheader("👨‍🌾 My Farm")
        st.write(
            "Future versions can store farm details, "
            "crop history and recommendations."
        )
        st.warning("Coming in a later development phase.")


# ==========================================================
# CURRENT MODEL
# ==========================================================

st.divider()

st.header("✅ Current ML Baseline")

with st.container(border=True):

    st.subheader(
        "Random Forest Crop Recommendation Model"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Test Accuracy",
        "93.53%",
    )

    col2.metric(
        "Top-3 Accuracy",
        "98.14%",
    )

    col3.metric(
        "Macro F1",
        "0.68",
    )

    col4.metric(
        "Test Records",
        "1,129",
    )

    st.write(
        "This is the current Model V1 baseline. "
        "The model will remain unchanged while the "
        "farmer application is developed."
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
        "The Smart Kisan home page and navigation foundation "
        "are complete."
    )

    st.info(
        "Next step: build the farmer-friendly "
        "Crop Recommendation page using the existing "
        "production ML model."
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
