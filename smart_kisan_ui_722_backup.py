"""Shared professional UI components for Smart Kisan."""

from pathlib import Path

import streamlit as st


APP_ROOT = Path(__file__).resolve().parent


# =========================================================
# PAGE CONFIGURATION
# =========================================================

def configure_page(title: str, icon: str = "🌾") -> None:
    """Configure the Streamlit page and apply the global design system."""

    st.set_page_config(
        page_title=f"{title} | Smart Kisan",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>

        /* =================================================
           SMART KISAN DESIGN SYSTEM
        ================================================= */

        :root {
            --forest: #123d2d;
            --forest-dark: #0b2b20;
            --leaf: #1b6b4b;
            --leaf-light: #2f8a62;
            --lime: #9fd56a;

            --cream: #f5f8f4;
            --white: #ffffff;

            --ink: #18352a;
            --muted: #64766c;

            --line: #dce8df;
            --line-dark: #c9dbce;

            --warning: #a66b00;
            --danger: #b5483f;
        }


        /* =================================================
           APP BACKGROUND
        ================================================= */

        .stApp {
            background:
                radial-gradient(
                    circle at 85% 0%,
                    rgba(159, 213, 106, 0.10),
                    transparent 28%
                ),
                var(--cream);
            color: var(--ink);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }


        /* =================================================
           SIDEBAR
        ================================================= */

        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #f0f7f1 0%,
                #eaf4ec 100%
            );

            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.4rem;
        }

        [data-testid="stSidebar"] .stPageLink a {
            border-radius: 12px;
            padding: 0.65rem 0.75rem;
            margin: 0.18rem 0;
            font-weight: 600;
            color: var(--ink);
            transition: all 0.15s ease;
        }

        [data-testid="stSidebar"] .stPageLink a:hover {
            background: #dcefdc;
            color: var(--forest);
        }


        /* =================================================
           TYPOGRAPHY
        ================================================= */

        h1,
        h2,
        h3 {
            color: var(--forest);
            letter-spacing: -0.025em;
        }

        h1 {
            font-weight: 750;
        }

        h2 {
            font-weight: 720;
        }

        h3 {
            font-weight: 700;
        }

        p {
            color: var(--ink);
        }


        /* =================================================
           HERO
        ================================================= */

        .hero {
            position: relative;
            overflow: hidden;

            background:
                radial-gradient(
                    circle at 92% 15%,
                    rgba(159, 213, 106, 0.18),
                    transparent 28%
                ),
                linear-gradient(
                    125deg,
                    var(--forest-dark),
                    var(--leaf)
                );

            border-radius: 24px;
            color: white;

            padding: 2.5rem 2.8rem;
            margin: 0 0 2rem;

            box-shadow:
                0 16px 40px rgba(18, 61, 45, 0.16);
        }

        .hero h1 {
            color: white !important;
            font-size: 2.35rem;
            margin: 0.15rem 0 0;
        }

        .hero p {
            color: white !important;
            opacity: 0.9;
            max-width: 780px;
            margin-top: 0.75rem;
            line-height: 1.65;
            font-size: 1.03rem;
        }

        .eyebrow {
            color: #c8edb6 !important;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin: 0;
        }


        /* =================================================
           SECTION HEADERS
        ================================================= */

        .section-kicker {
            color: var(--leaf) !important;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.1rem;
        }


        /* =================================================
           METRIC CARDS
        ================================================= */

        [data-testid="stMetric"] {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.15rem;

            box-shadow:
                0 5px 18px rgba(18, 61, 45, 0.045);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: var(--forest) !important;
            font-weight: 750;
        }


        /* =================================================
           CARDS
        ================================================= */

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: white;
            border-color: var(--line) !important;
            border-radius: 16px !important;
            box-shadow:
                0 5px 18px rgba(18, 61, 45, 0.04);
        }

        [data-testid="stExpander"] {
            border-color: var(--line) !important;
            border-radius: 14px !important;
            background: white;
        }


        /* =================================================
           FORM CARD
        ================================================= */

        .form-card {
            background: white;
            border: 1px solid var(--line);
            border-radius: 18px;

            padding: 1.45rem 1.6rem 1rem;
            margin-bottom: 1.2rem;

            box-shadow:
                0 5px 18px rgba(18, 61, 45, 0.045);
        }

        .form-card h2 {
            margin-top: 0;
            font-size: 1.35rem;
        }


        /* =================================================
           RECOMMENDATION CARD
        ================================================= */

        .recommendation-card {
            background: white;

            border: 1px solid var(--line);
            border-radius: 16px;

            padding: 1.25rem 1.35rem;

            min-height: 150px;

            box-shadow:
                0 5px 18px rgba(18, 61, 45, 0.045);
        }

        .recommendation-card--primary {
            border-color: #b9dbbf;
            border-top: 4px solid var(--leaf);
        }

        .result-hero {
            background: white;
            border: 1px solid #b9dbbf;
            border-left: 6px solid var(--leaf);

            border-radius: 18px;

            padding: 1.5rem 1.7rem;

            box-shadow:
                0 8px 25px rgba(18, 61, 45, 0.06);
        }


        /* =================================================
           CONFIDENCE BADGES
        ================================================= */

        .confidence-chip {
            display: inline-block;

            border-radius: 999px;

            padding: 0.34rem 0.72rem;

            font-size: 0.77rem;
            font-weight: 750;
        }

        .confidence-chip--high {
            background: #dff2d8;
            color: #17633e;
        }

        .confidence-chip--medium {
            background: #fff1cf;
            color: #8a6200;
        }

        .confidence-chip--low {
            background: #e8f0ed;
            color: #476257;
        }


        /* =================================================
           BUTTONS
        ================================================= */

        .stButton > button {
            border-radius: 11px;
            min-height: 2.75rem;
            font-weight: 700;

            border: 1px solid var(--line-dark);

            transition:
                transform 0.15s ease,
                box-shadow 0.15s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);

            box-shadow:
                0 6px 16px rgba(18, 61, 45, 0.10);
        }

        .stFormSubmitButton > button {
            background: var(--leaf) !important;
            border-color: var(--leaf) !important;

            color: white !important;

            border-radius: 12px !important;

            min-height: 3.15rem !important;

            font-size: 1rem !important;
            font-weight: 750 !important;

            box-shadow:
                0 8px 18px rgba(27, 107, 75, 0.20);
        }

        .stFormSubmitButton > button:hover {
            background: var(--forest) !important;
            border-color: var(--forest) !important;
        }


        /* =================================================
           TABLES
        ================================================= */

        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }


        /* =================================================
           TABS
        ================================================= */

        button[data-baseweb="tab"] {
            font-weight: 650;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--leaf);
        }


        /* =================================================
           ABOUT CARD
        ================================================= */

        .about-card {
            background:
                linear-gradient(
                    135deg,
                    #ffffff 0%,
                    #f2f9f2 100%
                );

            border: 1px solid #cfe3d3;
            border-radius: 18px;

            padding: 2rem 2.2rem;
            margin-top: 1.8rem;

            box-shadow:
                0 8px 24px rgba(18, 61, 45, 0.05);
        }

        .about-card__label {
            color: var(--leaf);

            font-size: 0.74rem;
            font-weight: 800;

            letter-spacing: 0.12em;
            text-transform: uppercase;

            margin-bottom: 0.45rem;
        }

        .about-card h2 {
            margin: 0;
            font-size: 1.9rem;
        }

        .about-card__subtitle {
            color: var(--forest);

            font-size: 1.15rem;
            font-weight: 650;

            margin: 0.35rem 0 1rem;
        }

        .about-card__description {
            color: var(--muted);

            max-width: 900px;

            line-height: 1.7;
            margin: 0;
        }

        .about-card__note {
            background: rgba(255,255,255,0.75);

            border-left: 3px solid var(--lime);

            border-radius: 0 10px 10px 0;

            color: var(--muted);

            font-size: 0.88rem;
            line-height: 1.55;

            padding: 0.75rem 0.95rem;

            margin-top: 1rem;
        }


        /* =================================================
           HIDE STREAMLIT FOOTER
        ================================================= */

        footer {
            visibility: hidden;
        }


        /* =================================================
           MOBILE
        ================================================= */

        @media (max-width: 640px) {

            .block-container {
                padding: 1rem;
            }

            .hero {
                padding: 1.7rem;
                border-radius: 18px;
            }

            .hero h1 {
                font-size: 1.8rem;
            }

            .form-card {
                padding: 1.15rem 1rem 0.8rem;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SIDEBAR
# =========================================================

def sidebar() -> None:
    """Render the clean Smart Kisan navigation sidebar."""

    with st.sidebar:

        st.markdown(
            """
            <div style="
                padding: 0.25rem 0 0.8rem;
            ">
                <div style="
                    font-size: 1.55rem;
                    font-weight: 800;
                    color: #123d2d;
                ">
                    🌾 Smart Kisan
                </div>

                <div style="
                    color: #64766c;
                    font-size: 0.86rem;
                    margin-top: 0.25rem;
                    line-height: 1.45;
                ">
                    AI-assisted crop decision support
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.caption("NAVIGATION")

        st.page_link(
            "app.py",
            label="Home",
            icon="🏠",
        )

        st.page_link(
            "pages/2_Crop_Recommendation.py",
            label="Crop Recommendation",
            icon="🌱",
        )

        st.page_link(
            "pages/3_Tehsil_Analysis.py",
            label="Tehsil Analysis",
            icon="📍",
        )

        st.page_link(
            "pages/4_Model_Performance.py",
            label="Model Performance",
            icon="📊",
        )

        st.divider()

        st.caption("MODEL")

        st.markdown(
            """
            <div style="
                background: #ffffff;
                border: 1px solid #dce8df;
                border-radius: 12px;
                padding: 0.85rem;
                margin-top: 0.4rem;
            ">

                <div style="
                    font-size: 0.72rem;
                    color: #64766c;
                    font-weight: 700;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                ">
                    Production model
                </div>

                <div style="
                    color: #123d2d;
                    font-size: 0.95rem;
                    font-weight: 750;
                    margin-top: 0.25rem;
                ">
                    Balanced Random Forest
                </div>

                <div style="
                    color: #64766c;
                    font-size: 0.78rem;
                    margin-top: 0.4rem;
                    line-height: 1.4;
                ">
                    Held-out accuracy
                    <strong style="color:#1b6b4b;">
                        93.53%
                    </strong>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# HERO
# =========================================================

def hero(
    kicker: str,
    title: str,
    description: str,
) -> None:
    """Render a professional page hero."""

    st.markdown(
        f"""
        <section class="hero">

            <p class="eyebrow">
                {kicker}
            </p>

            <h1>
                {title}
            </h1>

            <p>
                {description}
            </p>

        </section>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SECTION
# =========================================================

def section(
    title: str,
    description: str | None = None,
) -> None:
    """Render a consistent section heading."""

    st.markdown(
        f"""
        <p class="section-kicker">
            Smart Kisan
        </p>

        <h2>
            {title}
        </h2>
        """,
        unsafe_allow_html=True,
    )

    if description:
        st.caption(description)


# =========================================================
# FOOTER
# =========================================================

def footer() -> None:
    """Render the application footer."""

    st.divider()

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#64766c;
            font-size:0.78rem;
            padding:0.5rem 0 1rem;
        ">
            <strong style="color:#1b6b4b;">
                Smart Kisan
            </strong>
            · AI-powered crop decision support
            · Validate recommendations with local agricultural expertise.
        </div>
        """,
        unsafe_allow_html=True,
    )