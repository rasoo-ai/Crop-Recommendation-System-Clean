"""Shared presentation helpers for the Smart Kisan Streamlit application."""

from pathlib import Path

import streamlit as st


APP_ROOT = Path(__file__).parent


def configure_page(title: str, icon: str = "🌾") -> None:
    """Configure the global Smart Kisan page appearance."""

    st.set_page_config(
        page_title=f"{title} | Smart Kisan",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>

        /* =====================================================
           SMART KISAN DESIGN SYSTEM
        ===================================================== */

        :root {
            --forest: #123d2d;
            --forest-light: #1b6b4b;
            --leaf: #2f8f5b;
            --lime: #9fd56a;
            --cream: #f7faf6;
            --white: #ffffff;
            --ink: #19352a;
            --muted: #617469;
            --line: #dce9df;
            --soft-green: #edf7ef;
            --soft-yellow: #fff8e7;
        }


        /* =====================================================
           MAIN APPLICATION
        ===================================================== */

        .stApp {
            background: var(--cream);
            color: var(--ink);
        }

        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }


        /* =====================================================
           SIDEBAR
        ===================================================== */

        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #f0f7f1 0%,
                #e8f3ea 100%
            );

            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
        }

        [data-testid="stSidebar"] .stPageLink a {
            border-radius: 12px;
            padding: .65rem .75rem;
            margin: .2rem 0;
            font-weight: 600;
            transition: all .15s ease;
        }

        [data-testid="stSidebar"] .stPageLink a:hover {
            background: #dcefdc;
            color: var(--forest);
        }

        [data-testid="stSidebar"] hr {
            border-color: var(--line);
        }


        /* =====================================================
           TYPOGRAPHY
        ===================================================== */

        h1,
        h2,
        h3 {
            color: var(--forest);
            letter-spacing: -.025em;
        }

        h1 {
            font-weight: 750;
        }

        h2 {
            margin-top: .3rem;
        }


        /* =====================================================
           HERO
        ===================================================== */

        .hero {
            position: relative;
            overflow: hidden;

            background:
                radial-gradient(
                    circle at 90% 10%,
                    rgba(159, 213, 106, .22),
                    transparent 30%
                ),
                linear-gradient(
                    120deg,
                    #103d2c,
                    #247856
                );

            border-radius: 24px;
            color: white;

            padding: 2.5rem 2.8rem;
            margin: .2rem 0 2rem;

            box-shadow:
                0 14px 35px rgba(18, 61, 45, .16);
        }

        .hero h1,
        .hero p {
            color: white !important;
            margin: 0;
        }

        .hero h1 {
            font-size: clamp(2rem, 4vw, 3rem);
            margin-top: .25rem;
        }

        .hero p {
            opacity: .92;
            max-width: 760px;
            margin-top: .8rem;
            font-size: 1.05rem;
            line-height: 1.65;
        }

        .eyebrow {
            color: #c5edb1 !important;
            font-size: .75rem;
            font-weight: 800;
            letter-spacing: .14em;
            text-transform: uppercase;
        }


        /* =====================================================
           SECTION HEADINGS
        ===================================================== */

        .section-kicker {
            color: var(--leaf);
            font-size: .75rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin-bottom: .1rem;
        }


        /* =====================================================
           METRIC CARDS
        ===================================================== */

        [data-testid="stMetric"] {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 16px;

            padding: 1.15rem;

            box-shadow:
                0 5px 18px rgba(18, 61, 45, .045);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: var(--forest);
            font-weight: 750;
        }


        /* =====================================================
           CARDS / CONTAINERS
        ===================================================== */

        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stExpander"] {
            border-color: var(--line) !important;
            border-radius: 16px !important;
            background: var(--white);
        }


        /* =====================================================
           BUTTONS
        ===================================================== */

        .stButton > button {
            border-radius: 11px;
            min-height: 2.8rem;
            font-weight: 700;
        }

        .stButton > button:hover {
            border-color: var(--leaf);
        }


        /* =====================================================
           FORMS
        ===================================================== */

        .form-card {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.4rem 1.55rem .95rem;
            margin-bottom: 1.2rem;

            box-shadow:
                0 5px 18px rgba(18, 61, 45, .04);
        }

        .form-card h2 {
            font-size: 1.35rem;
            margin-top: 0;
        }

        .stFormSubmitButton > button {
            background: var(--leaf) !important;
            border-color: var(--leaf) !important;
            border-radius: 12px !important;

            box-shadow:
                0 8px 18px rgba(47, 143, 91, .20);

            font-size: 1rem !important;
            font-weight: 700 !important;
            min-height: 3.2rem !important;
        }

        .stFormSubmitButton > button:hover {
            background: var(--forest) !important;
            border-color: var(--forest) !important;
        }


        /* =====================================================
           RECOMMENDATION CARDS
        ===================================================== */

        .recommendation-card {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            height: 100%;

            box-shadow:
                0 5px 18px rgba(18, 61, 45, .04);
        }

        .recommendation-card--primary {
            border-color: #bfe0c3;
            border-top: 4px solid var(--leaf);
        }


        .confidence-chip {
            display: inline-block;
            border-radius: 999px;

            font-size: .78rem;
            font-weight: 700;

            padding: .32rem .7rem;
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


        /* =====================================================
           ABOUT CARD
        ===================================================== */

        .about-card {
            background:
                linear-gradient(
                    135deg,
                    #ffffff 0%,
                    #f2faf3 100%
                );

            border: 1px solid #cfe4d2;
            border-radius: 20px;

            padding: 2rem 2.2rem;
            margin-top: 1.75rem;

            box-shadow:
                0 8px 24px rgba(18, 61, 45, .06);
        }

        .about-card__label {
            color: var(--leaf);
            font-size: .75rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
            margin-bottom: .45rem;
        }

        .about-card h2 {
            font-size: 2rem;
            margin: 0;
        }

        .about-card__subtitle {
            color: var(--forest);
            font-size: 1.15rem;
            font-weight: 650;
            margin: .35rem 0 1rem;
        }

        .about-card__description {
            color: var(--muted);
            line-height: 1.7;
            max-width: 850px;
            margin: 0;
        }


        /* =====================================================
           STATUS / INFO
        ===================================================== */

        [data-testid="stAlert"] {
            border-radius: 13px;
        }


        /* =====================================================
           TABLES
        ===================================================== */

        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }


        /* =====================================================
           FOOTER
        ===================================================== */

        .smart-footer {
            margin-top: 3rem;
            padding-top: 1.2rem;

            border-top: 1px solid var(--line);

            color: var(--muted);
            font-size: .82rem;
            text-align: center;
        }


        /* =====================================================
           MOBILE
        ===================================================== */

        @media (max-width: 640px) {

            .block-container {
                padding: 1rem;
            }

            .hero {
                padding: 1.6rem;
                border-radius: 18px;
            }

            .hero h1 {
                font-size: 2rem;
            }

            .form-card {
                padding: 1.15rem 1rem .75rem;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar() -> None:
    """Render the Smart Kisan navigation sidebar."""

    with st.sidebar:

        st.markdown(
            """
            <div style="
                padding:.3rem 0 1rem;
            ">
                <div style="
                    font-size:1.45rem;
                    font-weight:800;
                    color:#123d2d;
                ">
                    🌾 Smart Kisan
                </div>

                <div style="
                    color:#617469;
                    font-size:.82rem;
                    margin-top:.25rem;
                    line-height:1.4;
                ">
                    AI-assisted agricultural intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.caption("NAVIGATION")

        st.page_link(
            "app.py",
            label="Home Dashboard",
            icon="⌂",
        )

        st.page_link(
            "pages/2_Crop_Recommendation.py",
            label="Crop Recommendation",
            icon="🌱",
        )

        st.page_link(
            "pages/3_Tehsil_Analysis.py",
            label="Regional Intelligence",
            icon="📍",
        )

        st.page_link(
            "pages/4_Model_Performance.py",
            label="Model Performance",
            icon="📊",
        )

        st.divider()

        st.caption("AI MODEL")

        st.markdown(
            """
            <div style="
                background:#ffffff;
                border:1px solid #dce9df;
                border-radius:14px;
                padding:1rem;
                margin-top:.5rem;
            ">

                <div style="
                    font-size:.72rem;
                    font-weight:800;
                    color:#2f8f5b;
                    letter-spacing:.1em;
                    text-transform:uppercase;
                ">
                    Production Model
                </div>

                <div style="
                    font-size:1rem;
                    font-weight:750;
                    color:#123d2d;
                    margin-top:.35rem;
                ">
                    Balanced Random Forest
                </div>

                <div style="
                    font-size:.8rem;
                    color:#617469;
                    margin-top:.45rem;
                    line-height:1.5;
                ">
                    Evaluated on 1,129 held-out records
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


def hero(
    kicker: str,
    title: str,
    description: str,
) -> None:
    """Render a large page hero."""

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


def footer() -> None:
    """Render the application footer."""

    st.markdown(
        """
        <div class="smart-footer">
            <strong>Smart Kisan</strong>
            · AI-powered crop decision support
            · Designed for informed agricultural decisions
        </div>
        """,
        unsafe_allow_html=True,
    )