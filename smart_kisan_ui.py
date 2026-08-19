import streamlit as st


def configure_page(title: str, icon: str = "🌾") -> None:
    st.set_page_config(
        page_title=f"{title} | Smart Kisan",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
<style>
:root {
    --forest: #123d2d;
    --leaf: #1b6b4b;
    --lime: #9fd56a;
    --cream: #f7faf6;
    --ink: #19352a;
    --muted: #64766c;
    --line: #dce9df;
}

.stApp {
    background: var(--cream);
    color: var(--ink);
}

.block-container {
    max-width: 1240px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #f0f7f1;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] .stPageLink a {
    border-radius: 10px;
    padding: .55rem .65rem;
    margin: .15rem 0;
}

[data-testid="stSidebar"] .stPageLink a:hover {
    background: #dff2d8;
    color: var(--forest);
}

h1, h2, h3 {
    color: var(--forest);
    letter-spacing: -.02em;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1rem;
    box-shadow: 0 3px 12px rgba(18,61,45,.035);
}

.stButton > button {
    border-radius: 10px;
    min-height: 2.7rem;
    font-weight: 650;
}

.stButton > button:hover {
    border-color: var(--leaf);
    color: var(--forest);
}

footer {
    visibility: hidden;
}

@media (max-width: 640px) {
    .block-container {
        padding: 1rem;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def sidebar() -> None:
    with st.sidebar:

        st.markdown(
            """
<div style="
    font-size:1.55rem;
    font-weight:800;
    color:#123d2d;
">
    🌾 Smart Kisan
</div>

<div style="
    color:#64766c;
    font-size:0.86rem;
    margin-top:0.25rem;
    line-height:1.45;
">
    AI-assisted crop decision support
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

        st.caption("MODEL STATUS")

        st.markdown(
            """
<div style="
    color:#64766c;
    font-size:0.70rem;
    font-weight:800;
    letter-spacing:0.08em;
    text-transform:uppercase;
">
    Production model
</div>

<div style="
    color:#123d2d;
    font-size:0.95rem;
    font-weight:750;
    margin-top:0.25rem;
">
    Balanced Random Forest
</div>

<div style="
    color:#64766c;
    font-size:0.78rem;
    margin-top:0.45rem;
">
    Held-out accuracy
</div>

<div style="
    color:#1b6b4b;
    font-size:1.35rem;
    font-weight:800;
    margin-top:0.05rem;
">
    93.53%
</div>

<div style="
    color:#64766c;
    font-size:0.78rem;
    margin-top:0.45rem;
">
    1,129 test records · 80/20 stratified split
</div>
            """,
            unsafe_allow_html=True,
        )


def section(title: str, description: str | None = None) -> None:
    st.markdown(
        f"""
<div style="
    color:#1b6b4b;
    font-size:0.78rem;
    font-weight:800;
    letter-spacing:0.10em;
    text-transform:uppercase;
    margin-bottom:0.15rem;
">
    Smart Kisan
</div>

<h2 style="
    color:#123d2d;
    margin-top:0;
    margin-bottom:0.3rem;
">
    {title}
</h2>
        """,
        unsafe_allow_html=True,
    )

    if description:
        st.caption(description)


def hero(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
<div style="
    background:linear-gradient(135deg,#103d2c 0%,#247856 100%);
    border-radius:24px;
    padding:3rem 3.2rem;
    margin-bottom:2rem;
    color:white;
    box-shadow:0 14px 32px rgba(18,61,45,.16);
">

<div style="
    color:#c5edb1;
    font-size:0.78rem;
    font-weight:800;
    letter-spacing:0.12em;
    text-transform:uppercase;
    margin-bottom:0.8rem;
">
    {kicker}
</div>

<h1 style="
    color:white!important;
    font-size:3rem;
    line-height:1.08;
    margin:0;
">
    {title}
</h1>

<p style="
    color:rgba(255,255,255,.90);
    font-size:1.05rem;
    line-height:1.65;
    max-width:760px;
    margin-top:1rem;
">
    {description}
</p>

</div>
        """,
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.divider()

    st.markdown(
        """
<div style="
    text-align:center;
    color:#64766c;
    font-size:0.82rem;
    padding:0.5rem;
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