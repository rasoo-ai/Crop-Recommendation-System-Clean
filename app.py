import streamlit as st
import requests

st.set_page_config(
    page_title="Smart Kisan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #f0f7ee 0%, #e8f5e2 50%, #f0f7ee 100%); }
.block-container { max-width: 1300px; padding-top: 1.5rem; padding-bottom: 3rem; }
footer { visibility: hidden; }
.hero-banner {
    background: linear-gradient(135deg, #1a5c2a 0%, #2d8a45 40%, #4aad5e 100%);
    border-radius: 20px; padding: 3rem 3.5rem; color: white;
    margin-bottom: 2rem; position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(26,92,42,0.3);
}
.hero-banner::before {
    content: "🌾"; position: absolute; right: 3rem; top: 50%;
    transform: translateY(-50%); font-size: 8rem; opacity: 0.15;
}
.hero-title { font-size: 3rem; font-weight: 700; margin: 0 0 0.5rem 0; letter-spacing: -1px; }
.hero-sub { font-size: 1.2rem; opacity: 0.9; margin: 0 0 1.5rem 0; font-weight: 300; }
.hero-badge {
    display: inline-block; background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3); border-radius: 30px;
    padding: 0.4rem 1.2rem; font-size: 0.85rem; font-weight: 500; margin-right: 0.5rem;
}
.stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }
.stat-card {
    background: white; border-radius: 16px; padding: 1.5rem;
    text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border-top: 4px solid #2d8a45;
}
.stat-number { font-size: 2.2rem; font-weight: 700; color: #1a5c2a; line-height: 1; margin-bottom: 0.3rem; }
.stat-label { font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }
.feature-card {
    background: white; border-radius: 16px; padding: 1.8rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #e8f5e2;
    height: 100%; position: relative; overflow: hidden;
}
.feature-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #2d8a45, #4aad5e); border-radius: 16px 16px 0 0;
}
.feature-icon { font-size: 2.5rem; margin-bottom: 1rem; }
.feature-title { font-size: 1.1rem; font-weight: 600; color: #1a5c2a; margin-bottom: 0.6rem; }
.feature-desc { font-size: 0.9rem; color: #555; line-height: 1.6; }
.feature-tag { display: inline-block; background: #e8f5e2; color: #2d8a45; border-radius: 20px; padding: 0.2rem 0.8rem; font-size: 0.75rem; font-weight: 600; margin-top: 1rem; }
.feature-tag-soon { display: inline-block; background: #fff8e6; color: #d97706; border-radius: 20px; padding: 0.2rem 0.8rem; font-size: 0.75rem; font-weight: 600; margin-top: 1rem; }
.progress-row { background: white; border-radius: 12px; padding: 1rem 1.5rem; margin-bottom: 0.8rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); display: flex; align-items: center; gap: 1rem; }
.progress-label { font-weight: 600; color: #333; min-width: 180px; font-size: 0.9rem; }
.progress-bar-bg { flex: 1; background: #e8f5e2; border-radius: 8px; height: 10px; overflow: hidden; }
.progress-bar-fill { height: 100%; border-radius: 8px; }
.progress-value { font-weight: 700; color: #1a5c2a; font-size: 0.9rem; min-width: 50px; text-align: right; }
.weather-card {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    border-radius: 16px; padding: 1.5rem; color: white;
    box-shadow: 0 4px 20px rgba(14,165,233,0.3); margin-bottom: 1rem;
}
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a5c2a 0%, #2d8a45 100%); }
[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.markdown("## 🌾 Smart Kisan")
    st.markdown("*AI-powered farming assistant*")
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    st.page_link("app.py",                         label="🏠 Home")
    st.page_link("pages/2_Crop_Recommendation.py",  label="🌱 Crop Recommendation")
    st.page_link("pages/3_Tehsil_Analysis.py",      label="📍 Tehsil Analysis")
    st.page_link("pages/4_Model_Performance.py",    label="📊 Model Performance")
    st.markdown("---")
    st.markdown("**Model V1 — Live**")
    st.markdown("🟢 Random Forest (balanced)")
    st.markdown("📈 Test Accuracy: **93.53%**")
    st.markdown("📋 Test Records: **1,129**")
    st.markdown("🎯 Macro F1: **0.682**")

# ==========================================================
# HERO
# ==========================================================

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🌾 Smart Kisan</div>
    <div class="hero-sub">AI-powered farming decisions — built for Indian agriculture</div>
    <span class="hero-badge">🤖 Machine Learning</span>
    <span class="hero-badge">🗺️ 20+ States</span>
    <span class="hero-badge">🌱 12 Crops</span>
    <span class="hero-badge">✅ 93.5% Accuracy</span>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# STATS
# ==========================================================

st.markdown("""
<div class="stat-row">
    <div class="stat-card"><div class="stat-number">93.53%</div><div class="stat-label">Test Accuracy</div></div>
    <div class="stat-card"><div class="stat-number">98.14%</div><div class="stat-label">Top-3 Accuracy</div></div>
    <div class="stat-card"><div class="stat-number">1,129</div><div class="stat-label">Test Records</div></div>
    <div class="stat-card"><div class="stat-number">0.682</div><div class="stat-label">Macro F1 Score</div></div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# LIVE WEATHER
# ==========================================================

st.markdown("### 🌦️ Live Weather for Farming")
st.caption("Check current conditions before planting, spraying or irrigating.")

WEATHER_API_KEY = "5454c305471a447ae8f570f68e62480c" # get free from openweathermap.org

col_city, col_btn = st.columns([4, 1])
with col_city:
    city = st.text_input("City", value="Hyderabad", label_visibility="collapsed", placeholder="Enter your city...")
with col_btn:
    check_weather = st.button("🌤️ Check Weather", use_container_width=True)

if check_weather and city:
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        r = requests.get(url, timeout=5).json()

        if r.get("cod") == 200:
            temp     = r["main"]["temp"]
            feels    = r["main"]["feels_like"]
            humidity = r["main"]["humidity"]
            wind     = r["wind"]["speed"]
            desc     = r["weather"][0]["description"].title()
            rain     = r.get("rain", {}).get("1h", 0)

            w1, w2, w3, w4, w5 = st.columns(5)
            w1.metric("🌡️ Temperature",  f"{temp}°C",   f"Feels {feels}°C")
            w2.metric("💧 Humidity",     f"{humidity}%")
            w3.metric("🌧️ Rainfall",     f"{rain} mm/hr")
            w4.metric("💨 Wind Speed",   f"{wind} m/s")
            w5.metric("☁️ Condition",    desc)

            # Smart farming advice
            if rain > 5:
                st.warning("🌧️ Heavy rainfall — avoid spraying pesticides or fertilizers today.")
            elif temp > 38:
                st.warning("🔥 Very high temperature — irrigate crops early morning or evening only.")
            elif humidity > 85:
                st.warning("💦 High humidity — watch for fungal diseases in crops.")
            elif wind > 10:
                st.warning("💨 High wind speed — avoid aerial spraying today.")
            else:
                st.success(f"✅ Good farming conditions in **{city}** today!")
        else:
            st.error("City not found. Try: Hyderabad, Delhi, Mumbai, Pune, Chennai")
    except Exception:
        st.error("Could not fetch weather. Check your internet connection.")

st.markdown("---")

# ==========================================================
# FEATURES
# ==========================================================

st.markdown("### 🌱 Platform Features")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""<div class="feature-card"><div class="feature-icon">🌱</div><div class="feature-title">Crop Recommendation</div><div class="feature-desc">Enter your soil profile, weather data and location. Get ML-powered crop recommendations with confidence scores and regional evidence.</div><span class="feature-tag">✅ Live</span></div>""", unsafe_allow_html=True)
    st.page_link("pages/2_Crop_Recommendation.py", label="→ Open Crop Recommendation")
with c2:
    st.markdown("""<div class="feature-card"><div class="feature-icon">📍</div><div class="feature-title">Tehsil Analysis</div><div class="feature-desc">Explore crop recommendations by state, district and tehsil. Interactive map shows ML predictions for every region.</div><span class="feature-tag">✅ Live</span></div>""", unsafe_allow_html=True)
    st.page_link("pages/3_Tehsil_Analysis.py", label="→ Open Tehsil Analysis")
with c3:
    st.markdown("""<div class="feature-card"><div class="feature-icon">📊</div><div class="feature-title">Model Performance</div><div class="feature-desc">Deep-dive into model evaluation — per-crop F1 scores, confusion matrix, feature importance, error analysis and ROC curves.</div><span class="feature-tag">✅ Live</span></div>""", unsafe_allow_html=True)
    st.page_link("pages/4_Model_Performance.py", label="→ Open Model Performance")

st.markdown("<br>", unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    st.markdown("""<div class="feature-card"><div class="feature-icon">🌦️</div><div class="feature-title">Weather Intelligence</div><div class="feature-desc">Live weather data for irrigation planning and seasonal farming decisions.</div><span class="feature-tag">✅ Live</span></div>""", unsafe_allow_html=True)
with c5:
    st.markdown("""<div class="feature-card"><div class="feature-icon">💰</div><div class="feature-title">Mandi Prices</div><div class="feature-desc">Real-time market prices to help farmers compare crops and identify best selling opportunities.</div><span class="feature-tag-soon">🔜 Coming Soon</span></div>""", unsafe_allow_html=True)
with c6:
    st.markdown("""<div class="feature-card"><div class="feature-icon">👨‍🌾</div><div class="feature-title">My Farm</div><div class="feature-desc">Store your farm profile, crop history, soil records and personalised recommendations.</div><span class="feature-tag-soon">🔜 Coming Soon</span></div>""", unsafe_allow_html=True)

# ==========================================================
# MODEL SNAPSHOT
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🤖 Model Performance Snapshot")
col_left, col_right = st.columns(2)
with col_left:
    st.markdown("**Per-crop model strength:**")
    crops_f1 = [("Cotton",0.97),("Rice",0.95),("Maize",0.93),("Wheat",0.91),("Pulses",0.88),("Mustard",0.85),("Sugarcane",0.72),("Vegetables",0.61),("Potato",0.48),("Barley",0.35)]
    for crop, score in crops_f1:
        pct = int(score * 100)
        color = "#16a34a" if score >= 0.7 else "#f59e0b" if score >= 0.5 else "#ef4444"
        st.markdown(f"""<div class="progress-row"><div class="progress-label">{crop}</div><div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{pct}%;background:{color}"></div></div><div class="progress-value" style="color:{color}">{score:.2f}</div></div>""", unsafe_allow_html=True)
with col_right:
    st.markdown("**What makes Smart Kisan work:**")
    with st.container(border=True):
        st.markdown("""
| Feature | Role |
|---|---|
| 🌡️ Temperature | Climate suitability |
| 🌧️ Rainfall | Water requirement |
| 🧪 Nitrogen (N) | Soil fertility |
| 🌱 Soil Moisture | Irrigation need |
| ⚗️ Soil pH | Nutrient availability |
| 🪨 Potassium (K) | Root development |
| 🌍 State / Zone | Regional context |
| 💧 Humidity | Disease risk |
""")
    st.info("🔑 **Top insight:** Temperature, Rainfall and Nitrogen account for over 40% of the model's decision-making power.")

# ==========================================================
# HOW IT WORKS
# ==========================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### ⚙️ How It Works")
h1, h2, h3, h4 = st.columns(4)
for col, icon, step, desc in [
    (h1,"📋","1. Enter Data","Input soil test results, location and weather"),
    (h2,"🤖","2. ML Processing","Random Forest analyses 19 features"),
    (h3,"🎯","3. Get Results","Top-3 recommendations with confidence"),
    (h4,"📊","4. Verify","Review regional data and model metrics"),
]:
    with col:
        with st.container(border=True):
            st.markdown(f"### {icon}")
            st.markdown(f"**{step}**")
            st.markdown(f"<small>{desc}</small>", unsafe_allow_html=True)

st.divider()
st.caption("🌾 Smart Kisan — data-driven agricultural guidance. Not a guarantee of yield or profit. Model V1 | Random Forest | 93.53% accuracy")