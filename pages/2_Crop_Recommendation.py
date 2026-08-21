import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Crop Recommendation - Smart Kisan", page_icon="🌱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #f0f7ee 0%, #e8f5e2 100%); }
.block-container { max-width: 1300px; padding-top: 1.5rem; padding-bottom: 3rem; }
footer { visibility: hidden; }
.page-header { background: linear-gradient(135deg, #1a5c2a, #2d8a45); border-radius: 16px; padding: 2rem 2.5rem; color: white; margin-bottom: 2rem; box-shadow: 0 4px 20px rgba(26,92,42,0.25); }
.page-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.page-header p { margin: 0.5rem 0 0; opacity: 0.85; }
.result-hero { background: linear-gradient(135deg, #1a5c2a, #2d8a45); border-radius: 20px; padding: 2.5rem; color: white; text-align: center; margin: 1.5rem 0; box-shadow: 0 8px 32px rgba(26,92,42,0.3); }
.result-hero .crop-name { font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; }
.result-hero .crop-icon { font-size: 4rem; }
.result-hero .confidence-badge { display: inline-block; background: rgba(255,255,255,0.2); border: 2px solid rgba(255,255,255,0.4); border-radius: 30px; padding: 0.5rem 1.5rem; font-size: 1.3rem; font-weight: 700; margin-top: 0.5rem; }
.medal-card { background: white; border-radius: 16px; padding: 1.5rem; text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.07); border: 1px solid #e8f5e2; }
.medal-crop { font-size: 1.2rem; font-weight: 700; color: #1a5c2a; margin: 0.5rem 0; }
.medal-score { font-size: 1.8rem; font-weight: 700; color: #2d8a45; }
.input-section { background: white; border-radius: 16px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border: 1px solid #e8f5e2; margin-bottom: 1rem; }
.input-section h3 { color: #1a5c2a; margin-top: 0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a5c2a 0%, #2d8a45 100%); }
[data-testid="stSidebar"] * { color: white !important; }
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

st.markdown("""<div class="page-header"><h1>🌱 Crop Recommendation</h1><p>Enter your farm conditions and get AI-powered crop recommendations with confidence scores.</p></div>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("output/crop_prediction_model_balanced.pkl")

@st.cache_data
def load_dataset():
    return pd.read_excel("output/Crop_Normalized.xlsx")

try:
    model = load_model()
    df    = load_dataset()
except Exception as e:
    st.error("Unable to load model or dataset.")
    st.exception(e); st.stop()

states     = sorted(df["State_Name"].dropna().astype(str).unique())
soil_types = sorted(df["Soil_Type"].dropna().astype(str).unique())
zones      = sorted(df["Agro_Climatic Zone"].dropna().astype(str).unique())

numeric_columns = ["pH_Value","Nitrogen_Value (N)","Phosphorus_Value (P)","Potassium_Value (K)","Electrical_Conductivity (EC)","Organic_Carbon (%)","Soil_Moisture (%)","Zinc (%)","Iron (%)","Manganese (%)","Copper (%)","Boron (%)","Sulphur (%)","Rainfall_cm","temperature_celsius","humidity_percentage"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

dataset_ranges = {col: {"min": float(df[col].dropna().min()), "max": float(df[col].dropna().max())} for col in numeric_columns if not df[col].dropna().empty}

CROP_EMOJI = {"Rice":"🌾","Wheat":"🌿","Maize":"🌽","Cotton":"🌸","Mustard":"🌻","Pulses":"🫘","Vegetables":"🥦","Apple":"🍎","Walnut":"🌰","Sugarcane":"🎋","Potato":"🥔","Barley":"🌾"}
def crop_emoji(crop): return CROP_EMOJI.get(crop, "🌱")
def confidence_level(v):
    if v >= 70: return ("High confidence","🟢","#16a34a")
    if v >= 50: return ("Moderate confidence","🟡","#d97706")
    return ("Low confidence","🔴","#dc2626")

EXAMPLES = {
    "🌾 Rice":    {"state":"Andhra Pradesh","zone":"Southern Plateau and Hills Region","soil_type":"Alluvial","ph":6.9,"nitrogen":30.7,"phosphorus":204.9,"potassium":53.4,"ec":0.4,"organic":0.8,"moisture":30.0,"rainfall":232.5,"temperature":27.9,"humidity":65.0,"zinc":0.6,"iron":3.2,"manganese":1.1,"copper":0.3,"boron":0.4,"sulphur":12.0},
    "🌽 Maize":   {"state":"Telangana","zone":"Southern Plateau and Hills Region","soil_type":"Alluvial Soil","ph":6.9,"nitrogen":14.9,"phosphorus":219.0,"potassium":35.2,"ec":0.4,"organic":0.8,"moisture":30.0,"rainfall":172.9,"temperature":27.5,"humidity":64.7,"zinc":0.6,"iron":3.2,"manganese":1.1,"copper":0.3,"boron":0.4,"sulphur":12.0},
    "🌿 Wheat":   {"state":"Telangana","zone":"Southern Plateau and Hills Region","soil_type":"Alluvial Soil","ph":6.6,"nitrogen":17.2,"phosphorus":221.1,"potassium":36.3,"ec":0.4,"organic":0.7,"moisture":17.2,"rainfall":130.0,"temperature":27.5,"humidity":63.9,"zinc":0.6,"iron":3.2,"manganese":1.1,"copper":0.3,"boron":0.4,"sulphur":12.0},
    "🌸 Cotton":  {"state":"Andhra Pradesh","zone":"Southern Plateau and Hills Region","soil_type":"Black Cotton Soil (Vertisols)","ph":7.8,"nitrogen":55.4,"phosphorus":178.6,"potassium":77.9,"ec":0.5,"organic":0.6,"moisture":16.8,"rainfall":59.4,"temperature":27.7,"humidity":64.2,"zinc":0.6,"iron":3.2,"manganese":1.1,"copper":0.3,"boron":0.4,"sulphur":12.0},
    "🌻 Mustard": {"state":"Jharkhand","zone":"Eastern Plateau and Hills Region","soil_type":"Red Sandy Soil","ph":6.2,"nitrogen":0.3,"phosphorus":230.6,"potassium":17.9,"ec":0.4,"organic":0.5,"moisture":17.3,"rainfall":1242.3,"temperature":25.4,"humidity":71.8,"zinc":0.6,"iron":3.2,"manganese":1.1,"copper":0.3,"boron":0.4,"sulphur":12.0},
    "🫘 Pulses":  {"state":"Andhra Pradesh","zone":"East Coast Plains and Hills Region","soil_type":"Red sandy loam","ph":7.1,"nitrogen":175.3,"phosphorus":77.9,"potassium":210.4,"ec":0.4,"organic":0.7,"moisture":16.8,"rainfall":455.3,"temperature":27.3,"humidity":66.7,"zinc":0.6,"iron":3.2,"manganese":1.1,"copper":0.3,"boron":0.4,"sulphur":12.0},
}

if "preset" not in st.session_state:
    st.session_state.preset = {}

st.markdown("#### ⚡ Quick Load Examples")
cols = st.columns(len(EXAMPLES))
for i, (label, values) in enumerate(EXAMPLES.items()):
    with cols[i]:
        if st.button(label, use_container_width=True):
            st.session_state.preset = values; st.rerun()

preset = st.session_state.preset
def gv(k, d): return preset.get(k, d)
def gi(k, opts): v = preset.get(k); return opts.index(v) if v in opts else 0

st.markdown("---")
mode = st.radio("Input mode", ["👨‍🌾 Quick Recommendation","🔬 Advanced Soil Analysis"], horizontal=True)

left_col, right_col = st.columns(2, gap="large")
with left_col:
    st.markdown('<div class="input-section"><h3>📍 Farm Location</h3>', unsafe_allow_html=True)
    state     = st.selectbox("State",              states,     index=gi("state",     states))
    zone      = st.selectbox("Agro-Climatic Zone", zones,      index=gi("zone",      zones))
    soil_type = st.selectbox("Soil Type",          soil_types, index=gi("soil_type", soil_types))
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-section"><h3>🌍 Weather Conditions</h3>', unsafe_allow_html=True)
    rainfall    = st.number_input("Rainfall (cm)",    min_value=0.0, value=float(gv("rainfall",    120.0)), step=1.0)
    temperature = st.number_input("Temperature (°C)", value=float(gv("temperature", 28.0)),                 step=0.1)
    humidity    = st.number_input("Humidity (%)",     min_value=0.0, max_value=100.0, value=float(gv("humidity", 75.0)), step=1.0)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="input-section"><h3>🧪 Primary Soil Properties</h3>', unsafe_allow_html=True)
    ph         = st.number_input("Soil pH",             min_value=0.0, max_value=14.0, value=float(gv("ph",         6.8)), step=0.1)
    nitrogen   = st.number_input("Nitrogen (N)",        min_value=0.0, value=float(gv("nitrogen",   120.0)), step=1.0)
    phosphorus = st.number_input("Phosphorus (P)",      min_value=0.0, value=float(gv("phosphorus",  40.0)), step=1.0)
    potassium  = st.number_input("Potassium (K)",       min_value=0.0, value=float(gv("potassium",  180.0)), step=1.0)
    ec         = st.number_input("Electrical Conductivity", min_value=0.0, value=float(gv("ec",      0.4)), step=0.1)
    organic    = st.number_input("Organic Carbon (%)",  min_value=0.0, value=float(gv("organic",     0.8)), step=0.1)
    moisture   = st.number_input("Soil Moisture (%)",   min_value=0.0, max_value=100.0, value=float(gv("moisture", 30.0)), step=1.0)
    st.markdown('</div>', unsafe_allow_html=True)

if mode == "🔬 Advanced Soil Analysis":
    st.markdown('<div class="input-section"><h3>🔬 Micronutrients</h3>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1: zinc = st.number_input("Zinc (%)", min_value=0.0, value=float(gv("zinc",0.6)), step=0.1); iron = st.number_input("Iron (%)", min_value=0.0, value=float(gv("iron",3.2)), step=0.1)
    with a2: manganese = st.number_input("Manganese (%)", min_value=0.0, value=float(gv("manganese",1.1)), step=0.1); copper = st.number_input("Copper (%)", min_value=0.0, value=float(gv("copper",0.3)), step=0.1)
    with a3: boron = st.number_input("Boron (%)", min_value=0.0, value=float(gv("boron",0.4)), step=0.1); sulphur = st.number_input("Sulphur (%)", min_value=0.0, value=float(gv("sulphur",12.0)), step=1.0)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    zinc=float(gv("zinc",0.6)); iron=float(gv("iron",3.2)); manganese=float(gv("manganese",1.1))
    copper=float(gv("copper",0.3)); boron=float(gv("boron",0.4)); sulphur=float(gv("sulphur",12.0))

st.markdown("---")
predict = st.button("🌾 Get Crop Recommendation", use_container_width=True, type="primary")

if predict:
    sample = pd.DataFrame([{"Soil_Type":soil_type,"pH_Value":float(ph),"Nitrogen_Value (N)":float(nitrogen),"Phosphorus_Value (P)":float(phosphorus),"Potassium_Value (K)":float(potassium),"Electrical_Conductivity (EC)":float(ec),"Organic_Carbon (%)":float(organic),"Soil_Moisture (%)":float(moisture),"Zinc (%)":float(zinc),"Iron (%)":float(iron),"Manganese (%)":float(manganese),"Copper (%)":float(copper),"Boron (%)":float(boron),"Sulphur (%)":float(sulphur),"Rainfall_cm":float(rainfall),"temperature_celsius":float(temperature),"humidity_percentage":float(humidity),"State_Name":state,"Agro_Climatic Zone":zone}])

    warnings_list = []
    for col in numeric_columns:
        if col not in sample.columns or col not in dataset_ranges: continue
        val = float(sample[col].iloc[0]); mn = dataset_ranges[col]["min"]; mx = dataset_ranges[col]["max"]
        if val < mn or val > mx: warnings_list.append(f"**{col}**: {val:.2f} (range: {mn:.2f}–{mx:.2f})")
    if warnings_list:
        with st.expander(f"⚠️ {len(warnings_list)} inputs outside training range"):
            for w in warnings_list: st.markdown(f"- {w}")

    try:
        prediction    = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]
        top_indices   = probabilities.argsort()[-3:][::-1]
        top3          = [(model.classes_[i], float(probabilities[i]*100)) for i in top_indices]
        top_crop, top_conf = top3[0]
        conf_text, conf_icon, conf_color = confidence_level(top_conf)

        state_count   = len(df[(df["State_Name"].astype(str)==state)&(df["Crop"].astype(str)==str(top_crop))])
        overall_count = len(df[df["Crop"].astype(str)==str(top_crop)])

        st.markdown(f"""
        <div class="result-hero">
            <div class="crop-icon">{crop_emoji(top_crop)}</div>
            <div class="crop-name">{top_crop}</div>
            <div style="opacity:0.85;margin-bottom:0.5rem">Recommended crop for your conditions</div>
            <div class="confidence-badge">{conf_icon} {top_conf:.1f}% — {conf_text}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 🏆 Top 3 Recommendations")
        medals = ["🥇","🥈","🥉"]
        cols3 = st.columns(3)
        for i, (crop, conf) in enumerate(top3):
            with cols3[i]:
                st.markdown(f"""<div class="medal-card"><div style="font-size:2rem">{medals[i]}</div><div style="font-size:1.8rem">{crop_emoji(crop)}</div><div class="medal-crop">{crop}</div><div class="medal-score">{conf:.1f}%</div></div>""", unsafe_allow_html=True)
                st.progress(min(conf/100, 1.0))

        st.markdown("#### 📍 Regional Evidence")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Model Confidence", f"{top_conf:.2f}%")
        e2.metric("State Records",    state_count)
        e3.metric("Total Records",    overall_count)
        regional = "Good 🟢" if state_count >= 30 else "Limited 🟡" if state_count >= 10 else "Low 🔴"
        e4.metric("Regional Support", regional)

        if top_conf >= 70 and state_count >= 30:   st.success("✅ Strong confidence with good regional training data.")
        elif top_conf >= 50 and state_count >= 10: st.warning("🟡 Moderate confidence — review alternatives before deciding.")
        else: st.error(f"🔴 Limited evidence. Only {state_count} records for {top_crop} in {state}.")

        st.markdown("#### 📋 Evidence Table")
        ev_rows = []
        for crop, conf in top3:
            sc = len(df[(df["State_Name"].astype(str)==state)&(df["Crop"].astype(str)==str(crop))])
            oc = len(df[df["Crop"].astype(str)==str(crop)])
            ev_rows.append({"Crop":crop,"Model Score (%)":round(conf,2),"State Records":sc,"Overall Records":oc,"Regional Support":"Good" if sc>=30 else "Limited" if sc>=10 else "Very limited"})
        st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)

        with st.expander("📋 Input Summary"):
            summary = pd.DataFrame({"Parameter":["State","Zone","Soil Type","pH","Nitrogen","Phosphorus","Potassium","EC","Organic Carbon","Soil Moisture","Rainfall","Temperature","Humidity","Zinc","Iron","Manganese","Copper","Boron","Sulphur"],"Value":[state,zone,soil_type,f"{ph:.2f}",f"{nitrogen:.2f}",f"{phosphorus:.2f}",f"{potassium:.2f}",f"{ec:.2f}",f"{organic:.2f}",f"{moisture:.2f}",f"{rainfall:.2f}",f"{temperature:.2f}",f"{humidity:.2f}",f"{zinc:.2f}",f"{iron:.2f}",f"{manganese:.2f}",f"{copper:.2f}",f"{boron:.2f}",f"{sulphur:.2f}"]})
            st.dataframe(summary, use_container_width=True, hide_index=True)

        st.info("🌾 ML recommendation only. Not a guarantee of yield or profit. Verify with local conditions and agricultural advice.")
    except Exception as e:
        st.error("Recommendation could not be generated."); st.exception(e)

st.divider()
with st.expander("ℹ️ About the AI Model"):
    c1,c2,c3 = st.columns(3)
    c1.metric("Test Accuracy","93.53%"); c2.metric("Top-3 Accuracy","98.14%"); c3.metric("Test Records","1,129")