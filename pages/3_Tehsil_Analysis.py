from pathlib import Path
import warnings
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Tehsil Analysis - Smart Kisan", page_icon="📍", layout="wide")

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
.stat-strip { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.stat-mini { background: white; border-radius: 12px; padding: 1rem 1.2rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-top: 3px solid #2d8a45; }
.stat-mini .num { font-size: 1.8rem; font-weight: 700; color: #1a5c2a; }
.stat-mini .lbl { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
.rec-result { background: linear-gradient(135deg, #1a5c2a, #2d8a45); border-radius: 14px; padding: 1.5rem; color: white; text-align: center; margin: 1rem 0; }
.rec-result .rec-crop { font-size: 1.8rem; font-weight: 700; }
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

st.markdown("""<div class="page-header"><h1>📍 Tehsil Analysis</h1><p>Explore ML crop recommendations for any tehsil — with interactive maps and regional data.</p></div>""", unsafe_allow_html=True)

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "output" / "Crop_Normalized.xlsx"

@st.cache_data
def load_data():
    if not DATA_FILE.exists(): raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    data = pd.read_excel(DATA_FILE)
    required = ["State_Name","District_Name","Tehsil_Name","Crop","State_Latitude","State_Longitude"]
    missing = [c for c in required if c not in data.columns]
    if missing: raise ValueError("Missing columns: " + ", ".join(missing))
    return data

try: df = load_data()
except Exception as e: st.error("Unable to load dataset."); st.exception(e); st.stop()

@st.cache_resource
def load_model():
    try: import joblib
    except: return None
    search_dirs = [BASE_DIR/"models", BASE_DIR/"model", BASE_DIR/"output", BASE_DIR]
    candidates = []
    for d in search_dirs:
        if not d.exists(): continue
        for ext in ["*.joblib","*.pkl","*.pickle","*.sav"]: candidates.extend(d.glob(ext))
    candidates = sorted(set(candidates), key=lambda p: (0 if any(x in p.name.lower() for x in ["crop","random","forest","model"]) else 1, len(str(p))))
    for f in candidates:
        try:
            m = joblib.load(f)
            if hasattr(m, "predict"): return m
        except: continue
    return None

model = load_model()

CROP_COLORS = {"Rice":"green","Wheat":"gold","Maize":"orange","Cotton":"lightblue","Mustard":"beige","Pulses":"darkgreen","Vegetables":"lime","Apple":"red","Walnut":"darkred","Sugarcane":"cadetblue","Potato":"gray","Barley":"lightgreen"}
CROP_EMOJI  = {"Rice":"🌾","Wheat":"🌿","Maize":"🌽","Cotton":"🌸","Mustard":"🌻","Pulses":"🫘","Vegetables":"🥦","Apple":"🍎","Walnut":"🌰","Sugarcane":"🎋","Potato":"🥔","Barley":"🌾"}
def safe_mode(s): v = s.dropna().astype(str); return v.mode().iloc[0] if not v.empty else "Unknown"
def crop_color(c): return CROP_COLORS.get(str(c), "blue")
def crop_icon(c):  return CROP_EMOJI.get(str(c), "🌱")

def make_prediction(tehsil_df):
    if model is None: return None, None, "Model not found"
    row = {"Soil_Type": safe_mode(tehsil_df["Soil_Type"])}
    for col in ["pH_Value","Nitrogen_Value (N)","Phosphorus_Value (P)","Potassium_Value (K)","Electrical_Conductivity (EC)","Organic_Carbon (%)","Soil_Moisture (%)","Zinc (%)","Iron (%)","Manganese (%)","Copper (%)","Boron (%)","Sulphur (%)","Rainfall_cm","temperature_celsius","humidity_percentage"]:
        vals = pd.to_numeric(tehsil_df.get(col, pd.Series()), errors="coerce")
        row[col] = float(vals.mean()) if not vals.dropna().empty else 0.0
    row["State_Name"] = str(tehsil_df["State_Name"].iloc[0])
    row["Agro_Climatic Zone"] = safe_mode(tehsil_df["Agro_Climatic Zone"]) if "Agro_Climatic Zone" in tehsil_df.columns else "Unknown"
    try:
        X = pd.DataFrame([row]); pred = model.predict(X)[0]
        conf = float(model.predict_proba(X)[0].max()*100) if hasattr(model,"predict_proba") else None
        return str(pred), conf, None
    except Exception as e: return None, None, str(e)

total_states  = df["State_Name"].nunique()
total_tehsils = df["Tehsil_Name"].nunique()
total_crops   = df["Crop"].nunique()
total_records = len(df)
st.markdown(f"""<div class="stat-strip"><div class="stat-mini"><div class="num">{total_states}</div><div class="lbl">States</div></div><div class="stat-mini"><div class="num">{total_tehsils:,}</div><div class="lbl">Tehsils</div></div><div class="stat-mini"><div class="num">{total_crops}</div><div class="lbl">Crop Types</div></div><div class="stat-mini"><div class="num">{total_records:,}</div><div class="lbl">Records</div></div></div>""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Tehsil Lookup","🗺️ State Map"])

with tab1:
    st.markdown("### 🔍 Find a Tehsil")
    states_list = sorted(df["State_Name"].dropna().astype(str).unique())
    sel_state   = st.selectbox("State", states_list)
    state_df    = df[df["State_Name"].astype(str)==sel_state].copy()
    districts   = sorted(state_df["District_Name"].dropna().astype(str).unique())
    sel_dist    = st.selectbox("District", districts)
    dist_df     = state_df[state_df["District_Name"].astype(str)==sel_dist].copy()
    tehsils     = sorted(dist_df["Tehsil_Name"].dropna().astype(str).unique())
    sel_tehsil  = st.selectbox("Tehsil", tehsils)
    tehsil_df   = dist_df[dist_df["Tehsil_Name"].astype(str)==sel_tehsil].copy()

    if not tehsil_df.empty:
        crop_counts = tehsil_df["Crop"].dropna().astype(str).value_counts()
        info_col, chart_col = st.columns(2)
        with info_col:
            st.markdown("#### 📊 Recorded Crops")
            if not crop_counts.empty:
                st.success(f"{crop_icon(crop_counts.index[0])} Most recorded: **{crop_counts.index[0]}** ({crop_counts.iloc[0]} records)")
            st.dataframe(crop_counts.rename("Records").reset_index().rename(columns={"index":"Crop"}), use_container_width=True, hide_index=True)
        with chart_col:
            st.markdown("#### 📈 Distribution")
            st.bar_chart(crop_counts)

        num_cols = ["pH_Value","Nitrogen_Value (N)","Rainfall_cm","temperature_celsius","humidity_percentage","Soil_Moisture (%)"]
        available = [c for c in num_cols if c in tehsil_df.columns]
        for col in available: tehsil_df[col] = pd.to_numeric(tehsil_df[col], errors="coerce")
        st.markdown("#### 🧪 Soil & Climate Profile")
        st.dataframe(tehsil_df[available].describe().loc[["mean","min","max"]].T.round(2).rename(columns={"mean":"Mean","min":"Min","max":"Max"}), use_container_width=True)

        if st.button("🤖 Generate ML Recommendation", use_container_width=True, type="primary"):
            with st.spinner("Running model..."):
                pred, conf, err = make_prediction(tehsil_df)
            if pred:
                conf_str = f"{conf:.1f}%" if conf else "N/A"
                st.markdown(f"""<div class="rec-result"><div style="font-size:3rem">{crop_icon(pred)}</div><div class="rec-crop">{pred}</div><div style="opacity:0.85">Confidence: {conf_str}</div></div>""", unsafe_allow_html=True)
                if conf and conf >= 70: st.success("✅ High confidence")
                elif conf and conf >= 50: st.warning("🟡 Moderate confidence")
                else: st.error("🔴 Low confidence")
            else:
                st.warning("Prediction could not be generated.")
                if err: st.code(err)

with tab2:
    st.markdown("### 🗺️ State Crop Recommendation Map")
    map_states = sorted(df["State_Name"].dropna().astype(str).unique())
    map_state  = st.selectbox("Select State", map_states, key="state_map_selector")
    if st.button("🗺️ Generate Map", key="gen_map", use_container_width=True, type="primary"):
        with st.spinner(f"Computing recommendations for {map_state}..."):
            map_df = df[df["State_Name"].astype(str)==map_state].copy()
            map_df["State_Latitude"]  = pd.to_numeric(map_df["State_Latitude"],  errors="coerce")
            map_df["State_Longitude"] = pd.to_numeric(map_df["State_Longitude"], errors="coerce")
            map_df = map_df.dropna(subset=["State_Latitude","State_Longitude"])
            if map_df.empty: st.error("No coordinates found."); st.stop()

            center_lat = float(map_df["State_Latitude"].mean())
            center_lon = float(map_df["State_Longitude"].mean())
            results, pred_errors = [], []

            for tehsil, tdf in map_df.groupby("Tehsil_Name", dropna=True):
                tehsil = str(tehsil)
                lat = pd.to_numeric(tdf["State_Latitude"], errors="coerce").mean()
                lon = pd.to_numeric(tdf["State_Longitude"], errors="coerce").mean()
                if pd.isna(lat) or pd.isna(lon): continue
                pred, conf, err = make_prediction(tdf)
                if pred is None:
                    cc = tdf["Crop"].dropna().astype(str).value_counts()
                    pred = cc.index[0] if not cc.empty else "Unknown"; conf = None
                    if err: pred_errors.append({"Tehsil":tehsil,"Error":err})
                results.append({"Tehsil":tehsil,"Crop":pred,"Confidence":conf,"Latitude":float(lat),"Longitude":float(lon),"Records":len(tdf)})

            if not results: st.error("No results generated."); st.stop()
            results_df = pd.DataFrame(results)

        conf_vals = pd.to_numeric(results_df["Confidence"], errors="coerce").dropna()
        avg_conf  = f"{conf_vals.mean():.1f}%" if not conf_vals.empty else "N/A"
        st.success(f"✅ Generated recommendations for **{len(results_df)} tehsils** in {map_state}")
        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Tehsils",len(results_df)); s2.metric("Unique Crops",results_df["Crop"].nunique())
        s3.metric("Avg Confidence",avg_conf); s4.metric("Records",len(map_df))

        crop_map = folium.Map(location=[center_lat,center_lon], zoom_start=6, tiles="CartoDB positron", control_scale=True)
        from folium.plugins import MarkerCluster
        cluster = MarkerCluster(name="Tehsil Recommendations").add_to(crop_map)
        for _, row in results_df.iterrows():
            crop = str(row["Crop"]); conf_str = f"{float(row['Confidence']):.1f}%" if pd.notna(row["Confidence"]) else "N/A"
            popup_html = f"""<div style="width:230px;font-family:Inter,sans-serif"><h4 style="color:#1a5c2a">{row['Tehsil']}</h4><hr><b>Crop:</b> {crop_icon(crop)} {crop}<br><b>Confidence:</b> {conf_str}<br><b>Records:</b> {int(row['Records'])}</div>"""
            folium.CircleMarker(location=[float(row["Latitude"]),float(row["Longitude"])], radius=9, color=crop_color(crop), fill=True, fill_color=crop_color(crop), fill_opacity=0.85, weight=2, tooltip=f"{row['Tehsil']} — {crop}", popup=folium.Popup(popup_html, max_width=280)).add_to(cluster)
        bounds = [[float(results_df["Latitude"].min()),float(results_df["Longitude"].min())],[float(results_df["Latitude"].max()),float(results_df["Longitude"].max())]]
        if results_df["Latitude"].nunique()>1 or results_df["Longitude"].nunique()>1: crop_map.fit_bounds(bounds)
        folium.LayerControl().add_to(crop_map)
        st_folium(crop_map, width="100%", height=650, returned_objects=[], key=f"map_{map_state}")

        st.markdown("#### 📊 Crop Distribution")
        st.bar_chart(results_df["Crop"].value_counts())
        display_df = results_df[["Tehsil","Crop","Confidence","Records"]].copy()
        display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{float(x):.1f}%" if pd.notna(x) else "N/A")
        st.dataframe(display_df.sort_values("Crop"), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", display_df.to_csv(index=False), file_name=f"{map_state}_recommendations.csv", mime="text/csv", use_container_width=True)

st.divider()
st.caption("📍 Smart Kisan Tehsil Analysis | Random Forest | 93.53% accuracy")