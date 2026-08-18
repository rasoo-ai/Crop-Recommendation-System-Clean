import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Crop Recommendation - Smart Kisan",
    page_icon="🌱",
    layout="wide",
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
        max-width: 1400px;
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
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "output/crop_prediction_model_balanced.pkl"
    )


@st.cache_data
def load_dataset():
    return pd.read_excel(
        "output/Crop_Normalized.xlsx"
    )


try:
    model = load_model()
    df    = load_dataset()
except Exception as error:
    st.error("Unable to load the production model or dataset.")
    st.exception(error)
    st.stop()


# ==========================================================
# DROPDOWN VALUES
# ==========================================================

states     = sorted(df["State_Name"].dropna().astype(str).unique())
soil_types = sorted(df["Soil_Type"].dropna().astype(str).unique())
zones      = sorted(df["Agro_Climatic Zone"].dropna().astype(str).unique())


# ==========================================================
# EXAMPLE PRESETS
# ==========================================================

EXAMPLES = {
    "🌾 Rice": {
        "state":       "Andhra Pradesh",
        "zone":        "Southern Plateau and Hills Region",
        "soil_type":   "Alluvial",
        "ph":          6.9,
        "nitrogen":    30.7,
        "phosphorus":  204.9,
        "potassium":   53.4,
        "ec":          0.4,
        "organic":     0.8,
        "moisture":    30.0,
        "rainfall":    232.5,
        "temperature": 27.9,
        "humidity":    65.0,
        "zinc":        0.6,
        "iron":        3.2,
        "manganese":   1.1,
        "copper":      0.3,
        "boron":       0.4,
        "sulphur":     12.0,
    },
    "🌽 Maize": {
        "state":       "Telangana",
        "zone":        "Southern Plateau and Hills Region",
        "soil_type":   "Alluvial Soil",
        "ph":          6.9,
        "nitrogen":    14.9,
        "phosphorus":  219.0,
        "potassium":   35.2,
        "ec":          0.4,
        "organic":     0.8,
        "moisture":    30.0,
        "rainfall":    172.9,
        "temperature": 27.5,
        "humidity":    64.7,
        "zinc":        0.6,
        "iron":        3.2,
        "manganese":   1.1,
        "copper":      0.3,
        "boron":       0.4,
        "sulphur":     12.0,
    },
    "🌿 Wheat": {
        "state":       "Telangana",
        "zone":        "Southern Plateau and Hills Region",
        "soil_type":   "Alluvial Soil",
        "ph":          6.6,
        "nitrogen":    17.2,
        "phosphorus":  221.1,
        "potassium":   36.3,
        "ec":          0.4,
        "organic":     0.7,
        "moisture":    17.2,
        "rainfall":    130.0,
        "temperature": 27.5,
        "humidity":    63.9,
        "zinc":        0.6,
        "iron":        3.2,
        "manganese":   1.1,
        "copper":      0.3,
        "boron":       0.4,
        "sulphur":     12.0,
    },
    "🌸 Cotton": {
        "state":       "Andhra Pradesh",
        "zone":        "Southern Plateau and Hills Region",
        "soil_type":   "Black Cotton Soil (Vertisols)",
        "ph":          7.8,
        "nitrogen":    55.4,
        "phosphorus":  178.6,
        "potassium":   77.9,
        "ec":          0.5,
        "organic":     0.6,
        "moisture":    16.8,
        "rainfall":    59.4,
        "temperature": 27.7,
        "humidity":    64.2,
        "zinc":        0.6,
        "iron":        3.2,
        "manganese":   1.1,
        "copper":      0.3,
        "boron":       0.4,
        "sulphur":     12.0,
    },
    "🌻 Mustard": {
        "state":       "Jharkhand",
        "zone":        "Eastern Plateau and Hills Region",
        "soil_type":   "Red Sandy Soil",
        "ph":          6.2,
        "nitrogen":    0.3,
        "phosphorus":  230.6,
        "potassium":   17.9,
        "ec":          0.4,
        "organic":     0.5,
        "moisture":    17.3,
        "rainfall":    1242.3,
        "temperature": 25.4,
        "humidity":    71.8,
        "zinc":        0.6,
        "iron":        3.2,
        "manganese":   1.1,
        "copper":      0.3,
        "boron":       0.4,
        "sulphur":     12.0,
    },
    "🫘 Pulses": {
        "state":       "Andhra Pradesh",
        "zone":        "East Coast Plains and Hills Region",
        "soil_type":   "Red sandy loam",
        "ph":          7.1,
        "nitrogen":    175.3,
        "phosphorus":  77.9,
        "potassium":   210.4,
        "ec":          0.4,
        "organic":     0.7,
        "moisture":    16.8,
        "rainfall":    455.3,
        "temperature": 27.3,
        "humidity":    66.7,
        "zinc":        0.6,
        "iron":        3.2,
        "manganese":   1.1,
        "copper":      0.3,
        "boron":       0.4,
        "sulphur":     12.0,
    },
}


# ==========================================================
# CROP EMOJI
# ==========================================================

CROP_EMOJI = {
    "Rice": "🌾", "Wheat": "🌿", "Maize": "🌽",
    "Cotton": "🌸", "Mustard": "🌻", "Pulses": "🫘",
    "Vegetables": "🥦", "Apple": "🍎", "Walnut": "🌰",
}

def crop_emoji(crop):
    return CROP_EMOJI.get(crop, "🌱")


# ==========================================================
# CONFIDENCE
# ==========================================================

def confidence_level(value):
    if value >= 70:
        return "High confidence", "🟢"
    if value >= 50:
        return "Moderate confidence", "🟡"
    return "Low confidence", "🔴"


# ==========================================================
# HEADER
# ==========================================================

st.title("🌱 Crop Recommendation")
st.write(
    "Enter your farm conditions and Smart Kisan will "
    "provide the model's top crop recommendations."
)
st.info(
    "For better results, use values from a recent soil test "
    "or reliable farm measurements."
)


# ==========================================================
# ONE-CLICK EXAMPLES
# ==========================================================

st.subheader("⚡ Quick Load Examples")
st.caption(
    "Click any example to instantly fill all fields "
    "with real agricultural values."
)

# Session state for preset values
if "preset" not in st.session_state:
    st.session_state.preset = None

ex_cols = st.columns(len(EXAMPLES))
for i, (label, values) in enumerate(EXAMPLES.items()):
    with ex_cols[i]:
        if st.button(label, use_container_width=True):
            st.session_state.preset = values
            st.rerun()

# Load preset into p dict or use defaults
p = st.session_state.preset or {}

def pval(key, default):
    """Return preset value if available, else default."""
    return p.get(key, default)

def pstate(key, options, default_idx=0):
    """Return index of preset value in options list."""
    val = p.get(key)
    if val and val in options:
        return options.index(val)
    return default_idx


# ==========================================================
# INPUT MODE
# ==========================================================

st.divider()

mode = st.radio(
    "Choose input mode",
    ["👨‍🌾 Quick Recommendation", "🔬 Advanced Soil Analysis"],
    horizontal=True,
)


# ==========================================================
# LOCATION
# ==========================================================

st.header("📍 Farm Location")

loc1, loc2, loc3 = st.columns(3)

with loc1:
    state = st.selectbox(
        "State",
        states,
        index=pstate("state", states),
    )

with loc2:
    zone = st.selectbox(
        "Agro-Climatic Zone",
        zones,
        index=pstate("zone", zones),
    )

with loc3:
    soil_type = st.selectbox(
        "Soil Type",
        soil_types,
        index=pstate("soil_type", soil_types),
    )


# ==========================================================
# PRIMARY SOIL + WEATHER
# ==========================================================

st.header("🌍 Soil & Weather")

soil1, soil2 = st.columns(2)

with soil1:
    ph          = st.number_input("Soil pH",                  min_value=0.0, max_value=14.0, value=pval("ph", 6.8),    step=0.1)
    nitrogen    = st.number_input("Nitrogen (N)",              min_value=0.0, value=pval("nitrogen", 120.0),            step=1.0)
    phosphorus  = st.number_input("Phosphorus (P)",            min_value=0.0, value=pval("phosphorus", 40.0),           step=1.0)
    potassium   = st.number_input("Potassium (K)",             min_value=0.0, value=pval("potassium", 180.0),           step=1.0)
    ec          = st.number_input("Electrical Conductivity",   min_value=0.0, value=pval("ec", 0.4),                   step=0.1)

with soil2:
    organic     = st.number_input("Organic Carbon (%)",        min_value=0.0, value=pval("organic", 0.8),              step=0.1)
    moisture    = st.number_input("Soil Moisture (%)",         min_value=0.0, max_value=100.0, value=pval("moisture", 30.0),  step=1.0)
    rainfall    = st.number_input("Rainfall (cm)",             min_value=0.0, value=pval("rainfall", 120.0),           step=1.0)
    temperature = st.number_input("Temperature (°C)",          value=pval("temperature", 28.0),                        step=0.1)
    humidity    = st.number_input("Humidity (%)",              min_value=0.0, max_value=100.0, value=pval("humidity", 75.0),  step=1.0)


# ==========================================================
# ADVANCED SOIL DATA
# ==========================================================

if mode == "🔬 Advanced Soil Analysis":
    st.header("🧪 Advanced Soil Nutrients")
    st.caption("Use this section when laboratory soil-test values are available.")

    adv1, adv2 = st.columns(2)
    with adv1:
        zinc      = st.number_input("Zinc (%)",      min_value=0.0, value=pval("zinc", 0.6),      step=0.1)
        iron      = st.number_input("Iron (%)",      min_value=0.0, value=pval("iron", 3.2),      step=0.1)
        manganese = st.number_input("Manganese (%)", min_value=0.0, value=pval("manganese", 1.1), step=0.1)
    with adv2:
        copper    = st.number_input("Copper (%)",    min_value=0.0, value=pval("copper", 0.3),    step=0.1)
        boron     = st.number_input("Boron (%)",     min_value=0.0, value=pval("boron", 0.4),     step=0.1)
        sulphur   = st.number_input("Sulphur (%)",   min_value=0.0, value=pval("sulphur", 12.0),  step=1.0)
else:
    zinc = pval("zinc", 0.6)
    iron = pval("iron", 3.2)
    manganese = pval("manganese", 1.1)
    copper = pval("copper", 0.3)
    boron = pval("boron", 0.4)
    sulphur = pval("sulphur", 12.0)


# ==========================================================
# PREDICT BUTTON
# ==========================================================

st.divider()

predict = st.button(
    "🌾 Get Crop Recommendation",
    use_container_width=True,
    type="primary",
)


if predict:

    try:

        sample = pd.DataFrame([{
            "Soil_Type":                     soil_type,
            "pH_Value":                      float(ph),
            "Nitrogen_Value (N)":            float(nitrogen),
            "Phosphorus_Value (P)":          float(phosphorus),
            "Potassium_Value (K)":           float(potassium),
            "Electrical_Conductivity (EC)":  float(ec),
            "Organic_Carbon (%)":            float(organic),
            "Soil_Moisture (%)":             float(moisture),
            "Zinc (%)":                      float(zinc),
            "Iron (%)":                      float(iron),
            "Manganese (%)":                 float(manganese),
            "Copper (%)":                    float(copper),
            "Boron (%)":                     float(boron),
            "Sulphur (%)":                   float(sulphur),
            "Rainfall_cm":                   float(rainfall),
            "temperature_celsius":           float(temperature),
            "humidity_percentage":           float(humidity),
            "State_Name":                    state,
            "Agro_Climatic Zone":            zone,
        }])

        prediction    = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]
        top_indices   = probabilities.argsort()[-3:][::-1]
        top3 = [(model.classes_[i], float(probabilities[i] * 100)) for i in top_indices]

        top_crop, top_confidence = top3[0]
        confidence_text, confidence_icon = confidence_level(top_confidence)

        # --------------------------------------------------
        # RESULT CARD
        # --------------------------------------------------

        st.divider()
        st.header("🌾 Recommendation Result")

        if top_confidence >= 70:
            st.success(f"{confidence_icon} {confidence_text}")
        elif top_confidence >= 50:
            st.warning(f"{confidence_icon} {confidence_text}")
        else:
            st.error(f"{confidence_icon} {confidence_text}")

        st.subheader(f"{crop_emoji(top_crop)} Recommended Crop: {top_crop}")
        st.metric("Model Confidence", f"{top_confidence:.2f}%")

        if top_confidence >= 70:
            st.success("The model shows relatively high confidence for these supplied conditions.")
        elif top_confidence >= 50:
            st.warning("The model shows moderate confidence. Please also consider the alternative crops.")
        else:
            st.error(
                "The model is uncertain for these conditions. "
                "Treat this as guidance and verify with local agricultural advice before planting."
            )

        # --------------------------------------------------
        # TOP 3 CARDS
        # --------------------------------------------------

        st.header("🏆 Top 3 Recommendations")

        medals = ["🥇", "🥈", "🥉"]
        rank1, rank2, rank3 = st.columns(3)
        ranking_columns = [rank1, rank2, rank3]

        for position, (crop, confidence) in enumerate(top3):
            with ranking_columns[position]:
                with st.container(border=True):
                    st.subheader(f"{medals[position]} {crop_emoji(crop)} {crop}")
                    st.metric("Model score", f"{confidence:.2f}%")
                    st.progress(min(confidence / 100, 1.0))

        # --------------------------------------------------
        # INPUT SUMMARY
        # --------------------------------------------------

        st.header("📋 Input Summary")

        summary = pd.DataFrame({
            "Parameter": [
                "State", "Agro-Climatic Zone", "Soil Type",
                "pH", "Nitrogen", "Phosphorus", "Potassium",
                "Electrical Conductivity", "Organic Carbon",
                "Soil Moisture", "Rainfall", "Temperature", "Humidity",
                "Zinc", "Iron", "Manganese", "Copper", "Boron", "Sulphur",
            ],
            "Value": [
                str(state), str(zone), str(soil_type),
                f"{ph:.2f}", f"{nitrogen:.2f}", f"{phosphorus:.2f}",
                f"{potassium:.2f}", f"{ec:.2f}", f"{organic:.2f}",
                f"{moisture:.2f}", f"{rainfall:.2f}", f"{temperature:.2f}",
                f"{humidity:.2f}", f"{zinc:.2f}", f"{iron:.2f}",
                f"{manganese:.2f}", f"{copper:.2f}", f"{boron:.2f}",
                f"{sulphur:.2f}",
            ],
        })

        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.info(
            "This recommendation is generated by the current machine-learning model. "
            "It is not a guarantee of yield or profitability. Consider local conditions, "
            "irrigation, season, weather and qualified agricultural advice before planting."
        )

    except Exception as error:
        st.error("The crop recommendation could not be generated.")
        st.exception(error)


# ==========================================================
# MODEL INFO
# ==========================================================

st.divider()

with st.expander("ℹ️ About the AI model"):
    st.write(
        "Smart Kisan Model V1 uses a Random Forest classifier "
        "trained on the project's agricultural dataset."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Test Accuracy",   "93.53%")
    m2.metric("Top-3 Accuracy",  "98.14%")
    m3.metric("Test Records",    "1,129")
    st.caption(
        "These are held-out evaluation results and are not "
        "a guarantee of prediction accuracy for every farm."
    )
