import joblib
import pandas as pd
import streamlit as st

from smart_kisan_ui import configure_page, footer, hero, section, sidebar


configure_page("Crop Recommendation", "🌱")
sidebar()


@st.cache_resource
def load_model():
    return joblib.load("output/crop_prediction_model_balanced.pkl")


@st.cache_data
def load_dataset():
    return pd.read_excel("output/Crop_Normalized.xlsx")


try:
    model, df = load_model(), load_dataset()
except Exception:
    st.error("Smart Kisan could not load its recommendation resources. Please check that the model and dataset files are available, then try again.")
    st.stop()


NUMERIC_COLUMNS = [
    "pH_Value", "Nitrogen_Value (N)", "Phosphorus_Value (P)", "Potassium_Value (K)",
    "Electrical_Conductivity (EC)", "Organic_Carbon (%)", "Soil_Moisture (%)", "Zinc (%)",
    "Iron (%)", "Manganese (%)", "Copper (%)", "Boron (%)", "Sulphur (%)", "Rainfall_cm",
    "temperature_celsius", "humidity_percentage",
]
for column in NUMERIC_COLUMNS:
    df[column] = pd.to_numeric(df[column], errors="coerce")

FEATURES = [
    "Soil_Type", "pH_Value", "Nitrogen_Value (N)", "Phosphorus_Value (P)",
    "Potassium_Value (K)", "Electrical_Conductivity (EC)", "Organic_Carbon (%)",
    "Soil_Moisture (%)", "Zinc (%)", "Iron (%)", "Manganese (%)", "Copper (%)",
    "Boron (%)", "Sulphur (%)", "Rainfall_cm", "temperature_celsius", "humidity_percentage",
    "State_Name", "Agro_Climatic Zone",
]
RANGES = {column: (float(df[column].min()), float(df[column].max())) for column in NUMERIC_COLUMNS if df[column].notna().any()}
CROP_ICONS = {"Rice": "🌾", "Wheat": "🌾", "Maize": "🌽", "Cotton": "🌸", "Mustard": "🌻", "Pulses": "🫘", "Vegetables": "🥬", "Apple": "🍎", "Potato": "🥔", "Sugarcane": "🎋"}


def crop_icon(crop):
    return CROP_ICONS.get(str(crop), "🌱")


def confidence_label(score):
    if score >= 70:
        return "High Confidence", "high", "The model strongly favours this crop for the supplied conditions."
    if score >= 50:
        return "Medium Confidence", "medium", "Review the alternatives and local conditions before making a decision."
    return "Low Confidence", "low", "The model has limited confidence for these farm conditions. Consider reviewing the inputs and consulting local agricultural guidance."


def number(label, key, value, step, help_text, maximum=None):
    return st.number_input(label, min_value=0.0, max_value=maximum, value=float(st.session_state.get(key, value)), step=step, key=key, help=help_text)


hero("Smart Kisan decision support", "AI Crop Recommendation", "Get data-driven crop insights based on your farm conditions.")
st.caption("Use current soil-test results and representative weather measurements for the most useful decision-support insight.")

with st.expander("Quick-start examples", expanded=False):
    st.caption("Examples only. They populate the form; the prediction always uses the values currently displayed.")
    examples = {
        "Rice profile": {"ph": 6.9, "nitrogen": 30.7, "phosphorus": 204.9, "potassium": 53.4, "moisture": 30.0, "rainfall": 232.5, "temperature": 27.9, "humidity": 65.0},
        "Maize profile": {"ph": 6.9, "nitrogen": 14.9, "phosphorus": 219.0, "potassium": 35.2, "moisture": 30.0, "rainfall": 172.9, "temperature": 27.5, "humidity": 64.7},
        "Wheat profile": {"ph": 6.6, "nitrogen": 17.2, "phosphorus": 221.1, "potassium": 36.3, "moisture": 17.2, "rainfall": 130.0, "temperature": 27.5, "humidity": 63.9},
    }
    for col, (label, values) in zip(st.columns(3), examples.items()):
        with col:
            if st.button(label, key=f"preset_{label}", width="stretch"):
                st.session_state.update(values)
                st.rerun()

states = sorted(df["State_Name"].dropna().astype(str).unique())
zones = sorted(df["Agro_Climatic Zone"].dropna().astype(str).unique())
soil_types = sorted(df["Soil_Type"].dropna().astype(str).unique())

with st.form("recommendation_form"):
    with st.container(border=True):
        section("Farm & Location", "Regional descriptors help tailor the model's existing recommendation to the farm context.")
        a, b, c = st.columns(3)
        with a: state = st.selectbox("State", states, help="Select the state where the farm is located.")
        with b: zone = st.selectbox("Agro-climatic Zone", zones, help="The farm's agro-climatic classification.")
        with c: soil_type = st.selectbox("Soil Type", soil_types, help="Select the soil classification reported for the farm.")

    with st.container(border=True):
        section("Soil Health", "Enter values exactly as reported by the latest soil test or field measurement.")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            ph = number("Soil pH", "ph", 6.8, .1, "Acidity or alkalinity on a 0–14 scale.", 14.0)
            nitrogen = number("Nitrogen (N)", "nitrogen", 120.0, 1.0, "Soil-test nitrogen value.")
        with s2:
            phosphorus = number("Phosphorus (P)", "phosphorus", 40.0, 1.0, "Soil-test phosphorus value.")
            potassium = number("Potassium (K)", "potassium", 180.0, 1.0, "Soil-test potassium value.")
        with s3:
            ec = number("Electrical Conductivity", "ec", .4, .1, "A soil salinity indicator.")
            organic = number("Organic Carbon (%)", "organic", .8, .1, "Organic carbon percentage in the soil.")
        with s4:
            moisture = number("Soil Moisture (%)", "moisture", 30.0, 1.0, "Current or representative soil moisture.", 100.0)

    with st.container(border=True):
        section("Weather Conditions", "Use representative recent or seasonal conditions for the farm.")
        w1, w2, w3 = st.columns(3)
        with w1: rainfall = number("Rainfall (cm)", "rainfall", 120.0, 1.0, "Rainfall measurement in centimetres.")
        with w2: temperature = st.number_input("Temperature (°C)", value=float(st.session_state.get("temperature", 28.0)), step=.1, key="temperature", help="Air temperature in degrees Celsius.")
        with w3: humidity = number("Humidity (%)", "humidity", 75.0, 1.0, "Relative humidity percentage.", 100.0)

    with st.expander("Advanced Nutrient Analysis", expanded=False):
        st.caption("Laboratory soil-test values; all six micronutrients remain part of the model input.")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1: zinc = number("Zinc (%)", "zinc", .6, .1, "Zinc measurement from the soil test.")
        with m2: iron = number("Iron (%)", "iron", 3.2, .1, "Iron measurement from the soil test.")
        with m3: manganese = number("Manganese (%)", "manganese", 1.1, .1, "Manganese measurement from the soil test.")
        with m4: copper = number("Copper (%)", "copper", .3, .1, "Copper measurement from the soil test.")
        with m5: boron = number("Boron (%)", "boron", .4, .1, "Boron measurement from the soil test.")
        with m6: sulphur = number("Sulphur (%)", "sulphur", 12.0, 1.0, "Sulphur measurement from the soil test.")
    submitted = st.form_submit_button("Get Crop Recommendation", type="primary", width="stretch")

if submitted:
    # This DataFrame preserves the production model's existing feature names and values.
    sample = pd.DataFrame([{"Soil_Type": soil_type, "pH_Value": float(ph), "Nitrogen_Value (N)": float(nitrogen), "Phosphorus_Value (P)": float(phosphorus), "Potassium_Value (K)": float(potassium), "Electrical_Conductivity (EC)": float(ec), "Organic_Carbon (%)": float(organic), "Soil_Moisture (%)": float(moisture), "Zinc (%)": float(zinc), "Iron (%)": float(iron), "Manganese (%)": float(manganese), "Copper (%)": float(copper), "Boron (%)": float(boron), "Sulphur (%)": float(sulphur), "Rainfall_cm": float(rainfall), "temperature_celsius": float(temperature), "humidity_percentage": float(humidity), "State_Name": state, "Agro_Climatic Zone": zone}])[FEATURES]
    warnings = [f"{column}: {sample[column].iloc[0]:.2f} is outside the dataset range ({low:.2f}–{high:.2f})." for column, (low, high) in RANGES.items() if not low <= float(sample[column].iloc[0]) <= high]
    if warnings:
        st.warning("Some values fall outside the range represented in the training data. The result is still generated, but should be interpreted carefully.")
        with st.expander("Review range checks"):
            st.write("\n".join(f"- {message}" for message in warnings))
    try:
        with st.spinner("Analyzing farm conditions…"):
            prediction = model.predict(sample)[0]
            probabilities = model.predict_proba(sample)[0]
        indices = probabilities.argsort()[-3:][::-1]
        top3 = [(model.classes_[index], float(probabilities[index] * 100)) for index in indices]
    except Exception:
        st.error("The recommendation could not be generated. Please verify the selected categories and measurements, then try again.")
        st.stop()

    top_crop, top_score = top3[0]
    label, level, explanation = confidence_label(top_score)
    state_count = len(df[(df["State_Name"].astype(str) == state) & (df["Crop"].astype(str) == str(top_crop))])
    overall_count = len(df[df["Crop"].astype(str) == str(top_crop)])
    support = "Good" if state_count >= 30 else "Limited" if state_count >= 10 else "Very limited"
    section("Recommended Crops", "Model probabilities and supporting data are shown exactly as generated.")
    st.markdown(f'<div class="result-hero"><span class="confidence-chip confidence-chip--{level}">{label}</span><h2>{top_crop}</h2><p class="muted">{explanation}</p></div>', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    r1.metric("Model confidence", f"{top_score:.2f}%")
    r2.metric("State + crop records", state_count)
    r3.metric("Regional support", support, help="Training records for this crop in the selected state.")

    section("Ranked Recommendations", "Compare the three highest-scoring crops before deciding.")
    for col, (rank, (crop, score)) in zip(st.columns(3), enumerate(top3, 1)):
        with col:
            with st.container(border=True):
                st.caption(f"RANK {rank}")
                st.subheader(str(crop))
                st.metric("Model score", f"{score:.2f}%")
                st.progress(min(score / 100, 1.0))

    evidence = pd.DataFrame([{"Crop": crop, "Model score (%)": round(score, 2), "State records": len(df[(df["State_Name"].astype(str) == state) & (df["Crop"].astype(str) == str(crop))]), "Overall records": len(df[df["Crop"].astype(str) == str(crop)])} for crop, score in top3])
    left, right = st.columns([1, 1.2])
    with left:
        section("Supporting evidence")
        st.dataframe(evidence, width="stretch", hide_index=True)
    with right:
        section("Farm Input Summary")
        display = sample.T.reset_index(); display.columns = ["Parameter", "Value"]
        display["Value"] = display["Value"].astype(str)
        st.dataframe(display, width="stretch", hide_index=True)
    st.info("This is decision support, not a guarantee of yield or profitability. Check seasonality, irrigation, market conditions, and local agricultural advice before planting.")

with st.expander("About this recommendation model"):
    st.write("Smart Kisan uses the project’s production balanced Random Forest classifier. The input features and model inference pipeline are unchanged.")
    x, y, z = st.columns(3)
    x.metric("Held-out accuracy", "93.53%")
    y.metric("Top-3 accuracy", "98.14%")
    z.metric("Evaluation split", "80/20 stratified")

footer()
