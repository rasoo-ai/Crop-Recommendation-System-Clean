import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# --------------------------------------------------
# Load Model and Dataset
# --------------------------------------------------
model = joblib.load("output/crop_prediction_model.pkl")
df = pd.read_excel("output/Crop_Normalized.xlsx")

soil_types = sorted(df["Soil_Type"].dropna().unique())
states = sorted(df["State_Name"].dropna().unique())
zones = sorted(df["Agro_Climatic Zone"].dropna().unique())

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🌾 Crop Recommendation System")
st.markdown("### Machine Learning Based Crop Recommendation Using Soil & Weather Parameters")

st.markdown("---")

# --------------------------------------------------
# Input Form
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:

    soil_type = st.selectbox("Soil Type", soil_types)

    state = st.selectbox("State", states)

    zone = st.selectbox("Agro Climatic Zone", zones)

    ph = st.number_input("pH Value", value=6.8)

    nitrogen = st.number_input("Nitrogen (N)", value=120.0)

    phosphorus = st.number_input("Phosphorus (P)", value=40.0)

    potassium = st.number_input("Potassium (K)", value=180.0)

    ec = st.number_input("Electrical Conductivity", value=0.4)

with col2:

    organic = st.number_input("Organic Carbon (%)", value=0.8)

    moisture = st.number_input("Soil Moisture (%)", value=30.0)

    zinc = st.number_input("Zinc (%)", value=0.6)

    iron = st.number_input("Iron (%)", value=3.2)

    manganese = st.number_input("Manganese (%)", value=1.1)

    copper = st.number_input("Copper (%)", value=0.3)

    boron = st.number_input("Boron (%)", value=0.4)

    sulphur = st.number_input("Sulphur (%)", value=12.0)

    rainfall = st.number_input("Rainfall (cm)", value=120.0)

    temperature = st.number_input("Temperature (°C)", value=28.0)

    humidity = st.number_input("Humidity (%)", value=75.0)

st.markdown("---")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Crop", use_container_width=True):

    sample = pd.DataFrame([{
        "Soil_Type": soil_type,
        "pH_Value": ph,
        "Nitrogen_Value (N)": nitrogen,
        "Phosphorus_Value (P)": phosphorus,
        "Potassium_Value (K)": potassium,
        "Electrical_Conductivity (EC)": ec,
        "Organic_Carbon (%)": organic,
        "Soil_Moisture (%)": moisture,
        "Zinc (%)": zinc,
        "Iron (%)": iron,
        "Manganese (%)": manganese,
        "Copper (%)": copper,
        "Boron (%)": boron,
        "Sulphur (%)": sulphur,
        "Rainfall_cm": rainfall,
        "temperature_celsius": temperature,
        "humidity_percentage": humidity,
        "State_Name": state,
        "Agro_Climatic Zone": zone
    }])

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    st.success(f"### 🌾 Recommended Crop: **{prediction}**")

    st.markdown("## Top 3 Recommendations")

    top3 = probabilities.argsort()[-3:][::-1]

    for i in top3:
        crop = model.classes_[i]
        score = probabilities[i] * 100

        st.write(f"**{crop}**")
        st.progress(float(probabilities[i]))
        st.write(f"{score:.2f}%")
        st.write("")

st.markdown("---")
st.caption("Crop Recommendation System using Machine Learning | Random Forest Classifier")