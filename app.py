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

@st.cache_resource
def load_model():
    return joblib.load("output/crop_prediction_model.pkl")


@st.cache_data
def load_dataset():
    return pd.read_excel("output/Crop_Normalized.xlsx")


model = load_model()
df = load_dataset()

# --------------------------------------------------
# Prepare Dropdown Values
# --------------------------------------------------

soil_types = sorted(
    df["Soil_Type"].dropna().unique()
)

states = sorted(
    df["State_Name"].dropna().unique()
)

zones = sorted(
    df["Agro_Climatic Zone"].dropna().unique()
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🌾 Crop Recommendation System")

st.markdown(
    "### Machine Learning Based Crop Recommendation "
    "Using Soil & Weather Parameters"
)

st.markdown("---")

# ==================================================
# SECTION 1 - INDIVIDUAL CROP PREDICTION
# ==================================================

st.header("🌱 Individual Crop Recommendation")

st.markdown(
    "Enter soil, weather and geographical parameters "
    "to predict the most suitable crop."
)

col1, col2 = st.columns(2)

with col1:

    soil_type = st.selectbox(
        "Soil Type",
        soil_types
    )

    state = st.selectbox(
        "State",
        states
    )

    zone = st.selectbox(
        "Agro Climatic Zone",
        zones
    )

    ph = st.number_input(
        "pH Value",
        value=6.8
    )

    nitrogen = st.number_input(
        "Nitrogen (N)",
        value=120.0
    )

    phosphorus = st.number_input(
        "Phosphorus (P)",
        value=40.0
    )

    potassium = st.number_input(
        "Potassium (K)",
        value=180.0
    )

    ec = st.number_input(
        "Electrical Conductivity",
        value=0.4
    )

with col2:

    organic = st.number_input(
        "Organic Carbon (%)",
        value=0.8
    )

    moisture = st.number_input(
        "Soil Moisture (%)",
        value=30.0
    )

    zinc = st.number_input(
        "Zinc (%)",
        value=0.6
    )

    iron = st.number_input(
        "Iron (%)",
        value=3.2
    )

    manganese = st.number_input(
        "Manganese (%)",
        value=1.1
    )

    copper = st.number_input(
        "Copper (%)",
        value=0.3
    )

    boron = st.number_input(
        "Boron (%)",
        value=0.4
    )

    sulphur = st.number_input(
        "Sulphur (%)",
        value=12.0
    )

    rainfall = st.number_input(
        "Rainfall (cm)",
        value=120.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        value=28.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        value=75.0
    )

# --------------------------------------------------
# Individual Prediction
# --------------------------------------------------

st.markdown("---")

if st.button(
    "🌾 Predict Crop",
    use_container_width=True
):

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

    try:
        prediction = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]

        st.success(
            f"### 🌾 Recommended Crop: **{prediction}**"
        )

        st.markdown("## 🏆 Top 3 Recommendations")

        top3 = probabilities.argsort()[-3:][::-1]

        for rank, i in enumerate(top3, start=1):

            crop = model.classes_[i]
            score = probabilities[i] * 100

            st.write(f"### {rank}. {crop}")

            st.progress(
                float(probabilities[i])
            )

            st.write(
                f"Confidence: **{score:.2f}%**"
            )

    except Exception as e:
        st.error("Prediction could not be completed.")
        st.exception(e)

# ==================================================
# SECTION 2 - STATE-WISE RECOMMENDATION
# ==================================================

st.markdown("---")

st.header("🇮🇳 State-wise Crop Recommendations")

st.markdown(
    "Generate representative crop recommendations for "
    "all states available in the dataset."
)

if st.button(
    "🇮🇳 Generate State-wise Recommendations",
    use_container_width=True
):

    with st.spinner(
        "Generating recommendations for all states..."
    ):

        numeric_cols = [
            "pH_Value",
            "Nitrogen_Value (N)",
            "Phosphorus_Value (P)",
            "Potassium_Value (K)",
            "Electrical_Conductivity (EC)",
            "Organic_Carbon (%)",
            "Soil_Moisture (%)",
            "Zinc (%)",
            "Iron (%)",
            "Manganese (%)",
            "Copper (%)",
            "Boron (%)",
            "Sulphur (%)",
            "Rainfall_cm",
            "temperature_celsius",
            "humidity_percentage"
        ]

        results = []

        for current_state in states:

            state_df = df[
                df["State_Name"] == current_state
            ]

            if state_df.empty:
                continue

            averages = state_df[numeric_cols].mean()

            soil_mode = (
                state_df["Soil_Type"]
                .dropna()
                .mode()
            )

            representative_soil = (
                soil_mode.iloc[0]
                if not soil_mode.empty
                else soil_types[0]
            )

            zone_mode = (
                state_df["Agro_Climatic Zone"]
                .dropna()
                .mode()
            )

            representative_zone = (
                zone_mode.iloc[0]
                if not zone_mode.empty
                else zones[0]
            )

            state_sample = pd.DataFrame([{
                "Soil_Type": representative_soil,
                "pH_Value": averages["pH_Value"],
                "Nitrogen_Value (N)": averages["Nitrogen_Value (N)"],
                "Phosphorus_Value (P)": averages["Phosphorus_Value (P)"],
                "Potassium_Value (K)": averages["Potassium_Value (K)"],
                "Electrical_Conductivity (EC)": averages[
                    "Electrical_Conductivity (EC)"
                ],
                "Organic_Carbon (%)": averages[
                    "Organic_Carbon (%)"
                ],
                "Soil_Moisture (%)": averages[
                    "Soil_Moisture (%)"
                ],
                "Zinc (%)": averages["Zinc (%)"],
                "Iron (%)": averages["Iron (%)"],
                "Manganese (%)": averages["Manganese (%)"],
                "Copper (%)": averages["Copper (%)"],
                "Boron (%)": averages["Boron (%)"],
                "Sulphur (%)": averages["Sulphur (%)"],
                "Rainfall_cm": averages["Rainfall_cm"],
                "temperature_celsius": averages[
                    "temperature_celsius"
                ],
                "humidity_percentage": averages[
                    "humidity_percentage"
                ],
                "State_Name": current_state,
                "Agro_Climatic Zone": representative_zone
            }])

            try:
                state_prediction = model.predict(
                    state_sample
                )[0]

                state_probabilities = model.predict_proba(
                    state_sample
                )[0]

                top3_state = (
                    state_probabilities
                    .argsort()[-3:][::-1]
                )

                results.append({
                    "State": current_state,
                    "Soil Type": representative_soil,
                    "Agro Climatic Zone": representative_zone,
                    "Recommended Crop": state_prediction,
                    "Confidence (%)": round(
                        state_probabilities[top3_state[0]] * 100,
                        2
                    ),
                    "Second Choice": model.classes_[
                        top3_state[1]
                    ],
                    "Third Choice": model.classes_[
                        top3_state[2]
                    ]
                })

            except Exception:
                continue

        state_results = pd.DataFrame(results)

        if state_results.empty:

            st.error(
                "No state-wise recommendations could be generated."
            )

        else:

            st.success(
                f"Successfully generated recommendations "
                f"for {len(state_results)} states."
            )

            st.dataframe(
                state_results,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("## 📊 Recommendation Summary")

            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:
                st.metric(
                    "States Analyzed",
                    len(state_results)
                )

            with summary_col2:
                st.metric(
                    "Different Recommended Crops",
                    state_results[
                        "Recommended Crop"
                    ].nunique()
                )

            st.markdown(
                "### 🌾 Recommended Crop Distribution"
            )

            crop_counts = (
                state_results[
                    "Recommended Crop"
                ]
                .value_counts()
            )

            st.bar_chart(crop_counts)

            csv_data = state_results.to_csv(
                index=False
            )

            st.download_button(
                label="📥 Download State-wise Results",
                data=csv_data,
                file_name="Statewise_Crop_Recommendations.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Crop Recommendation System using Machine Learning | "
    "Random Forest Classifier"
)