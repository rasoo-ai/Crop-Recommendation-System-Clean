import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# ==========================================================
# LOAD MODEL AND DATASET
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("output/crop_prediction_model.pkl")


@st.cache_data
def load_dataset():
    return pd.read_excel("output/Crop_Normalized.xlsx")


model = load_model()
df = load_dataset()

# ==========================================================
# PREPARE DROPDOWN VALUES
# ==========================================================

soil_types = sorted(
    df["Soil_Type"].dropna().astype(str).unique()
)

states = sorted(
    df["State_Name"].dropna().astype(str).unique()
)

zones = sorted(
    df["Agro_Climatic Zone"].dropna().astype(str).unique()
)

# ==========================================================
# TITLE
# ==========================================================

st.title("🌾 Crop Recommendation System")

st.markdown(
    "### Machine Learning Based Crop Recommendation "
    "Using Soil & Weather Parameters"
)

st.markdown("---")

# ==========================================================
# SECTION 1 - INDIVIDUAL CROP PREDICTION
# ==========================================================

st.header("🌱 Individual Crop Recommendation")

st.write(
    "Enter soil, weather and geographical parameters "
    "to predict the most suitable crop."
)

col1, col2 = st.columns(2)

# ----------------------------------------------------------
# LEFT COLUMN
# ----------------------------------------------------------

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
        value=6.8,
        step=0.1
    )

    nitrogen = st.number_input(
        "Nitrogen (N)",
        value=120.0,
        step=1.0
    )

    phosphorus = st.number_input(
        "Phosphorus (P)",
        value=40.0,
        step=1.0
    )

    potassium = st.number_input(
        "Potassium (K)",
        value=180.0,
        step=1.0
    )

    ec = st.number_input(
        "Electrical Conductivity",
        value=0.4,
        step=0.1
    )

# ----------------------------------------------------------
# RIGHT COLUMN
# ----------------------------------------------------------

with col2:

    organic = st.number_input(
        "Organic Carbon (%)",
        value=0.8,
        step=0.1
    )

    moisture = st.number_input(
        "Soil Moisture (%)",
        value=30.0,
        step=1.0
    )

    zinc = st.number_input(
        "Zinc (%)",
        value=0.6,
        step=0.1
    )

    iron = st.number_input(
        "Iron (%)",
        value=3.2,
        step=0.1
    )

    manganese = st.number_input(
        "Manganese (%)",
        value=1.1,
        step=0.1
    )

    copper = st.number_input(
        "Copper (%)",
        value=0.3,
        step=0.1
    )

    boron = st.number_input(
        "Boron (%)",
        value=0.4,
        step=0.1
    )

    sulphur = st.number_input(
        "Sulphur (%)",
        value=12.0,
        step=1.0
    )

    rainfall = st.number_input(
        "Rainfall (cm)",
        value=120.0,
        step=1.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        value=28.0,
        step=0.1
    )

    humidity = st.number_input(
        "Humidity (%)",
        value=75.0,
        step=1.0
    )

# ==========================================================
# INDIVIDUAL PREDICTION
# ==========================================================

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

        for rank, index in enumerate(top3, start=1):

            crop = model.classes_[index]

            confidence = probabilities[index] * 100

            st.write(
                f"### {rank}. {crop}"
            )

            st.progress(
                float(probabilities[index])
            )

            st.write(
                f"Confidence: **{confidence:.2f}%**"
            )

    except Exception as error:

        st.error(
            "Unable to generate the crop prediction."
        )

        st.exception(error)

# ==========================================================
# SECTION 2 - STATE-WISE CROP RECOMMENDATIONS
# ==========================================================

st.markdown("---")

st.header("🇮🇳 State-wise Crop Recommendations")

st.write(
    "Generate representative crop recommendations for "
    "all states available in the dataset."
)

if st.button(
    "🇮🇳 Generate State-wise Recommendations",
    use_container_width=True
):

    with st.spinner(
        "Calculating state-wise recommendations..."
    ):

        # --------------------------------------------------
        # NUMERIC FEATURES
        # --------------------------------------------------

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

        # --------------------------------------------------
        # PROCESS EVERY STATE
        # --------------------------------------------------

        for current_state in states:

            state_df = df[
                df["State_Name"].astype(str)
                == current_state
            ].copy()

            if state_df.empty:
                continue

            # ------------------------------------------------
            # SAFE NUMERIC CONVERSION
            # ------------------------------------------------

            state_numeric = state_df[
                numeric_cols
            ].apply(
                pd.to_numeric,
                errors="coerce"
            )

            # ------------------------------------------------
            # CALCULATE SAFE AVERAGES
            # ------------------------------------------------

            averages = state_numeric.mean(
                skipna=True
            )

            # ------------------------------------------------
            # CHECK MISSING NUMERIC VALUES
            # ------------------------------------------------

            if averages.isna().all():
                continue

            # Fill any missing averages from the
            # overall dataset averages
            overall_numeric = df[
                numeric_cols
            ].apply(
                pd.to_numeric,
                errors="coerce"
            )

            overall_averages = overall_numeric.mean(
                skipna=True
            )

            averages = averages.fillna(
                overall_averages
            )

            # ------------------------------------------------
            # MOST COMMON SOIL TYPE
            # ------------------------------------------------

            soil_mode = (
                state_df["Soil_Type"]
                .dropna()
                .astype(str)
                .mode()
            )

            if not soil_mode.empty:
                representative_soil = soil_mode.iloc[0]
            else:
                representative_soil = soil_types[0]

            # ------------------------------------------------
            # MOST COMMON AGRO-CLIMATIC ZONE
            # ------------------------------------------------

            zone_mode = (
                state_df["Agro_Climatic Zone"]
                .dropna()
                .astype(str)
                .mode()
            )

            if not zone_mode.empty:
                representative_zone = zone_mode.iloc[0]
            else:
                representative_zone = zones[0]

            # ------------------------------------------------
            # CREATE REPRESENTATIVE SAMPLE
            # ------------------------------------------------

            state_sample = pd.DataFrame([{
                "Soil_Type": representative_soil,

                "pH_Value":
                    float(averages["pH_Value"]),

                "Nitrogen_Value (N)":
                    float(
                        averages["Nitrogen_Value (N)"]
                    ),

                "Phosphorus_Value (P)":
                    float(
                        averages["Phosphorus_Value (P)"]
                    ),

                "Potassium_Value (K)":
                    float(
                        averages["Potassium_Value (K)"]
                    ),

                "Electrical_Conductivity (EC)":
                    float(
                        averages[
                            "Electrical_Conductivity (EC)"
                        ]
                    ),

                "Organic_Carbon (%)":
                    float(
                        averages["Organic_Carbon (%)"]
                    ),

                "Soil_Moisture (%)":
                    float(
                        averages["Soil_Moisture (%)"]
                    ),

                "Zinc (%)":
                    float(
                        averages["Zinc (%)"]
                    ),

                "Iron (%)":
                    float(
                        averages["Iron (%)"]
                    ),

                "Manganese (%)":
                    float(
                        averages["Manganese (%)"]
                    ),

                "Copper (%)":
                    float(
                        averages["Copper (%)"]
                    ),

                "Boron (%)":
                    float(
                        averages["Boron (%)"]
                    ),

                "Sulphur (%)":
                    float(
                        averages["Sulphur (%)"]
                    ),

                "Rainfall_cm":
                    float(
                        averages["Rainfall_cm"]
                    ),

                "temperature_celsius":
                    float(
                        averages["temperature_celsius"]
                    ),

                "humidity_percentage":
                    float(
                        averages[
                            "humidity_percentage"
                        ]
                    ),

                "State_Name": current_state,

                "Agro_Climatic Zone":
                    representative_zone
            }])

            # ------------------------------------------------
            # PREDICT STATE CROP
            # ------------------------------------------------

            try:

                prediction = model.predict(
                    state_sample
                )[0]

                probabilities = model.predict_proba(
                    state_sample
                )[0]

                # Top 3 predictions
                top3 = probabilities.argsort()[-3:][::-1]

                results.append({
                    "State":
                        current_state,

                    "Soil Type":
                        representative_soil,

                    "Agro Climatic Zone":
                        representative_zone,

                    "Recommended Crop":
                        model.classes_[top3[0]],

                    "Confidence (%)":
                        round(
                            probabilities[top3[0]] * 100,
                            2
                        ),

                    "Second Choice":
                        model.classes_[top3[1]],

                    "Third Choice":
                        model.classes_[top3[2]]
                })

            except Exception:
                continue

        # ==================================================
        # DISPLAY RESULTS
        # ==================================================

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

            # ------------------------------------------------
            # TABLE
            # ------------------------------------------------

            st.dataframe(
                state_results,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # SUMMARY METRICS
            # ------------------------------------------------

            st.markdown("## 📊 Recommendation Summary")

            metric1, metric2 = st.columns(2)

            with metric1:

                st.metric(
                    "States Analyzed",
                    len(state_results)
                )

            with metric2:

                st.metric(
                    "Different Recommended Crops",
                    state_results[
                        "Recommended Crop"
                    ].nunique()
                )

            # ------------------------------------------------
            # CROP DISTRIBUTION
            # ------------------------------------------------

            st.markdown(
                "### 🌾 Recommended Crop Distribution"
            )

            crop_counts = (
                state_results[
                    "Recommended Crop"
                ]
                .value_counts()
            )

            st.bar_chart(
                crop_counts
            )

            # ------------------------------------------------
            # DOWNLOAD RESULTS
            # ------------------------------------------------

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

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Crop Recommendation System using Machine Learning | "
    "Random Forest Classifier"
)