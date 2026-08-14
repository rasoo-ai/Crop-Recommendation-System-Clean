
import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Tehsil-wise Crop Analysis",
    page_icon="📍",
    layout="wide",
)

st.title("📍 Tehsil-wise Crop Recommendation")

st.write(
    "Select a state, district, and tehsil to view "
    "observed crop information and machine-learning recommendations."
)


# ==========================================================
# LOAD MODEL AND DATA
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "output/crop_prediction_model.pkl"
    )


@st.cache_data
def load_data():
    return pd.read_excel(
        "output/Crop_Normalized.xlsx"
    )


model = load_model()
df = load_data()


# ==========================================================
# FEATURES USED BY MODEL
# ==========================================================

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
    "humidity_percentage",
]


# ==========================================================
# LOCATION SELECTION
# ==========================================================

states = sorted(
    df["State_Name"]
    .dropna()
    .astype(str)
    .unique()
)

selected_state = st.selectbox(
    "Select State",
    states
)


state_df = df[
    df["State_Name"].astype(str)
    == selected_state
].copy()


districts = sorted(
    state_df["District_Name"]
    .dropna()
    .astype(str)
    .unique()
)

selected_district = st.selectbox(
    "Select District",
    districts
)


district_df = state_df[
    state_df["District_Name"].astype(str)
    == selected_district
].copy()


tehsils = sorted(
    district_df["Tehsil_Name"]
    .dropna()
    .astype(str)
    .unique()
)

selected_tehsil = st.selectbox(
    "Select Tehsil",
    tehsils
)


st.markdown("---")


# ==========================================================
# GENERATE RECOMMENDATION
# ==========================================================

if st.button(
    "📍 Generate Tehsil Recommendation",
    width="stretch",
):

    tehsil_df = district_df[
        district_df["Tehsil_Name"].astype(str)
        == selected_tehsil
    ].copy()

    if tehsil_df.empty:
        st.error(
            "No data found for the selected tehsil."
        )
        st.stop()


    # ======================================================
    # OBSERVED CROP INFORMATION
    # ======================================================

    st.subheader(
        "🌾 Observed Crop Information"
    )

    tehsil_df["ha"] = pd.to_numeric(
        tehsil_df["ha"],
        errors="coerce",
    )

    observed_crop_summary = (
        tehsil_df
        .groupby(
            "Crop",
            dropna=True
        )["ha"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    if observed_crop_summary.empty:

        st.info(
            "No crop-area information is available "
            "for this tehsil."
        )

    else:

        top_observed_crop = (
            observed_crop_summary.index[0]
        )

        top_observed_area = (
            observed_crop_summary.iloc[0]
        )

        st.success(
            f"Major recorded crop: "
            f"**{top_observed_crop}** "
            f"({top_observed_area:,.2f} ha)"
        )

        observed_table = (
            observed_crop_summary
            .reset_index()
            .rename(
                columns={
                    "Crop": "Crop",
                    "ha": "Recorded Area (ha)",
                }
            )
        )

        observed_table[
            "Recorded Area (ha)"
        ] = (
            observed_table[
                "Recorded Area (ha)"
            ].round(2)
        )

        st.dataframe(
            observed_table,
            width="stretch",
            hide_index=True,
        )


    # ======================================================
    # REPRESENTATIVE NUMERIC CONDITIONS
    # ======================================================

    tehsil_numeric = tehsil_df[
        numeric_cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    averages = tehsil_numeric.mean(
        skipna=True
    )


    # Overall fallback values
    overall_numeric = df[
        numeric_cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    overall_averages = (
        overall_numeric.mean(
            skipna=True
        )
    )

    averages = averages.fillna(
        overall_averages
    )


    # ======================================================
    # REPRESENTATIVE SOIL TYPE
    # ======================================================

    soil_mode = (
        tehsil_df["Soil_Type"]
        .dropna()
        .astype(str)
        .mode()
    )

    if soil_mode.empty:
        representative_soil = "Unknown"
    else:
        representative_soil = (
            soil_mode.iloc[0]
        )


    # ======================================================
    # REPRESENTATIVE AGRO-CLIMATIC ZONE
    # ======================================================

    zone_mode = (
        tehsil_df["Agro_Climatic Zone"]
        .dropna()
        .astype(str)
        .mode()
    )

    if zone_mode.empty:
        representative_zone = "Unknown"
    else:
        representative_zone = (
            zone_mode.iloc[0]
        )


    # ======================================================
    # CREATE MODEL INPUT
    # ======================================================

    sample = pd.DataFrame([{

        "Soil_Type":
            representative_soil,

        "pH_Value":
            float(
                averages["pH_Value"]
            ),

        "Nitrogen_Value (N)":
            float(
                averages["Nitrogen_Value (N)"]
            ),

        "Phosphorus_Value (P)":
            float(
                averages[
                    "Phosphorus_Value (P)"
                ]
            ),

        "Potassium_Value (K)":
            float(
                averages[
                    "Potassium_Value (K)"
                ]
            ),

        "Electrical_Conductivity (EC)":
            float(
                averages[
                    "Electrical_Conductivity (EC)"
                ]
            ),

        "Organic_Carbon (%)":
            float(
                averages[
                    "Organic_Carbon (%)"
                ]
            ),

        "Soil_Moisture (%)":
            float(
                averages[
                    "Soil_Moisture (%)"
                ]
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
                averages[
                    "temperature_celsius"
                ]
            ),

        "humidity_percentage":
            float(
                averages[
                    "humidity_percentage"
                ]
            ),

        "State_Name":
            selected_state,

        "Agro_Climatic Zone":
            representative_zone,
    }])


    # ======================================================
    # MACHINE LEARNING PREDICTION
    # ======================================================

    try:

        prediction = model.predict(
            sample
        )[0]

        probabilities = (
            model.predict_proba(
                sample
            )[0]
        )

        top3 = (
            probabilities
            .argsort()[-3:][::-1]
        )

        confidence = (
            probabilities[top3[0]]
            * 100
        )


        # ==================================================
        # RELIABILITY LEVEL
        # ==================================================

        if confidence >= 70:

            reliability = "High"

        elif confidence >= 50:

            reliability = "Moderate"

        else:

            reliability = "Low"


        # ==================================================
        # LOCATION SUMMARY
        # ==================================================

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "State",
                selected_state
            )

        with col2:

            st.metric(
                "District",
                selected_district
            )

        with col3:

            st.metric(
                "Tehsil",
                selected_tehsil
            )


        # ==================================================
        # MAIN ML RESULT
        # ==================================================

        st.subheader(
            "🤖 Machine Learning Recommendation"
        )

        st.success(
            f"Recommended Crop for "
            f"**{selected_tehsil}**: "
            f"**{prediction}**"
        )

        col_a, col_b = st.columns(2)

        with col_a:

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

        with col_b:

            st.metric(
                "Prediction Reliability",
                reliability
            )


        # ==================================================
        # TOP 3 RECOMMENDATIONS
        # ==================================================

        st.markdown("---")

        st.subheader(
            "🏆 Top 3 Model Recommendations"
        )

        for rank, index in enumerate(
            top3,
            start=1
        ):

            crop = model.classes_[index]

            crop_confidence = (
                probabilities[index]
                * 100
            )

            st.write(
                f"### {rank}. {crop}"
            )

            st.progress(
                float(
                    probabilities[index]
                )
            )

            st.write(
                f"Confidence: "
                f"**{crop_confidence:.2f}%**"
            )


        # ==================================================
        # REPRESENTATIVE CONDITIONS
        # ==================================================

        st.markdown("---")

        st.subheader(
            "🌱 Representative Conditions"
        )

        condition_col1, condition_col2 = (
            st.columns(2)
        )

        with condition_col1:

            st.write(
                f"**Soil Type:** "
                f"{representative_soil}"
            )

            st.write(
                f"**pH:** "
                f"{averages['pH_Value']:.2f}"
            )

            st.write(
                f"**Nitrogen:** "
                f"{averages['Nitrogen_Value (N)']:.2f}"
            )

            st.write(
                f"**Phosphorus:** "
                f"{averages['Phosphorus_Value (P)']:.2f}"
            )

            st.write(
                f"**Potassium:** "
                f"{averages['Potassium_Value (K)']:.2f}"
            )

        with condition_col2:

            st.write(
                f"**Rainfall:** "
                f"{averages['Rainfall_cm']:.2f} cm"
            )

            st.write(
                f"**Temperature:** "
                f"{averages['temperature_celsius']:.2f} °C"
            )

            st.write(
                f"**Humidity:** "
                f"{averages['humidity_percentage']:.2f}%"
            )

            st.write(
                f"**Agro Climatic Zone:** "
                f"{representative_zone}"
            )


        # ==================================================
        # DATASET EVIDENCE VS MODEL
        # ==================================================

        st.markdown("---")

        st.subheader(
            "🔎 Dataset Evidence vs Model"
        )

        if not observed_crop_summary.empty:

            observed_crop = (
                observed_crop_summary.index[0]
            )

            if observed_crop == prediction:

                st.success(
                    f"✅ The ML recommendation "
                    f"(**{prediction}**) matches "
                    f"the major crop recorded in "
                    f"the dataset for this tehsil."
                )

            else:

                st.warning(
                    f"⚠️ The ML recommendation "
                    f"(**{prediction}**) differs "
                    f"from the major crop recorded "
                    f"in the dataset "
                    f"(**{observed_crop}**)."
                )


        # ==================================================
        # INTERPRETATION
        # ==================================================

        if reliability == "High":

            st.success(
                "The model assigns a relatively high "
                "probability to its top recommendation. "
                "This is a model-confidence indicator, "
                "not a guarantee of crop suitability."
            )

        elif reliability == "Moderate":

            st.warning(
                "The model shows moderate confidence. "
                "The alternative crops should also be "
                "considered before making a decision."
            )

        else:

            st.warning(
                "The model has low confidence. "
                "Treat this recommendation cautiously "
                "and consider the top alternative crops."
            )


        # ==================================================
        # DISCLAIMER
        # ==================================================

        st.info(
            "This recommendation is based on the available "
            "dataset and representative environmental "
            "conditions. Actual crop suitability can vary "
            "with irrigation, season, soil variability, "
            "and local farming practices."
        )

        st.caption(
            f"Analysis based on "
            f"{len(tehsil_df)} dataset record(s) "
            f"for this tehsil."
        )


    except Exception as error:

        st.error(
            "Unable to generate the tehsil recommendation."
        )

        st.exception(error)

