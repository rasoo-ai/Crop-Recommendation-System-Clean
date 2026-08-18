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
# LOAD PRODUCTION MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "output/crop_prediction_model_balanced.pkl"
    )


# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_dataset():
    return pd.read_excel(
        "output/Crop_Normalized.xlsx"
    )


try:
    model = load_model()
    df = load_dataset()

except Exception as error:
    st.error(
        "Unable to load the production model or dataset."
    )
    st.exception(error)
    st.stop()


# ==========================================================
# PREPARE DROPDOWNS
# ==========================================================

states = sorted(
    df["State_Name"]
    .dropna()
    .astype(str)
    .unique()
)

soil_types = sorted(
    df["Soil_Type"]
    .dropna()
    .astype(str)
    .unique()
)

zones = sorted(
    df["Agro_Climatic Zone"]
    .dropna()
    .astype(str)
    .unique()
)


# ==========================================================
# CROP ICONS
# ==========================================================

CROP_EMOJI = {
    "Rice": "🌾",
    "Wheat": "🌿",
    "Maize": "🌽",
    "Cotton": "🌸",
    "Mustard": "🌻",
    "Pulses": "🫘",
    "Vegetables": "🥦",
    "Apple": "🍎",
    "Walnut": "🌰",
    "Sugarcane": "🎋",
    "Potato": "🥔",
    "Barley": "🌾",
}


def crop_emoji(crop):
    return CROP_EMOJI.get(
        crop,
        "🌱",
    )


# ==========================================================
# CONFIDENCE CLASSIFICATION
# ==========================================================

def confidence_level(value):

    if value >= 70:
        return (
            "High confidence",
            "🟢",
        )

    if value >= 50:
        return (
            "Moderate confidence",
            "🟡",
        )

    return (
        "Low confidence",
        "🔴",
    )


# ==========================================================
# NUMERIC COLUMNS
# ==========================================================

numeric_columns = [
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


# Convert numeric columns safely
for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ==========================================================
# DATASET RANGES
# ==========================================================

dataset_ranges = {}

for column in numeric_columns:

    series = df[column].dropna()

    if not series.empty:

        dataset_ranges[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
        }


# ==========================================================
# EXAMPLE PRESETS
# ==========================================================

EXAMPLES = {
    "🌾 Rice": {
        "state": "Andhra Pradesh",
        "zone": "Southern Plateau and Hills Region",
        "soil_type": "Alluvial",
        "ph": 6.9,
        "nitrogen": 30.7,
        "phosphorus": 204.9,
        "potassium": 53.4,
        "ec": 0.4,
        "organic": 0.8,
        "moisture": 30.0,
        "rainfall": 232.5,
        "temperature": 27.9,
        "humidity": 65.0,
        "zinc": 0.6,
        "iron": 3.2,
        "manganese": 1.1,
        "copper": 0.3,
        "boron": 0.4,
        "sulphur": 12.0,
    },
    "🌽 Maize": {
        "state": "Telangana",
        "zone": "Southern Plateau and Hills Region",
        "soil_type": "Alluvial Soil",
        "ph": 6.9,
        "nitrogen": 14.9,
        "phosphorus": 219.0,
        "potassium": 35.2,
        "ec": 0.4,
        "organic": 0.8,
        "moisture": 30.0,
        "rainfall": 172.9,
        "temperature": 27.5,
        "humidity": 64.7,
        "zinc": 0.6,
        "iron": 3.2,
        "manganese": 1.1,
        "copper": 0.3,
        "boron": 0.4,
        "sulphur": 12.0,
    },
    "🌿 Wheat": {
        "state": "Telangana",
        "zone": "Southern Plateau and Hills Region",
        "soil_type": "Alluvial Soil",
        "ph": 6.6,
        "nitrogen": 17.2,
        "phosphorus": 221.1,
        "potassium": 36.3,
        "ec": 0.4,
        "organic": 0.7,
        "moisture": 17.2,
        "rainfall": 130.0,
        "temperature": 27.5,
        "humidity": 63.9,
        "zinc": 0.6,
        "iron": 3.2,
        "manganese": 1.1,
        "copper": 0.3,
        "boron": 0.4,
        "sulphur": 12.0,
    },
    "🌸 Cotton": {
        "state": "Andhra Pradesh",
        "zone": "Southern Plateau and Hills Region",
        "soil_type": "Black Cotton Soil (Vertisols)",
        "ph": 7.8,
        "nitrogen": 55.4,
        "phosphorus": 178.6,
        "potassium": 77.9,
        "ec": 0.5,
        "organic": 0.6,
        "moisture": 16.8,
        "rainfall": 59.4,
        "temperature": 27.7,
        "humidity": 64.2,
        "zinc": 0.6,
        "iron": 3.2,
        "manganese": 1.1,
        "copper": 0.3,
        "boron": 0.4,
        "sulphur": 12.0,
    },
    "🌻 Mustard": {
        "state": "Jharkhand",
        "zone": "Eastern Plateau and Hills Region",
        "soil_type": "Red Sandy Soil",
        "ph": 6.2,
        "nitrogen": 0.3,
        "phosphorus": 230.6,
        "potassium": 17.9,
        "ec": 0.4,
        "organic": 0.5,
        "moisture": 17.3,
        "rainfall": 1242.3,
        "temperature": 25.4,
        "humidity": 71.8,
        "zinc": 0.6,
        "iron": 3.2,
        "manganese": 1.1,
        "copper": 0.3,
        "boron": 0.4,
        "sulphur": 12.0,
    },
    "🫘 Pulses": {
        "state": "Andhra Pradesh",
        "zone": "East Coast Plains and Hills Region",
        "soil_type": "Red sandy loam",
        "ph": 7.1,
        "nitrogen": 175.3,
        "phosphorus": 77.9,
        "potassium": 210.4,
        "ec": 0.4,
        "organic": 0.7,
        "moisture": 16.8,
        "rainfall": 455.3,
        "temperature": 27.3,
        "humidity": 66.7,
        "zinc": 0.6,
        "iron": 3.2,
        "manganese": 1.1,
        "copper": 0.3,
        "boron": 0.4,
        "sulphur": 12.0,
    },
}


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
# QUICK EXAMPLES
# ==========================================================

st.subheader("⚡ Quick Load Examples")

st.caption(
    "These presets are application examples. "
    "Use verified farm or soil-test values for real decisions."
)


if "preset" not in st.session_state:
    st.session_state.preset = None


example_columns = st.columns(
    len(EXAMPLES)
)

for index, (
    label,
    values,
) in enumerate(EXAMPLES.items()):

    with example_columns[index]:

        if st.button(
            label,
            use_container_width=True,
        ):

            st.session_state.preset = values

            st.rerun()


preset = (
    st.session_state.preset
    if st.session_state.preset
    else {}
)


def get_value(
    key,
    default,
):
    return preset.get(
        key,
        default,
    )


def get_index(
    key,
    options,
):
    value = preset.get(key)

    if value in options:
        return options.index(value)

    return 0


# ==========================================================
# INPUT MODE
# ==========================================================

st.divider()

mode = st.radio(
    "Choose input mode",
    [
        "👨‍🌾 Quick Recommendation",
        "🔬 Advanced Soil Analysis",
    ],
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
        index=get_index(
            "state",
            states,
        ),
    )


with loc2:

    zone = st.selectbox(
        "Agro-Climatic Zone",
        zones,
        index=get_index(
            "zone",
            zones,
        ),
    )


with loc3:

    soil_type = st.selectbox(
        "Soil Type",
        soil_types,
        index=get_index(
            "soil_type",
            soil_types,
        ),
    )


# ==========================================================
# PRIMARY INPUTS
# ==========================================================

st.header("🌍 Soil & Weather")

input1, input2 = st.columns(2)


with input1:

    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=float(
            get_value(
                "ph",
                6.8,
            )
        ),
        step=0.1,
    )

    nitrogen = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        value=float(
            get_value(
                "nitrogen",
                120.0,
            )
        ),
        step=1.0,
    )

    phosphorus = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        value=float(
            get_value(
                "phosphorus",
                40.0,
            )
        ),
        step=1.0,
    )

    potassium = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        value=float(
            get_value(
                "potassium",
                180.0,
            )
        ),
        step=1.0,
    )

    ec = st.number_input(
        "Electrical Conductivity",
        min_value=0.0,
        value=float(
            get_value(
                "ec",
                0.4,
            )
        ),
        step=0.1,
    )


with input2:

    organic = st.number_input(
        "Organic Carbon (%)",
        min_value=0.0,
        value=float(
            get_value(
                "organic",
                0.8,
            )
        ),
        step=0.1,
    )

    moisture = st.number_input(
        "Soil Moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(
            get_value(
                "moisture",
                30.0,
            )
        ),
        step=1.0,
    )

    rainfall = st.number_input(
        "Rainfall (cm)",
        min_value=0.0,
        value=float(
            get_value(
                "rainfall",
                120.0,
            )
        ),
        step=1.0,
    )

    temperature = st.number_input(
        "Temperature (°C)",
        value=float(
            get_value(
                "temperature",
                28.0,
            )
        ),
        step=0.1,
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(
            get_value(
                "humidity",
                75.0,
            )
        ),
        step=1.0,
    )


# ==========================================================
# ADVANCED SOIL DATA
# ==========================================================

if mode == "🔬 Advanced Soil Analysis":

    st.header("🧪 Advanced Soil Nutrients")

    st.caption(
        "Use this section when laboratory soil-test "
        "values are available."
    )

    adv1, adv2 = st.columns(2)

    with adv1:

        zinc = st.number_input(
            "Zinc (%)",
            min_value=0.0,
            value=float(
                get_value(
                    "zinc",
                    0.6,
                )
            ),
            step=0.1,
        )

        iron = st.number_input(
            "Iron (%)",
            min_value=0.0,
            value=float(
                get_value(
                    "iron",
                    3.2,
                )
            ),
            step=0.1,
        )

        manganese = st.number_input(
            "Manganese (%)",
            min_value=0.0,
            value=float(
                get_value(
                    "manganese",
                    1.1,
                )
            ),
            step=0.1,
        )

    with adv2:

        copper = st.number_input(
            "Copper (%)",
            min_value=0.0,
            value=float(
                get_value(
                    "copper",
                    0.3,
                )
            ),
            step=0.1,
        )

        boron = st.number_input(
            "Boron (%)",
            min_value=0.0,
            value=float(
                get_value(
                    "boron",
                    0.4,
                )
            ),
            step=0.1,
        )

        sulphur = st.number_input(
            "Sulphur (%)",
            min_value=0.0,
            value=float(
                get_value(
                    "sulphur",
                    12.0,
                )
            ),
            step=1.0,
        )

else:

    # Temporary defaults until lab/IoT integration.
    zinc = float(
        get_value(
            "zinc",
            0.6,
        )
    )

    iron = float(
        get_value(
            "iron",
            3.2,
        )
    )

    manganese = float(
        get_value(
            "manganese",
            1.1,
        )
    )

    copper = float(
        get_value(
            "copper",
            0.3,
        )
    )

    boron = float(
        get_value(
            "boron",
            0.4,
        )
    )

    sulphur = float(
        get_value(
            "sulphur",
            12.0,
        )
    )


# ==========================================================
# PREDICTION
# ==========================================================

st.divider()

predict = st.button(
    "🌾 Get Crop Recommendation",
    use_container_width=True,
    type="primary",
)


if predict:

    # ======================================================
    # BUILD INPUT
    # ======================================================

    sample = pd.DataFrame(
        [
            {
                "Soil_Type": soil_type,
                "pH_Value": float(ph),
                "Nitrogen_Value (N)": float(nitrogen),
                "Phosphorus_Value (P)": float(phosphorus),
                "Potassium_Value (K)": float(potassium),
                "Electrical_Conductivity (EC)": float(ec),
                "Organic_Carbon (%)": float(organic),
                "Soil_Moisture (%)": float(moisture),
                "Zinc (%)": float(zinc),
                "Iron (%)": float(iron),
                "Manganese (%)": float(manganese),
                "Copper (%)": float(copper),
                "Boron (%)": float(boron),
                "Sulphur (%)": float(sulphur),
                "Rainfall_cm": float(rainfall),
                "temperature_celsius": float(temperature),
                "humidity_percentage": float(humidity),
                "State_Name": state,
                "Agro_Climatic Zone": zone,
            }
        ]
    )


    # ======================================================
    # INPUT VALIDATION
    # ======================================================

    warnings = []

    for column in numeric_columns:

        if column not in sample.columns:
            continue

        value = float(
            sample[column].iloc[0]
        )

        if column not in dataset_ranges:
            continue

        minimum = dataset_ranges[column]["min"]
        maximum = dataset_ranges[column]["max"]

        if value < minimum:
            warnings.append(
                f"{column}: {value:.2f} is below the "
                f"dataset range ({minimum:.2f}–{maximum:.2f})."
            )

        elif value > maximum:
            warnings.append(
                f"{column}: {value:.2f} is above the "
                f"dataset range ({minimum:.2f}–{maximum:.2f})."
            )


    if warnings:

        st.warning(
            "⚠️ Some inputs are outside the range represented "
            "in the training dataset."
        )

        with st.expander(
            "View input range warnings"
        ):

            for warning in warnings:
                st.write(
                    f"- {warning}"
                )

        st.info(
            "Please verify these measurements before using "
            "the recommendation."
        )


    # ======================================================
    # PREDICT
    # ======================================================

    try:

        prediction = model.predict(
            sample
        )[0]

        probabilities = model.predict_proba(
            sample
        )[0]

        top_indices = (
            probabilities
            .argsort()[-3:][::-1]
        )

        top3 = [
            (
                model.classes_[index],
                float(
                    probabilities[index] * 100
                ),
            )
            for index in top_indices
        ]

        top_crop, top_confidence = top3[0]

        confidence_text, confidence_icon = (
            confidence_level(
                top_confidence
            )
        )


        # ==================================================
        # REGIONAL EVIDENCE
        # ==================================================

        state_crop_count = len(
            df[
                (df["State_Name"].astype(str) == state)
                & (df["Crop"].astype(str) == str(top_crop))
            ]
        )

        overall_crop_count = len(
            df[
                df["Crop"].astype(str) == str(top_crop)
            ]
        )


        if state_crop_count >= 30:

            regional_support = "Good"
            regional_icon = "🟢"

        elif state_crop_count >= 10:

            regional_support = "Limited"
            regional_icon = "🟡"

        else:

            regional_support = "Very limited"
            regional_icon = "🔴"


        # ==================================================
        # RESULT
        # ==================================================

        st.divider()

        st.header("🌾 Recommendation Result")


        if top_confidence >= 70:

            st.success(
                f"{confidence_icon} {confidence_text}"
            )

        elif top_confidence >= 50:

            st.warning(
                f"{confidence_icon} {confidence_text}"
            )

        else:

            st.error(
                f"{confidence_icon} {confidence_text}"
            )


        st.subheader(
            f"{crop_emoji(top_crop)} "
            f"Recommended Crop: {top_crop}"
        )

        st.metric(
            "Model Confidence",
            f"{top_confidence:.2f}%",
        )


        # ==================================================
        # EVIDENCE METRICS
        # ==================================================

        evidence1, evidence2, evidence3 = (
            st.columns(3)
        )

        with evidence1:

            st.metric(
                "State + Crop Records",
                state_crop_count,
            )

        with evidence2:

            st.metric(
                "Overall Crop Records",
                overall_crop_count,
            )

        with evidence3:

            st.metric(
                "Regional Support",
                f"{regional_icon} {regional_support}",
            )


        # ==================================================
        # SAFETY MESSAGE
        # ==================================================

        if (
            top_confidence >= 70
            and state_crop_count >= 30
        ):

            st.success(
                "✅ This prediction has relatively strong "
                "model confidence and reasonable regional "
                "training support."
            )

        elif (
            top_confidence >= 50
            and state_crop_count >= 10
        ):

            st.warning(
                "🟡 The model has moderate confidence and "
                "limited regional evidence. Review the "
                "alternative crops before planting."
            )

        else:

            st.error(
                "🔴 Limited evidence for this recommendation."
            )

            st.info(
                f"The model has {top_confidence:.1f}% confidence "
                f"and there are {state_crop_count} training "
                f"records for {top_crop} in {state}. "
                "Treat this result as guidance and verify "
                "local agricultural conditions before planting."
            )


        # ==================================================
        # TOP 3
        # ==================================================

        st.header(
            "🏆 Top 3 Recommendations"
        )

        medals = [
            "🥇",
            "🥈",
            "🥉",
        ]

        result_columns = st.columns(3)

        for position, (
            crop,
            confidence,
        ) in enumerate(top3):

            with result_columns[position]:

                with st.container(
                    border=True
                ):

                    st.subheader(
                        f"{medals[position]} "
                        f"{crop_emoji(crop)} "
                        f"{crop}"
                    )

                    st.metric(
                        "Model score",
                        f"{confidence:.2f}%",
                    )

                    st.progress(
                        min(
                            confidence / 100,
                            1.0,
                        )
                    )


        # ==================================================
        # REGIONAL EVIDENCE TABLE
        # ==================================================

        st.header(
            "📍 Recommendation Evidence"
        )

        evidence_rows = []

        for crop, confidence in top3:

            state_count = len(
                df[
                    (df["State_Name"].astype(str) == state)
                    & (df["Crop"].astype(str) == str(crop))
                ]
            )

            overall_count = len(
                df[
                    df["Crop"].astype(str) == str(crop)
                ]
            )

            if state_count >= 30:
                support = "Good"
            elif state_count >= 10:
                support = "Limited"
            else:
                support = "Very limited"

            evidence_rows.append(
                {
                    "Crop": crop,
                    "Model Score (%)": round(
                        confidence,
                        2,
                    ),
                    "State Records": state_count,
                    "Overall Records": overall_count,
                    "Regional Support": support,
                }
            )

        evidence_df = pd.DataFrame(
            evidence_rows
        )

        st.dataframe(
            evidence_df,
            use_container_width=True,
            hide_index=True,
        )


        # ==================================================
        # INPUT SUMMARY
        # ==================================================

        st.header(
            "📋 Input Summary"
        )

        summary = pd.DataFrame(
            {
                "Parameter": [
                    "State",
                    "Agro-Climatic Zone",
                    "Soil Type",
                    "pH",
                    "Nitrogen",
                    "Phosphorus",
                    "Potassium",
                    "Electrical Conductivity",
                    "Organic Carbon",
                    "Soil Moisture",
                    "Rainfall",
                    "Temperature",
                    "Humidity",
                    "Zinc",
                    "Iron",
                    "Manganese",
                    "Copper",
                    "Boron",
                    "Sulphur",
                ],
                "Value": [
                    str(state),
                    str(zone),
                    str(soil_type),
                    f"{ph:.2f}",
                    f"{nitrogen:.2f}",
                    f"{phosphorus:.2f}",
                    f"{potassium:.2f}",
                    f"{ec:.2f}",
                    f"{organic:.2f}",
                    f"{moisture:.2f}",
                    f"{rainfall:.2f}",
                    f"{temperature:.2f}",
                    f"{humidity:.2f}",
                    f"{zinc:.2f}",
                    f"{iron:.2f}",
                    f"{manganese:.2f}",
                    f"{copper:.2f}",
                    f"{boron:.2f}",
                    f"{sulphur:.2f}",
                ],
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )


        # ==================================================
        # DISCLAIMER
        # ==================================================

        st.info(
            "🌾 This recommendation is generated by the "
            "current machine-learning model. It is not a "
            "guarantee of yield, profitability, or crop "
            "success. Consider local soil conditions, "
            "irrigation, season, weather and qualified "
            "agricultural advice before planting."
        )


    except Exception as error:

        st.error(
            "The crop recommendation could not be generated."
        )

        st.exception(error)


# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.divider()

with st.expander(
    "ℹ️ About the AI model"
):

    st.write(
        "Smart Kisan Model V1 uses the production "
        "Random Forest classifier trained on the "
        "project's agricultural dataset."
    )

    model_col1, model_col2, model_col3 = (
        st.columns(3)
    )

    model_col1.metric(
        "Test Accuracy",
        "93.53%",
    )

    model_col2.metric(
        "Top-3 Accuracy",
        "98.14%",
    )

    model_col3.metric(
        "Test Records",
        "1,129",
    )

    st.caption(
        "These are held-out evaluation results. "
        "They are not a guarantee of prediction accuracy "
        "for every individual farm."
    )
