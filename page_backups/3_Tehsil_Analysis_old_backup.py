from pathlib import Path
import warnings

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Tehsil Analysis",
    page_icon="📍",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "output" / "Crop_Normalized.xlsx"

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    data = pd.read_excel(DATA_FILE)

    required = [
        "State_Name",
        "District_Name",
        "Tehsil_Name",
        "Crop",
        "State_Latitude",
        "State_Longitude",
    ]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            "Dataset is missing columns: "
            + ", ".join(missing)
        )

    return data


try:
    df = load_data()
except Exception as e:
    st.error("Unable to load the crop dataset.")
    st.exception(e)
    st.stop()

# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():
    """
    Search common model locations automatically.
    Returns None if a compatible model cannot be found.
    """

    try:
        import joblib
    except ImportError:
        return None

    search_dirs = [
        BASE_DIR / "models",
        BASE_DIR / "model",
        BASE_DIR / "output",
        BASE_DIR,
    ]

    candidates = []

    extensions = [
        "*.joblib",
        "*.pkl",
        "*.pickle",
        "*.sav",
    ]

    for directory in search_dirs:
        if not directory.exists():
            continue

        for ext in extensions:
            candidates.extend(directory.glob(ext))

    # Prefer files whose names look like crop models
    candidates = sorted(
        set(candidates),
        key=lambda p: (
            0 if any(
                x in p.name.lower()
                for x in ["crop", "random", "forest", "model"]
            ) else 1,
            len(str(p)),
        ),
    )

    for model_file in candidates:
        try:
            model = joblib.load(model_file)

            if hasattr(model, "predict"):
                return model

        except Exception:
            continue

    return None


model = load_model()

# ============================================================
# HELPERS
# ============================================================

FEATURE_COLUMNS = [
    "Soil_Type",
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
    "State_Name",
    "Agro_Climatic Zone",
]


def safe_mode(series):
    values = series.dropna().astype(str)

    if values.empty:
        return "Unknown"

    return values.mode().iloc[0]


def crop_symbol(crop):
    symbols = {
        "Rice": "🌾",
        "Wheat": "🌾",
        "Maize": "🌽",
        "Cotton": "Cotton",
        "Mustard": "Mustard",
        "Pulses": "Pulses",
        "Vegetables": "Vegetables",
        "Apple": "Apple",
        "Walnut": "Walnut",
        "Sugarcane": "Sugarcane",
        "Potato": "Potato",
        "Barley": "Barley",
        "Coconut": "Coconut",
        "Rajma": "Rajma",
        "Basmati Rice": "🌾",
    }

    return symbols.get(str(crop), "Crop")


def crop_color(crop):
    colors = {
        "Rice": "green",
        "Wheat": "gold",
        "Maize": "orange",
        "Cotton": "lightblue",
        "Mustard": "beige",
        "Pulses": "darkgreen",
        "Vegetables": "lime",
        "Apple": "red",
        "Walnut": "darkred",
        "Sugarcane": "cadetblue",
        "Potato": "gray",
        "Barley": "lightgreen",
        "Coconut": "darkgreen",
        "Rajma": "purple",
        "Basmati Rice": "green",
    }

    return colors.get(str(crop), "blue")


def make_prediction(tehsil_df):
    """
    Aggregate one tehsil into one representative ML input.
    """

    if model is None:
        return None, None, "Model not found"

    row = {}

    # Categorical soil
    row["Soil_Type"] = safe_mode(tehsil_df["Soil_Type"])

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

    for column in numeric_columns:
        if column in tehsil_df.columns:
            values = pd.to_numeric(
                tehsil_df[column],
                errors="coerce",
            )

            value = values.mean()

            if pd.isna(value):
                value = 0.0

            row[column] = float(value)
        else:
            row[column] = 0.0

    row["State_Name"] = str(tehsil_df["State_Name"].iloc[0])

    if "Agro_Climatic Zone" in tehsil_df.columns:
        row["Agro_Climatic Zone"] = safe_mode(
            tehsil_df["Agro_Climatic Zone"]
        )
    else:
        row["Agro_Climatic Zone"] = "Unknown"

    X = pd.DataFrame([row])

    try:
        prediction = model.predict(X)[0]

        confidence = None

        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(X)[0]
                confidence = float(probabilities.max() * 100)
            except Exception:
                confidence = None

        return str(prediction), confidence, None

    except Exception as e:
        return None, None, str(e)


# ============================================================
# HEADER
# ============================================================

st.title("Tehsil-wise Crop Analysis")

st.write(
    "Select a state and explore crop recommendations for "
    "its tehsils on an interactive map."
)

tab1, tab2 = st.tabs(
    [
        "Tehsil Lookup",
        "State Map",
    ]
)

# ============================================================
# TAB 1 - TEHSIL LOOKUP
# ============================================================

with tab1:

    st.subheader("Tehsil Lookup")

    states = sorted(
        df["State_Name"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_state = st.selectbox(
        "Select State",
        states,
    )

    state_df = df[
        df["State_Name"].astype(str) == selected_state
    ].copy()

    districts = sorted(
        state_df["District_Name"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_district = st.selectbox(
        "Select District",
        districts,
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
        tehsils,
    )

    tehsil_df = district_df[
        district_df["Tehsil_Name"].astype(str)
        == selected_tehsil
    ].copy()

    if not tehsil_df.empty:

        st.subheader("Observed Crop Information")

        crop_counts = (
            tehsil_df["Crop"]
            .dropna()
            .astype(str)
            .value_counts()
        )

        if not crop_counts.empty:
            observed_crop = crop_counts.index[0]

            st.success(
                f"Most recorded crop: {crop_symbol(observed_crop)} "
                f"{observed_crop}"
            )

            st.dataframe(
                crop_counts.rename("Records")
                .reset_index()
                .rename(columns={"index": "Crop"}),
                use_container_width=True,
                hide_index=True,
            )

        if st.button(
            "Generate Tehsil Recommendation",
            use_container_width=True,
            type="primary",
        ):

            prediction, confidence, error = make_prediction(
                tehsil_df
            )

            if prediction:

                st.subheader("ML Recommendation")

                if confidence is not None:
                    st.success(
                        f"Recommended crop: "
                        f"{crop_symbol(prediction)} "
                        f"**{prediction}**"
                    )

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.1f}%",
                    )
                else:
                    st.success(
                        f"Recommended crop: "
                        f"{crop_symbol(prediction)} "
                        f"**{prediction}**"
                    )

            else:
                st.warning(
                    "ML prediction could not be generated."
                )

                if error:
                    st.code(error)


# ============================================================
# TAB 2 - STATE MAP
# ============================================================

with tab2:

    st.subheader("Crop Recommendation Map")

    st.write(
        "Select a state to generate a map showing "
        "recommendations for its tehsils."
    )

    map_states = sorted(
        df["State_Name"]
        .dropna()
        .astype(str)
        .unique()
    )

    map_state = st.selectbox(
        "Select State for Map",
        map_states,
        key="state_map_selector",
    )

    generate_map = st.button(
        "Generate State Map",
        key="generate_state_map",
        use_container_width=True,
        type="primary",
    )

    if generate_map:

        with st.spinner(
            f"Generating crop recommendations for {map_state}..."
        ):

            map_df = df[
                df["State_Name"].astype(str) == map_state
            ].copy()

            if map_df.empty:
                st.error(
                    f"No records found for {map_state}."
                )
                st.stop()

            # ------------------------------------------------
            # VALID COORDINATES
            # ------------------------------------------------

            map_df["State_Latitude"] = pd.to_numeric(
                map_df["State_Latitude"],
                errors="coerce",
            )

            map_df["State_Longitude"] = pd.to_numeric(
                map_df["State_Longitude"],
                errors="coerce",
            )

            map_df = map_df.dropna(
                subset=[
                    "State_Latitude",
                    "State_Longitude",
                ]
            )

            if map_df.empty:
                st.error(
                    "This state has no valid latitude/longitude "
                    "coordinates in Crop_Normalized.xlsx."
                )
                st.stop()

            # ------------------------------------------------
            # FIND STATE CENTER
            # ------------------------------------------------

            center_lat = float(
                map_df["State_Latitude"].mean()
            )

            center_lon = float(
                map_df["State_Longitude"].mean()
            )

            # ------------------------------------------------
            # GENERATE ONE RESULT PER TEHSIL
            # ------------------------------------------------

            results = []
            prediction_errors = []

            grouped = map_df.groupby(
                "Tehsil_Name",
                dropna=True,
            )

            for tehsil, tehsil_data in grouped:

                tehsil = str(tehsil)

                lat = pd.to_numeric(
                    tehsil_data["State_Latitude"],
                    errors="coerce",
                ).mean()

                lon = pd.to_numeric(
                    tehsil_data["State_Longitude"],
                    errors="coerce",
                ).mean()

                if pd.isna(lat) or pd.isna(lon):
                    continue

                # Try ML prediction
                prediction, confidence, error = make_prediction(
                    tehsil_data
                )

                # IMPORTANT:
                # Never silently discard a tehsil.
                # If ML fails, use observed crop as fallback.
                if prediction is None:

                    crop_counts = (
                        tehsil_data["Crop"]
                        .dropna()
                        .astype(str)
                        .value_counts()
                    )

                    if not crop_counts.empty:
                        prediction = crop_counts.index[0]
                    else:
                        prediction = "Unknown"

                    confidence = None

                    if error:
                        prediction_errors.append(
                            {
                                "Tehsil": tehsil,
                                "Error": error,
                            }
                        )

                results.append(
                    {
                        "Tehsil": tehsil,
                        "Crop": prediction,
                        "Confidence": confidence,
                        "Latitude": float(lat),
                        "Longitude": float(lon),
                        "Records": len(tehsil_data),
                    }
                )

            # ------------------------------------------------
            # CHECK RESULTS
            # ------------------------------------------------

            if not results:
                st.error(
                    "No tehsil results could be generated."
                )
                st.stop()

            results_df = pd.DataFrame(results)

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.success(
                f"Generated recommendations for "
                f"{len(results_df)} tehsils in {map_state}."
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Tehsils",
                len(results_df),
            )

            c2.metric(
                "Unique Crops",
                results_df["Crop"].nunique(),
            )

            confidence_values = pd.to_numeric(
                results_df["Confidence"],
                errors="coerce",
            ).dropna()

            if not confidence_values.empty:
                c3.metric(
                    "Average Confidence",
                    f"{confidence_values.mean():.1f}%",
                )
            else:
                c3.metric(
                    "Average Confidence",
                    "N/A",
                )

            c4.metric(
                "Dataset Records",
                len(map_df),
            )

            # ------------------------------------------------
            # CREATE MAP
            # ------------------------------------------------

            crop_map = folium.Map(
                location=[
                    center_lat,
                    center_lon,
                ],
                zoom_start=6,
                tiles="CartoDB positron",
                control_scale=True,
            )

            # Marker cluster makes overlapping state-level
            # coordinates usable.
            from folium.plugins import MarkerCluster

            marker_cluster = MarkerCluster(
                name="Tehsil Recommendations"
            ).add_to(crop_map)

            for _, row in results_df.iterrows():

                crop = str(row["Crop"])

                confidence_text = (
                    f"{float(row['Confidence']):.1f}%"
                    if pd.notna(row["Confidence"])
                    else "N/A"
                )

                popup_html = f"""
                <div style="width:230px">
                    <h4>{row['Tehsil']}</h4>
                    <hr>
                    <b>Recommended Crop:</b><br>
                    {crop}
                    <br><br>
                    <b>Confidence:</b><br>
                    {confidence_text}
                    <br><br>
                    <b>Records:</b><br>
                    {int(row['Records'])}
                </div>
                """

                folium.CircleMarker(
                    location=[
                        float(row["Latitude"]),
                        float(row["Longitude"]),
                    ],
                    radius=9,
                    color=crop_color(crop),
                    fill=True,
                    fill_color=crop_color(crop),
                    fill_opacity=0.85,
                    weight=2,
                    tooltip=(
                        f"{row['Tehsil']} - {crop}"
                    ),
                    popup=folium.Popup(
                        popup_html,
                        max_width=280,
                    ),
                ).add_to(marker_cluster)

            # Fit map to all available points
            bounds = [
                [
                    float(results_df["Latitude"].min()),
                    float(results_df["Longitude"].min()),
                ],
                [
                    float(results_df["Latitude"].max()),
                    float(results_df["Longitude"].max()),
                ],
            ]

            # If all coordinates are identical, don't fit bounds
            # because Folium may zoom excessively.
            if (
                results_df["Latitude"].nunique() > 1
                or results_df["Longitude"].nunique() > 1
            ):
                crop_map.fit_bounds(bounds)

            folium.LayerControl().add_to(crop_map)

            # ------------------------------------------------
            # DISPLAY MAP
            # ------------------------------------------------

            st.subheader(
                f"{map_state} - Crop Recommendation Map"
            )

            st_folium(
                crop_map,
                width="100%",
                height=650,
                returned_objects=[],
                key=f"map_{map_state}",
            )

            # ------------------------------------------------
            # RESULTS TABLE
            # ------------------------------------------------

            st.subheader(
                "Tehsil-wise Recommendations"
            )

            display_df = results_df.copy()

            display_df["Confidence (%)"] = display_df[
                "Confidence"
            ].apply(
                lambda x:
                    f"{float(x):.1f}%"
                    if pd.notna(x)
                    else "N/A"
            )

            display_df = display_df[
                [
                    "Tehsil",
                    "Crop",
                    "Confidence (%)",
                    "Records",
                ]
            ].rename(
                columns={
                    "Crop": "Recommended Crop",
                }
            )

            st.dataframe(
                display_df.sort_values(
                    "Recommended Crop"
                ),
                use_container_width=True,
                hide_index=True,
            )

            # ------------------------------------------------
            # CROP DISTRIBUTION
            # ------------------------------------------------

            st.subheader("Crop Distribution")

            distribution = (
                results_df["Crop"]
                .value_counts()
            )

            st.bar_chart(distribution)

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            csv_data = display_df.to_csv(
                index=False
            )

            st.download_button(
                "Download Recommendations CSV",
                csv_data,
                file_name=(
                    f"{map_state}_"
                    f"tehsil_recommendations.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

            # ------------------------------------------------
            # MODEL WARNING
            # ------------------------------------------------

            if model is None:

                st.warning(
                    "ML model was not automatically found. "
                    "The map is therefore using the most frequently "
                    "recorded crop for each tehsil."
                )

            elif prediction_errors:

                with st.expander(
                    "View prediction warnings"
                ):

                    st.write(
                        f"{len(prediction_errors)} tehsils "
                        "used observed-crop fallback."
                    )

                    st.dataframe(
                        pd.DataFrame(
                            prediction_errors
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Kisan Tehsil Analysis | "
    "Random Forest Classifier | "
    "93.53% test accuracy"
)