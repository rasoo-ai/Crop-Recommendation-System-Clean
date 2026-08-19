import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Tehsil-wise Crop Analysis - Smart Kisan",
    page_icon="≡ƒôì",
    layout="wide",
)

# ==========================================================
# CSS
# ==========================================================

st.markdown("""
<style>
.stApp { background-color: #f7faf7; }
.block-container { max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL AND DATA
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("output/crop_prediction_model_balanced.pkl")

@st.cache_data
def load_data():
    return pd.read_excel("output/Crop_Normalized.xlsx")

model = load_model()
df    = load_data()

# ==========================================================
# NUMERIC COLS
# ==========================================================

numeric_cols = [
    "pH_Value", "Nitrogen_Value (N)", "Phosphorus_Value (P)",
    "Potassium_Value (K)", "Electrical_Conductivity (EC)",
    "Organic_Carbon (%)", "Soil_Moisture (%)", "Zinc (%)",
    "Iron (%)", "Manganese (%)", "Copper (%)", "Boron (%)",
    "Sulphur (%)", "Rainfall_cm", "temperature_celsius",
    "humidity_percentage",
]

# Pre-compute overall averages for fallback
overall_numeric  = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
overall_averages = overall_numeric.mean(skipna=True)

# ==========================================================
# CROP EMOJI + COLORS
# ==========================================================

CROP_EMOJI = {
    "Rice": "≡ƒî╛", "Wheat": "≡ƒî┐", "Maize": "≡ƒî╜",
    "Cotton": "≡ƒî╕", "Mustard": "≡ƒî╗", "Pulses": "≡ƒ½ÿ",
    "Vegetables": "≡ƒÑª", "Apple": "≡ƒìÄ", "Walnut": "≡ƒî░",
    "Sugarcane": "≡ƒÄï", "Potato": "≡ƒÑö", "Barley": "≡ƒî╛",
    "Coconut": "≡ƒÑÑ", "Rajma": "≡ƒ½ÿ", "Basmati Rice": "≡ƒî╛",
}

CROP_COLOR = {
    "Rice": "#2196F3", "Wheat": "#FF9800", "Maize": "#FFEB3B",
    "Cotton": "#E91E63", "Mustard": "#FFC107", "Pulses": "#8BC34A",
    "Vegetables": "#4CAF50", "Apple": "#F44336", "Walnut": "#795548",
    "Sugarcane": "#00BCD4", "Coconut": "#FF5722", "Barley": "#9C27B0",
}

def crop_emoji(crop):
    return CROP_EMOJI.get(crop, "≡ƒî▒")

def crop_color(crop):
    return CROP_COLOR.get(crop, "#607D8B")

# ==========================================================
# HEADER
# ==========================================================

st.title("≡ƒôì Tehsil-wise Crop Analysis")
st.write(
    "Select a state, district, and tehsil to view "
    "observed crop information and ML recommendations. "
    "Use the map tab to explore all tehsils visually."
)

# ==========================================================
# TABS
# ==========================================================

tab1, tab2 = st.tabs(["≡ƒöì Tehsil Lookup", "≡ƒù║∩╕Å State Map"])

# ==========================================================
# TAB 1 ΓÇö TEHSIL LOOKUP
# ==========================================================

with tab1:

    states   = sorted(df["State_Name"].dropna().astype(str).unique())
    loc1, loc2, loc3 = st.columns(3)

    with loc1:
        selected_state = st.selectbox("Select State", states)

    state_df = df[df["State_Name"].astype(str) == selected_state].copy()
    districts = sorted(state_df["District_Name"].dropna().astype(str).unique())

    with loc2:
        selected_district = st.selectbox("Select District", districts)

    district_df = state_df[state_df["District_Name"].astype(str) == selected_district].copy()
    tehsils = sorted(district_df["Tehsil_Name"].dropna().astype(str).unique())

    with loc3:
        selected_tehsil = st.selectbox("Select Tehsil", tehsils)

    st.divider()

    if st.button("≡ƒôì Generate Tehsil Recommendation", use_container_width=True, type="primary"):

        tehsil_df = district_df[
            district_df["Tehsil_Name"].astype(str) == selected_tehsil
        ].copy()

        if tehsil_df.empty:
            st.error("No data found for the selected tehsil.")
            st.stop()

        # --------------------------------------------------
        # OBSERVED CROPS
        # --------------------------------------------------

        st.subheader("≡ƒî╛ Observed Crop Information")

        tehsil_df["ha"] = pd.to_numeric(tehsil_df["ha"], errors="coerce")
        observed = (
            tehsil_df.groupby("Crop", dropna=True)["ha"]
            .sum().sort_values(ascending=False)
        )

        if observed.empty:
            st.info("No crop-area information available for this tehsil.")
        else:
            top_obs_crop = observed.index[0]
            top_obs_area = observed.iloc[0]
            st.success(f"Major recorded crop: **{crop_emoji(top_obs_crop)} {top_obs_crop}** ({top_obs_area:,.2f} ha)")

            obs_table = observed.reset_index().rename(columns={"ha": "Recorded Area (ha)"})
            obs_table["Recorded Area (ha)"] = obs_table["Recorded Area (ha)"].round(2)
            st.dataframe(obs_table, use_container_width=True, hide_index=True)

        # --------------------------------------------------
        # COMPUTE AVERAGES
        # --------------------------------------------------

        tehsil_numeric = tehsil_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        averages = tehsil_numeric.mean(skipna=True).fillna(overall_averages)

        soil_mode = tehsil_df["Soil_Type"].dropna().astype(str).mode()
        representative_soil = soil_mode.iloc[0] if not soil_mode.empty else "Unknown"

        zone_mode = tehsil_df["Agro_Climatic Zone"].dropna().astype(str).mode()
        representative_zone = zone_mode.iloc[0] if not zone_mode.empty else "Unknown"

        # --------------------------------------------------
        # MODEL INPUT
        # --------------------------------------------------

        sample = pd.DataFrame([{
            "Soil_Type":                    representative_soil,
            "pH_Value":                     float(averages["pH_Value"]),
            "Nitrogen_Value (N)":           float(averages["Nitrogen_Value (N)"]),
            "Phosphorus_Value (P)":         float(averages["Phosphorus_Value (P)"]),
            "Potassium_Value (K)":          float(averages["Potassium_Value (K)"]),
            "Electrical_Conductivity (EC)": float(averages["Electrical_Conductivity (EC)"]),
            "Organic_Carbon (%)":           float(averages["Organic_Carbon (%)"]),
            "Soil_Moisture (%)":            float(averages["Soil_Moisture (%)"]),
            "Zinc (%)":                     float(averages["Zinc (%)"]),
            "Iron (%)":                     float(averages["Iron (%)"]),
            "Manganese (%)":                float(averages["Manganese (%)"]),
            "Copper (%)":                   float(averages["Copper (%)"]),
            "Boron (%)":                    float(averages["Boron (%)"]),
            "Sulphur (%)":                  float(averages["Sulphur (%)"]),
            "Rainfall_cm":                  float(averages["Rainfall_cm"]),
            "temperature_celsius":          float(averages["temperature_celsius"]),
            "humidity_percentage":          float(averages["humidity_percentage"]),
            "State_Name":                   selected_state,
            "Agro_Climatic Zone":           representative_zone,
        }])

        try:
            prediction    = model.predict(sample)[0]
            probabilities = model.predict_proba(sample)[0]
            top3_idx      = probabilities.argsort()[-3:][::-1]
            top3          = [(model.classes_[i], float(probabilities[i]*100)) for i in top3_idx]
            confidence    = top3[0][1]
            reliability   = "High" if confidence >= 70 else "Moderate" if confidence >= 50 else "Low"
            rel_icon      = "≡ƒƒó" if reliability == "High" else "≡ƒƒí" if reliability == "Moderate" else "≡ƒö┤"

            # Location summary
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("State",    selected_state)
            c2.metric("District", selected_district)
            c3.metric("Tehsil",   selected_tehsil)

            # Main result
            st.subheader("≡ƒñû ML Recommendation")

            if confidence >= 70:
                st.success(f"≡ƒƒó Recommended: **{crop_emoji(prediction)} {prediction}** ΓÇö High confidence")
            elif confidence >= 50:
                st.warning(f"≡ƒƒí Recommended: **{crop_emoji(prediction)} {prediction}** ΓÇö Moderate confidence")
            else:
                st.error(f"≡ƒö┤ Recommended: **{crop_emoji(prediction)} {prediction}** ΓÇö Low confidence")

            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction Confidence", f"{confidence:.2f}%")
            m2.metric("Reliability",           f"{rel_icon} {reliability}")
            m3.metric("Dataset Records",        len(tehsil_df))

            # Top 3
            st.divider()
            st.subheader("≡ƒÅå Top 3 Recommendations")
            medals = ["≡ƒÑç", "≡ƒÑê", "≡ƒÑë"]
            r1, r2, r3 = st.columns(3)
            for pos, (crop, conf) in enumerate(top3):
                with [r1, r2, r3][pos]:
                    with st.container(border=True):
                        st.subheader(f"{medals[pos]} {crop_emoji(crop)} {crop}")
                        st.metric("Confidence", f"{conf:.2f}%")
                        st.progress(min(conf/100, 1.0))

            # Conditions
            st.divider()
            st.subheader("≡ƒî▒ Representative Conditions")
            cond1, cond2 = st.columns(2)
            with cond1:
                st.markdown(f"**Soil Type:** {representative_soil}")
                st.markdown(f"**pH:** {averages['pH_Value']:.2f}")
                st.markdown(f"**Nitrogen:** {averages['Nitrogen_Value (N)']:.2f}")
                st.markdown(f"**Phosphorus:** {averages['Phosphorus_Value (P)']:.2f}")
                st.markdown(f"**Potassium:** {averages['Potassium_Value (K)']:.2f}")
            with cond2:
                st.markdown(f"**Rainfall:** {averages['Rainfall_cm']:.2f} cm")
                st.markdown(f"**Temperature:** {averages['temperature_celsius']:.2f} ┬░C")
                st.markdown(f"**Humidity:** {averages['humidity_percentage']:.2f}%")
                st.markdown(f"**Agro Zone:** {representative_zone}")

            # Dataset vs Model
            st.divider()
            st.subheader("≡ƒöÄ Dataset Evidence vs Model")
            if not observed.empty:
                obs_crop = observed.index[0]
                if obs_crop == prediction:
                    st.success(f"Γ£à ML recommendation (**{prediction}**) matches the major recorded crop.")
                else:
                    st.warning(f"ΓÜá∩╕Å ML recommendation (**{prediction}**) differs from recorded crop (**{obs_crop}**).")

            st.info(
                "Recommendation based on available dataset and representative "
                "environmental conditions. Actual suitability varies with "
                "irrigation, season, and local practices."
            )
            st.caption(f"Analysis based on {len(tehsil_df)} record(s) for this tehsil.")

        except Exception as error:
            st.error("Unable to generate the tehsil recommendation.")
            st.exception(error)

# ==========================================================
# TAB 2 ΓÇö STATE MAP
# ==========================================================

with tab2:

    st.subheader("≡ƒù║∩╕Å Crop Recommendation Map")
    st.write("Select a state to see ML crop recommendations for all its tehsils on an interactive map.")

    map_states = sorted(df["State_Name"].dropna().astype(str).unique())
    map_state  = st.selectbox("Select State for Map", map_states, key="map_state")

    if st.button("≡ƒù║∩╕Å Generate State Map", use_container_width=True, type="primary"):

        try:
            import folium
            from streamlit_folium import st_folium

            map_df = df[df["State_Name"].astype(str) == map_state].copy()

            if map_df.empty:
                st.error("No data for this state.")
                st.stop()

            # Check if lat/lon available
            has_coords = (
                "State_Latitude" in map_df.columns and
                "State_Longitude" in map_df.columns and
                map_df["State_Latitude"].notna().any()
            )

            if not has_coords:
                st.warning("No GPS coordinates in dataset. Showing summary table instead.")

                # Fallback ΓÇö show table
                results = []
                for tehsil in map_df["Tehsil_Name"].dropna().unique()[:50]:
                    t_df = map_df[map_df["Tehsil_Name"].astype(str) == tehsil]
                    t_num = t_df[numeric_cols].apply(pd.to_numeric, errors="coerce").mean(skipna=True).fillna(overall_averages)
                    soil = t_df["Soil_Type"].mode()[0] if not t_df["Soil_Type"].mode().empty else "Unknown"
                    zone = t_df["Agro_Climatic Zone"].mode()[0] if not t_df["Agro_Climatic Zone"].mode().empty else "Unknown"
                    try:
                        s = pd.DataFrame([{
                            "Soil_Type": soil,
                            "pH_Value": float(t_num["pH_Value"]),
                            "Nitrogen_Value (N)": float(t_num["Nitrogen_Value (N)"]),
                            "Phosphorus_Value (P)": float(t_num["Phosphorus_Value (P)"]),
                            "Potassium_Value (K)": float(t_num["Potassium_Value (K)"]),
                            "Electrical_Conductivity (EC)": float(t_num["Electrical_Conductivity (EC)"]),
                            "Organic_Carbon (%)": float(t_num["Organic_Carbon (%)"]),
                            "Soil_Moisture (%)": float(t_num["Soil_Moisture (%)"]),
                            "Zinc (%)": float(t_num["Zinc (%)"]),
                            "Iron (%)": float(t_num["Iron (%)"]),
                            "Manganese (%)": float(t_num["Manganese (%)"]),
                            "Copper (%)": float(t_num["Copper (%)"]),
                            "Boron (%)": float(t_num["Boron (%)"]),
                            "Sulphur (%)": float(t_num["Sulphur (%)"]),
                            "Rainfall_cm": float(t_num["Rainfall_cm"]),
                            "temperature_celsius": float(t_num["temperature_celsius"]),
                            "humidity_percentage": float(t_num["humidity_percentage"]),
                            "State_Name": map_state,
                            "Agro_Climatic Zone": zone,
                        }])
                        pred  = model.predict(s)[0]
                        proba = model.predict_proba(s)[0]
                        conf  = round(float(proba.max() * 100), 1)
                        results.append({"Tehsil": tehsil, "Recommended Crop": pred, "Confidence (%)": conf, "Soil Type": soil})
                    except Exception:
                        continue

                if results:
                    res_df = pd.DataFrame(results).sort_values("Confidence (%)", ascending=False)
                    st.success(f"Generated recommendations for {len(res_df)} tehsils in {map_state}")

                    # Summary metrics
                    sm1, sm2, sm3 = st.columns(3)
                    sm1.metric("Tehsils Analysed", len(res_df))
                    sm2.metric("Unique Crops",     res_df["Recommended Crop"].nunique())
                    sm3.metric("Avg Confidence",   f"{res_df['Confidence (%)'].mean():.1f}%")

                    # Crop distribution
                    st.markdown("### ≡ƒî╛ Crop Distribution")
                    st.bar_chart(res_df["Recommended Crop"].value_counts())

                    # Full table
                    st.markdown("### ≡ƒôï Tehsil-wise Recommendations")
                    st.dataframe(res_df, use_container_width=True, hide_index=True)

                    # Download
                    st.download_button(
                        "≡ƒôÑ Download as CSV",
                        res_df.to_csv(index=False),
                        file_name=f"{map_state}_tehsil_recommendations.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            else:
                # Has coordinates ΓÇö show Folium map
                results = []
                for tehsil in map_df["Tehsil_Name"].dropna().unique():
                    t_df = map_df[map_df["Tehsil_Name"].astype(str) == tehsil]
                    t_num = t_df[numeric_cols].apply(pd.to_numeric, errors="coerce").mean(skipna=True).fillna(overall_averages)
                    soil  = t_df["Soil_Type"].mode()[0] if not t_df["Soil_Type"].mode().empty else "Unknown"
                    zone  = t_df["Agro_Climatic Zone"].mode()[0] if not t_df["Agro_Climatic Zone"].mode().empty else "Unknown"
                    lat   = t_df["State_Latitude"].mean()
                    lon   = t_df["State_Longitude"].mean()
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    try:
                        s = pd.DataFrame([{
                            "Soil_Type": soil,
                            "pH_Value": float(t_num["pH_Value"]),
                            "Nitrogen_Value (N)": float(t_num["Nitrogen_Value (N)"]),
                            "Phosphorus_Value (P)": float(t_num["Phosphorus_Value (P)"]),
                            "Potassium_Value (K)": float(t_num["Potassium_Value (K)"]),
                            "Electrical_Conductivity (EC)": float(t_num["Electrical_Conductivity (EC)"]),
                            "Organic_Carbon (%)": float(t_num["Organic_Carbon (%)"]),
                            "Soil_Moisture (%)": float(t_num["Soil_Moisture (%)"]),
                            "Zinc (%)": float(t_num["Zinc (%)"]),
                            "Iron (%)": float(t_num["Iron (%)"]),
                            "Manganese (%)": float(t_num["Manganese (%)"]),
                            "Copper (%)": float(t_num["Copper (%)"]),
                            "Boron (%)": float(t_num["Boron (%)"]),
                            "Sulphur (%)": float(t_num["Sulphur (%)"]),
                            "Rainfall_cm": float(t_num["Rainfall_cm"]),
                            "temperature_celsius": float(t_num["temperature_celsius"]),
                            "humidity_percentage": float(t_num["humidity_percentage"]),
                            "State_Name": map_state,
                            "Agro_Climatic Zone": zone,
                        }])
                        pred  = model.predict(s)[0]
                        proba = model.predict_proba(s)[0]
                        conf  = round(float(proba.max() * 100), 1)
                        results.append({"tehsil": tehsil, "crop": pred, "confidence": conf, "lat": lat, "lon": lon})
                    except Exception:
                        continue

                if results:
                    res_df = pd.DataFrame(results)
                    center_lat = res_df["lat"].mean()
                    center_lon = res_df["lon"].mean()

                    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")

                    for _, row in res_df.iterrows():
                        color = crop_color(row["crop"])
                        folium.CircleMarker(
                            location=[row["lat"], row["lon"]],
                            radius=10,
                            color=color,
                            fill=True,
                            fill_color=color,
                            fill_opacity=0.8,
                            popup=folium.Popup(
                                f"<b>{row['tehsil']}</b><br>"
                                f"Crop: {row['crop']}<br>"
                                f"Confidence: {row['confidence']}%",
                                max_width=200,
                            ),
                            tooltip=f"{row['tehsil']}: {row['crop']} ({row['confidence']}%)",
                        ).add_to(m)

                    st_folium(m, width=1200, height=600)

                    st.markdown("### ≡ƒôï Tehsil Recommendations")
                    display_df = res_df.rename(columns={"tehsil":"Tehsil","crop":"Recommended Crop","confidence":"Confidence (%)"}).drop(columns=["lat","lon"])
                    st.dataframe(display_df.sort_values("Confidence (%)", ascending=False), use_container_width=True, hide_index=True)

                    st.download_button(
                        "≡ƒôÑ Download as CSV",
                        display_df.to_csv(index=False),
                        file_name=f"{map_state}_map_recommendations.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        except ImportError:
            st.warning("Map libraries not installed. Showing table view instead.")
            st.code("pip install folium streamlit-folium")
        except Exception as e:
            st.error(f"Map generation failed: {e}")
            st.exception(e)

# ==========================================================
# FOOTER
# ==========================================================

st.divider()
st.caption("≡ƒôì Smart Kisan Tehsil Analysis | Random Forest Classifier | 93.53% test accuracy")
