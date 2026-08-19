import pandas as pd
import streamlit as st

from smart_kisan_ui import configure_page, footer, sidebar


# =========================================================
# PAGE
# =========================================================

configure_page("Tehsil Analysis", "📍")
sidebar()


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_excel("output/Crop_Normalized.xlsx")


try:
    data = load_data()
except Exception as error:
    st.error("Could not load the project dataset.")
    st.exception(error)
    st.stop()


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required = [
    "State_Name",
    "District_Name",
    "Tehsil_Name",
    "Crop",
]

missing = [column for column in required if column not in data.columns]

if missing:
    st.error("Required regional columns are missing from the dataset.")
    st.write("Missing columns:", missing)
    st.stop()


# Clean regional fields
data = data.dropna(subset=required).copy()

for column in required:
    data[column] = data[column].astype(str).str.strip()


# =========================================================
# PAGE HEADER
# =========================================================

st.caption("REGIONAL AGRICULTURAL INTELLIGENCE")

st.title("Tehsil Analysis")

st.write(
    "Explore crop patterns and agricultural records available "
    "in the Smart Kisan project dataset."
)

st.info(
    "This page presents descriptive insights from the offline "
    "project dataset. It is not a live GIS service or real-time "
    "agricultural advisory."
)


# =========================================================
# REGION SELECTION
# =========================================================

st.header("Select a region")

st.caption(
    "Choose a state, district, and tehsil to explore the available records."
)

col1, col2, col3 = st.columns(3)

with col1:
    states = sorted(data["State_Name"].unique())
    state = st.selectbox("State", states)

state_data = data[data["State_Name"] == state]

with col2:
    districts = sorted(state_data["District_Name"].unique())
    district = st.selectbox("District", districts)

district_data = state_data[
    state_data["District_Name"] == district
]

with col3:
    tehsils = sorted(district_data["Tehsil_Name"].unique())
    tehsil = st.selectbox("Tehsil", tehsils)

tehsil_data = district_data[
    district_data["Tehsil_Name"] == tehsil
].copy()


# =========================================================
# CROP COUNTS
# =========================================================

crop_counts = (
    tehsil_data["Crop"]
    .value_counts()
    .rename_axis("Crop")
    .reset_index(name="Records")
)

record_count = len(tehsil_data)

crop_count = tehsil_data["Crop"].nunique()

district_records = len(district_data)

district_crop_count = district_data["Crop"].nunique()

tehsil_count = district_data["Tehsil_Name"].nunique()

if crop_counts.empty:
    top_crop = "No data"
else:
    top_crop = str(crop_counts.iloc[0]["Crop"])


# =========================================================
# REGIONAL OVERVIEW
# =========================================================

st.header("Regional overview")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Agricultural records",
        f"{record_count:,}",
    )

with m2:
    st.metric(
        "Crop categories",
        f"{crop_count:,}",
    )

with m3:
    st.metric(
        "Top recorded crop",
        top_crop,
    )

with m4:
    st.metric(
        "Tehsils in district",
        f"{tehsil_count:,}",
    )


# =========================================================
# CROP INTELLIGENCE
# =========================================================

st.header("Crop intelligence")

st.subheader("Recorded crop distribution")

st.caption(
    "Number of records associated with each crop "
    "in the selected tehsil."
)

if crop_counts.empty:

    st.info("No crop records are available for this selection.")

else:

    # Simple native Streamlit chart.
    # No custom HTML.
    st.bar_chart(
        crop_counts.set_index("Crop")["Records"]
    )

    st.dataframe(
        crop_counts,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# TOP CROPS
# =========================================================

st.subheader("Most recorded crops")

if crop_counts.empty:

    st.info("No crop information is available.")

else:

    for position, row in enumerate(
        crop_counts.head(5).itertuples(index=False),
        start=1,
    ):

        st.write(
            f"**{position}. {row.Crop}** — "
            f"{int(row.Records):,} records"
        )


# =========================================================
# REGIONAL COVERAGE
# =========================================================

st.header("Regional coverage")

st.caption(
    "Understand how the selected tehsil compares "
    "with the surrounding district."
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Records in selected tehsil",
        f"{record_count:,}",
    )

with c2:
    st.metric(
        "Records in selected district",
        f"{district_records:,}",
    )

with c3:
    st.metric(
        "Crop categories in district",
        f"{district_crop_count:,}",
    )

with c4:

    if district_records > 0:
        share = record_count / district_records * 100
    else:
        share = 0

    st.metric(
        "Tehsil share of district records",
        f"{share:.1f}%",
    )

st.caption(
    "A larger number of records indicates greater representation "
    "in this dataset. It does not independently establish crop "
    "suitability, yield, profitability, or agricultural success."
)


# =========================================================
# AGRICULTURAL MEASUREMENTS
# =========================================================

st.header("Agricultural measurement snapshot")

st.caption(
    "Median values from available records for the selected tehsil."
)

measurement_columns = [
    "pH_Value",
    "Nitrogen_Value (N)",
    "Phosphorus_Value (P)",
    "Potassium_Value (K)",
    "Soil_Moisture (%)",
    "Rainfall_cm",
    "temperature_celsius",
    "humidity_percentage",
]

available_columns = [
    column
    for column in measurement_columns
    if column in tehsil_data.columns
]

if available_columns:

    numeric_data = tehsil_data[available_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    median_data = (
        numeric_data
        .median()
        .dropna()
        .round(2)
        .rename_axis("Measurement")
        .reset_index(name="Median value")
    )

    if median_data.empty:

        st.info(
            "No numeric measurement values are available "
            "for this tehsil."
        )

    else:

        st.dataframe(
            median_data,
            use_container_width=True,
            hide_index=True,
        )

else:

    st.info(
        "No agricultural measurement fields are available "
        "for this selection."
    )


# =========================================================
# REGIONAL RECORDS
# =========================================================

with st.expander("View regional records"):

    columns_to_show = [
        column
        for column in [
            "State_Name",
            "District_Name",
            "Tehsil_Name",
            "Crop",
            "pH_Value",
            "Nitrogen_Value (N)",
            "Phosphorus_Value (P)",
            "Potassium_Value (K)",
            "Soil_Moisture (%)",
            "Rainfall_cm",
            "temperature_celsius",
            "humidity_percentage",
        ]
        if column in tehsil_data.columns
    ]

    st.dataframe(
        tehsil_data[columns_to_show],
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# ABOUT
# =========================================================

st.header("About this analysis")

st.write(
    "Tehsil Analysis summarizes the regional records contained "
    "in the Smart Kisan project dataset. The results are descriptive "
    "and should not be interpreted as live field measurements or "
    "guaranteed crop recommendations."
)


# =========================================================
# FOOTER
# =========================================================

footer()