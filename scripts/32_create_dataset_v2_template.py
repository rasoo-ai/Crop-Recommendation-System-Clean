import pandas as pd

SOURCE = "output/Crop_Normalized.xlsx"
OUTPUT = "output/Dataset_V2_New_Records_Template.xlsx"

df = pd.read_excel(SOURCE)

columns = [
    "Crop",
    "State_Name",
    "District_Name",
    "Tehsil_Name",
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
    "Agro_Climatic Zone",
]

template = pd.DataFrame(
    columns=[
        c for c in columns
        if c in df.columns
    ]
)

template.to_excel(
    OUTPUT,
    index=False
)

print("Dataset V2 template created:")
print(OUTPUT)

print("\nColumns:")
for c in template.columns:
    print("-", c)
