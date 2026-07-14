import pandas as pd

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_excel("output/Crop_Normalized.xlsx")

print("Original Shape:", df.shape)

# Remove rows where Crop is missing
df = df.dropna(subset=["Crop"])

# Features used by the model
features = [
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
    "Agro_Climatic Zone"
]

target = "Crop"

# Keep only required columns
df = df[features + [target]]

# Remove rows with missing values in these columns
df = df.dropna()

print("Shape After Cleaning:", df.shape)

X = df[features]
y = df[target]