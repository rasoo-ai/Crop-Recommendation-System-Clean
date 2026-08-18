import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load("output/crop_prediction_model_balanced.pkl")

# Sample input
sample = pd.DataFrame([{
    "Soil_Type": "Loamy",
    "pH_Value": 6.8,
    "Nitrogen_Value (N)": 250,
    "Phosphorus_Value (P)": 35,
    "Potassium_Value (K)": 220,
    "Electrical_Conductivity (EC)": 0.4,
    "Organic_Carbon (%)": 0.75,
    "Soil_Moisture (%)": 25,
    "Zinc (%)": 0.8,
    "Iron (%)": 4.5,
    "Manganese (%)": 2.0,
    "Copper (%)": 0.4,
    "Boron (%)": 0.5,
    "Sulphur (%)": 12,
    "Rainfall_cm": 120,
    "temperature_celsius": 28,
    "humidity_percentage": 70,
    "State_Name": "Telangana",
    "Agro_Climatic Zone": "Deccan Plateau"
}])

# Predict Top 3
prob = model.predict_proba(sample)[0]

classes = model.classes_

top = np.argsort(prob)[::-1][:3]

print("=" * 60)
print("TOP 3 RECOMMENDED CROPS")
print("=" * 60)

for i in top:
    print(f"{classes[i]:20} {prob[i]*100:.2f}%")
