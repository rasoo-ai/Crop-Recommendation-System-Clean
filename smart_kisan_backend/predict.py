import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "output" / "crop_prediction_model_balanced.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"WARNING: Could not load model: {e}")

CROP_ADVICE = {
    "Rice":       "Ensure adequate water supply. Best in warm humid conditions with heavy rainfall.",
    "Wheat":      "Requires cool weather during growth. Ensure well-drained loamy soil.",
    "Maize":      "Needs well-drained fertile soil. Ensure adequate nitrogen fertilisation.",
    "Cotton":     "Thrives in black cotton soil with moderate rainfall. Avoid waterlogging.",
    "Mustard":    "Best in cool dry weather. Requires well-drained sandy loam soil.",
    "Pulses":     "Fix nitrogen naturally. Good for soil health. Needs moderate rainfall.",
    "Vegetables": "Requires fertile soil with good drainage. Regular irrigation recommended.",
    "Apple":      "Needs cold winters and cool summers. Best in hilly regions.",
    "Walnut":     "Requires deep fertile soil. Best in temperate climate regions.",
    "Sugarcane":  "Needs warm humid climate with heavy rainfall. Requires irrigation.",
    "Potato":     "Needs cool climate and well-drained sandy loam soil.",
    "Barley":     "Tolerates poor soil. Needs cool dry weather.",
}

def get_advice(crop: str) -> str:
    return CROP_ADVICE.get(crop, "Consult a local agricultural expert for best practices.")

def run_prediction(data) -> dict:
    if model is None:
        raise Exception("ML model not loaded. Check model path.")
    sample = pd.DataFrame([{
        "Soil_Type": data.soil_type,
        "pH_Value": float(data.ph),
        "Nitrogen_Value (N)": float(data.nitrogen),
        "Phosphorus_Value (P)": float(data.phosphorus),
        "Potassium_Value (K)": float(data.potassium),
        "Electrical_Conductivity (EC)": float(data.ec or 0.4),
        "Organic_Carbon (%)": float(data.organic or 0.8),
        "Soil_Moisture (%)": float(data.moisture or 30.0),
        "Zinc (%)": float(data.zinc or 0.6),
        "Iron (%)": float(data.iron or 3.2),
        "Manganese (%)": float(data.manganese or 1.1),
        "Copper (%)": float(data.copper or 0.3),
        "Boron (%)": float(data.boron or 0.4),
        "Sulphur (%)": float(data.sulphur or 12.0),
        "Rainfall_cm": float(data.rainfall),
        "temperature_celsius": float(data.temperature),
        "humidity_percentage": float(data.humidity),
        "State_Name": data.state,
        "Agro_Climatic Zone": data.agro_zone or "Unknown",
    }])
    prediction    = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    top_indices   = probabilities.argsort()[-3:][::-1]
    top3 = [{"crop": model.classes_[i], "confidence": round(float(probabilities[i]*100), 2)} for i in top_indices]
    return {"top_crop": str(prediction), "confidence": round(float(probabilities.max()*100), 2), "top3": top3, "advice": get_advice(str(prediction))}
