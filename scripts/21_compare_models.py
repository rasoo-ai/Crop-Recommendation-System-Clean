import pandas as pd
import joblib

DATA_FILE = "output/Crop_Normalized.xlsx"
OLD_MODEL_FILE = "output/crop_prediction_model_BACKUP.pkl"
NEW_MODEL_FILE = "output/crop_prediction_model_balanced.pkl"

FEATURES = [
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

df = pd.read_excel(DATA_FILE)

old_model = joblib.load(OLD_MODEL_FILE)
new_model = joblib.load(NEW_MODEL_FILE)

# Choose Sivakasi plus representative tehsils from different states.
requested = [
    ("Tamilnadu", "Virudhunagar", "Sivakasi"),
]

# Add more locations automatically.
location_df = (
    df[["State_Name", "District_Name", "Tehsil_Name"]]
    .dropna()
    .drop_duplicates()
)

extra = location_df.sample(
    n=min(9, len(location_df)),
    random_state=42,
)

locations = requested + [
    tuple(x)
    for x in extra.itertuples(index=False, name=None)
]

results = []

for state, district, tehsil in locations:

    mask = (
        (df["State_Name"].astype(str).str.strip() == str(state).strip())
        & (df["District_Name"].astype(str).str.strip() == str(district).strip())
        & (df["Tehsil_Name"].astype(str).str.strip() == str(tehsil).strip())
    )

    local = df[mask].copy()

    if local.empty:
        continue

    # Use the first available local record.
    row = local.iloc[[0]][FEATURES]

    old_pred = old_model.predict(row)[0]
    new_pred = new_model.predict(row)[0]

    old_prob = old_model.predict_proba(row)[0]
    new_prob = new_model.predict_proba(row)[0]

    old_idx = old_prob.argmax()
    new_idx = new_prob.argmax()

    old_conf = old_prob[old_idx] * 100
    new_conf = new_prob[new_idx] * 100

    results.append({
        "State": state,
        "District": district,
        "Tehsil": tehsil,
        "Observed Crop": local.iloc[0]["Crop"],
        "Old Prediction": old_pred,
        "Old Confidence": round(old_conf, 2),
        "New Prediction": new_pred,
        "New Confidence": round(new_conf, 2),
        "Changed": old_pred != new_pred,
    })

result_df = pd.DataFrame(results)

print("=" * 90)
print("OLD MODEL VS NEW MODEL")
print("=" * 90)

print(
    result_df.to_string(index=False)
)

print("\nChanged predictions:",
      int(result_df["Changed"].sum()))

print(
    "Total locations:",
    len(result_df)
)

result_df.to_excel(
    "output/Old_vs_New_Model_Comparison.xlsx",
    index=False
)

print(
    "\nSaved to:"
    " output/Old_vs_New_Model_Comparison.xlsx"
)
