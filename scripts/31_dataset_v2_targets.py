import pandas as pd

FILE = "output/Crop_Normalized.xlsx"

df = pd.read_excel(FILE)

target_groups = {
    "Jammu And Kashmir": [
        "Maize",
        "Rice",
        "Wheat",
        "Apple",
        "Walnut",
        "Vegetables",
        "Mustard",
        "Pulses",
    ],
    "Jharkhand": [
        "Maize",
        "Mustard",
        "Pulses",
        "Rice",
    ],
}

TARGET = 50

rows = []

for state, crops in target_groups.items():

    for crop in crops:

        current = len(
            df[
                (df["State_Name"] == state)
                & (df["Crop"] == crop)
            ]
        )

        needed = max(
            TARGET - current,
            0
        )

        rows.append({
            "State": state,
            "Crop": crop,
            "Current_Records": current,
            "Target_Records": TARGET,
            "Additional_Records_Needed": needed,
        })

result = pd.DataFrame(rows)

print("=" * 75)
print("DATASET V2 TARGETS")
print("=" * 75)

print(
    result.to_string(index=False)
)

result.to_excel(
    "output/Dataset_V2_Targets.xlsx",
    index=False
)

print(
    "\nSaved to:"
    " output/Dataset_V2_Targets.xlsx"
)
