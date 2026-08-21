import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

print("=" * 78)
print("SMART KISAN - PROBABILITY ANALYSIS")
print("=" * 78)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(BASE_DIR, "output", "Crop_Normalized.xlsx")

print()
print("SAFE ANALYSIS")
print("- No .pkl will be created")
print("- Existing model will not be changed")
print("- Streamlit will not be changed")

df = pd.read_excel(DATASET)

print()
print(f"Raw records: {len(df):,}")

# Exact benchmark crop set
crops = [
    "Rice",
    "Maize",
    "Wheat",
    "Pulses",
    "Cotton",
    "Mustard",
    "Vegetables",
    "Apple",
    "Walnut"
]

df = df[df["Crop"].isin(crops)].copy()

numeric_features = [
    "pH_Value",
    "Nitrogen_Value (N)",
    "Phosphorus_Value (P)",
    "Potassium_Value (K)",
    "Electrical_Conductivity (EC)",
    "Organic_Carbon (%)",
    "Zinc (%)",
    "Iron (%)",
    "Manganese (%)",
    "Copper (%)",
    "Boron (%)",
    "Sulphur (%)",
    "Rainfall_cm",
    "temperature_celsius",
    "humidity_percentage",
    "Soil_Moisture (%)"
]

categorical_features = [
    "State_Name",
    "Soil_Type"
]

features = numeric_features + categorical_features

df = df.dropna(
    subset=features + ["Crop"]
).copy()

print(f"Benchmark records: {len(df):,}")

X = df[features]
y = df["Crop"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()
print("=" * 78)
print("DATA LEAKAGE CHECK")
print("=" * 78)

overlap = set(X_train.index) & set(X_test.index)

print(f"Train/test index overlap: {len(overlap)}")

if overlap:
    raise RuntimeError("DATA LEAKAGE DETECTED")

print("Leakage check: PASSED")

# Full oversampling
train_df = X_train.copy()
train_df["Crop"] = y_train.values

max_count = train_df["Crop"].value_counts().max()

parts = []

for crop, group in train_df.groupby("Crop"):

    if len(group) < max_count:
        group = group.sample(
            n=max_count,
            replace=True,
            random_state=42
        )

    parts.append(group)

balanced = pd.concat(parts, ignore_index=True)

X_balanced = balanced[features]
y_balanced = balanced["Crop"]

print()
print("=" * 78)
print("FULL OVERSAMPLING")
print("=" * 78)

print(f"Training records : {len(X_train):,}")
print(f"Balanced records : {len(X_balanced):,}")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

model = RandomForestClassifier(
    n_estimators=700,
    max_depth=50,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

print()
print("Training...")
pipeline.fit(X_balanced, y_balanced)
print("Training complete.")

prob = pipeline.predict_proba(X_test)
classes = pipeline.named_steps["model"].classes_

pred = classes[np.argmax(prob, axis=1)]

print()
print("=" * 78)
print("BASELINE")
print("=" * 78)

print(f"Accuracy : {accuracy_score(y_test, pred) * 100:.2f}%")
print(f"Macro F1 : {f1_score(y_test, pred, average='macro'):.4f}")
print(
    f"Weighted F1 : "
    f"{f1_score(y_test, pred, average='weighted'):.4f}"
)

weak = ["Apple", "Mustard", "Vegetables", "Walnut"]

rows = []

for crop in weak:

    idx = np.where(y_test.values == crop)[0]

    crop_probs = prob[idx]

    crop_indices = [
        np.where(classes == c)[0][0]
        for c in classes
    ]

    own_index = np.where(classes == crop)[0][0]

    own_probs = crop_probs[:, own_index]

    rows.append({
        "Crop": crop,
        "Test Samples": len(idx),
        "Mean Own Probability": own_probs.mean(),
        "Median Own Probability": np.median(own_probs),
        "Min Own Probability": own_probs.min(),
        "Max Own Probability": own_probs.max(),
        "Correct Predictions": np.sum(
            pred[idx] == crop
        )
    })

prob_df = pd.DataFrame(rows)

print()
print("=" * 78)
print("WEAK-CROP PROBABILITY ANALYSIS")
print("=" * 78)

print(
    prob_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print()
print("=" * 78)
print("MOST CONFUSING PREDICTION PAIRS")
print("=" * 78)

for actual in weak:

    idx = np.where(y_test.values == actual)[0]

    wrong = idx[pred[idx] != actual]

    if len(wrong) == 0:
        continue

    counts = pd.Series(
        pred[wrong]
    ).value_counts()

    print()
    print(f"{actual} misclassified as:")

    for target, count in counts.items():
        print(f"  {target:<15} {count}")

# Save analysis only
output = os.path.join(
    BASE_DIR,
    "output",
    "Probability_Analysis.xlsx"
)

prob_df.to_excel(
    output,
    index=False
)

print()
print("=" * 78)
print("ANALYSIS COMPLETE")
print("=" * 78)

print(f"Report saved to: {output}")

print()
print("NO .pkl MODEL WAS CREATED.")
print("NO EXISTING MODEL WAS OVERWRITTEN.")
print("NO STREAMLIT FILE WAS CHANGED.")
print()
print("DO NOT DEPLOY YET.")
print("=" * 78)