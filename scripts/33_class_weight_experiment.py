import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# ==========================================================
# CONFIG
# ==========================================================

DATA_FILE = "output/Crop_Normalized.xlsx"
RANDOM_STATE = 42


print("=" * 75)
print("SMART KISAN - CLASS WEIGHT EXPERIMENT")
print("=" * 75)

print("""
SAFE EXPERIMENT
- Streamlit app will NOT be changed.
- Existing .pkl model will NOT be changed.
- No model will be saved.
""")


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_excel(DATA_FILE)

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
    "Agro_Climatic Zone",
]

numeric_cols = [
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

categorical_cols = [
    "Soil_Type",
    "State_Name",
    "Agro_Climatic Zone",
]

target = "Crop"


# ==========================================================
# CLEAN
# ==========================================================

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = df.dropna(
    subset=features + [target]
).drop_duplicates()

df[target] = df[target].replace({
    "Sugar Cane": "Sugarcane",
    "Paddy (Rice)": "Rice",
    "Oilseeds (Mustard)": "Mustard",
    "Pulses (Arhar)": "Pulses",
})


# ==========================================================
# REMOVE VERY SMALL CLASSES
# ==========================================================

counts = df[target].value_counts()

valid_classes = counts[counts >= 20].index

df = df[
    df[target].isin(valid_classes)
].copy()

df = df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ==========================================================
# SPLIT
# ==========================================================

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ==========================================================
# FULL OVERSAMPLING
# ==========================================================

train_df = X_train.copy()
train_df[target] = y_train.values

max_count = train_df[target].value_counts().max()

balanced_parts = []

for crop, group in train_df.groupby(target):

    if len(group) < max_count:

        group = resample(
            group,
            replace=True,
            n_samples=max_count,
            random_state=RANDOM_STATE,
        )

    balanced_parts.append(group)


balanced_train = pd.concat(
    balanced_parts,
    ignore_index=True
)

balanced_train = balanced_train.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

X_balanced = balanced_train[features]
y_balanced = balanced_train[target]


print(f"\nDataset records : {len(df):,}")
print(f"Training        : {len(X_train):,}")
print(f"Testing         : {len(X_test):,}")
print(f"Balanced train  : {len(X_balanced):,}")


# ==========================================================
# PREPROCESSOR
# ==========================================================

def make_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_cols,
            ),
            (
                "num",
                "passthrough",
                numeric_cols,
            ),
        ]
    )


# ==========================================================
# EXPERIMENTS
# ==========================================================

experiments = {

    "No Class Weight": None,

    "Balanced": "balanced",

    "Balanced Less Overfit": "balanced_subsample",

}


results = []


# ==========================================================
# RUN EXPERIMENTS
# ==========================================================

for name, class_weight in experiments.items():

    print("\n")
    print("=" * 75)
    print(f"TESTING: {name}")
    print("=" * 75)

    print(
        f"Class Weight: {class_weight}"
    )

    preprocessor = make_preprocessor()

    X_train_encoded = preprocessor.fit_transform(
        X_balanced
    )

    X_test_encoded = preprocessor.transform(
        X_test
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining...")

    model.fit(
        X_train_encoded,
        y_balanced
    )

    pred = model.predict(
        X_test_encoded
    )

    accuracy = accuracy_score(
        y_test,
        pred
    )

    macro_f1 = f1_score(
        y_test,
        pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Method": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
    })

    print(
        f"\nAccuracy    : {accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1    : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : {weighted_f1:.4f}"
    )

    # ------------------------------------------------------
    # PER-CROP REPORT
    # ------------------------------------------------------

    report = classification_report(
        y_test,
        pred,
        output_dict=True,
        zero_division=0,
    )

    print("\nPer-Crop Performance")
    print("-" * 75)

    for crop in sorted(valid_classes):

        if crop not in report:
            continue

        print(
            f"{crop:<20}"
            f" Precision: {report[crop]['precision']:.3f}"
            f" | Recall: {report[crop]['recall']:.3f}"
            f" | F1: {report[crop]['f1-score']:.3f}"
            f" | Samples: {int(report[crop]['support'])}"
        )


# ==========================================================
# FINAL COMPARISON
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Macro F1",
    ascending=False
)

print("\n")
print("=" * 75)
print("FINAL CLASS WEIGHT COMPARISON")
print("=" * 75)

display_df = results_df.copy()

display_df["Accuracy"] = (
    display_df["Accuracy"] * 100
).round(2)

display_df["Macro F1"] = (
    display_df["Macro F1"]
).round(4)

display_df["Weighted F1"] = (
    display_df["Weighted F1"]
).round(4)

print(
    display_df.to_string(
        index=False
    )
)


# ==========================================================
# BEST
# ==========================================================

best = results_df.iloc[0]

print("\n")
print("=" * 75)
print("BEST CLASS-WEIGHT CONFIGURATION")
print("=" * 75)

print(
    f"Method      : {best['Method']}"
)

print(
    f"Accuracy    : {best['Accuracy'] * 100:.2f}%"
)

print(
    f"Macro F1    : {best['Macro F1']:.4f}"
)

print(
    f"Weighted F1 : {best['Weighted F1']:.4f}"
)


# ==========================================================
# BASELINE COMPARISON
# ==========================================================

print("\n")
print("=" * 75)
print("CURRENT BASELINE")
print("=" * 75)

print("Accuracy    : 93.09%")
print("Macro F1    : 0.6466")
print("Weighted F1 : 0.9404")


print("\n")
print("=" * 75)
print("EXPERIMENT COMPLETE")
print("=" * 75)

print("\nNo .pkl file was created.")
print("No existing model was overwritten.")
print("No Streamlit files were changed.")

print("\nDo NOT deploy anything yet.")
print("Use the results to choose the next model.")