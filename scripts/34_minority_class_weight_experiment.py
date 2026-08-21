"""
SMART KISAN - MINORITY CLASS WEIGHT EXPERIMENT

SAFE EXPERIMENT ONLY

- Does NOT modify Streamlit.
- Does NOT modify the existing .pkl model.
- Does NOT save a new model.
- Uses the SAME dataset and preprocessing as script 19.
- Tests targeted class weights for minority crops.
"""

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample


# ==========================================================
# CONFIG
# ==========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ==========================================================
# HEADER
# ==========================================================

print("=" * 78)
print("SMART KISAN - MINORITY CLASS WEIGHT EXPERIMENT")
print("=" * 78)

print()
print("SAFE EXPERIMENT")
print("- Existing Streamlit app will NOT be changed.")
print("- Existing .pkl model will NOT be changed.")
print("- No model will be saved.")
print("- Same dataset/preprocessing as script 19.")
print()


# ==========================================================
# LOAD EXACT PROJECT DATASET
# ==========================================================

df = pd.read_excel(
    "output/Crop_Normalized.xlsx"
)

print(
    f"Dataset loaded: {len(df):,} records"
)


# ==========================================================
# EXACT FEATURES FROM SCRIPT 19
# ==========================================================

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
# CLEAN EXACTLY LIKE SCRIPT 19
# ==========================================================

for col in numeric_cols:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = df.dropna(
    subset=features + [target]
)


df = df.drop_duplicates()


# ==========================================================
# CROP NORMALIZATION
# ==========================================================

crop_mapping = {
    "Sugar Cane": "Sugarcane",
    "Paddy (Rice)": "Rice",
    "Oilseeds (Mustard)": "Mustard",
    "Pulses (Arhar)": "Pulses",
}


df[target] = df[target].replace(
    crop_mapping
)


# ==========================================================
# REMOVE EXTREMELY RARE CLASSES
# ==========================================================

counts = df[target].value_counts()

valid_classes = counts[
    counts >= 20
].index

df = df[
    df[target].isin(valid_classes)
].copy()


# ==========================================================
# DATASET INFORMATION
# ==========================================================

print()
print("=" * 78)
print("DATASET INFORMATION")
print("=" * 78)

print(
    f"Total records : {len(df):,}"
)

print(
    f"Crop classes  : {df[target].nunique()}"
)

print()
print("Crop distribution:")

print(
    df[target].value_counts()
)


# ==========================================================
# TRAIN / TEST
# ==========================================================

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


print()
print(
    f"Training records: {len(X_train):,}"
)

print(
    f"Testing records : {len(X_test):,}"
)


# ==========================================================
# FULL OVERSAMPLING
# SAME METHOD AS SCRIPT 19
# ==========================================================

train_df = X_train.copy()

train_df[target] = y_train.values


max_count = (
    train_df[target]
    .value_counts()
    .max()
)


balanced_parts = []


for crop, group in train_df.groupby(
    target
):

    if len(group) < max_count:

        group = resample(
            group,
            replace=True,
            n_samples=max_count,
            random_state=RANDOM_STATE,
        )

    balanced_parts.append(
        group
    )


balanced_train = pd.concat(
    balanced_parts,
    ignore_index=True
)


balanced_train = balanced_train.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(
    drop=True
)


X_balanced = balanced_train[
    features
]

y_balanced = balanced_train[
    target
]


print()
print(
    f"Balanced training records: "
    f"{len(X_balanced):,}"
)


# ==========================================================
# PREPROCESSOR
# ==========================================================

preprocessor = ColumnTransformer(
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

    "Baseline Balanced": None,

    "Minority Mild": {
        "Apple": 1.5,
        "Mustard": 1.5,
        "Vegetables": 1.5,
        "Walnut": 1.5,
    },

    "Minority Medium": {
        "Apple": 2.0,
        "Mustard": 2.0,
        "Vegetables": 2.0,
        "Walnut": 2.0,
    },

    "Minority Strong": {
        "Apple": 3.0,
        "Mustard": 3.0,
        "Vegetables": 3.0,
        "Walnut": 3.0,
    },

    "Weak Crop Focus": {
        "Apple": 3.0,
        "Mustard": 2.5,
        "Vegetables": 3.0,
        "Walnut": 3.0,
    },

    "Apple Walnut Focus": {
        "Apple": 4.0,
        "Walnut": 4.0,
        "Mustard": 2.0,
        "Vegetables": 2.0,
    },
}


# ==========================================================
# RESULTS
# ==========================================================

results = []


# ==========================================================
# RUN EXPERIMENTS
# ==========================================================

for experiment_name, class_weights in experiments.items():

    print()
    print("=" * 78)
    print(
        f"TESTING: {experiment_name}"
    )
    print("=" * 78)

    if class_weights is None:

        print(
            "Class Weight: None"
        )

    else:

        print(
            "Class Weight:"
        )

        for crop, weight in class_weights.items():

            print(
                f"  {crop}: {weight}"
            )


    # ------------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),

            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=30,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight=class_weights,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


    # ------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------

    print()
    print("Training...")

    model.fit(
        X_balanced,
        y_balanced
    )


    # ------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )


    print()
    print(
        f"Accuracy    : {accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1    : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : {weighted_f1:.4f}"
    )


    # ------------------------------------------------------
    # PER-CROP
    # ------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )


    print()
    print("Per-Crop Performance")
    print("-" * 78)


    for crop in sorted(
        y_test.unique()
    ):

        if crop not in report:
            continue

        precision = report[crop][
            "precision"
        ]

        recall = report[crop][
            "recall"
        ]

        f1 = report[crop][
            "f1-score"
        ]

        samples = int(
            report[crop]["support"]
        )

        print(
            f"{crop:<20}"
            f"Precision: {precision:.3f} | "
            f"Recall: {recall:.3f} | "
            f"F1: {f1:.3f} | "
            f"Samples: {samples}"
        )


    results.append(
        {
            "Model": experiment_name,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
        }
    )


# ==========================================================
# FINAL COMPARISON
# ==========================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Macro F1",
    ascending=False
)


print()
print()
print("=" * 78)
print("FINAL MINORITY WEIGHT COMPARISON")
print("=" * 78)


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


print()
print()
print("=" * 78)
print("BEST MINORITY-WEIGHT CONFIGURATION")
print("=" * 78)

print(
    f"Model       : {best['Model']}"
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
# BASELINE
# ==========================================================

baseline_rows = results_df[
    results_df["Model"]
    == "Baseline Balanced"
]


if not baseline_rows.empty:

    baseline = baseline_rows.iloc[0]

    improvement = (
        best["Macro F1"]
        - baseline["Macro F1"]
    )

    print()
    print("=" * 78)
    print("BASELINE COMPARISON")
    print("=" * 78)

    print(
        f"Baseline Macro F1 : "
        f"{baseline['Macro F1']:.4f}"
    )

    print(
        f"Best Macro F1     : "
        f"{best['Macro F1']:.4f}"
    )

    print(
        f"Improvement       : "
        f"{improvement:+.4f}"
    )


# ==========================================================
# SAFETY
# ==========================================================

print()
print()
print("=" * 78)
print("EXPERIMENT COMPLETE")
print("=" * 78)

print()
print("NO .pkl MODEL WAS CREATED.")
print("NO EXISTING MODEL WAS OVERWRITTEN.")
print("NO STREAMLIT FILE WAS CHANGED.")
print()
print(
    "Use the results above to choose the next model."
)

print("=" * 78)