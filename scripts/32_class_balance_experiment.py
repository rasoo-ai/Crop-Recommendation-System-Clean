import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# ==========================================================
# CONFIG
# ==========================================================

DATA_FILE = "output/Crop_Normalized.xlsx"
RANDOM_STATE = 42


# ==========================================================
# HEADER
# ==========================================================

print("=" * 75)
print("SMART KISAN - CLASS BALANCE EXPERIMENT")
print("=" * 75)

print("\nSAFE EXPERIMENT")
print("- Existing Streamlit app will NOT be changed.")
print("- Existing .pkl model will NOT be changed.")
print("- No model file will be saved.\n")


# ==========================================================
# LOAD
# ==========================================================

df = pd.read_excel(DATA_FILE)

print(f"Raw records: {len(df):,}")


# ==========================================================
# FEATURES
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
# CLEAN
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
# CROP NAME FIX
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
# REMOVE CLASSES WITH < 20 RECORDS
# ==========================================================

counts = df[target].value_counts()

valid_classes = counts[
    counts >= 20
].index

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
# EXPERIMENT 1
# CURRENT FULL OVERSAMPLING
# ==========================================================

def full_oversample(X_train, y_train):

    train_df = X_train.copy()

    train_df[target] = y_train.values

    max_count = (
        train_df[target]
        .value_counts()
        .max()
    )

    parts = []

    for crop, group in train_df.groupby(target):

        if len(group) < max_count:

            group = resample(
                group,
                replace=True,
                n_samples=max_count,
                random_state=RANDOM_STATE,
            )

        parts.append(group)

    result = pd.concat(
        parts,
        ignore_index=True
    )

    return result.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)


# ==========================================================
# EXPERIMENT 2
# MODERATE OVERSAMPLING
# ==========================================================

def moderate_oversample(X_train, y_train):

    train_df = X_train.copy()

    train_df[target] = y_train.values

    counts = (
        train_df[target]
        .value_counts()
    )

    # Target approximately 50% of majority class
    target_count = int(
        counts.max() * 0.50
    )

    parts = []

    for crop, group in train_df.groupby(target):

        desired = min(
            target_count,
            max(len(group), target_count)
        )

        if len(group) < target_count:

            group = resample(
                group,
                replace=True,
                n_samples=target_count,
                random_state=RANDOM_STATE,
            )

        parts.append(group)

    result = pd.concat(
        parts,
        ignore_index=True
    )

    return result.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)


# ==========================================================
# EXPERIMENT 3
# MINORITY FOCUSED OVERSAMPLING
# ==========================================================

def minority_oversample(X_train, y_train):

    train_df = X_train.copy()

    train_df[target] = y_train.values

    counts = (
        train_df[target]
        .value_counts()
    )

    majority = counts.max()

    parts = []

    for crop, group in train_df.groupby(target):

        count = len(group)

        if count < 50:

            desired = min(
                250,
                majority
            )

            group = resample(
                group,
                replace=True,
                n_samples=desired,
                random_state=RANDOM_STATE,
            )

        elif count < 100:

            desired = min(
                300,
                majority
            )

            group = resample(
                group,
                replace=True,
                n_samples=desired,
                random_state=RANDOM_STATE,
            )

        parts.append(group)

    result = pd.concat(
        parts,
        ignore_index=True
    )

    return result.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)


# ==========================================================
# PREPARE DATASETS
# ==========================================================

balance_methods = {
    "Full Oversampling": full_oversample(
        X_train,
        y_train
    ),

    "Moderate Oversampling": moderate_oversample(
        X_train,
        y_train
    ),

    "Minority Focused": minority_oversample(
        X_train,
        y_train
    ),

    "No Oversampling": X_train.assign(
        **{target: y_train.values}
    ),
}


# ==========================================================
# RESULTS
# ==========================================================

results = []


# ==========================================================
# RUN
# ==========================================================

for method_name, train_data in balance_methods.items():

    print("\n")
    print("=" * 75)
    print(f"TESTING: {method_name}")
    print("=" * 75)

    y_method = train_data[target]

    X_method = train_data[
        features
    ]

    print(
        f"Training records: "
        f"{len(X_method):,}"
    )

    print("\nTraining distribution:")

    print(
        y_method
        .value_counts()
        .sort_index()
    )

    # Fresh preprocessing for each method
    processor = ColumnTransformer(
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

    X_train_encoded = processor.fit_transform(
        X_method
    )

    X_test_encoded = processor.transform(
        X_test
    )

    # ------------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining Random Forest...")

    model.fit(
        X_train_encoded,
        y_method
    )

    pred = model.predict(
        X_test_encoded
    )

    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        pred
    )

    macro_f1 = f1_score(
        y_test,
        pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        pred,
        average="weighted",
        zero_division=0,
    )

    results.append({
        "Method": method_name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
    })

    print(
        f"\nAccuracy    : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1    : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : "
        f"{weighted_f1:.4f}"
    )

    # ------------------------------------------------------
    # PER-CROP
    # ------------------------------------------------------

    report = classification_report(
        y_test,
        pred,
        output_dict=True,
        zero_division=0,
    )

    print("\nPer-Crop F1:")

    for crop in sorted(valid_classes):

        if crop in report:

            print(
                f"{crop:<20}"
                f" F1={report[crop]['f1-score']:.3f}"
                f" | Recall={report[crop]['recall']:.3f}"
            )


# ==========================================================
# FINAL COMPARISON
# ==========================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)


print("\n")
print("=" * 75)
print("FINAL CLASS-BALANCE COMPARISON")
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
print("BEST BALANCING METHOD")
print("=" * 75)

print(
    f"Method       : {best['Method']}"
)

print(
    f"Accuracy     : "
    f"{best['Accuracy'] * 100:.2f}%"
)

print(
    f"Macro F1     : "
    f"{best['Macro F1']:.4f}"
)

print(
    f"Weighted F1  : "
    f"{best['Weighted F1']:.4f}"
)


# ==========================================================
# FINISH
# ==========================================================

print("\n")
print("=" * 75)
print("EXPERIMENT COMPLETE")
print("=" * 75)

print(
    "\nNo Streamlit files were changed."
)

print(
    "No existing .pkl model was changed."
)

print(
    "No model was saved."
)

print(
    "\nIMPORTANT:"
)

print(
    "Choose the method based primarily on Macro F1 "
    "and minority-crop performance, not accuracy alone."
)

print("=" * 75)