import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# ==========================================================
# CONFIG
# ==========================================================

DATA_FILE = "output/Crop_Normalized.xlsx"
RANDOM_STATE = 42


# ==========================================================
# PAGE / EXPERIMENT HEADER
# ==========================================================

print("=" * 75)
print("SMART KISAN - RANDOM FOREST TUNING + PER-CROP ANALYSIS")
print("=" * 75)

print("\nIMPORTANT:")
print("- Existing Streamlit app will NOT be changed.")
print("- Existing .pkl model will NOT be changed.")
print("- This script only runs experiments.\n")


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_excel(DATA_FILE)

print(f"Dataset loaded: {len(df):,} records")


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
# CLEAN DATA
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
# CROP NAME STANDARDIZATION
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
# REMOVE VERY SMALL CLASSES
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
# DATASET INFORMATION
# ==========================================================

print("\n" + "=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

print(f"Total records : {len(df):,}")
print(f"Crop classes  : {len(valid_classes)}")

print("\nOriginal crop distribution:")
print(
    df[target]
    .value_counts()
    .sort_index()
)


# ==========================================================
# TRAIN / TEST SPLIT
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

print("\nTrain records:", len(X_train))
print("Test records :", len(X_test))


# ==========================================================
# BALANCE TRAINING DATA ONLY
# ==========================================================

train_df = X_train.copy()

train_df[target] = y_train.values

max_count = (
    train_df[target]
    .value_counts()
    .max()
)

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

print("\nBalanced training records:")
print(f"{len(X_balanced):,}")


# ==========================================================
# PREPROCESSING
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


X_train_encoded = preprocessor.fit_transform(
    X_balanced
)

X_test_encoded = preprocessor.transform(
    X_test
)

print(
    f"\nEncoded feature count: "
    f"{X_train_encoded.shape[1]}"
)


# ==========================================================
# RANDOM FOREST EXPERIMENTS
# ==========================================================

experiments = [

    {
        "name": "Current",
        "n_estimators": 500,
        "max_depth": 40,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },

    {
        "name": "Less Overfit",
        "n_estimators": 500,
        "max_depth": 30,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },

    {
        "name": "More Trees",
        "n_estimators": 800,
        "max_depth": 40,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },

    {
        "name": "Deeper",
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },

    {
        "name": "Regularized",
        "n_estimators": 500,
        "max_depth": 30,
        "min_samples_split": 10,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },

    {
        "name": "Balanced Strong",
        "n_estimators": 700,
        "max_depth": 30,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
]


# ==========================================================
# STORAGE
# ==========================================================

results = []

best_model = None
best_name = None
best_macro_f1 = -1
best_predictions = None


# ==========================================================
# RUN EXPERIMENTS
# ==========================================================

for config in experiments:

    print("\n")
    print("=" * 75)
    print(f"TESTING: {config['name']}")
    print("=" * 75)

    print(
        f"Trees={config['n_estimators']} | "
        f"Depth={config['max_depth']} | "
        f"MinSplit={config['min_samples_split']} | "
        f"MinLeaf={config['min_samples_leaf']} | "
        f"Features={config['max_features']}"
    )

    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=config["min_samples_split"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
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
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        pred,
        average="weighted",
        zero_division=0,
    )

    results.append({
        "Model": config["name"],
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
    # PER-CROP REPORT FOR THIS MODEL
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

        precision = report[crop]["precision"]
        recall = report[crop]["recall"]
        f1 = report[crop]["f1-score"]
        support = int(report[crop]["support"])

        print(
            f"{crop:<25}"
            f" Precision: {precision:.3f}"
            f" | Recall: {recall:.3f}"
            f" | F1: {f1:.3f}"
            f" | Samples: {support}"
        )

    # ------------------------------------------------------
    # KEEP BEST MODEL IN MEMORY ONLY
    # ------------------------------------------------------

    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1
        best_model = model
        best_name = config["name"]
        best_predictions = pred


# ==========================================================
# FINAL RESULTS
# ==========================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by=["Macro F1", "Accuracy"],
    ascending=False
)


print("\n\n")
print("=" * 75)
print("FINAL MODEL COMPARISON")
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
# BEST MODEL DETAILS
# ==========================================================

print("\n")
print("=" * 75)
print("BEST MODEL")
print("=" * 75)

best_row = results_df.iloc[0]

print(
    f"Model       : {best_row['Model']}"
)

print(
    f"Accuracy    : "
    f"{best_row['Accuracy'] * 100:.2f}%"
)

print(
    f"Macro F1    : "
    f"{best_row['Macro F1']:.4f}"
)

print(
    f"Weighted F1 : "
    f"{best_row['Weighted F1']:.4f}"
)


# ==========================================================
# BEST MODEL - PER CROP ANALYSIS
# ==========================================================

print("\n")
print("=" * 75)
print(
    f"PER-CROP ANALYSIS - {best_name}"
)
print("=" * 75)

best_report = classification_report(
    y_test,
    best_predictions,
    output_dict=True,
    zero_division=0,
)

crop_results = []

for crop in sorted(valid_classes):

    if crop not in best_report:
        continue

    crop_results.append({
        "Crop": crop,
        "Precision": best_report[crop]["precision"],
        "Recall": best_report[crop]["recall"],
        "F1": best_report[crop]["f1-score"],
        "Test Samples": int(
            best_report[crop]["support"]
        ),
    })

crop_results_df = pd.DataFrame(
    crop_results
)

crop_results_df = crop_results_df.sort_values(
    "F1"
)

print(
    crop_results_df.to_string(
        index=False,
        formatters={
            "Precision": "{:.3f}".format,
            "Recall": "{:.3f}".format,
            "F1": "{:.3f}".format,
        },
    )
)


# ==========================================================
# STRONG / WEAK CROPS
# ==========================================================

print("\n")
print("=" * 75)
print("CROP INSIGHTS")
print("=" * 75)

strong_crops = crop_results_df[
    crop_results_df["F1"] >= 0.70
]

moderate_crops = crop_results_df[
    (crop_results_df["F1"] >= 0.50)
    & (crop_results_df["F1"] < 0.70)
]

weak_crops = crop_results_df[
    crop_results_df["F1"] < 0.50
]

print(
    f"\nStrong crops (F1 >= 0.70): "
    f"{len(strong_crops)}"
)

if len(strong_crops) > 0:
    print(
        ", ".join(
            strong_crops["Crop"].tolist()
        )
    )

print(
    f"\nModerate crops "
    f"(0.50 <= F1 < 0.70): "
    f"{len(moderate_crops)}"
)

if len(moderate_crops) > 0:
    print(
        ", ".join(
            moderate_crops["Crop"].tolist()
        )
    )

print(
    f"\nWeak crops (F1 < 0.50): "
    f"{len(weak_crops)}"
)

if len(weak_crops) > 0:
    print(
        ", ".join(
            weak_crops["Crop"].tolist()
        )
    )


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

print("\n")
print("=" * 75)
print("TOP MISCLASSIFICATIONS")
print("=" * 75)

labels = sorted(valid_classes)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=labels,
)

misses = []

for i, actual in enumerate(labels):

    for j, predicted in enumerate(labels):

        if i != j and cm[i, j] > 0:

            misses.append({
                "Actual": actual,
                "Predicted": predicted,
                "Count": int(cm[i, j]),
            })


misses_df = pd.DataFrame(
    misses
)

if len(misses_df) > 0:

    misses_df = misses_df.sort_values(
        "Count",
        ascending=False
    ).head(15)

    print(
        misses_df.to_string(
            index=False
        )
    )

else:

    print(
        "No misclassifications found."
    )


# ==========================================================
# FINAL MESSAGE
# ==========================================================

print("\n")
print("=" * 75)
print("EXPERIMENT COMPLETE")
print("=" * 75)

print(
    "\nNo existing model was overwritten."
)

print(
    "No Streamlit application files were changed."
)

print(
    "No new .pkl file was created."
)

print(
    "\nUse the results above to decide whether "
    "the improved configuration is worth deploying."
)

print("=" * 75)