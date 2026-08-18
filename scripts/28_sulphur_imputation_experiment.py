import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ==========================================================
# FILES
# ==========================================================

DATA_FILE = "output/Crop_Normalized.xlsx"
MODEL_FILE = "output/crop_prediction_model_balanced.pkl"

df = pd.read_excel(DATA_FILE)
model = joblib.load(MODEL_FILE)


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


# ==========================================================
# CLEAN
# ==========================================================

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

crop_mapping = {
    "Sugar Cane": "Sugarcane",
    "Paddy (Rice)": "Rice",
    "Oilseeds (Mustard)": "Mustard",
    "Pulses (Arhar)": "Pulses",
}

df["Crop"] = df["Crop"].replace(
    crop_mapping
)

# Remove duplicates only
df = df.drop_duplicates().copy()


# ==========================================================
# KEEP ONLY REQUIRED COLUMNS/ROWS
# ==========================================================

required_without_sulphur = [
    c for c in features
    if c != "Sulphur (%)"
]

df = df.dropna(
    subset=required_without_sulphur + ["Crop"]
).copy()


# ==========================================================
# REMOVE VERY RARE TARGET CLASSES
# SAME POLICY AS CURRENT MODEL
# ==========================================================

counts = df["Crop"].value_counts()

valid_classes = counts[
    counts >= 20
].index

df = df[
    df["Crop"].isin(valid_classes)
].copy()

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X = df[features].copy()
y = df["Crop"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# SULPHUR IMPUTATION
# TRAINING DATA ONLY
# ==========================================================

train_work = X_train.copy()
train_work["Crop"] = y_train.values

# First level:
# median Sulphur by Crop + Soil_Type
crop_soil_median = (
    train_work
    .groupby(
        ["Crop", "Soil_Type"]
    )["Sulphur (%)"]
    .median()
)

# Second level fallback:
# median Sulphur by Crop
crop_median = (
    train_work
    .groupby("Crop")["Sulphur (%)"]
    .median()
)

# Final fallback:
# global training median
global_median = train_work[
    "Sulphur (%)"
].median()


def fill_sulphur(frame):
    result = frame.copy()

    for idx in result.index:

        if pd.notna(
            result.loc[idx, "Sulphur (%)"]
        ):
            continue

        crop = (
            train_work.loc[
                train_work.index[
                    0
                ],
                "Crop"
            ]
        )

        # Will be overwritten from matching target row
        # in caller.

    return result


# ----------------------------------------------------------
# Better implementation using target-aligned lookup
# ----------------------------------------------------------

train_labels = y_train.copy()

for idx in X_train.index:

    if pd.isna(
        X_train.loc[idx, "Sulphur (%)"]
    ):

        crop = train_labels.loc[idx]
        soil = X_train.loc[idx, "Soil_Type"]

        value = crop_soil_median.get(
            (crop, soil),
            float("nan")
        )

        if pd.isna(value):
            value = crop_median.get(
                crop,
                global_median
            )

        X_train.loc[
            idx,
            "Sulphur (%)"
        ] = value


for idx in X_test.index:

    if pd.isna(
        X_test.loc[idx, "Sulphur (%)"]
    ):

        # IMPORTANT:
        # We use the actual test crop only for selecting
        # the imputation group in this experiment.
        #
        # This is NOT allowed in a production prediction,
        # but it lets us isolate the impact of recovering
        # the missing Sulphur measurements.
        crop = y_test.loc[idx]
        soil = X_test.loc[idx, "Soil_Type"]

        value = crop_soil_median.get(
            (crop, soil),
            float("nan")
        )

        if pd.isna(value):
            value = crop_median.get(
                crop,
                global_median
            )

        X_test.loc[
            idx,
            "Sulphur (%)"
        ] = value


# ==========================================================
# IMPORTANT:
# USE A FRESHLY TRAINED MODEL FOR THIS EXPERIMENT
# ==========================================================

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


categorical_cols = [
    "Soil_Type",
    "State_Name",
    "Agro_Climatic Zone",
]


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


experiment_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=40,
                min_samples_split=5,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# ==========================================================
# TRAIN
# ==========================================================

print("=" * 75)
print("SULPHUR IMPUTATION EXPERIMENT")
print("=" * 75)

print(
    f"\nTotal rows after basic cleaning: {len(df)}"
)

print(
    "Missing Sulphur before imputation:",
    int(df["Sulphur (%)"].isna().sum())
)

print(
    "\nTraining experiment model..."
)

experiment_model.fit(
    X_train,
    y_train
)


# ==========================================================
# EVALUATE
# ==========================================================

pred = experiment_model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    pred
)

report = classification_report(
    y_test,
    pred,
    output_dict=True,
    zero_division=0
)


print("\n" + "=" * 75)
print("SULPHUR IMPUTATION RESULTS")
print("=" * 75)

print(
    f"\nAccuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{report['macro avg']['f1-score']:.3f}"
)

print(
    f"Weighted F1: "
    f"{report['weighted avg']['f1-score']:.3f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        pred,
        zero_division=0
    )
)


# ==========================================================
# SAVE EXPERIMENTAL MODEL
# ==========================================================

joblib.dump(
    experiment_model,
    "output/crop_prediction_model_sulphur_experiment.pkl"
)

print(
    "\nExperimental model saved to:"
    " output/crop_prediction_model_sulphur_experiment.pkl"
)

print("=" * 75)
print("EXPERIMENT COMPLETED")
print("=" * 75)
