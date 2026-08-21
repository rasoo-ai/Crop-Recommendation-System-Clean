import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample


print("=" * 78)
print("SMART KISAN - MAX FEATURES EXPERIMENT")
print("=" * 78)

print("""
SAFE EXPERIMENT
- Existing Streamlit app will NOT be changed.
- Existing .pkl model will NOT be changed.
- No model will be saved.
- No application files will be changed.
- Agro_Climatic Zone is excluded.
- Full oversampling is retained.
- max_depth = 50.
- class_weight = None.
""")


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_excel(
    "output/Crop_Normalized.xlsx"
)

print(f"Dataset loaded: {len(df):,}")


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
).copy()


df = df.drop_duplicates()


crop_mapping = {
    "Sugar Cane": "Sugarcane",
    "Paddy (Rice)": "Rice",
    "Oilseeds (Mustard)": "Mustard",
    "Pulses (Arhar)": "Pulses",
}


df[target] = df[target].replace(
    crop_mapping
)


counts = df[target].value_counts()


valid_classes = counts[
    counts >= 20
].index


df = df[
    df[target].isin(valid_classes)
].copy()


# ==========================================================
# SPLIT
# ==========================================================

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# ==========================================================
# LEAKAGE CHECK
# ==========================================================

print()
print("=" * 78)
print("DATA LEAKAGE CHECK")
print("=" * 78)


overlap = len(
    set(X_train.index).intersection(
        set(X_test.index)
    )
)


print(f"Train/test index overlap: {overlap}")


if overlap != 0:
    raise RuntimeError(
        "DATA LEAKAGE DETECTED"
    )


print("Leakage check: PASSED")


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
            random_state=42,
        )

    balanced_parts.append(group)


balanced_train = pd.concat(
    balanced_parts,
    ignore_index=True
)


balanced_train = balanced_train.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


X_balanced = balanced_train[features]
y_balanced = balanced_train[target]


print()
print("=" * 78)
print("DATASET INFORMATION")
print("=" * 78)

print(f"Total records    : {len(df):,}")
print(f"Training records : {len(X_train):,}")
print(f"Testing records  : {len(X_test):,}")
print(f"Balanced records : {len(X_balanced):,}")


# ==========================================================
# BALANCING VALIDATION
# ==========================================================

print()
print("Balanced distribution:")
print(
    y_balanced.value_counts()
)


# ==========================================================
# EXPERIMENTS
# ==========================================================

experiments = {

    "Current sqrt": "sqrt",

    "Log2": "log2",

    "Half Features": 0.5,

    "Three Quarter Features": 0.75,

    "All Features": None,
}


# ==========================================================
# TRAINING
# ==========================================================

results = []


for name, max_features_value in experiments.items():

    print()
    print("=" * 78)
    print(f"TESTING: {name}")
    print("=" * 78)

    print(f"max_features: {max_features_value}")
    print("max_depth: 50")
    print("n_estimators: 700")
    print("min_samples_split: 5")
    print("min_samples_leaf: 2")
    print("class_weight: None")
    print("Training...")


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


    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=700,
                    max_depth=50,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features=max_features_value,
                    random_state=42,
                    n_jobs=-1,
                    class_weight=None,
                ),
            ),
        ]
    )


    model.fit(
        X_balanced,
        y_balanced
    )


    predictions = model.predict(
        X_test
    )


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


    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )


    weak_crops = [
        "Apple",
        "Mustard",
        "Vegetables",
        "Walnut",
    ]


    weak_f1_values = []


    print()
    print(f"Accuracy    : {accuracy * 100:.2f}%")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")


    print()
    print("Weak-Crop Performance")
    print("-" * 78)


    for crop in weak_crops:

        if crop not in report:
            continue

        precision = report[crop]["precision"]
        recall = report[crop]["recall"]
        f1 = report[crop]["f1-score"]

        weak_f1_values.append(f1)

        print(
            f"{crop:<15}"
            f" Precision: {precision:.3f}"
            f" | Recall: {recall:.3f}"
            f" | F1: {f1:.3f}"
        )


    weak_crop_f1 = (
        sum(weak_f1_values)
        / len(weak_f1_values)
        if weak_f1_values
        else 0
    )


    print(
        f"Average Weak-Crop F1: "
        f"{weak_crop_f1:.4f}"
    )


    results.append({
        "Model": name,
        "Max Features": max_features_value,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Weak Crop F1": weak_crop_f1,
    })


# ==========================================================
# FINAL COMPARISON
# ==========================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by=[
        "Macro F1",
        "Weak Crop F1",
        "Accuracy",
    ],
    ascending=False,
)


print()
print("=" * 78)
print("FINAL MAX FEATURES COMPARISON")
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


display_df["Weak Crop F1"] = (
    display_df["Weak Crop F1"]
).round(4)


print(
    display_df.to_string(
        index=False
    )
)


# ==========================================================
# BEST CONFIGURATION
# ==========================================================

best = results_df.iloc[0]


print()
print("=" * 78)
print("BEST MAX FEATURES CONFIGURATION")
print("=" * 78)


print(
    f"Model          : {best['Model']}"
)

print(
    f"Max Features   : {best['Max Features']}"
)

print(
    f"Accuracy       : "
    f"{best['Accuracy'] * 100:.2f}%"
)

print(
    f"Macro F1       : "
    f"{best['Macro F1']:.4f}"
)

print(
    f"Weighted F1    : "
    f"{best['Weighted F1']:.4f}"
)

print(
    f"Weak Crop F1   : "
    f"{best['Weak Crop F1']:.4f}"
)


# ==========================================================
# CURRENT BENCHMARK
# ==========================================================

current_accuracy = 0.9300
current_macro_f1 = 0.6879
current_weighted_f1 = 0.9403
current_weak_f1 = 0.3762


macro_change = (
    best["Macro F1"]
    - current_macro_f1
)


accuracy_change = (
    best["Accuracy"]
    - current_accuracy
)


weak_change = (
    best["Weak Crop F1"]
    - current_weak_f1
)


print()
print("=" * 78)
print("CURRENT BENCHMARK COMPARISON")
print("=" * 78)


print(
    f"Current Accuracy   : "
    f"{current_accuracy * 100:.2f}%"
)

print(
    f"New Accuracy       : "
    f"{best['Accuracy'] * 100:.2f}%"
)

print(
    f"Accuracy change    : "
    f"{accuracy_change * 100:+.2f} percentage points"
)


print()


print(
    f"Current Macro F1   : "
    f"{current_macro_f1:.4f}"
)

print(
    f"New Macro F1       : "
    f"{best['Macro F1']:.4f}"
)

print(
    f"Macro F1 change    : "
    f"{macro_change:+.4f}"
)


print()


print(
    f"Current Weak F1    : "
    f"{current_weak_f1:.4f}"
)

print(
    f"New Weak F1        : "
    f"{best['Weak Crop F1']:.4f}"
)

print(
    f"Weak F1 change     : "
    f"{weak_change:+.4f}"
)


# ==========================================================
# SAFETY
# ==========================================================

print()
print("=" * 78)
print("EXPERIMENT COMPLETE")
print("=" * 78)

print()
print("NO .pkl MODEL WAS CREATED.")
print("NO EXISTING MODEL WAS OVERWRITTEN.")
print("NO STREAMLIT FILE WAS CHANGED.")
print()
print("DO NOT DEPLOY YET.")
print("=" * 78)