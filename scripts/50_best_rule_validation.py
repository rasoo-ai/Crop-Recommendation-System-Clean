import os
import sys
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

print("=" * 78)
print("SMART KISAN - BEST DECISION RULE VALIDATION")
print("=" * 78)

print("""
SAFE VALIDATION
- Existing Streamlit app will NOT be changed.
- Existing .pkl model will NOT be changed.
- No model will be saved.
- No application files will be changed.
- Exact benchmark preprocessing is retained.
- Exact benchmark split seed is retained.
- Exact full oversampling is retained.
- Random Forest configuration is retained.
- Only Maize -> Walnut boost 1.15 is validated.
- Multiple deterministic repetitions are performed.
""")

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR,
    "output",
    "Crop_Normalized.xlsx"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "Best_Rule_Validation.xlsx"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def oversample_full(X, y, random_state=42):
    """
    Deterministic full oversampling to the maximum class size.
    """
    df = X.copy()
    df["_TARGET_"] = y.values

    counts = df["_TARGET_"].value_counts()
    target_size = counts.max()

    pieces = []

    for cls in sorted(counts.index):
        part = df[df["_TARGET_"] == cls]

        if len(part) < target_size:
            part = part.sample(
                n=target_size,
                replace=True,
                random_state=random_state
            )
        else:
            part = part.sample(
                n=target_size,
                replace=False,
                random_state=random_state
            )

        pieces.append(part)

    result = pd.concat(pieces, ignore_index=True)

    y_balanced = result.pop("_TARGET_")
    X_balanced = result

    return X_balanced, y_balanced


def apply_maize_walnut_boost(
    probabilities,
    classes,
    source="Maize",
    target="Walnut",
    boost=1.15
):
    """
    Increase Walnut probability only when Maize is the original
    argmax prediction.

    This preserves the intended directional rule:
        Maize -> Walnut

    It does NOT globally boost Walnut.
    """

    probabilities = probabilities.copy()

    class_to_index = {
        cls: i for i, cls in enumerate(classes)
    }

    maize_idx = class_to_index[source]
    walnut_idx = class_to_index[target]

    baseline_idx = np.argmax(probabilities, axis=1)

    for row_idx in range(len(probabilities)):
        if baseline_idx[row_idx] == maize_idx:
            probabilities[row_idx, walnut_idx] *= boost

    final_idx = np.argmax(probabilities, axis=1)

    return np.asarray(classes)[final_idx]


def evaluate(y_true, predictions):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0
        )
    }


# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------

banner("LOADING DATASET")

print("Dataset path:", DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_excel(DATA_PATH)

print("Raw records:", len(df))

# ----------------------------------------------------------------------
# TARGET DETECTION
# ----------------------------------------------------------------------

target_candidates = [
    "Crop",
    "crop",
    "Label",
    "label",
    "Target",
    "target"
]

target_col = None

for col in target_candidates:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError(
        "Could not find crop target column."
    )

# ----------------------------------------------------------------------
# BENCHMARK CROP FILTER
# ----------------------------------------------------------------------

benchmark_crops = [
    "Apple",
    "Cotton",
    "Maize",
    "Mustard",
    "Pulses",
    "Rice",
    "Vegetables",
    "Walnut",
    "Wheat"
]

df[target_col] = df[target_col].astype(str).str.strip()

df = df[
    df[target_col].isin(benchmark_crops)
].copy()

print(
    "After benchmark crop filtering:",
    len(df)
)

# ----------------------------------------------------------------------
# REMOVE MISSING VALUES
# ----------------------------------------------------------------------

before_drop = len(df)

df = df.dropna().reset_index(drop=False)

removed = before_drop - len(df)

print(
    "Rows removed for missing values:",
    removed
)

print(
    "Benchmark records:",
    len(df)
)

# Preserve original row index for leakage verification.
original_index = df["index"].copy()

# ----------------------------------------------------------------------
# FEATURES / TARGET
# ----------------------------------------------------------------------

X = df.drop(
    columns=[
        target_col,
        "index"
    ],
    errors="ignore"
)

y = df[target_col].copy()

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print()
print("Feature count:")
print("Numeric     :", len(numeric_columns))
print("Categorical :", len(categorical_columns))
print("Total       :", len(X.columns))

# ----------------------------------------------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------------------------------------------

banner("TRAIN / TEST SPLIT")

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X,
    y,
    original_index,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records :", len(X_train))
print("Testing records  :", len(X_test))

overlap = len(
    set(idx_train).intersection(set(idx_test))
)

print("Index overlap    :", overlap)

if overlap != 0:
    raise RuntimeError(
        "DATA LEAKAGE DETECTED."
    )

print("Leakage check: PASSED")

# ----------------------------------------------------------------------
# PREPROCESSING
# ----------------------------------------------------------------------

banner("PREPROCESSING")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numeric_columns
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(
    "Processed training features:",
    X_train_processed.shape[1]
)

print(
    "Processed testing features :",
    X_test_processed.shape[1]
)

# ----------------------------------------------------------------------
# OVERSAMPLING
# ----------------------------------------------------------------------

banner("FULL OVERSAMPLING")

X_train_processed_df = pd.DataFrame(
    X_train_processed.toarray()
    if hasattr(X_train_processed, "toarray")
    else X_train_processed
)

X_test_processed_df = pd.DataFrame(
    X_test_processed.toarray()
    if hasattr(X_test_processed, "toarray")
    else X_test_processed
)

X_balanced, y_balanced = oversample_full(
    X_train_processed_df,
    y_train.reset_index(drop=True),
    random_state=42
)

print(
    "Original training records :",
    len(X_train_processed_df)
)

print(
    "Balanced training records :",
    len(X_balanced)
)

print()
print("Balanced distribution:")
print(y_balanced.value_counts())

# ----------------------------------------------------------------------
# VALIDATION RUNS
# ----------------------------------------------------------------------

banner("VALIDATING BASELINE VS MAIZE -> WALNUT 1.15")

results = []

for run in range(1, 4):

    print()
    print("-" * 78)
    print(
        f"RUN {run} | Maize -> Walnut boost = 1.15"
    )
    print("-" * 78)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_balanced,
        y_balanced
    )

    probabilities = model.predict_proba(
        X_test_processed_df
    )

    classes = np.asarray(model.classes_)

    baseline_predictions = classes[
        np.argmax(probabilities, axis=1)
    ]

    boosted_predictions = apply_maize_walnut_boost(
        probabilities,
        classes,
        source="Maize",
        target="Walnut",
        boost=1.15
    )

    baseline_metrics = evaluate(
        y_test,
        baseline_predictions
    )

    boosted_metrics = evaluate(
        y_test,
        boosted_predictions
    )

    print()
    print("BASELINE")
    print(
        f"Accuracy    : "
        f"{baseline_metrics['accuracy']:.4f}"
    )
    print(
        f"Macro F1    : "
        f"{baseline_metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1 : "
        f"{baseline_metrics['weighted_f1']:.4f}"
    )

    print()
    print("MAIZE -> WALNUT 1.15")
    print(
        f"Accuracy    : "
        f"{boosted_metrics['accuracy']:.4f}"
    )
    print(
        f"Macro F1    : "
        f"{boosted_metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1 : "
        f"{boosted_metrics['weighted_f1']:.4f}"
    )

    results.append({
        "run": run,
        "model": "Baseline",
        **baseline_metrics
    })

    results.append({
        "run": run,
        "model": "Maize -> Walnut 1.15",
        **boosted_metrics
    })

# ----------------------------------------------------------------------
# RESULTS
# ----------------------------------------------------------------------

results_df = pd.DataFrame(results)

banner("VALIDATION SUMMARY")

print(results_df.to_string(index=False))

# ----------------------------------------------------------------------
# MEANS / STANDARD DEVIATIONS
# ----------------------------------------------------------------------

summary = (
    results_df
    .groupby("model")
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std")
    )
    .reset_index()
)

print()
print(summary.to_string(index=False))

# ----------------------------------------------------------------------
# DIFFERENCE
# ----------------------------------------------------------------------

baseline_summary = summary[
    summary["model"] == "Baseline"
].iloc[0]

boosted_summary = summary[
    summary["model"] == "Maize -> Walnut 1.15"
].iloc[0]

accuracy_change = (
    boosted_summary["accuracy_mean"]
    - baseline_summary["accuracy_mean"]
)

macro_change = (
    boosted_summary["macro_f1_mean"]
    - baseline_summary["macro_f1_mean"]
)

weighted_change = (
    boosted_summary["weighted_f1_mean"]
    - baseline_summary["weighted_f1_mean"]
)

banner("BOOST VS BASELINE")

print(
    f"Accuracy change    : "
    f"{accuracy_change:+.6f}"
)

print(
    f"Macro F1 change    : "
    f"{macro_change:+.6f}"
)

print(
    f"Weighted F1 change : "
    f"{weighted_change:+.6f}"
)

# ----------------------------------------------------------------------
# DECISION
# ----------------------------------------------------------------------

banner("VALIDATION DECISION")

if (
    macro_change > 0
    and accuracy_change >= 0
    and weighted_change >= 0
):
    decision = (
        "PROMISING - rule improves or preserves all "
        "primary metrics."
    )
elif macro_change > 0:
    decision = (
        "MIXED - Macro F1 improves, but another "
        "primary metric decreases."
    )
else:
    decision = (
        "REJECT - rule does not consistently improve "
        "the benchmark."
    )

print(decision)

# ----------------------------------------------------------------------
# SAVE REPORT
# ----------------------------------------------------------------------

banner("SAVING VALIDATION REPORT")

with pd.ExcelWriter(
    REPORT_PATH,
    engine="openpyxl"
) as writer:

    results_df.to_excel(
        writer,
        sheet_name="Run Results",
        index=False
    )

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    pd.DataFrame({
        "Metric": [
            "Accuracy change",
            "Macro F1 change",
            "Weighted F1 change",
            "Decision"
        ],
        "Value": [
            accuracy_change,
            macro_change,
            weighted_change,
            decision
        ]
    }).to_excel(
        writer,
        sheet_name="Decision",
        index=False
    )

print(
    "Report saved to:"
)
print(REPORT_PATH)

print()
print("=" * 78)
print("BEST RULE VALIDATION COMPLETE")
print("=" * 78)

print()
print("NO .pkl MODEL WAS CREATED.")
print("NO EXISTING MODEL WAS OVERWRITTEN.")
print("NO STREAMLIT FILE WAS CHANGED.")

print()
print("DO NOT DEPLOY YET.")

print("=" * 78)