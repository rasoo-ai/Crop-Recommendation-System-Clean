"""
SMART KISAN - STEP 56
BEST CONFUSION-PAIR RULE VALIDATION

SAFE VALIDATION
- Uses the canonical Step 52 pipeline.
- Uses the canonical Step 53 benchmark configuration.
- Validates ONLY the Step 55 candidate:
      Pulses -> Mustard
      Boost = 1.10
- No .pkl model is created.
- Existing .pkl model is NOT changed.
- Streamlit is NOT changed.
- No application files are changed.
- Multiple deterministic repetitions are performed.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "output"
    / "Crop_Normalized.xlsx"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "output"
    / "Best_Rule_Validation.xlsx"
)

TARGET = "Crop"

RECOGNIZED_CROPS = [
    "Apple",
    "Cotton",
    "Maize",
    "Mustard",
    "Pulses",
    "Rice",
    "Vegetables",
    "Walnut",
    "Wheat",
]

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "State_Name",
    "Soil_Type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42
TEST_SIZE = 0.20

OVERSAMPLE_RANDOM_STATE = 42

PAIR_SOURCE = "Pulses"
PAIR_TARGET = "Mustard"
BOOST = 1.10

REPETITIONS = 3


# ============================================================================
# RANDOM FOREST CONFIGURATION
# ============================================================================

RF_CONFIG = {
    "n_estimators": 300,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================================
# HELPERS
# ============================================================================

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def fail(message):
    print("\n" + "!" * 78)
    print("VALIDATION FAILED")
    print("!" * 78)
    raise RuntimeError(message)


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def prepare_dataset():
    section("LOADING DATASET")

    print(f"Dataset path: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        fail(f"Dataset not found:\n{DATASET_PATH}")

    df = pd.read_excel(DATASET_PATH)

    print(f"Raw records: {len(df):,}")
    print(f"Raw columns: {len(df.columns)}")

    # ------------------------------------------------------------------------
    # REQUIRED COLUMN CHECK
    # ------------------------------------------------------------------------

    section("REQUIRED COLUMN CHECK")

    required = [TARGET] + FEATURES

    missing_columns = [
        col for col in required
        if col not in df.columns
    ]

    if missing_columns:
        fail(
            "Missing canonical columns:\n"
            + "\n".join(missing_columns)
        )

    print("All canonical columns are present.")

    # ------------------------------------------------------------------------
    # CROP FILTER
    # ------------------------------------------------------------------------

    section("BENCHMARK CROP FILTER")

    before_filter = len(df)

    df = df[
        df[TARGET].isin(RECOGNIZED_CROPS)
    ].copy()

    print(f"Before crop filtering : {before_filter:,}")
    print(f"After crop filtering  : {len(df):,}")
    print(f"Rows removed          : {before_filter - len(df):,}")

    # ------------------------------------------------------------------------
    # NUMERIC NORMALIZATION
    # ------------------------------------------------------------------------

    section("NUMERIC TYPE NORMALIZATION")

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    print(
        'All canonical numeric features converted with '
        'pd.to_numeric(..., errors="coerce")'
    )

    # ------------------------------------------------------------------------
    # MISSING VALUES
    # ------------------------------------------------------------------------

    section("MISSING VALUE HANDLING")

    missing_before = df[FEATURES].isna().sum()

    print("Missing values before cleaning:")

    any_missing = False

    for col, count in missing_before.items():
        if count > 0:
            any_missing = True
            print(f"{col:30s}: {int(count)}")

    if not any_missing:
        print("None")

    before_drop = len(df)

    df = df.dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    removed = before_drop - len(df)

    print(
        f"\nRows before missing-value removal : {before_drop:,}"
    )
    print(
        f"Rows removed                       : {removed:,}"
    )
    print(
        f"Benchmark records                  : {len(df):,}"
    )

    if len(df) != 5644:
        fail(
            f"Canonical benchmark record count mismatch. "
            f"Expected 5644, got {len(df)}."
        )

    # ------------------------------------------------------------------------
    # VALIDATE TYPES
    # ------------------------------------------------------------------------

    section("CANONICAL BENCHMARK VALIDATION")

    for col in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            fail(
                f"Numeric feature is not numeric: {col}"
            )

    for col in CATEGORICAL_FEATURES:
        if not (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        ):
            fail(
                f"Categorical feature has unexpected dtype: {col}"
            )

    print("Benchmark records: 5,644 [PASS]")
    print("Numeric feature validation: PASS")
    print("Categorical feature validation: PASS")

    return df


# ============================================================================
# SPLIT + PREPROCESS
# ============================================================================

def create_split_and_preprocess(df):
    section("TRAIN / TEST SPLIT")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_indices = set(X_train.index)
    test_indices = set(X_test.index)

    overlap = train_indices.intersection(test_indices)

    print(f"Training records : {len(X_train):,}")
    print(f"Testing records  : {len(X_test):,}")
    print(f"Index overlap    : {len(overlap)}")

    if overlap:
        fail("Train/test leakage detected.")

    print("Leakage check: PASSED")

    section("PREPROCESSING")

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    print(
        "Processed training shape: "
        f"{X_train_processed.shape}"
    )

    print(
        "Processed testing shape: "
        f"{X_test_processed.shape}"
    )

    if X_train_processed.shape[1] != 216:
        fail(
            "Processed feature count mismatch. "
            f"Expected 216, got {X_train_processed.shape[1]}."
        )

    print("Processed feature count: 216 [PASS]")

    return (
        X_train_processed,
        X_test_processed,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


# ============================================================================
# FULL RANDOM OVERSAMPLING
# ============================================================================

def full_oversample(X_train, y_train):
    section("FULL OVERSAMPLING")

    train_df = pd.DataFrame(
        X_train.toarray()
        if hasattr(X_train, "toarray")
        else X_train
    )

    train_df["_TARGET_"] = y_train.values

    max_count = (
        train_df["_TARGET_"]
        .value_counts()
        .max()
    )

    pieces = []

    for crop in RECOGNIZED_CROPS:
        crop_df = train_df[
            train_df["_TARGET_"] == crop
        ]

        sampled = resample(
            crop_df,
            replace=True,
            n_samples=max_count,
            random_state=OVERSAMPLE_RANDOM_STATE,
        )

        pieces.append(sampled)

    balanced = pd.concat(
        pieces,
        ignore_index=True,
    )

    y_balanced = balanced["_TARGET_"]
    X_balanced = balanced.drop(
        columns=["_TARGET_"]
    )

    print(
        f"Original training records : {len(train_df):,}"
    )

    print(
        f"Balanced training records : {len(balanced):,}"
    )

    print("\nBalanced distribution:")

    print(
        y_balanced.value_counts()
        .sort_index()
        .to_string()
    )

    if len(balanced) != 19593:
        fail(
            "Balanced training count mismatch. "
            f"Expected 19593, got {len(balanced)}."
        )

    return (
        X_balanced,
        y_balanced.reset_index(drop=True),
    )


# ============================================================================
# TRAIN MODEL
# ============================================================================

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        **RF_CONFIG
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


# ============================================================================
# DECISION RULE
# ============================================================================

def apply_pair_boost(
    probabilities,
    model_classes,
):
    """
    Only applies:

        Pulses -> Mustard
        boost = 1.10

    Rule is deliberately conservative.

    The boost is applied only when the current
    argmax prediction is Pulses.

    The Mustard probability is multiplied by 1.10,
    then probabilities are renormalized before
    selecting the final class.
    """

    probs = probabilities.copy()

    source_index = np.where(
        model_classes == PAIR_SOURCE
    )[0][0]

    target_index = np.where(
        model_classes == PAIR_TARGET
    )[0][0]

    original_predictions = np.argmax(
        probs,
        axis=1,
    )

    affected = (
        original_predictions == source_index
    )

    probs[affected, target_index] *= BOOST

    row_sums = probs.sum(axis=1)

    row_sums[row_sums == 0] = 1.0

    probs = probs / row_sums[:, None]

    final_predictions = np.argmax(
        probs,
        axis=1,
    )

    return (
        final_predictions,
        int(affected.sum()),
    )


# ============================================================================
# SINGLE VALIDATION RUN
# ============================================================================

def run_validation(
    X_train,
    y_train,
    X_test,
    y_test,
):
    model = train_model(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )

    classes = model.classes_

    baseline_indices = np.argmax(
        probabilities,
        axis=1,
    )

    baseline_predictions = classes[
        baseline_indices
    ]

    rule_indices, affected = apply_pair_boost(
        probabilities,
        classes,
    )

    rule_predictions = classes[
        rule_indices
    ]

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions,
    )

    rule_metrics = calculate_metrics(
        y_test,
        rule_predictions,
    )

    return (
        baseline_metrics,
        rule_metrics,
        affected,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)
    print("SMART KISAN - STEP 56 BEST RULE VALIDATION")
    print("=" * 78)

    print(
        """
SAFE VALIDATION

- Exact Step 52 canonical preprocessing is used.
- Exact Step 53 benchmark split is used.
- Exact full oversampling is used.
- Exact Random Forest configuration is used.
- Only Pulses -> Mustard boost 1.10 is validated.
- No .pkl model will be created.
- Existing .pkl model will NOT be changed.
- Streamlit will NOT be changed.
- No application files will be changed.
- Three deterministic repetitions are performed.
"""
    )

    # ------------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------------

    df = prepare_dataset()

    (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
    ) = create_split_and_preprocess(df)

    # ------------------------------------------------------------------------
    # OVERSAMPLING
    # ------------------------------------------------------------------------

    (
        X_balanced,
        y_balanced,
    ) = full_oversample(
        X_train_processed,
        y_train,
    )

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    section(
        "VALIDATING BASELINE VS "
        "PULSES -> MUSTARD 1.10"
    )

    all_results = []

    for run in range(
        1,
        REPETITIONS + 1,
    ):

        print("\n" + "-" * 78)
        print(
            f"RUN {run} | "
            f"Pulses -> Mustard boost = {BOOST}"
        )
        print("-" * 78)

        (
            baseline,
            rule,
            affected,
        ) = run_validation(
            X_balanced,
            y_balanced,
            X_test_processed,
            y_test,
        )

        print("\nBASELINE")

        print(
            f"Accuracy    : "
            f"{baseline['accuracy']:.4f}"
        )

        print(
            f"Macro F1    : "
            f"{baseline['macro_f1']:.4f}"
        )

        print(
            f"Weighted F1 : "
            f"{baseline['weighted_f1']:.4f}"
        )

        print(
            f"\nPULSES -> MUSTARD 1.10"
        )

        print(
            f"Accuracy    : "
            f"{rule['accuracy']:.4f}"
        )

        print(
            f"Macro F1    : "
            f"{rule['macro_f1']:.4f}"
        )

        print(
            f"Weighted F1 : "
            f"{rule['weighted_f1']:.4f}"
        )

        print(
            f"\nAffected predictions: {affected}"
        )

        all_results.append(
            {
                "run": run,
                "model": "Baseline",
                "accuracy": baseline["accuracy"],
                "macro_f1": baseline["macro_f1"],
                "weighted_f1": baseline["weighted_f1"],
            }
        )

        all_results.append(
            {
                "run": run,
                "model": (
                    "Pulses -> Mustard 1.10"
                ),
                "accuracy": rule["accuracy"],
                "macro_f1": rule["macro_f1"],
                "weighted_f1": rule["weighted_f1"],
            }
        )

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    section("VALIDATION SUMMARY")

    results_df = pd.DataFrame(
        all_results
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    summary_df = (
        results_df
        .groupby("model")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            weighted_f1_mean=("weighted_f1", "mean"),
            weighted_f1_std=("weighted_f1", "std"),
        )
        .reset_index()
    )

    summary_df = summary_df.fillna(0)

    print(
        "\n"
        + summary_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------------
    # COMPARE
    # ------------------------------------------------------------------------

    section("BOOST VS BASELINE")

    baseline_summary = summary_df[
        summary_df["model"] == "Baseline"
    ].iloc[0]

    rule_summary = summary_df[
        summary_df["model"]
        == "Pulses -> Mustard 1.10"
    ].iloc[0]

    accuracy_change = (
        rule_summary["accuracy_mean"]
        - baseline_summary["accuracy_mean"]
    )

    macro_change = (
        rule_summary["macro_f1_mean"]
        - baseline_summary["macro_f1_mean"]
    )

    weighted_change = (
        rule_summary["weighted_f1_mean"]
        - baseline_summary["weighted_f1_mean"]
    )

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

    # ------------------------------------------------------------------------
    # DECISION
    # ------------------------------------------------------------------------

    section("VALIDATION DECISION")

    if (
        macro_change > 0
        and accuracy_change >= 0
        and weighted_change >= 0
    ):
        decision = (
            "PASS - rule consistently improves "
            "or preserves the canonical benchmark."
        )
    else:
        decision = (
            "REJECT - rule does not consistently "
            "improve the canonical benchmark."
        )

    print(decision)

    # ------------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------------

    section("SAVING VALIDATION REPORT")

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        REPORT_PATH,
        engine="openpyxl",
    ) as writer:

        results_df.to_excel(
            writer,
            sheet_name="Run_Results",
            index=False,
        )

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        comparison_df = pd.DataFrame(
            [
                {
                    "Candidate":
                        "Pulses -> Mustard 1.10",
                    "Baseline_Accuracy":
                        baseline_summary[
                            "accuracy_mean"
                        ],
                    "Rule_Accuracy":
                        rule_summary[
                            "accuracy_mean"
                        ],
                    "Accuracy_Change":
                        accuracy_change,
                    "Baseline_Macro_F1":
                        baseline_summary[
                            "macro_f1_mean"
                        ],
                    "Rule_Macro_F1":
                        rule_summary[
                            "macro_f1_mean"
                        ],
                    "Macro_F1_Change":
                        macro_change,
                    "Baseline_Weighted_F1":
                        baseline_summary[
                            "weighted_f1_mean"
                        ],
                    "Rule_Weighted_F1":
                        rule_summary[
                            "weighted_f1_mean"
                        ],
                    "Weighted_F1_Change":
                        weighted_change,
                    "Decision":
                        decision,
                }
            ]
        )

        comparison_df.to_excel(
            writer,
            sheet_name="Decision",
            index=False,
        )

    print(
        f"Report saved to:\n{REPORT_PATH}"
    )

    # ------------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------------

    section(
        "BEST RULE VALIDATION COMPLETE"
    )

    print(
        """
NO .pkl MODEL WAS CREATED.
NO EXISTING MODEL WAS OVERWRITTEN.
NO STREAMLIT FILE WAS CHANGED.

DO NOT DEPLOY YET.
"""
    )


if __name__ == "__main__":
    main()