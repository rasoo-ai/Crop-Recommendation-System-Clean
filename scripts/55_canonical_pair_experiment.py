"""
SMART KISAN - CANONICAL CONFUSION-PAIR EXPERIMENT
=================================================

SAFE EXPERIMENT

- Uses the exact Step-52 canonical preprocessing pipeline.
- Uses the exact Step-53 canonical train/test split.
- Uses the exact full oversampling procedure.
- Uses the same Random Forest configuration.
- Does NOT create a .pkl model.
- Does NOT modify the existing .pkl model.
- Does NOT modify Streamlit.
- Does NOT modify application files.
- Tests only confusion pairs identified by Step 54.
- Every rule is compared against the exact canonical baseline.

IMPORTANT:
This experiment only evaluates targeted probability decision rules.
No rule is approved for deployment here.
"""

from pathlib import Path
import sys
import warnings

import numpy as np
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


warnings.filterwarnings("ignore")


# ============================================================================
# PATHS
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
    / "Canonical_Confusion_Pair_Experiment.xlsx"
)


# ============================================================================
# CANONICAL DEFINITIONS FROM STEP 52
# ============================================================================

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


# ============================================================================
# RANDOM FOREST CONFIGURATION
# ============================================================================
#
# Keep this identical to the Random Forest configuration used by Step 53.
# If Step 53 contains additional parameters in your project, copy them here
# exactly rather than changing them.
# ============================================================================

RF_CONFIG = {
    "n_estimators": 300,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================================
# CONFUSION PAIRS FROM STEP 54
# ============================================================================
#
# These are supported by actual errors in the canonical confusion matrix.
#
# For directional pairs:
#     source -> target
#
# For Vegetables/Wheat we test both directions independently.
# ============================================================================

PAIR_CANDIDATES = [
    ("Maize", "Mustard"),
    ("Rice", "Mustard"),
    ("Maize", "Rice"),
    ("Pulses", "Mustard"),
    ("Rice", "Apple"),
    ("Vegetables", "Wheat"),
    ("Wheat", "Vegetables"),
    ("Rice", "Pulses"),
    ("Maize", "Wheat"),
    ("Maize", "Walnut"),
]


# ============================================================================
# BOOST VALUES
# ============================================================================

BOOST_VALUES = [
    1.05,
    1.10,
    1.15,
    1.20,
]


# ============================================================================
# UTILITY
# ============================================================================

def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fail(message):
    print()
    print("!" * 78)
    print("CANONICAL EXPERIMENT FAILED")
    print("!" * 78)
    print(message)
    raise RuntimeError(message)


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_columns(df):
    required = FEATURES + [TARGET]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        fail(
            "Missing required canonical columns:\n"
            + "\n".join(f" - {x}" for x in missing)
        )


# ============================================================================
# PREPROCESSING
# ============================================================================

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ============================================================================
# FULL RANDOM OVERSAMPLING
# ============================================================================

def full_random_oversample(X, y):
    """
    Exact full random oversampling.

    Every class is sampled with replacement until it reaches
    the size of the largest class.
    """

    rng = np.random.RandomState(RANDOM_STATE)

    y_series = pd.Series(
        y,
        index=X.index,
        name=TARGET,
    )

    class_counts = y_series.value_counts()

    target_count = class_counts.max()

    X_parts = []
    y_parts = []

    for crop in RECOGNIZED_CROPS:
        class_indices = y_series[
            y_series == crop
        ].index.to_numpy()

        if len(class_indices) == 0:
            fail(
                f"Oversampling failed: class '{crop}' "
                "is missing from training data."
            )

        if len(class_indices) < target_count:
            sampled_indices = rng.choice(
                class_indices,
                size=target_count,
                replace=True,
            )
        else:
            sampled_indices = class_indices

        X_parts.append(
            X.loc[sampled_indices]
        )

        y_parts.append(
            y_series.loc[sampled_indices]
        )

    X_balanced = pd.concat(
        X_parts,
        axis=0,
    ).reset_index(drop=True)

    y_balanced = pd.concat(
        y_parts,
        axis=0,
    ).reset_index(drop=True)

    return X_balanced, y_balanced


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    report = classification_report(
        y_true,
        y_pred,
        labels=RECOGNIZED_CROPS,
        output_dict=True,
        zero_division=0,
    )

    weak_f1_values = [
        report[crop]["f1-score"]
        for crop in RECOGNIZED_CROPS
        if report[crop]["support"] > 0
    ]

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=RECOGNIZED_CROPS,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=RECOGNIZED_CROPS,
            average="weighted",
            zero_division=0,
        ),
        "weak_crop_f1": (
            min(weak_f1_values)
            if weak_f1_values
            else 0.0
        ),
    }


# ============================================================================
# DECISION RULE
# ============================================================================

def apply_pair_boost(
    probabilities,
    classes,
    baseline_predictions,
    source_crop,
    target_crop,
    boost,
):
    """
    Targeted rule:

    Only when the baseline prediction is source_crop,
    multiply target_crop probability by the requested boost.

    Then re-select the class with maximum probability.

    This prevents unrelated predictions from being changed.
    """

    adjusted = probabilities.copy()

    class_to_index = {
        str(crop): index
        for index, crop in enumerate(classes)
    }

    if source_crop not in class_to_index:
        return baseline_predictions.copy()

    if target_crop not in class_to_index:
        return baseline_predictions.copy()

    source_index = class_to_index[source_crop]
    target_index = class_to_index[target_crop]

    result = baseline_predictions.copy()

    source_mask = (
        baseline_predictions == source_crop
    )

    if not np.any(source_mask):
        return result

    adjusted[source_mask, target_index] *= boost

    selected_indices = np.argmax(
        adjusted[source_mask],
        axis=1,
    )

    class_array = np.asarray(classes)

    result[source_mask] = class_array[
        selected_indices
    ]

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():

    section(
        "SMART KISAN - CANONICAL CONFUSION-PAIR EXPERIMENT"
    )

    print(
        """
SAFE EXPERIMENT

- Exact Step-52 canonical preprocessing is used.
- Exact Step-53 train/test split is used.
- Exact full oversampling is used.
- Exact canonical Random Forest configuration is used.
- No .pkl model will be created.
- Existing .pkl model will NOT be changed.
- Streamlit will NOT be changed.
- No application files will be changed.
- Only actual Step-54 confusion pairs are tested.
"""
    )

    # ========================================================================
    # DATASET
    # ========================================================================

    section("LOADING DATASET")

    print(f"Dataset path: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        fail(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_excel(DATASET_PATH)

    print(f"Raw records: {len(df):,}")
    print(f"Raw columns: {len(df.columns)}")

    validate_columns(df)

    print("Required canonical columns: PASS")

    # ========================================================================
    # CROP FILTER
    # ========================================================================

    section("BENCHMARK CROP FILTER")

    before_filter = len(df)

    df = df[
        df[TARGET].isin(RECOGNIZED_CROPS)
    ].copy()

    after_filter = len(df)

    print(
        f"Before crop filtering : {before_filter:,}"
    )
    print(
        f"After crop filtering  : {after_filter:,}"
    )
    print(
        f"Rows removed          : "
        f"{before_filter - after_filter:,}"
    )

    print()
    print(
        df[TARGET].value_counts()
        .reindex(RECOGNIZED_CROPS)
        .fillna(0)
        .astype(int)
        .to_string()
    )

    # ========================================================================
    # NUMERIC NORMALIZATION
    # ========================================================================

    section("NUMERIC TYPE NORMALIZATION")

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print(
        "All 16 canonical numeric features converted "
        "with pd.to_numeric(..., errors='coerce')."
    )

    # ========================================================================
    # MISSING VALUES
    # ========================================================================

    section("MISSING VALUE HANDLING")

    missing_before = (
        df[FEATURES]
        .isna()
        .sum()
    )

    print("Missing values before cleaning:")

    for column, count in missing_before.items():
        if count > 0:
            print(
                f"{column:<30}: {count}"
            )

    before_clean = len(df)

    df = df.dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    after_clean = len(df)

    print()
    print(
        f"Rows before missing-value removal : "
        f"{before_clean:,}"
    )
    print(
        f"Rows removed                       : "
        f"{before_clean - after_clean:,}"
    )
    print(
        f"Benchmark records                  : "
        f"{after_clean:,}"
    )

    if after_clean != 5644:
        fail(
            "Canonical benchmark record count mismatch.\n"
            f"Expected: 5644\n"
            f"Actual  : {after_clean}"
        )

    # ========================================================================
    # TARGET VALIDATION
    # ========================================================================

    section("CANONICAL TARGET VALIDATION")

    target_counts = (
        df[TARGET]
        .value_counts()
        .reindex(RECOGNIZED_CROPS)
        .fillna(0)
        .astype(int)
    )

    print(
        target_counts.to_string()
    )

    if target_counts.eq(0).any():
        fail(
            "At least one canonical crop has zero records."
        )

    # ========================================================================
    # TRAIN / TEST SPLIT
    # ========================================================================

    section("TRAIN / TEST SPLIT")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"Training records : {len(X_train):,}"
    )
    print(
        f"Testing records  : {len(X_test):,}"
    )

    train_indices = set(X_train.index)
    test_indices = set(X_test.index)

    overlap = train_indices.intersection(
        test_indices
    )

    print(
        f"Index overlap     : {len(overlap)}"
    )

    if overlap:
        fail(
            "Train/test leakage detected."
        )

    print("Leakage check: PASSED")

    if len(X_train) != 4515:
        fail(
            f"Expected 4,515 training records, "
            f"got {len(X_train)}."
        )

    if len(X_test) != 1129:
        fail(
            f"Expected 1,129 testing records, "
            f"got {len(X_test)}."
        )

    # ========================================================================
    # PREPROCESSING
    # ========================================================================

    section("PREPROCESSING")

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    print(
        f"Processed training shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed testing shape : "
        f"{X_test_processed.shape}"
    )

    processed_features = (
        X_train_processed.shape[1]
    )

    print(
        f"Processed feature count : "
        f"{processed_features}"
    )

    if processed_features != 216:
        fail(
            "Canonical processed feature count mismatch.\n"
            f"Expected: 216\n"
            f"Actual  : {processed_features}"
        )

    # ========================================================================
    # OVERSAMPLING
    # ========================================================================

    section("FULL OVERSAMPLING")

    X_train_balanced, y_train_balanced = (
        full_random_oversample(
            X_train,
            y_train,
        )
    )

    X_train_balanced_processed = (
        preprocessor.transform(
            X_train_balanced
        )
    )

    print(
        f"Original training records : "
        f"{len(X_train):,}"
    )

    print(
        f"Balanced training records : "
        f"{len(X_train_balanced):,}"
    )

    balanced_counts = (
        y_train_balanced
        .value_counts()
        .reindex(RECOGNIZED_CROPS)
    )

    print()
    print(
        balanced_counts.to_string()
    )

    if len(X_train_balanced) != 19593:
        fail(
            "Canonical balanced training count mismatch.\n"
            f"Expected: 19593\n"
            f"Actual  : {len(X_train_balanced)}"
        )

    # ========================================================================
    # RANDOM FOREST
    # ========================================================================

    section(
        "TRAINING CANONICAL RANDOM FOREST"
    )

    print("Training...")

    model = RandomForestClassifier(
        **RF_CONFIG
    )

    model.fit(
        X_train_balanced_processed,
        y_train_balanced,
    )

    print("Training complete.")

    # ========================================================================
    # BASELINE PREDICTIONS
    # ========================================================================

    section("BASELINE PREDICTIONS")

    probabilities = model.predict_proba(
        X_test_processed
    )

    classes = model.classes_

    baseline_indices = np.argmax(
        probabilities,
        axis=1,
    )

    baseline_predictions = np.asarray(
        classes
    )[baseline_indices]

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions,
    )

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

    print(
        f"Weak Crop F1: "
        f"{baseline_metrics['weak_crop_f1']:.4f}"
    )

    # ========================================================================
    # VERIFY BASELINE CONFUSION PAIRS
    # ========================================================================

    section(
        "VERIFYING STEP-54 CONFUSION PAIRS"
    )

    cm = confusion_matrix(
        y_test,
        baseline_predictions,
        labels=RECOGNIZED_CROPS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=RECOGNIZED_CROPS,
        columns=RECOGNIZED_CROPS,
    )

    actual_pairs = []

    for source_crop, target_crop in PAIR_CANDIDATES:

        source_to_target = int(
            cm_df.loc[
                source_crop,
                target_crop,
            ]
        )

        target_to_source = int(
            cm_df.loc[
                target_crop,
                source_crop,
            ]
        )

        total_pair_errors = (
            source_to_target
            + target_to_source
        )

        actual_pairs.append(
            {
                "Source Crop": source_crop,
                "Target Crop": target_crop,
                "Forward Errors": source_to_target,
                "Reverse Errors": target_to_source,
                "Total Pair Errors": total_pair_errors,
            }
        )

    pair_check_df = pd.DataFrame(
        actual_pairs
    )

    print(
        pair_check_df.to_string(
            index=False
        )
    )

    # ========================================================================
    # INDIVIDUAL PAIR EXPERIMENTS
    # ========================================================================

    section(
        "INDIVIDUAL CONFUSION-PAIR EXPERIMENTS"
    )

    results = []

    for (
        source_crop,
        target_crop,
    ) in PAIR_CANDIDATES:

        print()
        print("-" * 78)
        print(
            f"PAIR: {source_crop} -> {target_crop}"
        )
        print("-" * 78)

        source_index = (
            RECOGNIZED_CROPS.index(
                source_crop
            )
        )

        target_index = (
            RECOGNIZED_CROPS.index(
                target_crop
            )
        )

        forward_errors = int(
            cm_df.loc[
                source_crop,
                target_crop,
            ]
        )

        reverse_errors = int(
            cm_df.loc[
                target_crop,
                source_crop,
            ]
        )

        total_pair_errors = (
            forward_errors
            + reverse_errors
        )

        for boost in BOOST_VALUES:

            adjusted_predictions = (
                apply_pair_boost(
                    probabilities=probabilities,
                    classes=classes,
                    baseline_predictions=(
                        baseline_predictions.copy()
                    ),
                    source_crop=source_crop,
                    target_crop=target_crop,
                    boost=boost,
                )
            )

            metrics = calculate_metrics(
                y_test,
                adjusted_predictions,
            )

            changed_predictions = int(
                np.sum(
                    adjusted_predictions
                    != baseline_predictions
                )
            )

            print(
                f"{source_crop} -> {target_crop} "
                f"Boost {boost:.2f}    "
                f"Accuracy: "
                f"{metrics['accuracy']:.4f} | "
                f"Macro F1: "
                f"{metrics['macro_f1']:.4f} | "
                f"Weak F1: "
                f"{metrics['weak_crop_f1']:.4f}"
            )

            results.append(
                {
                    "Source Crop": source_crop,
                    "Target Crop": target_crop,
                    "Boost": boost,
                    "Forward Errors": forward_errors,
                    "Reverse Errors": reverse_errors,
                    "Total Pair Errors": total_pair_errors,
                    "Changed Predictions": changed_predictions,
                    "Accuracy": metrics["accuracy"],
                    "Macro F1": metrics["macro_f1"],
                    "Weighted F1": metrics[
                        "weighted_f1"
                    ],
                    "Weak Crop F1": metrics[
                        "weak_crop_f1"
                    ],
                    "Accuracy Change": (
                        metrics["accuracy"]
                        - baseline_metrics["accuracy"]
                    ),
                    "Macro F1 Change": (
                        metrics["macro_f1"]
                        - baseline_metrics["macro_f1"]
                    ),
                    "Weighted F1 Change": (
                        metrics["weighted_f1"]
                        - baseline_metrics["weighted_f1"]
                    ),
                    "Weak Crop F1 Change": (
                        metrics["weak_crop_f1"]
                        - baseline_metrics["weak_crop_f1"]
                    ),
                }
            )

    results_df = pd.DataFrame(
        results
    )

    # ========================================================================
    # RANKING
    # ========================================================================

    section(
        "EXPERIMENT RANKING"
    )

    ranking_df = (
        results_df
        .sort_values(
            by=[
                "Macro F1",
                "Weak Crop F1",
                "Accuracy",
                "Weighted F1",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    ranking_df.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(ranking_df) + 1,
        ),
    )

    display_columns = [
        "Rank",
        "Source Crop",
        "Target Crop",
        "Boost",
        "Changed Predictions",
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Weak Crop F1",
        "Macro F1 Change",
        "Weak Crop F1 Change",
    ]

    print(
        ranking_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================================
    # BEST CONFIGURATION
    # ========================================================================

    section(
        "BEST CONFIGURATION"
    )

    best = ranking_df.iloc[0]

    print(
        f"Pair           : "
        f"{best['Source Crop']} -> "
        f"{best['Target Crop']}"
    )

    print(
        f"Boost          : "
        f"{best['Boost']:.2f}"
    )

    print(
        f"Accuracy       : "
        f"{best['Accuracy']:.4f}"
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

    print(
        f"Macro F1 change: "
        f"{best['Macro F1 Change']:+.4f}"
    )

    print(
        f"Weak F1 change : "
        f"{best['Weak Crop F1 Change']:+.4f}"
    )

    # ========================================================================
    # IMPORTANT DECISION
    # ========================================================================

    section(
        "STEP 55 DECISION"
    )

    improvement = (
        best["Macro F1"]
        > baseline_metrics["macro_f1"]
    )

    weak_improvement = (
        best["Weak Crop F1"]
        > baseline_metrics["weak_crop_f1"]
    )

    accuracy_not_worse = (
        best["Accuracy"]
        >= baseline_metrics["accuracy"]
    )

    if (
        improvement
        and weak_improvement
        and accuracy_not_worse
    ):
        decision = (
            "CANDIDATE - requires Step 56 "
            "deterministic validation."
        )
    else:
        decision = (
            "NO CLEAR IMPROVEMENT - "
            "do not deploy."
        )

    print()
    print(decision)

    # ========================================================================
    # SAVE REPORT
    # ========================================================================

    section(
        "SAVING EXPERIMENT REPORT"
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_df = pd.DataFrame(
        [
            {
                "Model": "Canonical Baseline",
                "Accuracy": baseline_metrics[
                    "accuracy"
                ],
                "Macro F1": baseline_metrics[
                    "macro_f1"
                ],
                "Weighted F1": baseline_metrics[
                    "weighted_f1"
                ],
                "Weak Crop F1": baseline_metrics[
                    "weak_crop_f1"
                ],
            }
        ]
    )

    with pd.ExcelWriter(
        REPORT_PATH,
        engine="openpyxl",
    ) as writer:

        baseline_df.to_excel(
            writer,
            sheet_name="Baseline",
            index=False,
        )

        cm_df.to_excel(
            writer,
            sheet_name="Confusion Matrix",
        )

        pair_check_df.to_excel(
            writer,
            sheet_name="Pair Verification",
            index=False,
        )

        results_df.to_excel(
            writer,
            sheet_name="All Experiments",
            index=False,
        )

        ranking_df.to_excel(
            writer,
            sheet_name="Ranking",
            index=False,
        )

        pd.DataFrame(
            [
                {
                    "Decision": decision,
                    "Best Source": best[
                        "Source Crop"
                    ],
                    "Best Target": best[
                        "Target Crop"
                    ],
                    "Best Boost": best[
                        "Boost"
                    ],
                    "Baseline Accuracy": (
                        baseline_metrics[
                            "accuracy"
                        ]
                    ),
                    "Best Accuracy": best[
                        "Accuracy"
                    ],
                    "Baseline Macro F1": (
                        baseline_metrics[
                            "macro_f1"
                        ]
                    ),
                    "Best Macro F1": best[
                        "Macro F1"
                    ],
                    "Baseline Weak F1": (
                        baseline_metrics[
                            "weak_crop_f1"
                        ]
                    ),
                    "Best Weak F1": best[
                        "Weak Crop F1"
                    ],
                }
            ]
        ).to_excel(
            writer,
            sheet_name="Decision",
            index=False,
        )

    print(
        f"Report saved to:\n{REPORT_PATH}"
    )

    # ========================================================================
    # FINAL SAFETY MESSAGE
    # ========================================================================

    section(
        "CANONICAL CONFUSION-PAIR EXPERIMENT COMPLETE"
    )

    print(
        f"""
Baseline Accuracy    : {baseline_metrics['accuracy']:.4f}
Baseline Macro F1    : {baseline_metrics['macro_f1']:.4f}
Baseline Weighted F1 : {baseline_metrics['weighted_f1']:.4f}
Baseline Weak Crop F1: {baseline_metrics['weak_crop_f1']:.4f}

Best Pair:
    {best['Source Crop']} -> {best['Target Crop']}

Best Boost:
    {best['Boost']:.2f}

Best Accuracy:
    {best['Accuracy']:.4f}

Best Macro F1:
    {best['Macro F1']:.4f}

Decision:
    {decision}

NO .pkl MODEL CREATED.
NO EXISTING MODEL OVERWRITTEN.
NO STREAMLIT FILE CHANGED.

IMPORTANT:
The best rule is NOT automatically approved for deployment.
Step 56 must validate any candidate rule before deployment.
"""
    )


if __name__ == "__main__":
    main()