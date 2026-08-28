from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

TRAIN_CSV = ARTIFACTS_DIR / "train_features.csv"
TEST_CSV = ARTIFACTS_DIR / "test_features.csv"

# IMPORTANT:
# This is an experiment, so we DO NOT overwrite
# the current 7-class production model.
MODEL_PATH = (
    ARTIFACTS_DIR
    / "quality_model_6class.joblib"
)

METADATA_PATH = (
    ARTIFACTS_DIR
    / "feature_metadata_6class.json"
)

REPORT_PATH = (
    ARTIFACTS_DIR
    / "evaluation_report_6class.txt"
)


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

MODEL_VERSION = "3.0-6class"

RANDOM_STATE = 42

EXCLUDED_CLASS = "defect"


# ---------------------------------------------------------
# MODEL FEATURES
# ---------------------------------------------------------

FEATURE_COLUMNS = [
    "brightness",
    "contrast",
    "sharpness",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "saturation",
    "edge_density",
    "noise_level",
    "entropy",
    "gradient_mean",
    "gradient_std",
    "strong_edge_ratio",
    "local_contrast_mean",
    "local_contrast_std",
    "patch_intensity_std",
    "patch_intensity_range",
    "extreme_pixel_ratio",
    "laplacian_extreme_ratio",
]

TARGET_COLUMN = "label"


# ---------------------------------------------------------
# DATASET LOADING
# ---------------------------------------------------------

def load_dataset():
    """
    Load train and test feature CSV files.
    """

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {TRAIN_CSV}"
        )

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Testing CSV not found: {TEST_CSV}"
        )

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    test_df = pd.read_csv(
        TEST_CSV
    )

    return (
        train_df,
        test_df,
    )


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_features(
    train_df,
    test_df,
):
    """
    Verify required columns exist.
    """

    required_columns = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_train = [
        column
        for column in required_columns
        if column not in train_df.columns
    ]

    missing_test = [
        column
        for column in required_columns
        if column not in test_df.columns
    ]

    if missing_train:
        raise ValueError(
            "Training CSV is missing columns: "
            + ", ".join(missing_train)
        )

    if missing_test:
        raise ValueError(
            "Testing CSV is missing columns: "
            + ", ".join(missing_test)
        )


# ---------------------------------------------------------
# SIX-CLASS FILTERING
# ---------------------------------------------------------

def prepare_six_class_dataset(
    train_df,
    test_df,
):
    """
    Remove synthetic 'defect' samples from ML training
    and evaluation.

    The ML classifier will focus on global image-quality
    conditions:

    clean
    blur
    underexposed
    overexposed
    noise
    degraded
    """

    train_filtered = train_df[
        train_df[TARGET_COLUMN]
        != EXCLUDED_CLASS
    ].copy()

    test_filtered = test_df[
        test_df[TARGET_COLUMN]
        != EXCLUDED_CLASS
    ].copy()

    return (
        train_filtered,
        test_filtered,
    )


# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------

def train_model(
    x_train,
    y_train,
):
    """
    Train Random Forest using the same configuration
    as the previous 7-class experiment.

    Keeping the algorithm and parameters identical makes
    the comparison between 7-class and 6-class models fair.
    """

    print(
        "\nTraining 6-class Random Forest model..."
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
    )

    return model


# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------

def evaluate_model(
    model,
    x_test,
    y_test,
):
    """
    Evaluate only on the held-out six-class test set.
    """

    predictions = model.predict(
        x_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions,
        )
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=model.classes_,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y_test,
        predictions,
        labels=model.classes_,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_,
    )

    return {
        "predictions": predictions,
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "report_dict": report_dict,
        "matrix": matrix,
    }


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

def get_feature_importance(
    model,
):

    feature_importance = {}

    for (
        feature_name,
        importance,
    ) in zip(
        FEATURE_COLUMNS,
        model.feature_importances_,
    ):

        feature_importance[
            feature_name
        ] = round(
            float(importance),
            6,
        )

    return feature_importance


# ---------------------------------------------------------
# METADATA
# ---------------------------------------------------------

def save_metadata(
    model,
    feature_importance,
    train_count,
    test_count,
    metrics,
):

    metadata = {
        "model_type": (
            "RandomForestClassifier"
        ),

        "model_version": MODEL_VERSION,

        "purpose": (
            "Global image-quality classification"
        ),

        "excluded_ml_class": (
            EXCLUDED_CLASS
        ),

        "feature_count": len(
            FEATURE_COLUMNS
        ),

        "feature_names": (
            FEATURE_COLUMNS
        ),

        "classes": (
            model.classes_.tolist()
        ),

        "training_samples": (
            train_count
        ),

        "testing_samples": (
            test_count
        ),

        "accuracy": round(
            float(
                metrics["accuracy"]
            ),
            6,
        ),

        "balanced_accuracy": round(
            float(
                metrics[
                    "balanced_accuracy"
                ]
            ),
            6,
        ),

        "macro_f1": round(
            float(
                metrics["macro_f1"]
            ),
            6,
        ),

        "weighted_f1": round(
            float(
                metrics["weighted_f1"]
            ),
            6,
        ),

        "feature_importance": (
            feature_importance
        ),

        "random_state": (
            RANDOM_STATE
        ),

        "n_estimators": 300,

        "max_depth": None,

        "class_weight": (
            "balanced"
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )


# ---------------------------------------------------------
# EVALUATION REPORT
# ---------------------------------------------------------

def save_report(
    model,
    metrics,
    feature_importance,
    train_count,
    test_count,
):

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "VISION AI - 6 CLASS MODEL EVALUATION\n"
        )

        file.write(
            "=" * 65
            + "\n\n"
        )

        file.write(
            "Experiment Purpose\n"
        )

        file.write(
            "-" * 65
            + "\n"
        )

        file.write(
            "Evaluate global image-quality classification "
            "without using localized synthetic defects as "
            "a Random Forest class.\n\n"
        )

        file.write(
            "Excluded ML class: defect\n"
        )

        file.write(
            "Potential visual defects will be evaluated "
            "separately using localized image analysis "
            "if this experiment performs better.\n\n"
        )

        file.write(
            f"Model: RandomForestClassifier\n"
        )

        file.write(
            f"Model Version: {MODEL_VERSION}\n"
        )

        file.write(
            f"Feature Count: "
            f"{len(FEATURE_COLUMNS)}\n"
        )

        file.write(
            f"Training Samples: "
            f"{train_count}\n"
        )

        file.write(
            f"Testing Samples: "
            f"{test_count}\n"
        )

        file.write(
            f"Classes: "
            f"{list(model.classes_)}\n\n"
        )

        file.write(
            "OVERALL METRICS\n"
        )

        file.write(
            "-" * 65
            + "\n"
        )

        file.write(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}\n"
        )

        file.write(
            f"Balanced Accuracy: "
            f"{metrics['balanced_accuracy']:.4f}\n"
        )

        file.write(
            f"Macro F1: "
            f"{metrics['macro_f1']:.4f}\n"
        )

        file.write(
            f"Weighted F1: "
            f"{metrics['weighted_f1']:.4f}\n\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )

        file.write(
            "-" * 65
            + "\n"
        )

        file.write(
            metrics["report"]
        )

        file.write(
            "\n\nCONFUSION MATRIX\n"
        )

        file.write(
            "-" * 65
            + "\n"
        )

        file.write(
            f"Classes: "
            f"{list(model.classes_)}\n"
        )

        file.write(
            str(
                metrics["matrix"]
            )
        )

        file.write(
            "\n\nFEATURE IMPORTANCE\n"
        )

        file.write(
            "-" * 65
            + "\n"
        )

        sorted_features = sorted(
            feature_importance.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for (
            name,
            importance,
        ) in sorted_features:

            file.write(
                f"{name:25}: "
                f"{importance:.6f}\n"
            )


# ---------------------------------------------------------
# PRINT CLASS SUMMARY
# ---------------------------------------------------------

def print_class_summary(
    model,
    report_dict,
):
    """
    Print the metrics we specifically care about
    when deciding whether the experiment succeeded.
    """

    print(
        "\nPER-CLASS RECALL"
    )

    print(
        "-" * 60
    )

    for class_name in model.classes_:

        recall = (
            report_dict[
                class_name
            ]["recall"]
        )

        precision = (
            report_dict[
                class_name
            ]["precision"]
        )

        f1 = (
            report_dict[
                class_name
            ]["f1-score"]
        )

        print(
            f"{class_name:15} "
            f"Precision: "
            f"{precision:.4f}  "
            f"Recall: "
            f"{recall:.4f}  "
            f"F1: "
            f"{f1:.4f}"
        )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print(
        "\nVISION AI 6-CLASS MODEL EXPERIMENT"
    )

    print(
        "=" * 60
    )

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    train_df, test_df = (
        load_dataset()
    )

    validate_features(
        train_df,
        test_df,
    )

    print(
        f"\nOriginal training samples: "
        f"{len(train_df)}"
    )

    print(
        f"Original testing samples : "
        f"{len(test_df)}"
    )

    # -----------------------------------------------------
    # REMOVE DEFECT CLASS
    # -----------------------------------------------------

    (
        train_df,
        test_df,
    ) = prepare_six_class_dataset(
        train_df,
        test_df,
    )

    print(
        f"\n6-class training samples : "
        f"{len(train_df)}"
    )

    print(
        f"6-class testing samples  : "
        f"{len(test_df)}"
    )

    print(
        f"ML features              : "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        "\nTraining class counts:"
    )

    print(
        train_df[
            TARGET_COLUMN
        ].value_counts().sort_index()
    )

    print(
        "\nTesting class counts:"
    )

    print(
        test_df[
            TARGET_COLUMN
        ].value_counts().sort_index()
    )

    # -----------------------------------------------------
    # PREPARE FEATURES
    # -----------------------------------------------------

    x_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    x_test = test_df[
        FEATURE_COLUMNS
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model = train_model(
        x_train,
        y_train,
    )

    # -----------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------

    metrics = evaluate_model(
        model,
        x_test,
        y_test,
    )

    print(
        "\nMODEL EVALUATION"
    )

    print(
        "=" * 60
    )

    print(
        f"\nAccuracy          : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1       : "
        f"{metrics['weighted_f1']:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        metrics["report"]
    )

    print(
        "Confusion Matrix:"
    )

    print(
        metrics["matrix"]
    )

    print_class_summary(
        model,
        metrics["report_dict"],
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    feature_importance = (
        get_feature_importance(
            model
        )
    )

    print(
        "\nFeature Importance:"
    )

    sorted_features = sorted(
        feature_importance.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for (
        feature_name,
        importance,
    ) in sorted_features:

        print(
            f"  {feature_name:25}: "
            f"{importance:.6f}"
        )

    # -----------------------------------------------------
    # SAVE EXPERIMENT
    # -----------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH,
    )

    save_metadata(
        model,
        feature_importance,
        len(train_df),
        len(test_df),
        metrics,
    )

    save_report(
        model,
        metrics,
        feature_importance,
        len(train_df),
        len(test_df),
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"Experimental model saved to:\n"
        f"{MODEL_PATH}"
    )

    print(
        f"\nMetadata saved to:\n"
        f"{METADATA_PATH}"
    )

    print(
        f"\nEvaluation report saved to:\n"
        f"{REPORT_PATH}"
    )

    print(
        "\n6-CLASS EXPERIMENT COMPLETE ✅"
    )


if __name__ == "__main__":
    main()