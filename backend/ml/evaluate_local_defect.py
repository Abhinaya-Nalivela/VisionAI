from pathlib import Path
import sys

import cv2
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------
# MAKE BACKEND PACKAGE IMPORTABLE
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BACKEND_DIR),
    )


from app.services.local_anomaly import (
    extract_local_anomaly_features,
)


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATASET_DIR = (
    BACKEND_DIR
    / "dataset"
    / "generated"
)

TRAIN_DIR = (
    DATASET_DIR
    / "train"
)

TEST_DIR = (
    DATASET_DIR
    / "test"
)

REPORT_PATH = (
    BACKEND_DIR
    / "artifacts"
    / "local_defect_evaluation.txt"
)


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

# Conservative requirement:
# keep clean-image false positives at or below 10%
# while selecting the threshold from training data.
MAX_ALLOWED_TRAIN_FALSE_POSITIVE_RATE = 0.10


# ---------------------------------------------------------
# LOAD IMAGE PATHS
# ---------------------------------------------------------

def get_images(folder):
    """
    Return supported image files from a folder.
    """

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}"
        )

    images = [
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    return sorted(
        images
    )


# ---------------------------------------------------------
# ANALYZE ONE IMAGE
# ---------------------------------------------------------

def get_anomaly_score(
    image_path,
):
    """
    Load one image and calculate its local anomaly features.
    """

    image = cv2.imread(
        str(
            image_path
        )
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    result = (
        extract_local_anomaly_features(
            image
        )
    )

    return result


# ---------------------------------------------------------
# BUILD CLEAN VS DEFECT DATASET
# ---------------------------------------------------------

def build_binary_dataset(
    split_dir,
):
    """
    Build a binary evaluation dataset.

    clean  = 0
    defect = 1
    """

    rows = []

    class_mapping = {
        "clean": 0,
        "defect": 1,
    }

    for (
        class_name,
        target,
    ) in class_mapping.items():

        class_dir = (
            split_dir
            / class_name
        )

        image_paths = (
            get_images(
                class_dir
            )
        )

        print(
            f"{class_name:10}: "
            f"{len(image_paths)} images"
        )

        for image_path in image_paths:

            local_features = (
                get_anomaly_score(
                    image_path
                )
            )

            rows.append(
                {
                    "filename": (
                        image_path.name
                    ),

                    "label": (
                        target
                    ),

                    "class_name": (
                        class_name
                    ),

                    "score": float(
                        local_features[
                            "local_anomaly_score"
                        ]
                    ),

                    "features": (
                        local_features
                    ),
                }
            )

    return rows


# ---------------------------------------------------------
# THRESHOLD SELECTION
# ---------------------------------------------------------

def select_threshold(
    train_rows,
):
    """
    Select threshold using TRAINING DATA ONLY.

    Goal:
    1. Keep clean false-positive rate <= 10%
    2. Maximize defect recall
    3. Prefer higher F1 if recall ties
    """

    y_true = np.array(
        [
            row["label"]
            for row in train_rows
        ]
    )

    scores = np.array(
        [
            row["score"]
            for row in train_rows
        ],
        dtype=np.float32,
    )

    candidates = np.arange(
        0.30,
        0.951,
        0.01,
    )

    valid_results = []

    for threshold in candidates:

        predictions = (
            scores >= threshold
        ).astype(
            int
        )

        clean_mask = (
            y_true == 0
        )

        clean_count = int(
            np.sum(
                clean_mask
            )
        )

        false_positives = int(
            np.sum(
                predictions[
                    clean_mask
                ] == 1
            )
        )

        if clean_count > 0:
            false_positive_rate = (
                false_positives
                / clean_count
            )
        else:
            false_positive_rate = 0.0

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if (
            false_positive_rate
            <= MAX_ALLOWED_TRAIN_FALSE_POSITIVE_RATE
        ):

            valid_results.append(
                {
                    "threshold": round(
                        float(
                            threshold
                        ),
                        2,
                    ),

                    "false_positive_rate": (
                        false_positive_rate
                    ),

                    "precision": (
                        precision
                    ),

                    "recall": (
                        recall
                    ),

                    "f1": (
                        f1
                    ),
                }
            )

    if not valid_results:
        raise RuntimeError(
            "No threshold achieved the required "
            "training false-positive rate."
        )

    best = max(
        valid_results,
        key=lambda item: (
            item["recall"],
            item["f1"],
            item["threshold"],
        ),
    )

    return best


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate(
    rows,
    threshold,
):
    """
    Evaluate a fixed threshold on a dataset.
    """

    y_true = np.array(
        [
            row["label"]
            for row in rows
        ]
    )

    scores = np.array(
        [
            row["score"]
            for row in rows
        ],
        dtype=np.float32,
    )

    predictions = (
        scores >= threshold
    ).astype(
        int
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[
            0,
            1,
        ],
    )

    report = classification_report(
        y_true,
        predictions,
        target_names=[
            "clean",
            "defect",
        ],
        digits=4,
        zero_division=0,
    )

    tn, fp, fn, tp = (
        matrix.ravel()
    )

    if (
        fp + tn
    ) > 0:

        false_positive_rate = (
            fp
            / (
                fp
                + tn
            )
        )

    else:
        false_positive_rate = 0.0

    return {
        "accuracy": (
            accuracy
        ),

        "precision": (
            precision
        ),

        "recall": (
            recall
        ),

        "f1": (
            f1
        ),

        "false_positive_rate": (
            false_positive_rate
        ),

        "matrix": (
            matrix
        ),

        "report": (
            report
        ),

        "tn": int(
            tn
        ),

        "fp": int(
            fp
        ),

        "fn": int(
            fn
        ),

        "tp": int(
            tp
        ),
    }


# ---------------------------------------------------------
# SCORE DISTRIBUTION
# ---------------------------------------------------------

def print_score_summary(
    rows,
    title,
):
    """
    Print clean and defect anomaly-score distributions.
    """

    print(
        f"\n{title}"
    )

    print(
        "-" * 60
    )

    for (
        class_name,
        label,
    ) in [
        (
            "clean",
            0,
        ),
        (
            "defect",
            1,
        ),
    ]:

        scores = np.array(
            [
                row["score"]
                for row in rows
                if row["label"]
                == label
            ],
            dtype=np.float32,
        )

        print(
            f"{class_name:10} "
            f"count={len(scores):3d}  "
            f"mean={np.mean(scores):.4f}  "
            f"median={np.median(scores):.4f}  "
            f"min={np.min(scores):.4f}  "
            f"max={np.max(scores):.4f}"
        )


# ---------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------

def save_report(
    threshold_info,
    train_metrics,
    test_metrics,
):
    """
    Save the defect evaluation results.
    """

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "VISION AI LOCAL DEFECT EVALUATION\n"
        )

        file.write(
            "=" * 65
            + "\n\n"
        )

        file.write(
            "Method:\n"
        )

        file.write(
            "Patch-based local anomaly detection "
            "using robust within-image comparisons.\n\n"
        )

        file.write(
            "Threshold Selection:\n"
        )

        file.write(
            "Threshold selected using TRAINING DATA ONLY.\n"
        )

        file.write(
            "Maximum allowed training clean-image "
            "false-positive rate: 10%.\n\n"
        )

        file.write(
            f"Selected threshold: "
            f"{threshold_info['threshold']:.2f}\n"
        )

        file.write(
            f"Training false-positive rate: "
            f"{threshold_info['false_positive_rate']:.4f}\n"
        )

        file.write(
            f"Training defect precision: "
            f"{threshold_info['precision']:.4f}\n"
        )

        file.write(
            f"Training defect recall: "
            f"{threshold_info['recall']:.4f}\n"
        )

        file.write(
            f"Training defect F1: "
            f"{threshold_info['f1']:.4f}\n\n"
        )

        file.write(
            "FINAL HELD-OUT TEST RESULTS\n"
        )

        file.write(
            "-" * 65
            + "\n"
        )

        file.write(
            f"Accuracy: "
            f"{test_metrics['accuracy']:.4f}\n"
        )

        file.write(
            f"Defect Precision: "
            f"{test_metrics['precision']:.4f}\n"
        )

        file.write(
            f"Defect Recall: "
            f"{test_metrics['recall']:.4f}\n"
        )

        file.write(
            f"Defect F1: "
            f"{test_metrics['f1']:.4f}\n"
        )

        file.write(
            f"Clean False Positive Rate: "
            f"{test_metrics['false_positive_rate']:.4f}\n\n"
        )

        file.write(
            "Confusion Matrix\n"
        )

        file.write(
            "[[TN FP]\n"
            " [FN TP]]\n"
        )

        file.write(
            str(
                test_metrics[
                    "matrix"
                ]
            )
        )

        file.write(
            "\n\nClassification Report\n"
        )

        file.write(
            test_metrics[
                "report"
            ]
        )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print(
        "\nVISION AI LOCAL DEFECT EVALUATION"
    )

    print(
        "=" * 60
    )

    # -----------------------------------------------------
    # TRAIN SET
    # -----------------------------------------------------

    print(
        "\nBuilding TRAIN clean/defect dataset..."
    )

    train_rows = (
        build_binary_dataset(
            TRAIN_DIR
        )
    )

    # -----------------------------------------------------
    # TEST SET
    # -----------------------------------------------------

    print(
        "\nBuilding TEST clean/defect dataset..."
    )

    test_rows = (
        build_binary_dataset(
            TEST_DIR
        )
    )

    # -----------------------------------------------------
    # SCORE DISTRIBUTIONS
    # -----------------------------------------------------

    print_score_summary(
        train_rows,
        "TRAIN SCORE DISTRIBUTION",
    )

    print_score_summary(
        test_rows,
        "TEST SCORE DISTRIBUTION",
    )

    # -----------------------------------------------------
    # SELECT THRESHOLD USING TRAIN ONLY
    # -----------------------------------------------------

    print(
        "\nSelecting threshold using TRAINING DATA ONLY..."
    )

    threshold_info = (
        select_threshold(
            train_rows
        )
    )

    threshold = (
        threshold_info[
            "threshold"
        ]
    )

    print(
        f"\nSelected threshold             : "
        f"{threshold:.2f}"
    )

    print(
        f"Training false-positive rate   : "
        f"{threshold_info['false_positive_rate']:.4f}"
    )

    print(
        f"Training defect precision      : "
        f"{threshold_info['precision']:.4f}"
    )

    print(
        f"Training defect recall         : "
        f"{threshold_info['recall']:.4f}"
    )

    print(
        f"Training defect F1             : "
        f"{threshold_info['f1']:.4f}"
    )

    # -----------------------------------------------------
    # TRAIN PERFORMANCE
    # -----------------------------------------------------

    train_metrics = (
        evaluate(
            train_rows,
            threshold,
        )
    )

    # -----------------------------------------------------
    # HELD-OUT TEST PERFORMANCE
    # -----------------------------------------------------

    test_metrics = (
        evaluate(
            test_rows,
            threshold,
        )
    )

    print(
        "\nFINAL HELD-OUT TEST RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"\nAccuracy                 : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Defect Precision         : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Defect Recall            : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Defect F1                : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Clean False Positive Rate: "
        f"{test_metrics['false_positive_rate']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        test_metrics[
            "matrix"
        ]
    )

    print(
        "\nClassification Report:"
    )

    print(
        test_metrics[
            "report"
        ]
    )

    # -----------------------------------------------------
    # SAVE REPORT
    # -----------------------------------------------------

    save_report(
        threshold_info,
        train_metrics,
        test_metrics,
    )

    print(
        "\nEvaluation report saved to:"
    )

    print(
        REPORT_PATH
    )

    print(
        "\nLOCAL DEFECT EVALUATION COMPLETE ✅"
    )


if __name__ == "__main__":
    main()