from pathlib import Path
import sys
import json

import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ---------------------------------------------------------
# MAKE BACKEND IMPORTABLE
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BACKEND_DIR),
    )


from app.services.feature_extraction import (
    extract_features,
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

ARTIFACT_DIR = (
    BACKEND_DIR
    / "artifacts"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "defect_binary_model.joblib"
)

METADATA_PATH = (
    ARTIFACT_DIR
    / "defect_binary_metadata.json"
)

REPORT_PATH = (
    ARTIFACT_DIR
    / "defect_binary_evaluation.txt"
)


# ---------------------------------------------------------
# GLOBAL FEATURES
# ---------------------------------------------------------

GLOBAL_FEATURES = [
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


# ---------------------------------------------------------
# LOCAL FEATURES
# ---------------------------------------------------------

LOCAL_FEATURES = [
    "local_brightness_outlier",
    "local_contrast_outlier",
    "local_sharpness_outlier",
    "local_noise_outlier",
    "local_edge_outlier",
    "neighbor_inconsistency",
    "local_anomaly_score",
]


FEATURE_COLUMNS = (
    GLOBAL_FEATURES
    + LOCAL_FEATURES
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ---------------------------------------------------------
# GET IMAGES
# ---------------------------------------------------------

def get_images(folder):

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}"
        )

    return sorted(
        [
            path
            for path in folder.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )


# ---------------------------------------------------------
# EXTRACT COMBINED FEATURES
# ---------------------------------------------------------

def extract_combined_features(
    image_path,
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    global_features = (
        extract_features(
            image
        )
    )

    local_features = (
        extract_local_anomaly_features(
            image
        )
    )

    combined = {}

    for feature_name in GLOBAL_FEATURES:

        combined[
            feature_name
        ] = float(
            global_features[
                feature_name
            ]
        )

    for feature_name in LOCAL_FEATURES:

        combined[
            feature_name
        ] = float(
            local_features[
                feature_name
            ]
        )

    return combined


# ---------------------------------------------------------
# BUILD DATASET
# ---------------------------------------------------------

def build_dataset(
    split_dir,
):

    rows = []

    class_mapping = {
        "clean": 0,
        "defect": 1,
    }

    for (
        class_name,
        label,
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

        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):

            combined_features = (
                extract_combined_features(
                    image_path
                )
            )

            row = {
                "filename": (
                    image_path.name
                ),

                "label": (
                    label
                ),
            }

            row.update(
                combined_features
            )

            rows.append(
                row
            )

            if index % 50 == 0:
                print(
                    f"  processed "
                    f"{index}/"
                    f"{len(image_paths)}"
                )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------

def evaluate_model(
    model,
    dataframe,
):

    X = dataframe[
        FEATURE_COLUMNS
    ]

    y_true = dataframe[
        "label"
    ]

    y_pred = model.predict(
        X
    )

    probabilities = (
        model.predict_proba(
            X
        )
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=[
            "clean",
            "defect",
        ],
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matrix": matrix,
        "report": report,
        "probabilities": probabilities,
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print(
        "\nVISION AI BINARY DEFECT MODEL"
    )

    print(
        "=" * 60
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # BUILD TRAIN DATA
    # -----------------------------------------------------

    print(
        "\nBuilding TRAIN dataset..."
    )

    train_df = (
        build_dataset(
            TRAIN_DIR
        )
    )

    # -----------------------------------------------------
    # BUILD TEST DATA
    # -----------------------------------------------------

    print(
        "\nBuilding TEST dataset..."
    )

    test_df = (
        build_dataset(
            TEST_DIR
        )
    )

    print(
        "\nDATASET SUMMARY"
    )

    print(
        "-" * 60
    )

    print(
        f"Training samples : "
        f"{len(train_df)}"
    )

    print(
        f"Testing samples  : "
        f"{len(test_df)}"
    )

    print(
        f"Features         : "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        "\nTraining classes:"
    )

    print(
        train_df[
            "label"
        ].value_counts().sort_index()
    )

    print(
        "\nTesting classes:"
    )

    print(
        test_df[
            "label"
        ].value_counts().sort_index()
    )

    # -----------------------------------------------------
    # PREPARE TRAINING
    # -----------------------------------------------------

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        "label"
    ]

    # -----------------------------------------------------
    # TRAIN BINARY RANDOM FOREST
    # -----------------------------------------------------

    print(
        "\nTraining binary Random Forest..."
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    # -----------------------------------------------------
    # FINAL HELD-OUT TEST EVALUATION
    # -----------------------------------------------------

    results = (
        evaluate_model(
            model,
            test_df,
        )
    )

    print(
        "\nFINAL HELD-OUT TEST RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"\nAccuracy          : "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{results['balanced_accuracy']:.4f}"
    )

    print(
        f"Defect Precision  : "
        f"{results['precision']:.4f}"
    )

    print(
        f"Defect Recall     : "
        f"{results['recall']:.4f}"
    )

    print(
        f"Defect F1         : "
        f"{results['f1']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        results[
            "matrix"
        ]
    )

    print(
        "\nClassification Report:"
    )

    print(
        results[
            "report"
        ]
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "feature": (
                FEATURE_COLUMNS
            ),

            "importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print(
        "\nFEATURE IMPORTANCE"
    )

    print(
        "-" * 60
    )

    for _, row in importance_df.iterrows():

        print(
            f"{row['feature']:30} "
            f"{row['importance']:.6f}"
        )

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH,
    )

    # -----------------------------------------------------
    # SAVE METADATA
    # -----------------------------------------------------

    metadata = {
        "model_type": (
            "RandomForestClassifier"
        ),

        "task": (
            "binary_potential_visual_defect_detection"
        ),

        "classes": {
            "0": "clean",
            "1": "defect",
        },

        "feature_count": (
            len(
                FEATURE_COLUMNS
            )
        ),

        "features": (
            FEATURE_COLUMNS
        ),

        "training_samples": int(
            len(
                train_df
            )
        ),

        "testing_samples": int(
            len(
                test_df
            )
        ),

        "test_accuracy": round(
            float(
                results[
                    "accuracy"
                ]
            ),
            4,
        ),

        "test_balanced_accuracy": round(
            float(
                results[
                    "balanced_accuracy"
                ]
            ),
            4,
        ),

        "defect_precision": round(
            float(
                results[
                    "precision"
                ]
            ),
            4,
        ),

        "defect_recall": round(
            float(
                results[
                    "recall"
                ]
            ),
            4,
        ),

        "defect_f1": round(
            float(
                results[
                    "f1"
                ]
            ),
            4,
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

    # -----------------------------------------------------
    # SAVE REPORT
    # -----------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "VISION AI BINARY DEFECT MODEL\n"
        )

        file.write(
            "=" * 65
            + "\n\n"
        )

        file.write(
            "Task:\n"
            "Binary clean vs potential visual defect detection.\n\n"
        )

        file.write(
            f"Training samples: "
            f"{len(train_df)}\n"
        )

        file.write(
            f"Testing samples: "
            f"{len(test_df)}\n"
        )

        file.write(
            f"Feature count: "
            f"{len(FEATURE_COLUMNS)}\n\n"
        )

        file.write(
            "HELD-OUT TEST RESULTS\n"
        )

        file.write(
            "-" * 65
            + "\n"
        )

        file.write(
            f"Accuracy: "
            f"{results['accuracy']:.4f}\n"
        )

        file.write(
            f"Balanced Accuracy: "
            f"{results['balanced_accuracy']:.4f}\n"
        )

        file.write(
            f"Defect Precision: "
            f"{results['precision']:.4f}\n"
        )

        file.write(
            f"Defect Recall: "
            f"{results['recall']:.4f}\n"
        )

        file.write(
            f"Defect F1: "
            f"{results['f1']:.4f}\n\n"
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
                results[
                    "matrix"
                ]
            )
        )

        file.write(
            "\n\nClassification Report\n"
        )

        file.write(
            results[
                "report"
            ]
        )

        file.write(
            "\n\nFeature Importance\n"
        )

        for _, row in importance_df.iterrows():

            file.write(
                f"{row['feature']}: "
                f"{row['importance']:.6f}\n"
            )

    print(
        "\nArtifacts saved:"
    )

    print(
        MODEL_PATH
    )

    print(
        METADATA_PATH
    )

    print(
        REPORT_PATH
    )

    print(
        "\nBINARY DEFECT MODEL TRAINING COMPLETE ✅"
    )


if __name__ == "__main__":
    main()