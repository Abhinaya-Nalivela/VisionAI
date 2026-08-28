from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# MODEL LOCATION
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BACKEND_DIR
    / "artifacts"
    / "quality_model_6class.joblib"
)


# ---------------------------------------------------------
# MODEL FEATURES
# ---------------------------------------------------------

# IMPORTANT:
# These features must remain in the exact same order
# as the FEATURE_COLUMNS used during model training.
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


# ---------------------------------------------------------
# EXPECTED MODEL CLASSES
# ---------------------------------------------------------

EXPECTED_CLASSES = {
    "clean",
    "blur",
    "underexposed",
    "overexposed",
    "noise",
    "degraded",
}


# ---------------------------------------------------------
# MODEL CACHE
# ---------------------------------------------------------

_model = None


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

def load_model():
    """
    Load the trained 6-class Random Forest model.

    The model is cached after the first load so inference
    does not reload the file for every request.
    """

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ML model not found at: {MODEL_PATH}"
            )

        _model = joblib.load(
            MODEL_PATH
        )

        # -------------------------------------------------
        # VERIFY FEATURE COUNT
        # -------------------------------------------------

        expected_feature_count = len(
            FEATURE_COLUMNS
        )

        model_feature_count = getattr(
            _model,
            "n_features_in_",
            None,
        )

        if (
            model_feature_count is not None
            and model_feature_count
            != expected_feature_count
        ):
            raise ValueError(
                "Model feature mismatch. "
                f"Model expects {model_feature_count} features, "
                f"but inference service provides "
                f"{expected_feature_count}."
            )

        # -------------------------------------------------
        # VERIFY MODEL CLASSES
        # -------------------------------------------------

        model_classes = {
            str(class_name)
            for class_name in _model.classes_
        }

        if model_classes != EXPECTED_CLASSES:
            raise ValueError(
                "Unexpected model classes. "
                f"Expected: {sorted(EXPECTED_CLASSES)}. "
                f"Loaded: {sorted(model_classes)}."
            )

        print(
            "\n6-class Random Forest loaded successfully."
        )

        print(
            f"Model path: {MODEL_PATH}"
        )

        print(
            f"Classes: {sorted(model_classes)}"
        )

    return _model


# ---------------------------------------------------------
# VALIDATE FEATURES
# ---------------------------------------------------------

def validate_features(
    features: dict,
):
    """
    Verify that all features required by the trained model
    are available before prediction.
    """

    missing_features = [
        feature_name
        for feature_name in FEATURE_COLUMNS
        if feature_name not in features
    ]

    if missing_features:
        raise ValueError(
            "Missing required model features: "
            + ", ".join(
                missing_features
            )
        )


# ---------------------------------------------------------
# CREATE FEATURE VECTOR
# ---------------------------------------------------------

def create_feature_vector(
    features: dict,
):
    """
    Convert extracted image features into a one-row
    DataFrame using the exact training feature order.
    """

    validate_features(
        features
    )

    feature_vector = pd.DataFrame(
        [
            {
                feature_name: features[
                    feature_name
                ]
                for feature_name
                in FEATURE_COLUMNS
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    return feature_vector


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

def predict_quality(
    features: dict,
) -> dict:
    """
    Predict global image quality using the trained
    six-class Random Forest model.

    Supported classes:
        clean
        blur
        underexposed
        overexposed
        noise
        degraded
    """

    model = load_model()

    feature_vector = (
        create_feature_vector(
            features
        )
    )

    prediction = model.predict(
        feature_vector
    )[0]

    probabilities = (
        model.predict_proba(
            feature_vector
        )[0]
    )

    probability_map = {
        str(class_name): round(
            float(probability),
            4,
        )
        for (
            class_name,
            probability,
        ) in zip(
            model.classes_,
            probabilities,
        )
    }

    confidence = float(
        np.max(
            probabilities
        )
    )

    return {
        "predicted_class": str(
            prediction
        ),

        "confidence": round(
            confidence,
            4,
        ),

        "probabilities": (
            probability_map
        ),
    }