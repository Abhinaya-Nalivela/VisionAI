from pathlib import Path

import cv2

from app.services.feature_extraction import extract_features
from app.services.model_service import predict_quality


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
SAMPLE_IMAGE = PROJECT_ROOT / "sample_images" / "clean.jpg"

EXPECTED_CLASSES = {
    "blur",
    "clean",
    "degraded",
    "noise",
    "overexposed",
    "underexposed",
}


def test_model_inference():
    """Verify that the trained model can perform inference."""

    assert SAMPLE_IMAGE.exists(), f"Sample image not found: {SAMPLE_IMAGE}"

    image = cv2.imread(str(SAMPLE_IMAGE))

    assert image is not None, "OpenCV could not decode the sample image."

    features = extract_features(image)

    prediction = predict_quality(features)

    assert isinstance(prediction, dict)

    assert "predicted_class" in prediction
    assert "confidence" in prediction
    assert "probabilities" in prediction

    assert prediction["predicted_class"] in EXPECTED_CLASSES

    assert 0.0 <= prediction["confidence"] <= 1.0

    probabilities = prediction["probabilities"]

    assert set(probabilities.keys()) == EXPECTED_CLASSES

    for probability in probabilities.values():
        assert 0.0 <= probability <= 1.0

    assert abs(sum(probabilities.values()) - 1.0) < 1e-3