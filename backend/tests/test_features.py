from pathlib import Path

import cv2

from app.services.feature_extraction import extract_features


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
SAMPLE_IMAGE = PROJECT_ROOT / "sample_images" / "clean.jpg"


def test_feature_extraction():
    """Verify that CV features can be extracted from a valid sample image."""

    assert SAMPLE_IMAGE.exists(), f"Sample image not found: {SAMPLE_IMAGE}"

    image = cv2.imread(str(SAMPLE_IMAGE))

    assert image is not None, "OpenCV could not decode the sample image."

    features = extract_features(image)

    assert isinstance(features, dict)
    assert len(features) > 0

    expected_features = {
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
    }

    assert expected_features.issubset(features.keys())