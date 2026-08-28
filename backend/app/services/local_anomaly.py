import cv2
import numpy as np


def _safe_float(value):
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)


def _robust_outlier_score(values):
    """
    Returns robust z-scores using median absolute deviation.

    This is preferable to normal mean/std because one defective patch
    should not heavily distort the reference distribution.
    """
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return np.array([], dtype=np.float32)

    median = np.median(values)

    mad = np.median(np.abs(values - median))

    if mad < 1e-6:
        return np.zeros_like(values)

    robust_z = 0.6745 * np.abs(values - median) / mad

    return robust_z


def _patch_features(patch):
    """
    Extract local characteristics from a single image patch.
    """

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Local sharpness / high-frequency structure
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # Local noise estimate
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    noise = float(np.std(residual))

    # Local edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = float(np.mean(edges > 0))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "noise": noise,
        "edge_ratio": edge_ratio,
    }


def extract_local_anomaly_features(image, grid_size=8):
    """
    Analyze the image spatially using a grid of patches.

    Instead of asking whether the ENTIRE image has high texture,
    we look for patches that behave unusually compared with the rest
    of the same image.

    Returns interpretable local anomaly measurements.
    """

    if image is None:
        raise ValueError("Image cannot be None.")

    if len(image.shape) != 3:
        raise ValueError("Expected a BGR color image.")

    height, width = image.shape[:2]

    # Very small images do not support useful patch comparison.
    if height < grid_size * 8 or width < grid_size * 8:
        return {
            "local_brightness_outlier": 0.0,
            "local_contrast_outlier": 0.0,
            "local_sharpness_outlier": 0.0,
            "local_noise_outlier": 0.0,
            "local_edge_outlier": 0.0,
            "neighbor_inconsistency": 0.0,
            "local_anomaly_score": 0.0,
        }

    patch_height = height // grid_size
    patch_width = width // grid_size

    patches = []

    feature_maps = {
        "brightness": np.zeros((grid_size, grid_size), dtype=np.float32),
        "contrast": np.zeros((grid_size, grid_size), dtype=np.float32),
        "sharpness": np.zeros((grid_size, grid_size), dtype=np.float32),
        "noise": np.zeros((grid_size, grid_size), dtype=np.float32),
        "edge_ratio": np.zeros((grid_size, grid_size), dtype=np.float32),
    }

    # ---------------------------------------------------------
    # 1. Extract local features for every image region
    # ---------------------------------------------------------

    for row in range(grid_size):
        for col in range(grid_size):

            y1 = row * patch_height
            x1 = col * patch_width

            if row == grid_size - 1:
                y2 = height
            else:
                y2 = (row + 1) * patch_height

            if col == grid_size - 1:
                x2 = width
            else:
                x2 = (col + 1) * patch_width

            patch = image[y1:y2, x1:x2]

            values = _patch_features(patch)

            patches.append(values)

            for feature_name in feature_maps:
                feature_maps[feature_name][row, col] = values[feature_name]

    # ---------------------------------------------------------
    # 2. Robust global-within-image outliers
    # ---------------------------------------------------------

    feature_outliers = {}

    for feature_name in feature_maps:

        flattened = feature_maps[feature_name].flatten()

        z_scores = _robust_outlier_score(flattened)

        # Rather than using one single maximum patch,
        # take the mean of the top 3 strongest patches.
        top_count = min(3, len(z_scores))

        if top_count > 0:
            top_values = np.sort(z_scores)[-top_count:]
            score = float(np.mean(top_values))
        else:
            score = 0.0

        feature_outliers[feature_name] = score

    # ---------------------------------------------------------
    # 3. Compare each patch against immediate neighbours
    # ---------------------------------------------------------

    neighbour_scores = []

    feature_names = [
        "brightness",
        "contrast",
        "sharpness",
        "noise",
        "edge_ratio",
    ]

    for row in range(grid_size):
        for col in range(grid_size):

            current_vector = []

            neighbour_vectors = []

            for feature_name in feature_names:
                current_vector.append(
                    feature_maps[feature_name][row, col]
                )

            directions = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ]

            for dr, dc in directions:

                nr = row + dr
                nc = col + dc

                if (
                    0 <= nr < grid_size
                    and 0 <= nc < grid_size
                ):
                    vector = []

                    for feature_name in feature_names:
                        vector.append(
                            feature_maps[feature_name][nr, nc]
                        )

                    neighbour_vectors.append(vector)

            if not neighbour_vectors:
                continue

            current_vector = np.asarray(
                current_vector,
                dtype=np.float32,
            )

            neighbours = np.asarray(
                neighbour_vectors,
                dtype=np.float32,
            )

            neighbour_median = np.median(
                neighbours,
                axis=0,
            )

            # Normalize feature scales using the median magnitude.
            denominator = np.maximum(
                np.abs(neighbour_median),
                1.0,
            )

            relative_difference = (
                np.abs(current_vector - neighbour_median)
                / denominator
            )

            neighbour_scores.append(
                float(np.mean(relative_difference))
            )

    if neighbour_scores:

        sorted_neighbour_scores = np.sort(neighbour_scores)

        top_count = min(3, len(sorted_neighbour_scores))

        neighbour_inconsistency = float(
            np.mean(sorted_neighbour_scores[-top_count:])
        )

    else:
        neighbour_inconsistency = 0.0

    # ---------------------------------------------------------
    # 4. Convert raw measurements into 0–1 evidence scores
    # ---------------------------------------------------------

    def normalize_outlier(value, start=2.5, strong=6.0):
        if value <= start:
            return 0.0

        if value >= strong:
            return 1.0

        return (value - start) / (strong - start)

    brightness_score = normalize_outlier(
        feature_outliers["brightness"]
    )

    contrast_score = normalize_outlier(
        feature_outliers["contrast"]
    )

    sharpness_score = normalize_outlier(
        feature_outliers["sharpness"]
    )

    noise_score = normalize_outlier(
        feature_outliers["noise"]
    )

    edge_score = normalize_outlier(
        feature_outliers["edge_ratio"]
    )

    # Neighbour inconsistency uses its own scale.
    neighbour_score = min(
        1.0,
        neighbour_inconsistency / 1.5,
    )

    # Texture-based evidence matters more than brightness,
    # because normal scenes often contain naturally bright/dark areas.
    local_anomaly_score = (
        brightness_score * 0.05
        + contrast_score * 0.15
        + sharpness_score * 0.25
        + noise_score * 0.25
        + edge_score * 0.15
        + neighbour_score * 0.15
    )

    return {
        "local_brightness_outlier": round(
            _safe_float(brightness_score), 4
        ),
        "local_contrast_outlier": round(
            _safe_float(contrast_score), 4
        ),
        "local_sharpness_outlier": round(
            _safe_float(sharpness_score), 4
        ),
        "local_noise_outlier": round(
            _safe_float(noise_score), 4
        ),
        "local_edge_outlier": round(
            _safe_float(edge_score), 4
        ),
        "neighbor_inconsistency": round(
            _safe_float(neighbour_score), 4
        ),
        "local_anomaly_score": round(
            _safe_float(local_anomaly_score), 4
        ),
    }