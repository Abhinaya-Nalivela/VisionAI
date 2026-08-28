import cv2
import numpy as np


def extract_features(image: np.ndarray) -> dict:
    """
    Extract image-quality and localized defect features
    from an OpenCV image.

    Parameters:
        image: OpenCV image in BGR format.

    Returns:
        Dictionary containing image-quality statistics.
    """

    if image is None:
        raise ValueError(
            "Image is empty or could not be decoded."
        )

    if len(image.shape) != 3:
        raise ValueError(
            "Expected a color image."
        )

    height, width, channels = image.shape

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    total_pixels = gray.size


    # -----------------------------------------------------
    # 1. BRIGHTNESS
    # -----------------------------------------------------

    brightness = float(
        np.mean(gray)
    )


    # -----------------------------------------------------
    # 2. CONTRAST
    # -----------------------------------------------------

    contrast = float(
        np.std(gray)
    )


    # -----------------------------------------------------
    # 3. SHARPNESS
    # -----------------------------------------------------

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    sharpness = float(
        laplacian.var()
    )


    # -----------------------------------------------------
    # 4. DARK PIXEL RATIO
    # -----------------------------------------------------

    dark_pixels = np.sum(
        gray < 40
    )

    dark_pixel_ratio = float(
        dark_pixels
        / total_pixels
    )


    # -----------------------------------------------------
    # 5. BRIGHT PIXEL RATIO
    # -----------------------------------------------------

    bright_pixels = np.sum(
        gray > 215
    )

    bright_pixel_ratio = float(
        bright_pixels
        / total_pixels
    )


    # -----------------------------------------------------
    # 6. SATURATION
    # -----------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    saturation_channel = (
        hsv[:, :, 1]
    )

    saturation = float(
        np.mean(
            saturation_channel
        )
    )


    # -----------------------------------------------------
    # 7. EDGE DENSITY
    # -----------------------------------------------------

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_pixels = np.count_nonzero(
        edges
    )

    edge_density = float(
        edge_pixels
        / total_pixels
    )


    # -----------------------------------------------------
    # 8. NOISE LEVEL
    # -----------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    noise_residual = (
        gray.astype(np.float32)
        - blurred.astype(np.float32)
    )

    noise_level = float(
        np.std(
            noise_residual
        )
    )


    # -----------------------------------------------------
    # 9. ENTROPY
    # -----------------------------------------------------

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram.ravel()

    probability = (
        histogram
        / histogram.sum()
    )

    probability = (
        probability[
            probability > 0
        ]
    )

    entropy = float(
        -np.sum(
            probability
            * np.log2(
                probability
            )
        )
    )


    # -----------------------------------------------------
    # 10. GRADIENT MEAN
    # -----------------------------------------------------

    gradient_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    gradient_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    gradient_magnitude = (
        cv2.magnitude(
            gradient_x.astype(
                np.float32
            ),
            gradient_y.astype(
                np.float32
            )
        )
    )

    gradient_mean = float(
        np.mean(
            gradient_magnitude
        )
    )


    # -----------------------------------------------------
    # 11. GRADIENT STD
    # -----------------------------------------------------

    gradient_std = float(
        np.std(
            gradient_magnitude
        )
    )


    # -----------------------------------------------------
    # 12. STRONG EDGE RATIO
    # -----------------------------------------------------

    strong_edges = np.sum(
        gradient_magnitude > 150
    )

    strong_edge_ratio = float(
        strong_edges
        / total_pixels
    )


    # -----------------------------------------------------
    # 13. LOCAL CONTRAST VARIATION
    # -----------------------------------------------------

    local_mean = cv2.blur(
        gray.astype(
            np.float32
        ),
        (15, 15)
    )

    squared_gray = (
        gray.astype(
            np.float32
        )
        ** 2
    )

    local_squared_mean = cv2.blur(
        squared_gray,
        (15, 15)
    )

    local_variance = (
        local_squared_mean
        - local_mean ** 2
    )

    local_variance = np.maximum(
        local_variance,
        0
    )

    local_std = np.sqrt(
        local_variance
    )

    local_contrast_mean = float(
        np.mean(
            local_std
        )
    )

    local_contrast_std = float(
        np.std(
            local_std
        )
    )


    # -----------------------------------------------------
    # 14. PATCH INTENSITY VARIATION
    # -----------------------------------------------------

    patch_means = []

    grid_size = 4

    patch_height = max(
        1,
        height // grid_size
    )

    patch_width = max(
        1,
        width // grid_size
    )

    for row in range(
        grid_size
    ):

        for column in range(
            grid_size
        ):

            y1 = (
                row
                * patch_height
            )

            x1 = (
                column
                * patch_width
            )

            if row == (
                grid_size - 1
            ):
                y2 = height
            else:
                y2 = (
                    y1
                    + patch_height
                )

            if column == (
                grid_size - 1
            ):
                x2 = width
            else:
                x2 = (
                    x1
                    + patch_width
                )

            patch = gray[
                y1:y2,
                x1:x2
            ]

            if patch.size > 0:
                patch_means.append(
                    float(
                        np.mean(
                            patch
                        )
                    )
                )

    if patch_means:
        patch_intensity_std = float(
            np.std(
                patch_means
            )
        )

        patch_intensity_range = float(
            max(
                patch_means
            )
            - min(
                patch_means
            )
        )

    else:
        patch_intensity_std = 0.0
        patch_intensity_range = 0.0


    # -----------------------------------------------------
    # 15. EXTREME LOCAL PIXEL RATIO
    # -----------------------------------------------------

    extreme_dark = np.sum(
        gray < 15
    )

    extreme_bright = np.sum(
        gray > 240
    )

    extreme_pixel_ratio = float(
        (
            extreme_dark
            + extreme_bright
        )
        / total_pixels
    )


    # -----------------------------------------------------
    # 16. LAPLACIAN EXTREME RATIO
    # -----------------------------------------------------

    absolute_laplacian = np.abs(
        laplacian
    )

    laplacian_threshold = (
        np.mean(
            absolute_laplacian
        )
        + 3
        * np.std(
            absolute_laplacian
        )
    )

    extreme_laplacian_pixels = (
        np.sum(
            absolute_laplacian
            > laplacian_threshold
        )
    )

    laplacian_extreme_ratio = float(
        extreme_laplacian_pixels
        / total_pixels
    )


    # -----------------------------------------------------
    # RETURN FEATURES
    # -----------------------------------------------------

    return {
        "width": int(
            width
        ),

        "height": int(
            height
        ),

        "channels": int(
            channels
        ),

        "brightness": round(
            brightness,
            4
        ),

        "contrast": round(
            contrast,
            4
        ),

        "sharpness": round(
            sharpness,
            4
        ),

        "dark_pixel_ratio": round(
            dark_pixel_ratio,
            4
        ),

        "bright_pixel_ratio": round(
            bright_pixel_ratio,
            4
        ),

        "saturation": round(
            saturation,
            4
        ),

        "edge_density": round(
            edge_density,
            4
        ),

        "noise_level": round(
            noise_level,
            4
        ),

        "entropy": round(
            entropy,
            4
        ),

        "gradient_mean": round(
            gradient_mean,
            4
        ),

        "gradient_std": round(
            gradient_std,
            4
        ),

        "strong_edge_ratio": round(
            strong_edge_ratio,
            4
        ),

        "local_contrast_mean": round(
            local_contrast_mean,
            4
        ),

        "local_contrast_std": round(
            local_contrast_std,
            4
        ),

        "patch_intensity_std": round(
            patch_intensity_std,
            4
        ),

        "patch_intensity_range": round(
            patch_intensity_range,
            4
        ),

        "extreme_pixel_ratio": round(
            extreme_pixel_ratio,
            4
        ),

        "laplacian_extreme_ratio": round(
            laplacian_extreme_ratio,
            4
        ),
    }