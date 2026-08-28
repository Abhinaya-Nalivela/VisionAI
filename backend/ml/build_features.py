from pathlib import Path
import sys

import cv2
import pandas as pd


# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Allow imports from backend/
sys.path.insert(0, str(BACKEND_DIR))

from app.services.feature_extraction import extract_features


DATASET_DIR = BACKEND_DIR / "dataset" / "generated"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# Features that will be used by the ML model.
# Width, height and channels are kept as statistics but are
# intentionally not used as model inputs.
MODEL_FEATURES = [
    # Global image-quality features
    "brightness",
    "contrast",
    "sharpness",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "saturation",
    "edge_density",
    "noise_level",
    "entropy",

    # Localized defect-sensitive features
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


def process_split(split_name):
    """
    Extract features from every generated image
    belonging to either train or test.
    """

    split_dir = DATASET_DIR / split_name

    rows = []

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset split does not exist: {split_dir}"
        )

    print(f"\nProcessing {split_name.upper()} dataset...")

    # Each folder name represents the quality class
    for class_dir in sorted(split_dir.iterdir()):

        if not class_dir.is_dir():
            continue

        label = class_dir.name

        print(f"  Class: {label}")

        for image_path in sorted(class_dir.iterdir()):

            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                print(
                    f"    WARNING: Could not read {image_path.name}"
                )
                continue

            try:
                features = extract_features(image)

            except Exception as error:
                print(
                    f"    ERROR processing {image_path.name}: {error}"
                )
                continue

            row = {
                "filename": image_path.name,
                "label": label,
            }

            for feature_name in MODEL_FEATURES:
                row[feature_name] = features[feature_name]

            rows.append(row)

    return pd.DataFrame(rows)


def print_summary(dataframe, split_name):

    print("\n" + "-" * 50)

    print(
        f"{split_name.upper()} FEATURE DATASET"
    )

    print("-" * 50)

    print(
        f"Total samples: {len(dataframe)}"
    )

    print("\nSamples per class:")

    counts = dataframe["label"].value_counts()

    for label, count in counts.items():
        print(
            f"  {label:15}: {count}"
        )


def main():

    print("\nVISION AI FEATURE BUILDER")
    print("=" * 50)

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_dataframe = process_split(
        "train"
    )

    test_dataframe = process_split(
        "test"
    )

    print_summary(
        train_dataframe,
        "train"
    )

    print_summary(
        test_dataframe,
        "test"
    )

    train_output = (
        ARTIFACTS_DIR / "train_features.csv"
    )

    test_output = (
        ARTIFACTS_DIR / "test_features.csv"
    )

    train_dataframe.to_csv(
        train_output,
        index=False
    )

    test_dataframe.to_csv(
        test_output,
        index=False
    )

    print("\n" + "=" * 50)

    print(
        f"Training CSV saved to:\n{train_output}"
    )

    print(
        f"\nTesting CSV saved to:\n{test_output}"
    )

    print("\nFEATURE EXTRACTION COMPLETE ✅")


if __name__ == "__main__":
    main()