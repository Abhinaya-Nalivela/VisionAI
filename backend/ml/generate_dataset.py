from pathlib import Path
import random
import shutil

import cv2
import numpy as np


# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Original project images
ORIGINAL_SOURCE_DIR = BACKEND_DIR / "dataset" / "source"

# Selected high-quality KonIQ images
KONIQ_SOURCE_DIR = BACKEND_DIR / "dataset" / "koniq_source"

OUTPUT_DIR = BACKEND_DIR / "dataset" / "generated"

TRAIN_DIR = OUTPUT_DIR / "train"
TEST_DIR = OUTPUT_DIR / "test"


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

RANDOM_SEED = 42

# One variation is enough now because we have much more
# source-image diversity from KonIQ.
VARIATIONS_PER_TYPE = 1

TRAIN_RATIO = 0.80

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

CLASSES = [
    "clean",
    "blur",
    "underexposed",
    "overexposed",
    "noise",
    "degraded",
    "defect",
]


# ---------------------------------------------------------
# IMAGE DEGRADATION FUNCTIONS
# ---------------------------------------------------------

def create_blur(image):
    """
    Apply Gaussian blur with random strength.
    """

    kernel_size = random.choice(
        [5, 7, 9, 11, 15]
    )

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0,
    )


def create_underexposure(image):
    """
    Darken the image using random intensity scaling.
    """

    factor = random.uniform(
        0.20,
        0.55,
    )

    dark_image = (
        image.astype(np.float32)
        * factor
    )

    return np.clip(
        dark_image,
        0,
        255,
    ).astype(np.uint8)


def create_overexposure(image):
    """
    Brighten the image significantly.
    """

    factor = random.uniform(
        1.4,
        2.1,
    )

    offset = random.randint(
        20,
        60,
    )

    bright_image = (
        image.astype(np.float32)
        * factor
        + offset
    )

    return np.clip(
        bright_image,
        0,
        255,
    ).astype(np.uint8)


def create_noise(image):
    """
    Add Gaussian noise.
    """

    sigma = random.uniform(
        15,
        45,
    )

    noise = np.random.normal(
        0,
        sigma,
        image.shape,
    )

    noisy_image = (
        image.astype(np.float32)
        + noise
    )

    return np.clip(
        noisy_image,
        0,
        255,
    ).astype(np.uint8)


def create_degraded(image):
    """
    Produce severe degradation using resizing,
    blur and noise.
    """

    height, width = image.shape[:2]

    scale = random.uniform(
        0.15,
        0.40,
    )

    small_width = max(
        32,
        int(width * scale),
    )

    small_height = max(
        32,
        int(height * scale),
    )

    small = cv2.resize(
        image,
        (
            small_width,
            small_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    degraded = cv2.resize(
        small,
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    degraded = cv2.GaussianBlur(
        degraded,
        (7, 7),
        0,
    )

    sigma = random.uniform(
        10,
        30,
    )

    noise = np.random.normal(
        0,
        sigma,
        degraded.shape,
    )

    degraded = (
        degraded.astype(np.float32)
        + noise
    )

    return np.clip(
        degraded,
        0,
        255,
    ).astype(np.uint8)


def create_defect(image):
    """
    Add a localized synthetic visual defect.

    Possible defects:
    - scratch
    - dark spot
    - bright spot
    - damaged rectangular patch
    """

    defect_image = image.copy()

    height, width = defect_image.shape[:2]

    defect_type = random.choice(
        [
            "scratch",
            "dark_spot",
            "bright_spot",
            "patch",
        ]
    )

    # -----------------------------------------------------
    # SCRATCH
    # -----------------------------------------------------

    if defect_type == "scratch":

        start_x = random.randint(
            0,
            max(0, width - 1),
        )

        start_y = random.randint(
            0,
            max(0, height - 1),
        )

        end_x = random.randint(
            0,
            max(0, width - 1),
        )

        end_y = random.randint(
            0,
            max(0, height - 1),
        )

        thickness = random.randint(
            2,
            max(
                3,
                min(width, height) // 80,
            ),
        )

        if random.random() < 0.5:
            color = (
                random.randint(0, 50),
                random.randint(0, 50),
                random.randint(0, 50),
            )
        else:
            color = (
                random.randint(205, 255),
                random.randint(205, 255),
                random.randint(205, 255),
            )

        cv2.line(
            defect_image,
            (start_x, start_y),
            (end_x, end_y),
            color,
            thickness,
        )

    # -----------------------------------------------------
    # DARK SPOT
    # -----------------------------------------------------

    elif defect_type == "dark_spot":

        center_x = random.randint(
            0,
            max(0, width - 1),
        )

        center_y = random.randint(
            0,
            max(0, height - 1),
        )

        max_radius = max(
            5,
            min(width, height) // 10,
        )

        radius = random.randint(
            max(
                3,
                max_radius // 3,
            ),
            max_radius,
        )

        overlay = defect_image.copy()

        cv2.circle(
            overlay,
            (center_x, center_y),
            radius,
            (0, 0, 0),
            -1,
        )

        alpha = random.uniform(
            0.45,
            0.85,
        )

        defect_image = cv2.addWeighted(
            overlay,
            alpha,
            defect_image,
            1 - alpha,
            0,
        )

    # -----------------------------------------------------
    # BRIGHT SPOT
    # -----------------------------------------------------

    elif defect_type == "bright_spot":

        center_x = random.randint(
            0,
            max(0, width - 1),
        )

        center_y = random.randint(
            0,
            max(0, height - 1),
        )

        max_radius = max(
            5,
            min(width, height) // 10,
        )

        radius = random.randint(
            max(
                3,
                max_radius // 3,
            ),
            max_radius,
        )

        overlay = defect_image.copy()

        cv2.circle(
            overlay,
            (center_x, center_y),
            radius,
            (255, 255, 255),
            -1,
        )

        alpha = random.uniform(
            0.45,
            0.85,
        )

        defect_image = cv2.addWeighted(
            overlay,
            alpha,
            defect_image,
            1 - alpha,
            0,
        )

    # -----------------------------------------------------
    # DAMAGED PATCH
    # -----------------------------------------------------

    else:

        patch_width = max(
            8,
            int(
                width
                * random.uniform(
                    0.05,
                    0.20,
                )
            ),
        )

        patch_height = max(
            8,
            int(
                height
                * random.uniform(
                    0.05,
                    0.20,
                )
            ),
        )

        max_x = max(
            0,
            width - patch_width,
        )

        max_y = max(
            0,
            height - patch_height,
        )

        x1 = random.randint(
            0,
            max_x,
        )

        y1 = random.randint(
            0,
            max_y,
        )

        x2 = min(
            width,
            x1 + patch_width,
        )

        y2 = min(
            height,
            y1 + patch_height,
        )

        patch = defect_image[
            y1:y2,
            x1:x2
        ]

        if patch.size > 0:

            if random.random() < 0.5:

                damaged_patch = cv2.GaussianBlur(
                    patch,
                    (15, 15),
                    0,
                )

            else:

                damaged_patch = np.random.randint(
                    0,
                    256,
                    patch.shape,
                    dtype=np.uint8,
                )

            defect_image[
                y1:y2,
                x1:x2
            ] = damaged_patch

    return defect_image


# ---------------------------------------------------------
# DIRECTORY UTILITIES
# ---------------------------------------------------------

def prepare_directories():
    """
    Remove the old generated dataset and recreate
    train/test folders for every class.
    """

    if OUTPUT_DIR.exists():
        shutil.rmtree(
            OUTPUT_DIR
        )

    for split_dir in [
        TRAIN_DIR,
        TEST_DIR,
    ]:

        for class_name in CLASSES:

            directory = (
                split_dir
                / class_name
            )

            directory.mkdir(
                parents=True,
                exist_ok=True,
            )


def collect_images(
    directory,
    source_prefix,
):
    """
    Collect supported image files from a source folder.

    Each item contains:
    - path
    - source prefix

    Prefixes avoid filename collisions between datasets.
    """

    images = []

    if not directory.exists():
        print(
            f"WARNING: Source directory not found: "
            f"{directory}"
        )

        return images

    for path in directory.iterdir():

        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):

            images.append(
                {
                    "path": path,
                    "prefix": source_prefix,
                }
            )

    return images


def get_source_images():
    """
    Combine original VisionAI images with the
    selected high-quality KonIQ subset.
    """

    original_images = collect_images(
        ORIGINAL_SOURCE_DIR,
        "original",
    )

    koniq_images = collect_images(
        KONIQ_SOURCE_DIR,
        "koniq",
    )

    print(
        f"\nOriginal source images : "
        f"{len(original_images)}"
    )

    print(
        f"KonIQ source images    : "
        f"{len(koniq_images)}"
    )

    combined = (
        original_images
        + koniq_images
    )

    return combined


def save_image(
    directory,
    filename,
    image,
):

    output_path = (
        directory
        / filename
    )

    success = cv2.imwrite(
        str(output_path),
        image,
    )

    if not success:
        raise RuntimeError(
            f"Could not save image: {output_path}"
        )


# ---------------------------------------------------------
# DATASET GENERATION
# ---------------------------------------------------------

def generate_for_image(
    source_item,
    split_name,
):

    image_path = source_item["path"]
    source_prefix = source_item["prefix"]

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"WARNING: Could not read "
            f"{image_path.name}"
        )

        return 0

    destination_root = (
        TRAIN_DIR
        if split_name == "train"
        else TEST_DIR
    )

    # Prefix filenames so two datasets cannot overwrite
    # each other if they happen to contain identical names.
    base_name = (
        f"{source_prefix}_"
        f"{image_path.stem}"
    )

    generated_count = 0

    # -----------------------------------------------------
    # CLEAN
    # -----------------------------------------------------

    for index in range(
        VARIATIONS_PER_TYPE
    ):

        clean = image.copy()

        filename = (
            f"{base_name}_"
            f"clean_{index}.jpg"
        )

        save_image(
            destination_root / "clean",
            filename,
            clean,
        )

        generated_count += 1

    # -----------------------------------------------------
    # QUALITY DEGRADATIONS + DEFECT
    # -----------------------------------------------------

    degradation_functions = {
        "blur": create_blur,
        "underexposed": create_underexposure,
        "overexposed": create_overexposure,
        "noise": create_noise,
        "degraded": create_degraded,
        "defect": create_defect,
    }

    for (
        class_name,
        function,
    ) in degradation_functions.items():

        for index in range(
            VARIATIONS_PER_TYPE
        ):

            generated_image = function(
                image.copy()
            )

            filename = (
                f"{base_name}_"
                f"{class_name}_"
                f"{index}.jpg"
            )

            save_image(
                destination_root
                / class_name,
                filename,
                generated_image,
            )

            generated_count += 1

    return generated_count


# ---------------------------------------------------------
# REPORTING
# ---------------------------------------------------------

def report_class_counts():

    print(
        "\n========================================"
    )

    print(
        "GENERATED CLASS COUNTS"
    )

    print(
        "========================================"
    )

    for split_name, split_dir in [
        ("TRAIN", TRAIN_DIR),
        ("TEST", TEST_DIR),
    ]:

        print(
            f"\n{split_name}"
        )

        for class_name in CLASSES:

            class_dir = (
                split_dir
                / class_name
            )

            count = len(
                [
                    path
                    for path
                    in class_dir.iterdir()
                    if path.is_file()
                ]
            )

            print(
                f"{class_name:15}: "
                f"{count}"
            )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print(
        "\n========================================"
    )

    print(
        "VISION AI DATASET GENERATOR"
    )

    print(
        "========================================"
    )

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    source_images = (
        get_source_images()
    )

    if len(source_images) < 20:

        raise RuntimeError(
            "Not enough source images."
        )

    print(
        f"\nTotal source images    : "
        f"{len(source_images)}"
    )

    # -----------------------------------------------------
    # SOURCE-LEVEL TRAIN / TEST SPLIT
    #
    # IMPORTANT:
    # We split ORIGINAL images first and only then create
    # degradations. Therefore variants of the same original
    # cannot appear in both training and testing.
    # -----------------------------------------------------

    random.shuffle(
        source_images
    )

    split_index = int(
        len(source_images)
        * TRAIN_RATIO
    )

    train_sources = (
        source_images[
            :split_index
        ]
    )

    test_sources = (
        source_images[
            split_index:
        ]
    )

    print(
        f"Training source images : "
        f"{len(train_sources)}"
    )

    print(
        f"Testing source images  : "
        f"{len(test_sources)}"
    )

    prepare_directories()

    train_generated = 0
    test_generated = 0

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    print(
        "\nGenerating training images..."
    )

    for index, source_item in enumerate(
        train_sources,
        start=1,
    ):

        train_generated += generate_for_image(
            source_item,
            "train",
        )

        if (
            index % 25 == 0
            or index == len(train_sources)
        ):
            print(
                f"  Processed "
                f"{index}/"
                f"{len(train_sources)}"
            )

    # -----------------------------------------------------
    # TEST
    # -----------------------------------------------------

    print(
        "\nGenerating testing images..."
    )

    for index, source_item in enumerate(
        test_sources,
        start=1,
    ):

        test_generated += generate_for_image(
            source_item,
            "test",
        )

        if (
            index % 25 == 0
            or index == len(test_sources)
        ):
            print(
                f"  Processed "
                f"{index}/"
                f"{len(test_sources)}"
            )

    total_generated = (
        train_generated
        + test_generated
    )

    report_class_counts()

    print(
        "\n========================================"
    )

    print(
        "DATASET GENERATION COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Training generated : "
        f"{train_generated}"
    )

    print(
        f"Testing generated  : "
        f"{test_generated}"
    )

    print(
        f"Total generated    : "
        f"{total_generated}"
    )

    print(
        f"\nDataset location:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()