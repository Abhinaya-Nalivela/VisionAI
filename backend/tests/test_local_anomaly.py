from pathlib import Path

import cv2
import numpy as np

from app.services.local_anomaly import extract_local_anomaly_features


TEST_ROOT = Path("dataset/generated/test")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():

    results = {}

    for class_dir in sorted(TEST_ROOT.iterdir()):

        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        scores = []

        for image_path in sorted(class_dir.iterdir()):

            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Could not read: {image_path}")
                continue

            features = extract_local_anomaly_features(image)

            score = features["local_anomaly_score"]

            scores.append((image_path.name, score))

        if scores:
            results[class_name] = scores

    print("\n========================================")
    print("LOCAL ANOMALY VALIDATION")
    print("========================================")

    for class_name, scores in results.items():

        values = np.array(
            [score for _, score in scores],
            dtype=np.float32,
        )

        print(f"\nCLASS: {class_name}")
        print(f"Count   : {len(values)}")
        print(f"Mean    : {values.mean():.4f}")
        print(f"Median  : {np.median(values):.4f}")
        print(f"Min     : {values.min():.4f}")
        print(f"Max     : {values.max():.4f}")
        print(f"Std     : {values.std():.4f}")

    # ---------------------------------------------------------
    # Defect-specific ranking
    # ---------------------------------------------------------

    if "defect" in results:

        defect_scores = sorted(
            results["defect"],
            key=lambda item: item[1],
            reverse=True,
        )

        print("\n========================================")
        print("DEFECT IMAGES — HIGHEST TO LOWEST")
        print("========================================")

        for filename, score in defect_scores:
            print(f"{filename:35} {score:.4f}")

    # ---------------------------------------------------------
    # Clean-specific ranking
    # ---------------------------------------------------------

    if "clean" in results:

        clean_scores = sorted(
            results["clean"],
            key=lambda item: item[1],
            reverse=True,
        )

        print("\n========================================")
        print("CLEAN IMAGES — HIGHEST TO LOWEST")
        print("========================================")

        for filename, score in clean_scores:
            print(f"{filename:35} {score:.4f}")

    # ---------------------------------------------------------
    # Candidate threshold comparison
    # ---------------------------------------------------------

    if "defect" in results and "clean" in results:

        defect_values = np.array(
            [score for _, score in results["defect"]]
        )

        clean_values = np.array(
            [score for _, score in results["clean"]]
        )

        print("\n========================================")
        print("CANDIDATE THRESHOLDS")
        print("========================================")

        thresholds = [
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
        ]

        for threshold in thresholds:

            defect_detected = np.mean(
                defect_values >= threshold
            )

            clean_false_positive = np.mean(
                clean_values >= threshold
            )

            print(
                f"Threshold {threshold:.2f} | "
                f"Defect detected: {defect_detected * 100:6.2f}% | "
                f"Clean false positive: {clean_false_positive * 100:6.2f}%"
            )


if __name__ == "__main__":
    main()