from pathlib import Path
import shutil

import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

KONIQ_IMAGES = Path(r"C:\Users\Abhi\Downloads\koniq10k_512x384")

KONIQ_CSV = Path(
    r"C:\Users\Abhi\Downloads\koniq10k_scores_and_distributions.csv"
)

OUTPUT_DIR = Path("dataset/koniq_source")

SUBSET_SIZE = 250

RANDOM_SEED = 42


def find_column(columns, possible_names):
    """
    Find a column regardless of capitalization.
    """

    normalized = {
        str(col).strip().lower(): col
        for col in columns
    }

    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]

    return None


def main():

    print("\n========================================")
    print("VisionAI - KonIQ Dataset Preparation")
    print("========================================")

    # ---------------------------------------------------------
    # 1. Validate paths
    # ---------------------------------------------------------

    if not KONIQ_IMAGES.exists():
        raise FileNotFoundError(
            f"KonIQ image folder not found:\n{KONIQ_IMAGES}"
        )

    if not KONIQ_CSV.exists():
        raise FileNotFoundError(
            f"KonIQ CSV not found:\n{KONIQ_CSV}"
        )

    # ---------------------------------------------------------
    # 2. Load CSV
    # ---------------------------------------------------------

    df = pd.read_csv(KONIQ_CSV)

    print(f"\nCSV rows: {len(df)}")
    print("CSV columns:")
    print(list(df.columns))

    # ---------------------------------------------------------
    # 3. Detect filename and quality-score columns
    # ---------------------------------------------------------

    filename_column = find_column(
        df.columns,
        [
            "image_name",
            "filename",
            "file_name",
            "image",
        ],
    )

    score_column = find_column(
        df.columns,
        [
            "MOS",
            "mos",
            "MOS_zscore",
            "quality_score",
        ],
    )

    if filename_column is None:
        raise ValueError(
            "Could not automatically find the image filename column."
        )

    if score_column is None:
        raise ValueError(
            "Could not automatically find the MOS quality score column."
        )

    print(f"\nFilename column: {filename_column}")
    print(f"Quality column : {score_column}")

    # ---------------------------------------------------------
    # 4. Keep the higher-quality portion of KonIQ
    #
    # We do NOT simply use all images as 'clean'.
    # First take the top 30% according to human MOS,
    # then randomly select 250 examples from that pool.
    # ---------------------------------------------------------

    df = df.dropna(
        subset=[filename_column, score_column]
    ).copy()

    df[score_column] = pd.to_numeric(
        df[score_column],
        errors="coerce",
    )

    df = df.dropna(subset=[score_column])

    quality_threshold = df[score_column].quantile(0.70)

    high_quality = df[
        df[score_column] >= quality_threshold
    ].copy()

    print(
        f"\n70th percentile MOS threshold: "
        f"{quality_threshold:.4f}"
    )

    print(
        f"High-quality candidate images: "
        f"{len(high_quality)}"
    )

    if len(high_quality) < SUBSET_SIZE:
        raise ValueError(
            "Not enough high-quality candidates "
            f"for subset size {SUBSET_SIZE}."
        )

    selected = high_quality.sample(
        n=SUBSET_SIZE,
        random_state=RANDOM_SEED,
    )

    # ---------------------------------------------------------
    # 5. Create output directory
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Clear previous generated selection if script is rerun.
    for existing in OUTPUT_DIR.iterdir():
        if existing.is_file():
            existing.unlink()

    # ---------------------------------------------------------
    # 6. Build a filename lookup
    # ---------------------------------------------------------

    image_lookup = {}

    for path in KONIQ_IMAGES.rglob("*"):

        if path.is_file() and path.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
        }:
            image_lookup[path.name] = path

    print(
        f"\nImages found in extracted dataset: "
        f"{len(image_lookup)}"
    )

    # ---------------------------------------------------------
    # 7. Copy selected images
    # ---------------------------------------------------------

    copied = 0
    missing = []

    selection_records = []

    for _, row in selected.iterrows():

        filename = str(
            row[filename_column]
        ).strip()

        score = float(
            row[score_column]
        )

        source = image_lookup.get(filename)

        if source is None:
            missing.append(filename)
            continue

        destination = OUTPUT_DIR / filename

        shutil.copy2(
            source,
            destination,
        )

        copied += 1

        selection_records.append(
            {
                "filename": filename,
                "mos": score,
            }
        )

    # ---------------------------------------------------------
    # 8. Save selection metadata
    # ---------------------------------------------------------

    metadata_path = Path(
        "dataset/koniq_selection.csv"
    )

    pd.DataFrame(
        selection_records
    ).sort_values(
        "mos",
        ascending=False,
    ).to_csv(
        metadata_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 9. Report
    # ---------------------------------------------------------

    print("\n========================================")
    print("KONIQ PREPARATION COMPLETE")
    print("========================================")

    print(f"Requested images : {SUBSET_SIZE}")
    print(f"Copied images    : {copied}")
    print(f"Missing images   : {len(missing)}")

    print(
        f"\nImages saved to:\n"
        f"{OUTPUT_DIR.resolve()}"
    )

    print(
        f"\nSelection metadata:\n"
        f"{metadata_path.resolve()}"
    )

    if selection_records:

        selected_scores = [
            item["mos"]
            for item in selection_records
        ]

        print(
            f"\nSelected MOS range: "
            f"{min(selected_scores):.4f} "
            f"to "
            f"{max(selected_scores):.4f}"
        )

    if missing:
        print("\nFirst few missing files:")

        for filename in missing[:10]:
            print(filename)


if __name__ == "__main__":
    main()