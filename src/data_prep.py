"""Build the training/validation split from raw CBIS-DDSM files.

CBIS-DDSM ships as a set of case-description CSVs (one row per lesion, with a
pathology label) plus a directory of JPEG images keyed by a DICOM-derived
folder UID. This module joins the two, keeps only benign/malignant cases
(dropping the small number of unlabeled entries), and produces a stratified
80/20 train/validation split.

Expected layout after downloading + unzipping the Kaggle dataset
(awsaf49/cbis-ddsm-breast-cancer-image-dataset) into `data/raw/`:

    data/raw/csv/mass_case_description_train_set.csv
    data/raw/csv/mass_case_description_test_set.csv
    data/raw/csv/calc_case_description_train_set.csv
    data/raw/csv/calc_case_description_test_set.csv
    data/raw/jpeg/<uid>/*.jpg
"""

import glob
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP = {"BENIGN": 0, "MALIGNANT": 1}


def load_metadata(csv_dir: str) -> pd.DataFrame:
    """Load and combine the four CBIS-DDSM case-description CSVs."""
    csv_dir = Path(csv_dir)
    frames = [
        pd.read_csv(csv_dir / "mass_case_description_train_set.csv"),
        pd.read_csv(csv_dir / "mass_case_description_test_set.csv"),
        pd.read_csv(csv_dir / "calc_case_description_train_set.csv"),
        pd.read_csv(csv_dir / "calc_case_description_test_set.csv"),
    ]
    meta = pd.concat(frames, axis=0)

    # Keep only the two classes we're training on.
    meta = meta[meta["pathology"].isin(LABEL_MAP.keys())].copy()
    meta["label"] = meta["pathology"].map(LABEL_MAP)

    # The image folder UID is embedded in the cropped-image file path
    # (e.g. "Mass-Training_P_00001_LEFT_CC_1/<uid>/1-1.jpg").
    meta["uid"] = meta["cropped image file path"].apply(lambda x: x.split("/")[2])
    return meta


def link_images(meta: pd.DataFrame, jpeg_dir: str) -> pd.DataFrame:
    """Match every JPEG on disk to its metadata row via the folder UID."""
    all_images = glob.glob(f"{jpeg_dir}/**/*.jpg", recursive=True)
    images = pd.DataFrame(all_images, columns=["image_path"])
    images["uid"] = images["image_path"].apply(lambda x: x.split("/")[-2])

    df = images.merge(meta[["uid", "label"]], on="uid", how="inner")
    return df


def make_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified train/validation split so both sets keep the class balance."""
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_dataset(raw_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: CSVs + JPEGs on disk -> (train_df, val_df)."""
    raw_dir = Path(raw_dir)
    meta = load_metadata(raw_dir / "csv")
    df = link_images(meta, raw_dir / "jpeg")
    print(f"Metadata cases: {len(meta)} | Images linked: {len(df)}")
    return make_split(df)


if __name__ == "__main__":
    train_df, val_df = build_dataset()
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")
    print("Train class balance:\n", train_df["label"].value_counts())
    print("Val class balance:\n", val_df["label"].value_counts())
