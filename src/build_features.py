"""Build feature tables from images and masks."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config
from .features import extract_features


def ensure_dirs():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def build_features(split: str) -> pd.DataFrame:
    ensure_dirs()

    if split == "train":
        ids = [i for i in config.TRAIN_IDS if i not in config.EXCLUDED_TRAIN_IDS]
        img_dir = config.TRAIN_DIR
        mask_dir = config.TRAIN_MASK_DIR
        output_path = config.TRAIN_FEATURES_CSV
    elif split == "test":
        ids = list(config.TEST_IDS)
        img_dir = config.TEST_DIR
        mask_dir = config.TEST_MASK_DIR
        output_path = config.TEST_FEATURES_CSV
    else:
        raise ValueError(f"Unknown split: {split}")

    rows = []
    for idx, image_id in enumerate(ids, start=1):
        img_path = img_dir / f"{image_id}.JPG"
        mask_path = mask_dir / f"binary_{image_id}.tif"

        if not img_path.exists() or not mask_path.exists():
            print(f"  SKIP ID {image_id} (missing file)")
            continue

        if idx == 1 or idx % 25 == 0 or idx == len(ids):
            print(f"  [{split}] {idx}/{len(ids)} — ID {image_id}")

        feats = extract_features(img_path, mask_path)
        row = {config.ID_COLUMN: image_id}
        if split == "train":
            pass  # labels merged later
        row.update(feats)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(config.ID_COLUMN).reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    if split == "train":
        labels = pd.read_excel(config.LABEL_FILE)
        labels = labels.rename(columns={
            col: config.ID_COLUMN for col in labels.columns if col.strip().lower() == "id"
        })
        df = df.merge(
            labels[[config.ID_COLUMN, config.TARGET_COLUMN, config.SPECIES_COLUMN]],
            on=config.ID_COLUMN,
            how="left",
        )

    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} rows to {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build feature tables.")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    args = parser.parse_args()
    build_features(args.split)


if __name__ == "__main__":
    main()
