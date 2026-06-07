"""Generate final predictions on test set."""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from . import config
from .train_models import NON_FEATURE_COLUMNS


def generate_submission() -> pd.DataFrame:
    if not config.TEST_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Test features not found: {config.TEST_FEATURES_CSV}. "
            "Run: python -m src.build_features --split test"
        )
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {config.BEST_MODEL_PATH}. "
            "Run: python -m src.train_models"
        )

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(config.TEST_FEATURES_CSV)
    feature_cols = [c for c in test_df.columns if c not in NON_FEATURE_COLUMNS]
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    model = joblib.load(config.BEST_MODEL_PATH)
    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        config.ID_COLUMN: test_df[config.ID_COLUMN].astype(int),
        config.TARGET_COLUMN: predictions,
    })
    submission.to_csv(config.SUBMISSION_CSV, index=False)
    print(f"  Submission saved: {config.SUBMISSION_CSV} ({len(submission)} rows)")
    print(f"  Distribution:\n{submission[config.TARGET_COLUMN].value_counts().to_string()}")
    return submission


def main():
    parser = argparse.ArgumentParser(description="Generate submission CSV.")
    parser.parse_args()
    generate_submission()


if __name__ == "__main__":
    main()
