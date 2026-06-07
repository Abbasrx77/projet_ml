"""Train and evaluate supervised models."""

from __future__ import annotations

import argparse
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config

NON_FEATURE_COLUMNS = {config.ID_COLUMN, config.TARGET_COLUMN, config.SPECIES_COLUMN}


def _ensure_dirs():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(config.TRAIN_FEATURES_CSV)
    # Clean: exclude Dragonfly (1 sample), merge 'Bee & Bumblebee' into 'Bee'
    df = df[df[config.TARGET_COLUMN] != "Dragonfly"].copy()
    df[config.TARGET_COLUMN] = df[config.TARGET_COLUMN].replace("Bee & Bumblebee", "Bee")
    df = df.reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df[config.TARGET_COLUMN].astype(str)
    return X, y


def _scaled_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])


def _tree_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", classifier),
    ])


def model_registry() -> dict[str, Pipeline]:
    return {
        "svm_rbf": _scaled_pipeline(
            SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=config.RANDOM_STATE)
        ),
        "knn": _scaled_pipeline(
            KNeighborsClassifier(n_neighbors=5, weights="distance", metric="manhattan")
        ),
        "random_forest": _tree_pipeline(
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=config.RANDOM_STATE, n_jobs=-1)
        ),
        "gradient_boosting": _tree_pipeline(
            GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                       random_state=config.RANDOM_STATE)
        ),
        "mlp": _scaled_pipeline(
            MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                          alpha=0.001, max_iter=600, early_stopping=True,
                          random_state=config.RANDOM_STATE)
        ),
    }


def train_models():
    _ensure_dirs()
    X, y = load_training_data()

    min_class_count = y.value_counts().min()
    n_folds = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)

    rows = []
    for name, pipeline in model_registry().items():
        print(f"  Evaluating: {name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = cross_val_predict(pipeline, X, y, cv=cv)

        row = {
            "model": name,
            "accuracy": accuracy_score(y, y_pred),
            "macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, y_pred, average="weighted", zero_division=0),
            "macro_precision": precision_score(y, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y, y_pred, average="macro", zero_division=0),
        }
        rows.append(row)

        # Confusion matrix
        labels = sorted(y.unique())
        cm = confusion_matrix(y, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"Confusion Matrix: {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / f"confusion_matrix_{name}.png", dpi=150)
        plt.close(fig)

    comparison = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    comparison.to_csv(config.MODEL_COMPARISON_CSV, index=False)
    print(f"\n  Model comparison saved to {config.MODEL_COMPARISON_CSV}")
    print(comparison.to_string(index=False))

    # Train best model on full data
    best_name = comparison.iloc[0]["model"]
    best_pipeline = clone(model_registry()[best_name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_pipeline.fit(X, y)

    joblib.dump(best_pipeline, config.BEST_MODEL_PATH)

    info = {
        "best_model": best_name,
        "macro_f1": float(comparison.iloc[0]["macro_f1"]),
        "cv_folds": n_folds,
        "train_samples": len(X),
        "feature_count": X.shape[1],
    }
    config.BEST_MODEL_INFO_JSON.write_text(json.dumps(info, indent=2))
    print(f"\n  Best model: {best_name} (macro F1: {info['macro_f1']:.4f})")
    print(f"  Saved to {config.BEST_MODEL_PATH}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Train supervised models.")
    parser.parse_args()
    train_models()


if __name__ == "__main__":
    main()
