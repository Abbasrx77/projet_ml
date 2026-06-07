"""Generate data visualizations."""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE, Isomap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .train_models import NON_FEATURE_COLUMNS


def _load_and_scale() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(config.TRAIN_FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return df, pipe.fit_transform(X)


def _projection_plot(points, labels, path, title):
    fig, ax = plt.subplots(figsize=(9, 7))
    plot_df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "bug type": labels.values})
    sns.scatterplot(data=plot_df, x="x", y="y", hue="bug type", s=55,
                    edgecolor="white", linewidth=0.4, ax=ax)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_visualizations():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df, X = _load_and_scale()
    labels = df[config.TARGET_COLUMN]

    # Distribution plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, y=config.TARGET_COLUMN,
                  order=df[config.TARGET_COLUMN].value_counts().index, ax=axes[0])
    axes[0].set_title("Bug Type Distribution")
    sns.countplot(data=df, y=config.SPECIES_COLUMN,
                  order=df[config.SPECIES_COLUMN].value_counts().head(12).index, ax=axes[1])
    axes[1].set_title("Top 12 Species")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "distributions.png", dpi=150)
    plt.close(fig)

    # PCA
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    _projection_plot(X_pca, labels, config.FIGURES_DIR / "pca_2d.png",
                     f"PCA ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

    # t-SNE
    perplexity = min(30, len(df) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=config.RANDOM_STATE)
    X_tsne = tsne.fit_transform(X)
    _projection_plot(X_tsne, labels, config.FIGURES_DIR / "tsne_2d.png",
                     f"t-SNE (perplexity={perplexity})")

    # Isomap (non-linear)
    n_neighbors = min(10, len(df) - 1)
    isomap = Isomap(n_neighbors=n_neighbors, n_components=2)
    X_isomap = isomap.fit_transform(X)
    _projection_plot(X_isomap, labels, config.FIGURES_DIR / "isomap_2d.png",
                     f"Isomap (n_neighbors={n_neighbors})")

    print(f"  Visualizations saved to {config.FIGURES_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations.")
    parser.parse_args()
    generate_visualizations()


if __name__ == "__main__":
    main()
