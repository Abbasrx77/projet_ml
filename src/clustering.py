"""Clustering analysis."""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .train_models import NON_FEATURE_COLUMNS


def run_clustering():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.TRAIN_FEATURES_CSV)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X_raw = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X = pipe.fit_transform(X_raw)
    y = df[config.TARGET_COLUMN].astype(str)
    n_clusters = y.nunique()

    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    methods = {
        "kmeans": KMeans(n_clusters=n_clusters, n_init=30, random_state=config.RANDOM_STATE),
        "agglomerative": AgglomerativeClustering(n_clusters=n_clusters, linkage="ward"),
    }

    rows = []
    for name, estimator in methods.items():
        print(f"  Clustering: {name}")
        labels = estimator.fit_predict(X)
        sil = silhouette_score(X, labels)
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)
        rows.append({"method": name, "silhouette": sil, "ARI": ari, "NMI": nmi})

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Cluster": labels.astype(str)})
        sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Cluster",
                        palette="tab10", s=55, edgecolor="white", linewidth=0.4, ax=ax)
        ax.set_title(f"{name.title()} Clusters (PCA)")
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / f"clustering_{name}.png", dpi=150)
        plt.close(fig)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.CLUSTERING_METRICS_CSV, index=False)
    print(f"\n  Clustering metrics:")
    print(metrics.to_string(index=False))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run clustering.")
    parser.parse_args()
    run_clustering()


if __name__ == "__main__":
    main()
