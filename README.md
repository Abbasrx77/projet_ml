# Machine Learning Project: To bee or not to bee

**Cours** : IG.2412 — Machine Learning  
**Objectif** : Classification d'insectes pollinisateurs par type (bug type)

## Structure du projet

```
ML/
├── src/                         ← Code modulaire
│   ├── config.py                ← Chemins et constantes
│   ├── features.py              ← Extraction de features (obligatoires + extras)
│   ├── build_features.py        ← Construction des CSV de features
│   ├── train_models.py          ← Entraînement et évaluation supervisée
│   ├── clustering.py            ← Méthodes de clustering
│   ├── visualize.py             ← PCA, t-SNE, Isomap, distributions
│   └── predict.py               ← Prédiction sur le test set
│
├── notebooks/
│   ├── ML_final.ipynb           ← Notebook complet (version notebook)
│   └── ML_project.ipynb         ← Notebook original (features seulement)
│
├── scripts/
│   ├── run_train_pipeline.sh    ← Pipeline complète d'entraînement
│   └── run_submission_pipeline.sh ← Pipeline de soumission
│
├── train/                       ← (local, pas sur git)
├── test/                        ← (local, pas sur git)
├── data/processed/              ← Features CSV générées
├── models/                      ← Modèle sauvegardé (.joblib)
├── results/                     ← Métriques, comparaisons
└── reports/figures/             ← Visualisations PNG
```

## Installation

```bash
pip install -r requirements.txt
```

## Données

Placer les données (fournies par le prof) comme suit :
```
train/
  classif.xlsx
  1.JPG ... 250.JPG
  masks/
    binary_1.tif ... binary_250.tif
```

## Pipeline d'entraînement

```bash
python3 -m src.build_features --split train
python3 -m src.train_models
python3 -m src.clustering
python3 -m src.visualize
```

Ou en une commande : `bash scripts/run_train_pipeline.sh`

## Pipeline de soumission

Une fois le dossier `test/` disponible :

```bash
python3 -m src.build_features --split test
python3 -m src.predict
```

Le CSV final est écrit dans `results/submission.csv`.

## Features extraites

- **Obligatoires** : color_symmetry, shape_symmetry, bug_pixel_ratio, RGB stats (min/max/mean/median/std)
- **Shape** : aspect_ratio, circularity, solidity, eccentricity, extent, major/minor axis
- **Hu moments** : 7 invariants de forme
- **HSV** : mean et std de H, S, V
- **LAB** : mean et std de L, a, b
- **Texture** : edge_density, texture_contrast

## Résultats (cross-validation 5-fold)

| Modèle | F1 Macro | Accuracy |
|--------|:--------:|:--------:|
| SVM (RBF) | **0.76** | 0.86 |
| Random Forest | 0.71 | 0.82 |
| KNN | 0.60 | 0.78 |
| Gradient Boosting | 0.52 | 0.76 |
| MLP | 0.35 | 0.75 |
