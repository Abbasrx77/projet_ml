from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "train"
TRAIN_MASK_DIR = TRAIN_DIR / "masks"
TEST_DIR = PROJECT_ROOT / "test"
TEST_MASK_DIR = TEST_DIR / "masks"
LABEL_FILE = TRAIN_DIR / "classif.xlsx"

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_FEATURES_CSV = PROCESSED_DIR / "train_features.csv"
TEST_FEATURES_CSV = PROCESSED_DIR / "test_features.csv"

MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
CLUSTERING_METRICS_CSV = RESULTS_DIR / "clustering_metrics.csv"
BEST_MODEL_INFO_JSON = RESULTS_DIR / "best_model_info.json"
SUBMISSION_CSV = RESULTS_DIR / "submission.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

TRAIN_IDS = tuple(range(1, 251))
TEST_IDS = tuple(range(251, 348))
EXCLUDED_TRAIN_IDS = {154}
RANDOM_STATE = 42

TARGET_COLUMN = "bug type"
SPECIES_COLUMN = "species"
ID_COLUMN = "ID"
