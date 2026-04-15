"""Central configuration: paths and label vocabulary."""
from pathlib import Path

# Project root (parent of utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "placeholder_emotions.csv"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Must match training labels and suggestion keys
LABELS = ["Stress", "Anxiety", "Depression", "Neutral"]

# Default saved artifacts (baseline sklearn pipeline)
BASELINE_MODEL_PATH = MODELS_DIR / "tfidf_logistic_pipeline.pkl"
NAIVE_BAYES_PATH = MODELS_DIR / "tfidf_naive_bayes_pipeline.pkl"

# BERT optional paths
BERT_MODEL_DIR = MODELS_DIR / "bert_finetuned"

PREDICTIONS_LOG = LOGS_DIR / "predictions.log"
