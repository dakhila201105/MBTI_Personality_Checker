"""
Train baseline models (Logistic Regression + Naive Bayes), evaluate, and save with pickle.
Run from project root: python scripts/train.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import (  # noqa: E402
    BASELINE_MODEL_PATH,
    DATA_RAW,
    LABELS,
    MODELS_DIR,
    NAIVE_BAYES_PATH,
)
from utils.evaluator import evaluate_predictions, save_evaluation_json  # noqa: E402
from utils.models_baseline import (  # noqa: E402
    build_logistic_regression_pipeline,
    build_naive_bayes_pipeline,
)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: text, label")
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip()
    unknown = set(df["label"].unique()) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown labels {unknown}. Expected subset of {LABELS}")
    return df


def maybe_stratify(y: np.ndarray):
    """Return stratify array if every class has at least 2 samples."""
    _, counts = np.unique(y, return_counts=True)
    if np.min(counts) >= 2:
        return y
    return None


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> None:
    models = {
        "logistic_regression": build_logistic_regression_pipeline(),
        "naive_bayes": build_naive_bayes_pipeline(),
    }
    eval_dir = MODELS_DIR / "eval"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_true_str = label_encoder.inverse_transform(y_test)
        y_pred_str = label_encoder.inverse_transform(y_pred)
        metrics = evaluate_predictions(
            np.array(y_true_str),
            np.array(y_pred_str),
            labels=LABELS,
        )
        print(f"\n=== {name} ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision (macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (macro): {metrics['recall_macro']:.4f}")
        print(f"F1 (macro): {metrics['f1_macro']:.4f}")
        cm = metrics["confusion_matrix"]
        print("Confusion matrix (rows=true, cols=pred):")
        for row, lab in zip(cm, LABELS):
            print(f"  {lab}: {row}")
        save_evaluation_json(metrics, eval_dir / f"{name}_metrics.json")


def save_sklearn_artifact(
    pipeline,
    label_encoder: LabelEncoder,
    path: Path,
    model_name: str,
) -> None:
    """Persist pipeline + label encoder using pickle (joblib optional alternative)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "model_name": model_name,
    }
    with path.open("wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved artifact to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline emotion classifiers.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_RAW,
        help="Path to CSV with columns text,label",
    )
    args = parser.parse_args()

    df = load_dataset(args.data)
    le = LabelEncoder()
    le.fit(LABELS)
    y = le.transform(df["label"].values)
    X = df["text"].values

    strat = maybe_stratify(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=strat,
    )

    train_and_evaluate(X_train, X_test, y_train, y_test, le)

    # Fit final models on full data for deployment artifacts
    lr = build_logistic_regression_pipeline()
    lr.fit(X, y)
    save_sklearn_artifact(lr, le, BASELINE_MODEL_PATH, "tfidf_logistic_regression")

    nb = build_naive_bayes_pipeline()
    nb.fit(X, y)
    save_sklearn_artifact(nb, le, NAIVE_BAYES_PATH, "tfidf_naive_bayes")

    print("\nTraining complete. Start API with: uvicorn backend.main:app --reload")


if __name__ == "__main__":
    main()
