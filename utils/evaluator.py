"""
Model evaluation: accuracy, precision/recall/F1, confusion matrix.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """
    Compute standard metrics and structured classification report.
    """
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def save_evaluation_json(metrics: dict[str, Any], path: Path) -> None:
    """Persist metrics as JSON (numpy types converted)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # classification_report may contain nested dicts — ensure JSON-serializable
    serializable = json.loads(json.dumps(metrics, default=str))
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
