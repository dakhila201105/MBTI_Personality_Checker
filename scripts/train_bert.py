"""
Optional: fine-tune DistilBERT (or similar) for emotion classification.
Requires PyTorch + transformers. Small datasets may overfit—use more real data for production.

Usage (from project root):
  python scripts/train_bert.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import BERT_MODEL_DIR, DATA_RAW, LABELS  # noqa: E402
from utils.models_bert import train_bert_classifier  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_RAW)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=BERT_MODEL_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = df.dropna(subset=["text", "label"])
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).str.strip().tolist()
    bad = set(labels) - set(LABELS)
    if bad:
        raise ValueError(f"Invalid labels: {bad}")

    result = train_bert_classifier(
        texts=texts,
        labels=labels,
        label_list=LABELS,
        output_dir=args.out,
        epochs=args.epochs,
    )
    print("BERT training finished:", result)


if __name__ == "__main__":
    main()
