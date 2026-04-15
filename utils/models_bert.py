"""
Optional BERT fine-tuning and inference using Hugging Face Transformers.

Requires: torch, transformers, datasets (see requirements.txt).

This module is intentionally separate so baseline training works without GPU/PyTorch.
"""
from pathlib import Path
from typing import Any

import numpy as np

# Lazy imports inside functions to avoid import errors when extras not installed


def train_bert_classifier(
    texts: list[str],
    labels: list[str],
    label_list: list[str],
    output_dir: Path,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
) -> dict[str, Any]:
    """
    Fine-tune a small BERT-family model for multi-class text classification.
    Saves tokenizer + model to output_dir.
    """
    import torch
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    le = LabelEncoder()
    le.fit(label_list)
    y = le.transform(labels)

    X_train, X_val, y_train, y_val = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label={i: lab for i, lab in enumerate(le.classes_)},
        label2id={lab: i for i, lab in enumerate(le.classes_)},
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    train_ds = Dataset.from_dict({"text": X_train, "labels": y_train})
    val_ds = Dataset.from_dict({"text": X_val, "labels": y_val})
    train_ds = train_ds.map(tokenize_batch, batched=True)
    val_ds = val_ds.map(tokenize_batch, batched=True)
    cols_to_remove = ["text"]
    train_ds = train_ds.remove_columns([c for c in cols_to_remove if c in train_ds.column_names])
    val_ds = val_ds.remove_columns([c for c in cols_to_remove if c in val_ds.column_names])
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    args = TrainingArguments(
        output_dir=str(output_dir / "trainer_output"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels_ = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": (preds == labels_).mean()}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Save label encoder mapping
    import json

    meta = {"label_classes": list(le.classes_)}
    (output_dir / "label_encoder_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"output_dir": str(output_dir), "epochs": epochs}


class BertClassifierWrapper:
    """Load saved HF model + tokenizer for inference."""

    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = self.model.config.id2label

    def predict_proba(self, text: str) -> tuple[str, float]:
        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        label = self.id2label[idx]
        conf = float(probs[idx].item())
        return label, conf
