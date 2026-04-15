"""
Append-only logging of API predictions for audit and debugging.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from utils.config import LOGS_DIR, PREDICTIONS_LOG


def ensure_log_dir() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def log_prediction(
    text_snippet: str,
    emotion: str,
    confidence: float,
    model_name: str,
    extra: dict | None = None,
    log_path: Path | None = None,
) -> None:
    """
    Log one prediction as a single JSON line (JSONL style).
    Truncates input text in logs to reduce accidental PII retention.
    """
    ensure_log_dir()
    path = log_path or PREDICTIONS_LOG
    safe_text = (text_snippet or "")[:500]
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "emotion": emotion,
        "confidence": round(float(confidence), 6),
        "text_preview": safe_text,
    }
    if extra:
        record["extra"] = extra
    line = json.dumps(record, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
