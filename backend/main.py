"""
FastAPI application: POST /predict for emotion classification + suggestions.
Serves the static frontend from / for a single-process demo.
"""
from __future__ import annotations

import os
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import (  # noqa: E402
    BASELINE_MODEL_PATH,
    BERT_MODEL_DIR,
    LABELS,
)
from utils.prediction_logger import log_prediction  # noqa: E402
from utils.suggestions import extra_tips_for_label, suggestion_for_label  # noqa: E402

# Optional BERT
_bert_wrapper = None
_sklearn_artifact = None


def load_sklearn_model():
    global _sklearn_artifact
    if not BASELINE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing model at {BASELINE_MODEL_PATH}. Run: python scripts/train.py"
        )
    with BASELINE_MODEL_PATH.open("rb") as f:
        _sklearn_artifact = pickle.load(f)


def maybe_load_bert():
    global _bert_wrapper
    if os.getenv("USE_BERT", "").lower() not in ("1", "true", "yes"):
        return
    try:
        from utils.models_bert import BertClassifierWrapper
    except ImportError:
        return
    if not BERT_MODEL_DIR.is_dir() or not (BERT_MODEL_DIR / "config.json").is_file():
        return
    _bert_wrapper = BertClassifierWrapper(BERT_MODEL_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_sklearn_model()
    maybe_load_bert()
    yield


app = FastAPI(
    title="AI-Powered Mental Health Sentiment Analyzer",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User message to analyze")


class PredictResponse(BaseModel):
    emotion: str
    confidence: float
    suggestion: str
    extra_tips: list[str] = []
    model_used: str = "tfidf_logistic_regression"


def predict_sklearn(raw_text: str) -> tuple[str, float, str]:
    pipe = _sklearn_artifact["pipeline"]
    le: Any = _sklearn_artifact["label_encoder"]
    name = _sklearn_artifact.get("model_name", "sklearn_pipeline")
    idx = pipe.predict([raw_text])[0]
    emotion = le.inverse_transform(np.array([idx]))[0]
    proba = pipe.predict_proba([raw_text])[0]
    confidence = float(np.max(proba))
    return str(emotion), confidence, name


def predict_bert(raw_text: str) -> tuple[str, float, str] | None:
    if _bert_wrapper is None:
        return None
    label, conf = _bert_wrapper.predict_proba(raw_text)
    return str(label), float(conf), "bert_finetuned"


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    use_bert_first = os.getenv("USE_BERT", "").lower() in ("1", "true", "yes")
    emotion, confidence, model_used = None, 0.0, "unknown"

    if use_bert_first:
        bert_out = predict_bert(text)
        if bert_out is not None:
            emotion, confidence, model_used = bert_out

    if emotion is None:
        emotion, confidence, model_used = predict_sklearn(text)

    if emotion not in LABELS:
        emotion = "Neutral"

    suggestion = suggestion_for_label(emotion)
    tips = extra_tips_for_label(emotion)

    log_prediction(
        text_snippet=text,
        emotion=emotion,
        confidence=confidence,
        model_name=model_used,
        extra={"tips_count": len(tips)},
    )

    return PredictResponse(
        emotion=emotion,
        confidence=round(confidence, 4),
        suggestion=suggestion,
        extra_tips=tips,
        model_used=model_used,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "labels": LABELS,
        "bert_loaded": _bert_wrapper is not None,
        "sklearn_loaded": _sklearn_artifact is not None,
    }


FRONTEND_DIR = ROOT / "frontend"


@app.get("/")
def serve_index():
    """Serve single-page UI (same origin as API avoids CORS for /predict)."""
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(str(index))


if FRONTEND_DIR.is_dir():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static",
    )
