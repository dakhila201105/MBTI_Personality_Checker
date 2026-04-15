# AI-Powered Mental Health Sentiment Analyzer

**Repository:** [github.com/dakhila201105/MBTI_Personality_Checker](https://github.com/dakhila201105/MBTI_Personality_Checker)

An end-to-end NLP demo that classifies short user text into **Stress**, **Anxiety**, **Depression**, or **Neutral**, returns a **confidence score**, and surfaces **rule-based supportive suggestions**. A **FastAPI** backend serves a **baseline** TF-IDF + scikit-learn pipeline (Logistic Regression and Naive Bayes trained in one script), with an **optional** Hugging Face **BERT** fine-tuning path for experimentation.

> **Disclaimer:** This project is for education and prototyping only. It is **not** a medical or diagnostic tool. Always encourage users in crisis to contact qualified professionals or emergency services.

---

## Features

| Area | Details |
|------|---------|
| **Classification** | Four-way emotion-style labels with probability-based confidence |
| **Preprocessing** | Lowercasing, URL removal, punctuation stripping, stopword removal, lemmatization (NLTK) |
| **Baselines** | TF-IDF + Logistic Regression; TF-IDF + Multinomial Naive Bayes |
| **Advanced (optional)** | Fine-tune DistilBERT (or similar) via Hugging Face `transformers` |
| **Evaluation** | Accuracy; macro precision, recall, F1; confusion matrix; JSON metrics under `models/eval/` |
| **API** | `POST /predict` with JSON `{ "text": "..." }` |
| **Response** | `emotion`, `confidence`, `suggestion`, `extra_tips`, `model_used` |
| **Frontend** | Responsive HTML/CSS/JS UI (served by FastAPI for same-origin calls); Angular or other SPAs can call the same `POST /predict` API |
| **Ops** | Prediction logging (JSON lines), pickle model artifacts, modular `utils/` package |

---

## Tech Stack

- **Python 3.10+** (tested with 3.12)
- **FastAPI** + **Uvicorn**
- **scikit-learn**, **pandas**, **NumPy**
- **NLTK** (tokenization, stopwords, WordNet lemmatizer)
- **Optional:** **PyTorch** + **Hugging Face Transformers** (`requirements-bert.txt`)
- **Optional:** **spaCy** for alternative preprocessing (not installed by default; see `requirements.txt` comments)

---

## Project Structure

```
AI-Powered-Mental-Health-Sentiment-Analyzer/
├── backend/
│   └── main.py              # FastAPI app, /predict, /health, static UI
├── data/
│   ├── raw/
│   │   └── placeholder_emotions.csv
│   └── DATASET_INSTRUCTIONS.md
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── models/                   # Generated .pkl files (gitignored)
├── scripts/
│   ├── train.py              # Baseline training + evaluation + pickle export
│   └── train_bert.py         # Optional BERT fine-tuning
├── utils/
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models_baseline.py
│   ├── models_bert.py
│   ├── suggestions.py
│   ├── evaluator.py
│   └── prediction_logger.py
├── requirements.txt          # Core API + baseline ML
├── requirements-bert.txt     # Optional BERT stack
└── README.md
```

---

## Setup

### 1. Virtual environment

```bash
cd AI-Powered-Mental-Health-Sentiment-Analyzer
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Train baseline models and save pickles

```bash
python scripts/train.py
```

This will:

- Load `data/raw/placeholder_emotions.csv`
- Hold out a test split, print **accuracy / precision / recall / F1** and **confusion matrices**
- Write metrics to `models/eval/*.json`
- Fit on full data and save:
  - `models/tfidf_logistic_pipeline.pkl` (default API model)
  - `models/tfidf_naive_bayes_pipeline.pkl`

First run downloads **NLTK** data (punkt, stopwords, wordnet, etc.); ensure outbound network access.

### 3. Run the API + UI

From the project root (so `utils` imports resolve):

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000/** for the UI. The API exposes:

- `POST /predict` — body: `{ "text": "your message" }`
- `GET /health` — model/status probe

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"I feel overwhelmed and cannot sleep\"}"
```

### 4. Optional BERT fine-tuning

Install the extra stack:

```bash
pip install -r requirements-bert.txt
```

Train (small data will overfit—use real datasets for meaningful results):

```bash
python scripts/train_bert.py --epochs 3
```

Serve with BERT **if** `models/bert_finetuned/config.json` exists:

```bash
set USE_BERT=true
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

When `USE_BERT` is enabled, the API uses BERT when the artifact is present; otherwise it falls back to the sklearn pipeline.

---

## Dataset

- Placeholder: `data/raw/placeholder_emotions.csv` (`text`, `label`).
- Replacing with **Kaggle**, **Reddit**, or **Hugging Face** data: see **`data/DATASET_INSTRUCTIONS.md`**.

---

## Logging

- Each prediction appends one JSON line to **`logs/predictions.log`** (input truncated for privacy).
- Add log rotation or external logging for production deployments.

---

## Deployment Notes

### Render (example)

1. Set **Build Command**: `pip install -r requirements.txt && python scripts/train.py`  
   (Or bake a trained `models/` artifact in CI and skip training on Render.)
2. Set **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
3. Ensure **Root Directory** is this repository and **Python version** matches local dev.

### Hugging Face Spaces

- **Gradio** or **Docker** Spaces can wrap the same FastAPI app (e.g., run Uvicorn as `CMD`).
- For a minimal Space, a thin **Gradio** interface can call the same `utils` prediction helpers; alternatively use a **Docker** Space with `EXPOSE 7860` and a process manager running Uvicorn.

---

## Future Improvements

- Curate a larger, ethically sourced, multi-label or intensity-scored dataset
- Calibration (temperature scaling / Platt) for better probability estimates
- Model cards, bias evaluation, and clinician-in-the-loop review
- Replace or augment lexicon-based suggestions with templated clinical resources where appropriate
- Add authentication, rate limits, and red-team testing for misuse scenarios
- Export to **ONNX** or **TorchScript** for BERT serving optimization

---

## License

Use this repository for learning and portfolio work; verify licenses of any third-party datasets you add.
