# Replacing the Placeholder Dataset

The file `data/raw/placeholder_emotions.csv` is a **minimal synthetic example** for development and smoke-testing. For research-grade or production use, replace it with a real mental-health–related text dataset.

## Required CSV format

| Column | Description |
|--------|-------------|
| `text` | Raw user utterance or post (string) |
| `label` | One of: `Stress`, `Anxiety`, `Depression`, `Neutral` (match spelling in `utils/config.py` if you change classes) |

## Where to find data

1. **Kaggle**  
   Search for: emotion classification, mental health tweets, stress detection, depression Reddit.  
   Download CSV, rename columns to `text` and `label`, map your label names to the four categories above (or update `LABELS` in code).

2. **Reddit**  
   Use public datasets derived from mental-health subreddits (e.g., r/depression, r/anxiety) with proper licensing and ethics review. Preprocess to remove PII.

3. **Hugging Face Datasets**  
   Browse [https://huggingface.co/datasets](https://huggingface.co/datasets) for emotion or mental-health text; use `datasets` library to load and export to CSV.

## Ethics & safety

- Mental health labels are sensitive; avoid harmful generalizations.  
- This demo is **not** a clinical tool—disclaimers belong in your product copy.  
- Comply with dataset licenses and institutional review if applicable.

After replacing data, re-run:

```bash
python scripts/train.py
```

Then restart the API so it loads the new `models/` artifacts.
