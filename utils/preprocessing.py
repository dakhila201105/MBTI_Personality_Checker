"""
Text preprocessing for mental-health sentiment analysis.
Steps: lowercase, URL removal, punctuation stripping, stopword removal, lemmatization.
"""
import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Lazy NLTK data download (runs once per environment)
_nltk_ready = False


def ensure_nltk_data() -> None:
    """Download required NLTK corpora if missing."""
    global _nltk_ready
    if _nltk_ready:
        return

    def _ensure(resource: str, find_path: str) -> None:
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.download(resource, quiet=True)

    _ensure("punkt_tab", "tokenizers/punkt_tab")
    _ensure("punkt", "tokenizers/punkt")
    _ensure("stopwords", "corpora/stopwords")
    _ensure("wordnet", "corpora/wordnet")
    _ensure("omw-1.4", "corpora/omw-1.4")
    _nltk_ready = True


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    ensure_nltk_data()
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _stop_words() -> set:
    ensure_nltk_data()
    return set(stopwords.words("english"))


URL_PATTERN = re.compile(
    r"http\S+|www\.\S+",
    re.IGNORECASE,
)


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline for a single document.
    Returns a space-separated string of lemmatized tokens.
    """
    ensure_nltk_data()
    if not text or not str(text).strip():
        return ""

    lowered = text.lower().strip()
    lowered = remove_urls(lowered)
    # Remove punctuation by character (keep intra-word apostrophes handled by stripping)
    lowered = lowered.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(lowered)
    stops = _stop_words()
    lem = _lemmatizer()
    out = []
    for tok in tokens:
        if tok.isdigit():
            continue
        if tok in stops:
            continue
        if len(tok) < 2:
            continue
        out.append(lem.lemmatize(tok))
    return " ".join(out)


def preprocess_batch(texts: list[str]) -> list[str]:
    """Apply preprocess_text to a list of raw strings."""
    return [preprocess_text(t) for t in texts]


def sklearn_preprocess_raw(X) -> list[str]:
    """
    Adapter for sklearn Pipeline / FunctionTransformer: raw texts -> preprocessed strings.
    Accepts 1d/2d array-like of strings (e.g. shape (n_samples,) or (n_samples, 1)).
    """
    import numpy as np

    arr = np.asarray(X, dtype=object)
    return [preprocess_text(str(x)) for x in arr.ravel()]
