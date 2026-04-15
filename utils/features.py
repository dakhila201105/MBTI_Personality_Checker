"""
Feature extraction: TF-IDF vectorization for baseline models.
Optional BERT tokenization is handled in utils/models_bert.py.
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """
    TF-IDF with unigrams and bigrams.
    Text should already be preprocessed (see utils.preprocessing).
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
    )
