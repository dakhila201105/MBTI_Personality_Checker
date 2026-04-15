"""
Baseline classifiers: Logistic Regression and Multinomial Naive Bayes on TF-IDF features.
Pipelines include preprocessing -> TF-IDF -> classifier for end-to-end training/inference.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from utils.features import build_tfidf_vectorizer
from utils.preprocessing import sklearn_preprocess_raw


def build_logistic_regression_pipeline(
    max_iter: int = 2000,
    C: float = 1.0,
) -> Pipeline:
    """Preprocess + TF-IDF + multinomial logistic regression."""
    return Pipeline(
        [
            (
                "preprocess",
                FunctionTransformer(sklearn_preprocess_raw, validate=False),
            ),
            ("tfidf", build_tfidf_vectorizer()),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter,
                    C=C,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_naive_bayes_pipeline(alpha: float = 1.0) -> Pipeline:
    """Preprocess + TF-IDF + Multinomial Naive Bayes."""
    return Pipeline(
        [
            (
                "preprocess",
                FunctionTransformer(sklearn_preprocess_raw, validate=False),
            ),
            ("tfidf", build_tfidf_vectorizer()),
            ("clf", MultinomialNB(alpha=alpha)),
        ],
    )
