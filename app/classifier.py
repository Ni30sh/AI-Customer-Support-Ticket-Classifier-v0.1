from pathlib import Path
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from app.dataset import SAMPLE_TICKETS


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ticket_classifier.joblib"


def train_classifier() -> tuple[object, dict[str, float]]:
    texts = [text for text, _ in SAMPLE_TICKETS]
    labels = [label for _, label in SAMPLE_TICKETS]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    model = TfidfVectorizer(
        sublinear_tf=True,
        ngram_range=(1, 2),
        analyzer="char_wb",
        min_df=1,
    )
    X_train = model.fit_transform(train_texts)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, train_labels)

    predictions = classifier.predict(model.transform(test_texts))
    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, classifier), MODEL_PATH)
    return model, metrics


def load_classifier() -> tuple[object, object, dict[str, float]]:
    if not MODEL_PATH.exists():
        _, metrics = train_classifier()
    else:
        metrics = evaluate_saved_classifier()

    vectorizer, classifier = joblib.load(MODEL_PATH)
    return vectorizer, classifier, metrics


def evaluate_saved_classifier() -> dict[str, float]:
    vectorizer, classifier = joblib.load(MODEL_PATH)
    texts = [text for text, _ in SAMPLE_TICKETS]
    labels = [label for _, label in SAMPLE_TICKETS]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    predictions = classifier.predict(vectorizer.transform(test_texts))
    return {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
    }
