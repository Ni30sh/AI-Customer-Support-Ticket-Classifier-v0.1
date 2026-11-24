from fastapi.testclient import TestClient

from app.classifier import load_classifier
from app.main import app
from app.rag import FAQRetriever


client = TestClient(app)


def test_root_page():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["tickets_endpoint"] == "/tickets"


def test_valid_ticket_input():
    response = client.post("/tickets", json={"message": "I was charged twice"})
    assert response.status_code == 200
    assert set(response.json()) == {
        "category",
        "confidence",
        "draft_response",
        "sources",
        "needs_human_review",
    }


def test_invalid_ticket_input():
    response = client.post("/tickets", json={"message": ""})
    assert response.status_code == 422


def test_classification():
    vectorizer, classifier, _ = load_classifier()
    prediction = classifier.predict(vectorizer.transform(["The app crashes on upload"]))
    assert prediction[0] == "technical"


def test_retrieval_returns_three_documents():
    sources = FAQRetriever().retrieve("How do I reset my password")
    assert len(sources) == 3
    assert all(source.endswith(".md") for source in sources)


def test_unsupported_question_needs_human_review():
    response = client.post(
        "/tickets",
        json={"message": "What is the weather in Mars tomorrow"},
    )
    assert response.status_code == 200
    assert response.json()["needs_human_review"] is True
    assert response.json()["sources"] == []
