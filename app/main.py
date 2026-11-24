from pathlib import Path

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from app.classifier import load_classifier
from app.rag import FAQRetriever


class TicketRequest(BaseModel):
    message: str = Field(min_length=3, max_length=2000)


vectorizer, classifier, METRICS = load_classifier()
retriever = FAQRetriever()

app = FastAPI(title="AI Customer Support Ticket Classifier")


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "AI Customer Support Ticket Classifier",
        "tickets_endpoint": "/tickets",
        "documentation": "/docs",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


def build_draft(category: str, sources: list[str]) -> str:
    if not sources:
        return "I am unable to find a relevant answer. A support specialist should review this ticket."
    if category == "billing":
        return "Please review the billing FAQ below and confirm the charge, invoice, or refund details."
    if category == "technical":
        return "Please follow these troubleshooting steps and provide the error details if the issue continues."
    return "Please follow the account security steps below. Do not share passwords or verification codes."


@app.post("/tickets")
def create_ticket(request: TicketRequest) -> dict:
    vector = vectorizer.transform([request.message])
    category = str(classifier.predict(vector)[0])
    confidence = float(max(classifier.predict_proba(vector)[0]))
    sources = retriever.retrieve(request.message)
    needs_human_review = not sources
    return {
        "category": category,
        "confidence": round(confidence, 4),
        "draft_response": build_draft(category, sources),
        "sources": sources,
        "needs_human_review": needs_human_review,
    }
