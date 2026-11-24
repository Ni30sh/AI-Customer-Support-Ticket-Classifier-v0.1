# AI Customer Support Ticket Classifier

A complete Python example for classifying customer-support tickets and generating
draft responses with a small retrieval-augmented generation (RAG) workflow.

The project combines:

- a 30-message customer-support dataset;
- a scikit-learn TF-IDF and Logistic Regression classifier;
- a ChromaDB collection of FAQ documents;
- a FastAPI `POST /tickets` endpoint;
- five automated API, model, and retrieval tests.

## Categories

Each ticket is classified into exactly one category:

- `billing`
- `technical`
- `account`

If no relevant FAQ document is retrieved, the API marks the request for human
review instead of inventing an answer.

## Project structure

```text
.
├── app/
│   ├── __init__.py
│   ├── classifier.py    # TF-IDF and Logistic Regression training/evaluation
│   ├── dataset.py       # In-memory copy of the 30 sample tickets
│   ├── main.py          # FastAPI application and POST /tickets
│   └── rag.py           # ChromaDB indexing and FAQ retrieval
├── data/
│   └── tickets.csv      # 30 labeled training/evaluation messages
├── faq/
│   └── *.md             # 10 FAQ source documents
├── tests/
│   └── test_api.py      # Five required tests
├── models/
│   └── ticket_classifier.joblib
├── requirements.txt
└── README.md
```

The model and ChromaDB data are generated locally. The `models/` and `.chroma/`
directories should not be treated as source data.

## Dataset

`data/tickets.csv` contains 30 messages, with 10 messages for each category.
The same sample set is available as `SAMPLE_TICKETS` in `app/dataset.py`.

The sample intentionally includes the common words and phrases found in ticket
workflows, such as duplicate charges, application crashes, server errors,
password resets, profile changes, and account lockouts.

## Machine-learning classifier

The training pipeline is:

1. read the 30 labeled messages;
2. split the data into an 80/20 stratified train/test set;
3. transform text with unigram and bigram TF-IDF features;
4. train a Logistic Regression model;
5. save the vectorizer and classifier with `joblib`;
6. calculate accuracy and macro F1-score on the test set.

Training is automatic when the saved model does not exist. To retrain explicitly,
run:

```bash
python -c "from app.classifier import train_classifier; print(train_classifier()[1])"
```

### Evaluation metrics

The deterministic 80/20 split uses `random_state=42` and stratified sampling.
The current test-split metrics are:

| Metric | Score |
|---|---:|
| Accuracy | 0.5000 |
| Macro F1 | 0.5222 |

These values demonstrate behavior on this small demonstration dataset. They are
not an estimate of production accuracy, and larger real-world validation sets
would be required before deployment.

## RAG with ChromaDB

The `faq/` directory contains 10 Markdown documents. `FAQRetriever` starts a
persistent ChromaDB client at `.chroma/`, creates the `support_faq` collection,
and embeds each document with a deterministic local hash vector.

The retriever accepts a question and asks ChromaDB for the top three results.
The cosine distance is limited to `0.75`; if no result passes this threshold,
the question is unsupported and ChromaDB sources are withheld.

The demo intentionally uses a local embedding implementation so it has no
external API key dependency. For production use, replace
`stable_embedding()` and ChromaDB query embeddings with a managed or
domain-tuned embedding model.

## FastAPI endpoint

### Start the API

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open the interactive API documentation at:

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

### `POST /tickets`

Request body:

```json
{
  "message": "I cannot connect to the server"
}
```

`message` is required and must contain 3 to 2,000 characters.

Success response:

```json
{
  "category": "technical",
  "confidence": 0.89,
  "draft_response": "Please follow these troubleshooting steps and provide the error details if the issue continues.",
  "sources": [
    "technical_crash.md",
    "technical_api.md",
    "technical_sync.md"
  ],
  "needs_human_review": false
}
```

The response fields are:

- `category`: the predicted billing, technical, or account category;
- `confidence`: the highest Logistic Regression class probability;
- `draft_response`: a category-based draft answer;
- `sources`: up to three relevant FAQ Markdown filenames;
- `needs_human_review`: `true` when no relevant FAQ is found.

Empty or too-short messages receive HTTP 422 validation responses from Pydantic.

Example request using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/tickets" \
  -H "Content-Type: application/json" \
  -d '{"message":"I was charged twice and need a refund"}'
```

## Tests

The test suite has six cases:

1. the root page returns HTTP 200 and endpoint links;
2. a valid ticket returns HTTP 200 and the expected response shape;
3. an empty message receives HTTP 422 validation;
4. the model directly predicts a technical ticket;
5. FAQ retrieval returns three Markdown source names;
6. an unsupported question returns `needs_human_review=true` and no sources.

Run all tests:

```bash
pytest -q
```

## Implementation notes

- The API is synchronous and is suitable for local demonstration or a small
  development deployment.
- ChromaDB is persisted between process starts under `.chroma/`.
- The model is automatically retrained and saved if `models/ticket_classifier.joblib`
  is missing.
- In production, consider authentication, rate limiting, request logging,
  model monitoring, stronger safety rules, and a real hosted embedding model.

## Current limitations

- The fixed 30-message dataset is intentionally small.
- The hash embedding is not semantically trained.
- Draft responses are templates, not text generated by an LLM.
- Retrieval similarity is demonstrated, but it should be evaluated on a larger
  FAQ and query set.
