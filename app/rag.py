import hashlib
import math
from pathlib import Path
from typing import Iterable

import chromadb


FAQ_DIR = Path(__file__).resolve().parent.parent / "faq"
CHROMA_DIR = Path(__file__).resolve().parent.parent / ".chroma"
EMBEDDING_DIMENSIONS = 128


def stable_embedding(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    tokens = "".join(char.lower() if char.isalnum() else " " for char in text).split()
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % EMBEDDING_DIMENSIONS
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class FAQRetriever:
    def __init__(self) -> None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_or_create_collection(
            name="support_faq",
            metadata={"hnsw:space": "cosine"},
        )
        self._index_documents()

    def _index_documents(self) -> None:
        documents = sorted(FAQ_DIR.glob("*.md"))
        if not documents:
            return
        ids = [path.stem for path in documents]
        texts = [path.read_text(encoding="utf-8") for path in documents]
        embeddings = [stable_embedding(text) for text in texts]
        existing = set(self.collection.get(include=[])["ids"])
        missing = [index for index, faq_id in enumerate(ids) if faq_id not in existing]
        if missing:
            self.collection.add(
                ids=[ids[index] for index in missing],
                embeddings=[embeddings[index] for index in missing],
                documents=[texts[index] for index in missing],
                metadatas=[{"source": documents[index].name} for index in missing],
            )

    def retrieve(self, question: str, limit: int = 3) -> list[str]:
        result = self.collection.query(
            query_embeddings=[stable_embedding(question)],
            n_results=min(limit, self.collection.count()),
            include=["metadatas", "documents"],
        )
        metadatas = result["metadatas"][0]
        documents = result["documents"][0]
        stopwords = {
            "a", "an", "and", "do", "does", "for", "how", "i", "in", "is",
            "my", "of", "the", "to", "with", "what",
        }
        question_words = {
            word
            for word in "".join(
                char.lower() if char.isalnum() else " "
                for char in question
            ).split()
            if word not in stopwords
        }

        ranked: list[tuple[int, str]] = []
        for metadata, document in zip(metadatas, documents):
            document_words = set(
                "".join(
                    char.lower() if char.isalnum() else " "
                    for char in document
                ).split()
            )
            overlap = len(question_words & document_words)
            ranked.append((overlap, metadata["source"]))

        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked or ranked[0][0] == 0:
            return []
        return [source for _, source in ranked[:limit]]
