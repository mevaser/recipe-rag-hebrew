from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:
        logging.info("Could not load embedding model from local cache; trying remote download: %s", exc)
        return SentenceTransformer(model_name)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    prefix: str = PASSAGE_PREFIX,
    show_progress_bar: bool = True,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    prefixed_texts = [f"{prefix}{text}" for text in texts]
    embeddings = model.encode(
        prefixed_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
    )
    return normalize_embeddings(np.asarray(embeddings, dtype=np.float32))


def embed_passages(
    texts: list[str],
    model: SentenceTransformer | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    embedding_model = model or load_embedding_model()
    return embed_texts(texts, model=embedding_model, batch_size=batch_size, prefix=PASSAGE_PREFIX)


def embed_query(query: str, model: SentenceTransformer | None = None) -> np.ndarray:
    embedding_model = model or load_embedding_model()
    return embed_texts(
        [query],
        model=embedding_model,
        batch_size=1,
        prefix=QUERY_PREFIX,
        show_progress_bar=False,
    )[0]
