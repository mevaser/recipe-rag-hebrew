from __future__ import annotations


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}_chunk_{chunk_index:03d}"


def word_count(text: str) -> int:
    return len(text.split())


def make_indexed_text(text: str, metadata: dict) -> str:
    source = metadata.get("source", "")
    category = metadata.get("category", "")
    return f"שם מקור: {source}\nקטגוריה: {category}\nתוכן:\n{text}"


def make_chunk_metadata(
    document_metadata: dict,
    chunk_strategy: str,
    chunk_index: int,
    chunk_size_words: int,
    overlap_words: int | None,
) -> dict:
    metadata = dict(document_metadata)
    metadata.update(
        {
            "chunk_strategy": chunk_strategy,
            "chunk_index": chunk_index,
            "chunk_size_words": chunk_size_words,
            "overlap_words": overlap_words,
        }
    )
    return metadata


def chunk_full_document(documents: list[dict]) -> list[dict]:
    chunks: list[dict] = []

    for document in documents:
        text = document.get("text", "").strip()
        if not text:
            continue

        doc_id = document["doc_id"]
        chunks.append(
            {
                "chunk_id": make_chunk_id(doc_id, 0),
                "doc_id": doc_id,
                "text": text,
                "indexed_text": make_indexed_text(text, document.get("metadata", {})),
                "metadata": make_chunk_metadata(
                    document_metadata=document.get("metadata", {}),
                    chunk_strategy="full_document",
                    chunk_index=0,
                    chunk_size_words=word_count(text),
                    overlap_words=None,
                ),
            }
        )

    return chunks


def chunk_fixed_size(
    documents: list[dict],
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[dict] = []
    step = chunk_size - overlap

    for document in documents:
        text = document.get("text", "").strip()
        if not text:
            continue

        words = text.split()
        if not words:
            continue

        doc_id = document["doc_id"]
        chunk_index = 0

        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                continue

            chunk_text = " ".join(chunk_words).strip()
            if not chunk_text:
                continue

            chunks.append(
                {
                    "chunk_id": make_chunk_id(doc_id, chunk_index),
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "indexed_text": make_indexed_text(chunk_text, document.get("metadata", {})),
                    "metadata": make_chunk_metadata(
                        document_metadata=document.get("metadata", {}),
                        chunk_strategy="fixed_size",
                        chunk_index=chunk_index,
                        chunk_size_words=len(chunk_words),
                        overlap_words=overlap,
                    ),
                }
            )
            chunk_index += 1

            if start + chunk_size >= len(words):
                break

    return chunks
