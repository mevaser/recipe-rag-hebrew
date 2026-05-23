# AGENTS.md — Development Instructions for Codex

## Project

Hebrew Recipe RAG Assistant.

This is a custom Retrieval-Augmented Generation (RAG) pipeline for a mid-course assignment.
The system works over a local Hebrew recipe corpus and must answer questions using retrieved context with citations.

---

## Main Rule

Do not turn this project into a black-box LangChain or LlamaIndex wrapper.

Implement the main RAG components manually:

- document loading
- cleaning
- chunking
- embeddings
- indexing
- retrieval
- generation
- evaluation

Frameworks may be used only for low-level utilities, not as the whole RAG pipeline.

---

## Coding Standards

- Use Python.
- Use clear, modular files.
- Use type hints.
- Use pathlib for paths.
- Use logging instead of many print statements, except in CLI scripts.
- Use dataclasses or pydantic only where useful.
- All code comments must be in English.
- Keep functions small and readable.
- Avoid hardcoded absolute paths.
- Avoid Windows-only assumptions.
- Use UTF-8 everywhere.

---

## Hebrew Text Rules

The corpus is in Hebrew.

Text cleaning must preserve Hebrew content.

Do not:

- remove Hebrew letters
- transliterate Hebrew
- apply English-only stemming or lemmatization
- remove useful punctuation from ingredients or instructions
- aggressively alter Hebrew text

Allowed cleaning:

- normalize whitespace
- remove invisible RTL/LTR marks
- remove repeated blank lines
- strip empty text

---

## Supported Files

First version supports:

- `.docx`
- `.pdf`

First version ignores:

- `.doc`
- videos
- images
- OCR-only documents

If unsupported files are found, log them and skip safely.

---

## Default Paths

Use these default paths:

```text
data/raw
data/processed
data/processed/recipes_inventory.csv
data/processed/chunks.json
data/processed/index.faiss
eval/gold_set.jsonl
```

Make paths configurable through constants or CLI arguments when useful.

---

## Required Public Functions

The project must expose:

```python
def answer(question: str) -> dict:
    return {
        "answer": str,
        "sources": list[str],
        "retrieved_chunks": list[dict]
    }
```

Retriever function:

```python
def retrieve(query: str, k: int = 5) -> list[dict]:
    ...
```

---

## Required Data Formats

Document:

```python
{
    "doc_id": str,
    "text": str,
    "metadata": dict
}
```

Chunk:

```python
{
    "chunk_id": str,
    "doc_id": str,
    "text": str,
    "metadata": dict
}
```

Retrieved result:

```python
{
    "chunk_id": str,
    "text": str,
    "score": float,
    "metadata": dict
}
```

Metadata should include where available:

```python
{
    "source": str,
    "file_type": str,
    "relative_path": str,
    "category": str,
    "page": int | None,
    "language": "he"
}
```

---

## Embedding Model

Default embedding model:

```text
intfloat/multilingual-e5-small
```

Use `sentence-transformers`.

E5 models usually work best with prefixes:

```text
query: <question>
passage: <chunk text>
```

Apply this consistently if using E5.

---

## Vector Index

Use FAISS.

Save:

```text
data/processed/index.faiss
data/processed/chunks.json
```

The index must be reproducible.

If the index is deleted, running:

```bash
python src/build_index.py
```

should rebuild it.

---

## Generation

Generation should use a strict prompt.

The model must:

1. Answer only from the retrieved context.
2. Say the information was not found if context is insufficient.
3. Cite source chunks.
4. Avoid unsupported claims.
5. Prefer Hebrew answers for Hebrew questions.

Do not require an API key for the basic pipeline to run.

If no LLM API key exists, return a placeholder answer that still includes sources.

---

## Evaluation

Evaluation file:

```text
eval/gold_set.jsonl
```

Each line:

```json
{
  "question": "...",
  "reference_answer": "...",
  "must_cite_chunk_ids": ["..."],
  "category": "factual"
}
```

Implement:

```bash
python eval/run_eval.py
```

It should compute simple Hit@5.

A question is a hit if at least one expected chunk ID appears in the retrieved top 5 chunks.

---

## Development Workflow

Work in small steps.

For each task:

1. Explain briefly what changed.
2. Modify only relevant files.
3. Keep the code runnable.
4. Add simple checks where useful.
5. Do not implement advanced features before the foundation works.

---

## Do Not Do Yet

Do not implement these in the first phase:

- OCR
- video transcription
- `.doc` conversion
- advanced reranking
- hybrid retrieval
- UI
- database
- Docker
- cloud deployment

These can be future improvements.

---

## First Task

Create the project skeleton and implement corpus scanning.

The command:

```bash
python src/scan_corpus.py
```

must:

1. Recursively scan `data/raw`
2. Count `.docx` and `.pdf`
3. Print counts by extension
4. Save:

```text
data/processed/recipes_inventory.csv
```
