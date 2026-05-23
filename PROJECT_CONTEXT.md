# Project Context — Hebrew Recipe RAG Assistant

## Project Goal

This project is a mid-course assignment for building a custom Retrieval-Augmented Generation (RAG) pipeline over a private Hebrew recipe corpus.

The final system should answer questions about recipes using only retrieved context from the local recipe collection, and it must cite the source chunks used in the answer.

The corpus contains personal Hebrew recipe documents stored locally by the user.

## Current System Status

The project has moved well beyond the initial course skeleton:

- dense retrieval with `intfloat/multilingual-e5-small`
- BM25 retrieval
- weighted dense + BM25 hybrid retrieval
- metadata-aware indexing through `indexed_text`
- grounded Hebrew answer generation with Ollama
- strict retrieval evaluation and manual answer evaluation

Current best retrieval configuration:

```bash
python eval/run_eval.py --mode hybrid --candidate-k 50 --rrf-k 30 --dense-weight 0.5 --bm25-weight 2.0
```

Current best retrieval metrics:

- Hit@1: 78%
- Hit@3: 90%
- Hit@5: 92%
- Hit@10: 94%
- Hit@20: 94%
- MRR: 0.8412

Current verified manual answer evaluation on 10 selected questions:

- correct: 8
- partial: 2
- incorrect: 0

---

## Corpus

- Domain: Cooking recipes
- Language: Hebrew
- File types:
  - DOCX: 247 files
  - PDF: 99 files
  - DOC: 12 files

---

## First Version Scope

Supported in version 1:

- `.docx`
- `.pdf`

Excluded from version 1:

- `.doc`
- video files
- images
- OCR-only documents

These unsupported formats may be added later as future improvements.

---

## Assignment Requirement

The project must expose the following function:

```python
def answer(question: str) -> dict:
    return {
        "answer": str,
        "sources": list[str],
        "retrieved_chunks": list[dict]
    }
```

---

## Required Document Format

```python
{
    "doc_id": str,
    "text": str,
    "metadata": dict
}
```

---

## Required Chunk Format

```python
{
    "chunk_id": str,
    "doc_id": str,
    "text": str,
    "metadata": dict
}
```

---

## Required Retrieved Result Format

```python
{
    "chunk_id": str,
    "text": str,
    "score": float,
    "metadata": dict
}
```

---

## Metadata Fields

Metadata should include where available:

- source filename
- file type
- relative path
- parent folder / category
- page number for PDFs
- language: `"he"`

---

## Example Questions

The system should support questions such as:

- Which recipe contains chicken and potatoes?
- Which recipes do not contain dairy?
- How long should the cake be baked?
- Which recipe is faster: shakshuka or vegetable pie?
- What can I cook with rice and mushrooms?
- Which recipes include flour but not yeast?

---

## Design Principles

- Keep the system modular.
- Avoid black-box RAG frameworks in the first version.
- Implement the RAG components manually.
- Keep code readable and beginner-friendly.
- Keep all code comments in English.
- Preserve Hebrew text during cleaning and preprocessing.

```

```
