# Project Plan — Hebrew Recipe RAG

## Final Deliverable

A complete custom RAG pipeline over Hebrew recipe documents, including:

1. Corpus scanning
2. Document loading
3. Text cleaning
4. Chunking
5. Embedding
6. FAISS indexing
7. Retrieval
8. Answer generation
9. Source citation
10. Gold set evaluation
11. Ablation experiments
12. Final report

---

## Milestone 0 — Project Skeleton

Goal: Create the initial repository structure.

Deliverables:

- Folder structure
- README.md
- requirements.txt
- .gitignore
- PROJECT_CONTEXT.md
- PLAN.md
- AGENTS.md
- TASKS.md
- Python skeleton files

Done when:

- The project imports successfully.
- `python src/scan_corpus.py` runs without crashing.

---

## Milestone 1 — Corpus Inventory

Goal: Scan the local recipe corpus and create an inventory.

Tasks:

- Recursively scan `data/raw`
- Count supported files:
  - `.docx`
  - `.pdf`

- Ignore:
  - `.doc`
  - videos
  - images

- Export inventory to:

```text
data/processed/recipes_inventory.csv
```

Inventory columns:

- file_name
- extension
- relative_path
- parent_folder
- file_size_bytes
- last_modified

Done when:

- Counts are printed.
- CSV inventory is created.

---

## Milestone 2 — Document Loading

Goal: Load DOCX and PDF files into a unified internal format.

Document format:

```python
{
    "doc_id": str,
    "text": str,
    "metadata": dict
}
```

Tasks:

- Implement DOCX loader using `python-docx`
- Implement PDF loader using `pypdf` or `PyMuPDF`
- Preserve Hebrew text
- Include metadata:
  - source
  - file_type
  - relative_path
  - category
  - page number for PDFs
  - language

Done when:

- At least 10 DOCX files load successfully.
- At least 10 PDF files load successfully.
- Empty or unreadable files are logged and skipped safely.

---

## Milestone 3 — Text Cleaning

Goal: Normalize text without damaging Hebrew content.

Cleaning should:

- Remove invisible RTL/LTR marks
- Normalize whitespace
- Remove repeated blank lines
- Keep Hebrew letters intact
- Avoid aggressive preprocessing

Done when:

- Cleaned text is readable.
- Hebrew text is not corrupted.

---

## Milestone 4 — Chunking

Goal: Implement and compare two chunking strategies.

Strategy A:

- Full document / full recipe chunking

Strategy B:

- Fixed-size word chunking with overlap

Chunk format:

```python
{
    "chunk_id": str,
    "doc_id": str,
    "text": str,
    "metadata": dict
}
```

Done when:

- Chunks are created and saved to JSON.
- Each chunk has stable IDs.
- Each chunk preserves source metadata.

---

## Milestone 5 — Embeddings and FAISS Index

Goal: Convert chunks to embeddings and build a vector index.

Embedding model:

```text
intfloat/multilingual-e5-small
```

Vector index:

```text
FAISS
```

Output files:

```text
data/processed/chunks.json
data/processed/index.faiss
```

Done when:

- `python src/build_index.py` builds the index from scratch.
- The index can be deleted and rebuilt reproducibly.

---

## Milestone 6 — Retrieval

Goal: Retrieve the top-k most relevant chunks for a Hebrew question.

Required function:

```python
def retrieve(query: str, k: int = 5) -> list[dict]:
    ...
```

Each result must include:

```python
{
    "chunk_id": str,
    "text": str,
    "score": float,
    "metadata": dict
}
```

Done when:

- A Hebrew query returns relevant chunks.
- Scores are included.
- Source metadata is included.

---

## Milestone 7 — Answer Generation

Goal: Generate an answer using only retrieved context.

Rules:

1. Answer only from retrieved context.
2. If the answer is not found, say so.
3. Cite source chunks.
4. Avoid unsupported claims.
5. Prefer Hebrew answers for Hebrew questions.

Done when:

- `generate_answer(question, retrieved_chunks)` returns a grounded answer.
- The answer includes cited chunk IDs or source references.

---

## Milestone 8 — RAG Interface

Goal: Implement the required assignment interface.

Required function:

```python
def answer(question: str) -> dict:
    return {
        "answer": str,
        "sources": list[str],
        "retrieved_chunks": list[dict]
    }
```

Done when:

- `answer()` works end-to-end.
- It returns answer, sources, and retrieved chunks.

---

## Milestone 9 — Gold Set

Goal: Create an evaluation set of at least 50 Hebrew questions.

File:

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

Question categories:

- factual
- numerical
- temporal
- negation / absence
- comparison

Done when:

- At least 50 questions exist.
- Each question has at least one expected chunk ID.

---

## Milestone 10 — Evaluation

Goal: Evaluate retrieval and inspect answer quality.

Retrieval metric:

- Hit@5

Answer inspection:

- Manually inspect at least 10 answers
- Classify:
  - Correct
  - Partially correct
  - Incorrect
  - Unsupported / hallucinated

Done when:

- `python eval/run_eval.py` prints Hit@5.
- Manual answer inspection table is prepared.

---

## Milestone 11 — Ablation Study

Goal: Compare at least two system variations.

Recommended experiments:

1. Full recipe chunking vs fixed-size chunks
2. top-k = 3 vs top-k = 8
3. chunk size 300 vs chunk size 700
4. overlap 0 vs overlap 100

Table format:

| Experiment | Hit@5 | Answer Accuracy | Notes |
| ---------- | ----: | --------------: | ----- |

Done when:

- At least two ablation experiments are reported.

---

## Milestone 12 — Final Report

Goal: Write a short final report.

Report sections:

1. Corpus description
2. System architecture
3. Chunking strategy
4. Embedding and vector index choice
5. Retrieval method
6. Prompt design
7. Evaluation results
8. Ablation table
9. Failure analysis
10. Future improvements

Done when:

- Report is complete.
- README has exact run instructions.
