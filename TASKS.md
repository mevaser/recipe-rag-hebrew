# TASKS.md

## Current Phase

Foundation phase.

The goal is to create a clean project skeleton and implement corpus scanning only.

Do not move to full indexing, retrieval, or answer generation before the scanner works.

---

## Task 1 — Project Skeleton

Create folders:

```text
data/raw
data/processed
src
eval
report
```

Create files:

```text
README.md
requirements.txt
.gitignore
PROJECT_CONTEXT.md
PLAN.md
AGENTS.md
TASKS.md
data/MANIFEST.md
src/scan_corpus.py
src/document_loader.py
src/chunking.py
src/embeddings.py
src/build_index.py
src/retrieval.py
src/generation.py
src/rag_system.py
src/utils.py
eval/gold_set.jsonl
eval/run_eval.py
report/report.md
```

Done when:

- The folder structure exists.
- All required files exist.
- The project can be opened and edited normally in VS Code.

---

## Task 2 — Corpus Scanner

Implement:

```bash
python src/scan_corpus.py
```

Expected behavior:

- Scan `data/raw` recursively.
- Count supported files:
  - `.docx`
  - `.pdf`

- Ignore unsupported files for now:
  - `.doc`
  - videos
  - images
  - temporary Word files such as `~$file.docx`

- Print a clear summary to the terminal.
- Export inventory CSV to:

```text
data/processed/recipes_inventory.csv
```

CSV columns:

- file_name
- extension
- relative_path
- parent_folder
- file_size_bytes
- last_modified

Done when:

- Running `python src/scan_corpus.py` does not crash.
- The script prints counts by extension.
- The script prints total supported documents.
- `data/processed/recipes_inventory.csv` is created.

---

## Task 3 — Basic Utility Functions

Create basic utility functions in `src/utils.py`.

Required functions:

```python
from pathlib import Path


def ensure_dir(path: Path) -> None:
    ...


def setup_logging() -> None:
    ...


def clean_text(text: str) -> str:
    ...
```

`clean_text` should:

- Normalize whitespace.
- Remove invisible RTL/LTR marks.
- Preserve Hebrew text.
- Avoid aggressive preprocessing.

Done when:

- `scan_corpus.py` can use `ensure_dir` and logging if needed.
- `clean_text` exists but does not need advanced logic yet.

---

## Task 4 — Skeleton Function Signatures

Create function signatures only. Do not fully implement advanced RAG logic yet.

In `src/document_loader.py`:

```python
from pathlib import Path


def load_docx(path: Path, raw_dir: Path) -> dict:
    ...


def load_pdf(path: Path, raw_dir: Path) -> list[dict]:
    ...


def load_documents(raw_dir: Path) -> list[dict]:
    ...
```

In `src/chunking.py`:

```python
def chunk_full_document(documents: list[dict]) -> list[dict]:
    ...


def chunk_fixed_size(
    documents: list[dict],
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[dict]:
    ...
```

In `src/retrieval.py`:

```python
def retrieve(query: str, k: int = 5) -> list[dict]:
    ...
```

In `src/generation.py`:

```python
def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    ...


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    ...
```

In `src/rag_system.py`:

```python
def answer(question: str) -> dict:
    ...
```

Done when:

- All files import without syntax errors.
- Placeholder functions return safe placeholder values or raise clear `NotImplementedError` messages.

---

## Task 5 — Minimal Placeholder Evaluation File

Create `eval/gold_set.jsonl` with 5 placeholder Hebrew questions.

Example format:

```json
{
  "question": "איזה מתכון כולל עוף ותפוחי אדמה?",
  "reference_answer": "Placeholder answer.",
  "must_cite_chunk_ids": ["placeholder_chunk_id"],
  "category": "factual"
}
```

Use these categories across the placeholder examples:

- factual
- numerical
- negation
- comparison
- temporal

Done when:

- `eval/gold_set.jsonl` exists.
- The file is valid JSONL.
- Each line contains exactly one JSON object.

---

## Task 6 — Minimal README

Create a basic `README.md` with:

- Project name
- Short description
- Setup instructions
- How to place documents in `data/raw`
- How to run the scanner

Required commands:

```bash
pip install -r requirements.txt
python src/scan_corpus.py
```

Done when:

- A new developer can understand how to start.
- The README does not describe features that are not implemented yet as already working.

---

## Task 7 — Requirements File

Create `requirements.txt` for the foundation phase.

Include only packages needed soon:

```text
python-docx
pypdf
pymupdf
sentence-transformers
faiss-cpu
numpy
pandas
python-dotenv
tqdm
```

Do not add LangChain or LlamaIndex.

Done when:

- `pip install -r requirements.txt` can install the needed dependencies.

---

## Completion Criteria for Current Phase

This phase is complete when:

```bash
python src/scan_corpus.py
```

runs successfully and creates:

```text
data/processed/recipes_inventory.csv
```

After that, move to document loading and text extraction in the next phase.
