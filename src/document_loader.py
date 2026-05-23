from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from utils import clean_text


SUPPORTED_EXTENSIONS = {".docx", ".pdf"}


@dataclass
class LoadStats:
    docx_documents: int = 0
    pdf_page_documents: int = 0
    skipped_files: int = 0
    failed_files: int = 0

    @property
    def total_documents(self) -> int:
        return self.docx_documents + self.pdf_page_documents


@dataclass
class LoadResult:
    documents: list[dict]
    stats: LoadStats


def is_word_temp_file(path: Path) -> bool:
    return path.name.startswith("~$") and path.suffix.lower() == ".docx"


def make_relative_path(path: Path, raw_dir: Path) -> str:
    return path.relative_to(raw_dir).as_posix()


def make_metadata(path: Path, raw_dir: Path, page: int | None) -> dict:
    relative_path = make_relative_path(path, raw_dir)
    return {
        "source": path.name,
        "file_type": path.suffix.lower().lstrip("."),
        "relative_path": relative_path,
        "category": path.parent.name,
        "page": page,
        "language": "he",
    }


def make_doc_id(path: Path, raw_dir: Path, page: int | None = None) -> str:
    relative_path = make_relative_path(path, raw_dir)
    file_type = path.suffix.lower().lstrip(".")
    if page is None:
        return f"{file_type}:{relative_path}"
    return f"{file_type}:{relative_path}:page-{page}"


def iter_corpus_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        logging.warning("Raw data directory does not exist: %s", raw_dir)
        return []

    files: list[Path] = []
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        files.append(path)

    return sorted(files, key=lambda item: item.relative_to(raw_dir).as_posix().lower())


def iter_loadable_files(raw_dir: Path) -> list[Path]:
    return [
        path
        for path in iter_corpus_files(raw_dir)
        if path.suffix.lower() in SUPPORTED_EXTENSIONS and not is_word_temp_file(path)
    ]


def load_docx(path: Path, raw_dir: Path) -> dict:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("python-docx is required to load DOCX files.") from exc

    try:
        document = Document(path)
        paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        text = clean_text("\n".join(paragraphs))
    except Exception as exc:
        logging.warning("Failed to load DOCX file %s: %s", path, exc)
        raise

    if not text:
        logging.warning("Skipping DOCX file with no extracted text: %s", path)
        raise ValueError("DOCX file has no extracted text.")

    return {
        "doc_id": make_doc_id(path, raw_dir),
        "text": text,
        "metadata": make_metadata(path, raw_dir, page=None),
    }


def load_pdf(path: Path, raw_dir: Path) -> list[dict]:
    try:
        return load_pdf_with_pymupdf(path, raw_dir)
    except ImportError:
        logging.info("PyMuPDF is not available; falling back to pypdf.")
        return load_pdf_with_pypdf(path, raw_dir)


def load_pdf_with_pymupdf(path: Path, raw_dir: Path) -> list[dict]:
    try:
        import fitz
    except ImportError:
        raise

    documents: list[dict] = []
    try:
        with fitz.open(path) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                text = clean_text(page.get_text("text") or "")
                if not text:
                    logging.debug("Skipping PDF page with no extracted text: %s page %s", path, page_index)
                    continue
                documents.append(
                    {
                        "doc_id": make_doc_id(path, raw_dir, page=page_index),
                        "text": text,
                        "metadata": make_metadata(path, raw_dir, page=page_index),
                    }
                )
    except Exception as exc:
        logging.warning("Failed to load PDF file with PyMuPDF %s: %s", path, exc)
        raise

    return documents


def load_pdf_with_pypdf(path: Path, raw_dir: Path) -> list[dict]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PyMuPDF or pypdf is required to load PDF files.") from exc

    documents: list[dict] = []
    try:
        reader = PdfReader(str(path))
        for page_index, page in enumerate(reader.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if not text:
                logging.debug("Skipping PDF page with no extracted text: %s page %s", path, page_index)
                continue
            documents.append(
                {
                    "doc_id": make_doc_id(path, raw_dir, page=page_index),
                    "text": text,
                    "metadata": make_metadata(path, raw_dir, page=page_index),
                }
            )
    except Exception as exc:
        logging.warning("Failed to load PDF file with pypdf %s: %s", path, exc)
        raise

    return documents


def load_documents(raw_dir: Path) -> list[dict]:
    return load_documents_with_stats(raw_dir).documents


def load_documents_with_stats(raw_dir: Path) -> LoadResult:
    documents: list[dict] = []
    stats = LoadStats()
    raw_dir = raw_dir.resolve()

    for path in iter_corpus_files(raw_dir):
        suffix = path.suffix.lower()
        if is_word_temp_file(path):
            stats.skipped_files += 1
            logging.debug("Skipping temporary Word file: %s", path)
            continue
        if suffix not in SUPPORTED_EXTENSIONS:
            stats.skipped_files += 1
            logging.debug("Ignoring unsupported file: %s", path)
            continue

        if suffix == ".docx":
            try:
                document = load_docx(path, raw_dir)
            except Exception:
                stats.failed_files += 1
                continue
            documents.append(document)
            stats.docx_documents += 1
        elif suffix == ".pdf":
            try:
                page_documents = load_pdf(path, raw_dir)
            except Exception:
                stats.failed_files += 1
                continue
            if not page_documents:
                stats.skipped_files += 1
                logging.debug("Skipping PDF file with no extracted text pages: %s", path)
                continue
            documents.extend(page_documents)
            stats.pdf_page_documents += len(page_documents)

    return LoadResult(documents=documents, stats=stats)
