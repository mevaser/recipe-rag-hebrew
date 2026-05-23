from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

from utils import ensure_dir, setup_logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "indexed_chunks.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval" / "gold_candidates.jsonl"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"
ALLOWED_CATEGORIES = {"factual", "ingredients", "numerical", "process", "comparison", "negation"}
FORBIDDEN_QUESTION_PHRASES = (
    "מה הגודל",
    "מה המרכיב העיקרי",
    "recept",
)
FORBIDDEN_TEXT_FRAGMENTS = (
    "recept",
    "kováh",
    "shahar",
    "uxtap",
    "כ .או",
    "כהרף",
    "או6",
    "קובה או6",
    "ללבש",
    "לא מוזכר",
    "לא מוזכרת",
    "מה שמכיל",
    "מה שהופך",
    "מתרככות",
    "נושאים",
    "האופיה",
    "שסחוט",
    "לכנת",
)
VAGUE_QUESTION_STARTS = (
    "מה הגודל",
    "מה עושה ",
    "מה שהופך",
)
HEBREW_STOPWORDS = {
    "של",
    "את",
    "עם",
    "על",
    "או",
    "אם",
    "יש",
    "אין",
    "מה",
    "כמה",
    "איך",
    "איזה",
    "אילו",
    "באיזה",
    "צריך",
    "צריכה",
    "צריכים",
    "במתכון",
    "מתכון",
    "המתכון",
    "כולל",
    "כוללת",
    "להכין",
    "מכינים",
}
LATIN_LETTER_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
HEBREW_TOKEN_PATTERN = re.compile(r"[\u0590-\u05ff0-9]+")
UNEXPECTED_SCRIPT_PATTERN = re.compile(r"[^\u0590-\u05ff0-9\s.,;:!?()\[\]{}\"'״׳/%+\-–—]")


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draft candidate gold-set questions from indexed chunks.")
    parser.add_argument("--contains", help="Optional text substring filter.")
    parser.add_argument("--category", help="Optional category substring filter.")
    parser.add_argument("--source", help="Optional source substring filter.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum chunks to use. Defaults to 10.")
    parser.add_argument(
        "--questions-per-chunk",
        type=int,
        default=2,
        help="Number of candidate questions per chunk. Defaults to 2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Candidate JSONL output path. Defaults to eval/gold_candidates.jsonl.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Indexed chunks JSON path. Defaults to data/processed/indexed_chunks.json.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append candidates to the output JSONL file instead of overwriting it.",
    )
    return parser.parse_args()


def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as input_file:
        chunks = json.load(input_file)
    if not isinstance(chunks, list):
        raise ValueError(f"Expected a list of chunks in {path}.")
    return chunks


def contains_filter(value: str | None, needle: str | None) -> bool:
    if not needle:
        return True
    return needle.casefold() in (value or "").casefold()


def chunk_matches(chunk: dict, contains: str | None, category: str | None, source: str | None) -> bool:
    metadata = chunk.get("metadata", {})
    return (
        contains_filter(chunk.get("text"), contains)
        and contains_filter(metadata.get("category"), category)
        and contains_filter(metadata.get("source"), source)
    )


def select_chunks(
    chunks: list[dict],
    contains: str | None,
    category: str | None,
    source: str | None,
    limit: int,
) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit must be greater than 0.")

    selected: list[dict] = []
    for chunk in chunks:
        if chunk_matches(chunk, contains=contains, category=category, source=source):
            selected.append(chunk)
            if len(selected) >= limit:
                break
    return selected


def build_prompt(chunk: dict, questions_per_chunk: int) -> str:
    metadata = chunk.get("metadata", {})
    return f"""You draft candidate retrieval gold-set questions from one Hebrew recipe chunk.

Rules:
- Generate natural Hebrew questions that sound like a real user asking about recipes.
- The answer must be fully supported by the provided chunk text.
- Do not create questions that require outside knowledge.
- Prefer practical recipe questions.
- For numerical questions, use quantities, times, temperatures, or weights only if they clearly appear in the chunk.
- Do not invent ingredients or steps.
- Do not use retrieval output.
- Avoid broken or partial answers.
- Avoid English words unless they appear as ingredient names in the recipe.
- Never use the word "recept".
- Avoid vague questions like "מה הגודל..." or "מה המרכיב העיקרי..." unless the answer is very explicit and complete.
- Prefer practical question patterns such as:
  - "איזה מתכון כולל X?"
  - "כמה X צריך במתכון?"
  - "כמה זמן צריך לבשל/לאפות/להמתין?"
  - "באיזו טמפרטורה אופים?"
  - "מה עושים אחרי שלב X?"
  - "אילו מצרכים יש בבצק/מלית/רוטב?"
- If the chunk does not contain enough clear information to create high-quality questions, return [].
- Return a JSON array only. Do not add markdown, comments, or explanations.
- Return at most {questions_per_chunk} objects.
- Allowed category values: factual, ingredients, numerical, process, comparison, negation.

Example valid output:
[
  {{
    "question": "כמה כוסות בורגול צריך במתכון קובה בורגול מיוחדת?",
    "reference_answer": "במתכון קובה בורגול מיוחדת צריך 2 כוסות בורגול בינוני.",
    "category": "numerical"
  }}
]

Chunk metadata:
chunk_id: {chunk.get("chunk_id", "")}
source: {metadata.get("source", "")}
category: {metadata.get("category", "")}
page: {metadata.get("page")}

Chunk text:
{chunk.get("text", "")}
"""


def call_ollama(prompt: str) -> str:
    load_dotenv()
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    model_name = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("response", "")


def extract_json_text(text: str) -> str | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()

    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    array_start = stripped.find("[")
    array_end = stripped.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        return stripped[array_start : array_end + 1]

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        return stripped[object_start : object_end + 1]

    return None


def parse_ollama_candidates(response_text: str) -> list[dict]:
    json_text = extract_json_text(response_text)
    if json_text is None:
        raise ValueError("No JSON object or array found in Ollama response.")

    parsed = json.loads(json_text)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError("Ollama response JSON must be an object or array of objects.")
    return parsed


def count_words(text: str) -> int:
    return len(text.split())


def english_word_ratio(text: str) -> float:
    words = re.findall(r"\b[\w'-]+\b", text)
    if not words:
        return 0.0
    english_words = [word for word in words if re.search(r"[A-Za-z]", word)]
    return len(english_words) / len(words)


def has_latin_text(text: str) -> bool:
    return bool(LATIN_LETTER_PATTERN.search(text))


def has_unexpected_script(text: str) -> bool:
    return bool(UNEXPECTED_SCRIPT_PATTERN.search(text))


def has_forbidden_fragment(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    return any(fragment.casefold() in normalized for fragment in FORBIDDEN_TEXT_FRAGMENTS)


def has_broken_wording(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    if "\ufffd" in normalized:
        return True
    broken_patterns = (
        r"[\u0590-\u05ff]\s+\.\s*[\u0590-\u05ff]",
        r"\b[\u0590-\u05ff]\s+[.,;:]\s*[\u0590-\u05ff]",
        r"\d+[\u0590-\u05ff]{1,2}\b",
        r"\d+\s+\.\s+\d+",
        r"\b[\u0590-\u05ff]{1,2}\s+\.[\u0590-\u05ff]{1,3}\b",
    )
    return any(re.search(pattern, normalized) for pattern in broken_patterns)


def starts_with_vague_question(question: str) -> bool:
    question_lower = question.casefold().strip()
    return any(question_lower.startswith(phrase.casefold()) for phrase in VAGUE_QUESTION_STARTS)


def meaningful_tokens(text: str) -> set[str]:
    tokens = {token.casefold() for token in HEBREW_TOKEN_PATTERN.findall(text)}
    return {token for token in tokens if len(token) > 1 and token not in HEBREW_STOPWORDS}


def answer_appears_related(question: str, reference_answer: str) -> bool:
    question_tokens = meaningful_tokens(question)
    answer_tokens = meaningful_tokens(reference_answer)
    if not question_tokens or not answer_tokens:
        return False
    if question_tokens & answer_tokens:
        return True
    numerical_question_tokens = {"זמן", "טמפרטורה", "כוסות", "גרם", "דקות", "שעות"}
    if re.search(r"\d", reference_answer) and question_tokens & numerical_question_tokens:
        return True
    return False


def candidate_rejection_reason(candidate: dict) -> str | None:
    question = str(candidate.get("question", "")).strip()
    reference_answer = str(candidate.get("reference_answer", "")).strip()
    category = str(candidate.get("category", "")).strip()
    question_lower = question.casefold()

    if not question:
        return "empty question"
    if not reference_answer:
        return "empty reference_answer"
    if category not in ALLOWED_CATEGORIES:
        return "invalid category"
    if has_forbidden_fragment(question) or has_forbidden_fragment(reference_answer):
        return "candidate contains forbidden or broken fragment"
    if has_latin_text(question) or has_latin_text(reference_answer):
        return "candidate contains non-Hebrew Latin text"
    if has_unexpected_script(question) or has_unexpected_script(reference_answer):
        return "candidate contains unexpected non-Hebrew script"
    if any(phrase in question_lower for phrase in FORBIDDEN_QUESTION_PHRASES):
        return "question is vague or forbidden"
    if starts_with_vague_question(question):
        return "question starts with vague wording"
    if has_broken_wording(question) or has_broken_wording(reference_answer):
        return "candidate has broken wording"
    if english_word_ratio(question) > 0.15:
        return "question contains too much English"
    if english_word_ratio(reference_answer) > 0.20:
        return "reference_answer contains too much English"
    if count_words(reference_answer) < 4:
        return "reference_answer has fewer than 4 words"
    if count_words(question) < 5:
        return "question has fewer than 5 words"
    if not answer_appears_related(question, reference_answer):
        return "reference_answer appears unrelated to question"
    return None


def normalize_candidate(candidate: dict, chunk: dict) -> tuple[dict | None, str | None]:
    rejection_reason = candidate_rejection_reason(candidate)
    if rejection_reason:
        return None, rejection_reason

    category = candidate.get("category", "factual")
    question = str(candidate.get("question", "")).strip()
    reference_answer = str(candidate.get("reference_answer", "")).strip()

    metadata = chunk.get("metadata", {})
    return {
        "question": question,
        "reference_answer": reference_answer,
        "must_cite_chunk_ids": [chunk.get("chunk_id", "")],
        "category": category,
        "source": metadata.get("source", ""),
        "review_status": "needs_review",
    }, None


def draft_candidates_for_chunk(chunk: dict, questions_per_chunk: int) -> tuple[list[dict], int, int]:
    prompt = build_prompt(chunk, questions_per_chunk=questions_per_chunk)
    response_text = call_ollama(prompt)
    raw_candidates = parse_ollama_candidates(response_text)
    candidates: list[dict] = []
    rejected = 0
    for raw_candidate in raw_candidates[:questions_per_chunk]:
        if not isinstance(raw_candidate, dict):
            rejected += 1
            continue
        candidate, rejection_reason = normalize_candidate(raw_candidate, chunk)
        if candidate:
            candidates.append(candidate)
        else:
            rejected += 1
            logging.info("Rejected candidate for chunk %s: %s", chunk.get("chunk_id", ""), rejection_reason)
    return candidates, len(raw_candidates), rejected


def save_jsonl(candidates: list[dict], path: Path, append: bool = False) -> None:
    ensure_dir(path.parent)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as output_file:
        for candidate in candidates:
            output_file.write(json.dumps(candidate, ensure_ascii=False) + "\n")


def print_summary(
    chunks_loaded: int,
    chunks_selected: int,
    raw_candidates_generated: int,
    candidates_accepted: int,
    candidates_rejected: int,
    skipped_chunks: int,
    output_path: Path,
    append: bool,
) -> None:
    print("Gold candidate drafting summary")
    print("===============================")
    print(f"Chunks loaded: {chunks_loaded}")
    print(f"Matching chunks selected: {chunks_selected}")
    print(f"Raw candidates generated: {raw_candidates_generated}")
    print(f"Candidates accepted: {candidates_accepted}")
    print(f"Candidates rejected: {candidates_rejected}")
    print(f"Skipped chunks: {skipped_chunks}")
    print(f"Write mode: {'append' if append else 'overwrite'}")
    print(f"Output path: {output_path}")


def main() -> None:
    configure_stdout()
    setup_logging()
    args = parse_args()

    chunks = load_chunks(args.chunks_path.resolve())
    selected_chunks = select_chunks(
        chunks,
        contains=args.contains,
        category=args.category,
        source=args.source,
        limit=args.limit,
    )

    candidates: list[dict] = []
    skipped_chunks = 0
    raw_candidates_generated = 0
    candidates_rejected = 0
    for chunk in selected_chunks:
        try:
            chunk_candidates, raw_count, rejected_count = draft_candidates_for_chunk(chunk, args.questions_per_chunk)
            candidates.extend(chunk_candidates)
            raw_candidates_generated += raw_count
            candidates_rejected += rejected_count
        except Exception as exc:
            skipped_chunks += 1
            logging.warning("Skipping chunk %s after generation/parsing failure: %s", chunk.get("chunk_id", ""), exc)

    output_path = args.output.resolve()
    save_jsonl(candidates, output_path, append=args.append)
    print_summary(
        chunks_loaded=len(chunks),
        chunks_selected=len(selected_chunks),
        raw_candidates_generated=raw_candidates_generated,
        candidates_accepted=len(candidates),
        candidates_rejected=candidates_rejected,
        skipped_chunks=skipped_chunks,
        output_path=output_path,
        append=args.append,
    )


if __name__ == "__main__":
    main()
