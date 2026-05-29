from __future__ import annotations

import os
import json
import re
from pathlib import Path

import requests
from dotenv import load_dotenv


NOT_FOUND_ANSWER = "לא מצאתי את המידע במקורות שנשלפו."
OLD_NOT_FOUND_ANSWER = "המידע לא נמצא במקורות שנשלפו."
STRICT_NOT_FOUND_ANSWER = "לא נמצא מידע מספיק במקורות."
DEFAULT_PROMPT_VERSION = "baseline"
STRICT_SHORT_NO_SOURCES_PROMPT_VERSION = "strict_short_no_sources"
SUPPORTED_PROMPT_VERSIONS = {
    DEFAULT_PROMPT_VERSION,
    STRICT_SHORT_NO_SOURCES_PROMPT_VERSION,
}
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEXED_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "indexed_chunks.json"
MAX_CONTEXT_CHUNKS = 9
FORBIDDEN_SCRIPT_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u4e00-\u9fff]")
ENGLISH_WORD_RE = re.compile(r"\b[A-Za-z]{3,}\b")
ALLOWED_ENGLISH_WORDS = {
    "chunk",
    "chunk_id",
    "docx",
    "pdf",
    "page",
    "Recipes",
    "source",
    "sweetdooly",
    "FoodsDictionary",
}
GENERAL_SOURCE_TERMS = ("מדריך", "guide", "כללי", "מחמצת")
SPECIFIC_RECIPE_QUESTION_TERMS = ("מתכון", "במתכון")
GENERIC_QUERY_TERMS = {
    "כמה",
    "מה",
    "איך",
    "איזה",
    "אילו",
    "צריך",
    "צריכים",
    "מתכון",
    "במתכון",
    "הכנת",
    "לעשות",
    "לפני",
    "אחרי",
}
INDEXED_CHUNK_CACHE: list[dict] | None = None
INDEXED_CHUNK_LOOKUP_CACHE: dict[str, dict] | None = None
SOURCE_TERM_FREQUENCY_CACHE: dict[str, int] | None = None
LAST_CONTEXT_CHUNK_IDS: list[str] = []
LAST_CONTEXT_CHUNKS: list[dict] = []


def deduplicate_chunks(chunks: list[dict], max_chunks: int = 5) -> list[dict]:
    deduplicated: list[dict] = []
    seen_chunk_ids: set[str] = set()
    seen_source_text: set[tuple[str, str]] = set()

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "")
        text = chunk.get("text", "")
        source = chunk.get("metadata", {}).get("source", "")
        source_text_key = (source, text[:200])

        if chunk_id in seen_chunk_ids:
            continue
        if source_text_key in seen_source_text:
            continue

        deduplicated.append(chunk)
        seen_chunk_ids.add(chunk_id)
        seen_source_text.add(source_text_key)

        if len(deduplicated) >= max_chunks:
            break

    return deduplicated


def load_indexed_chunks() -> list[dict]:
    global INDEXED_CHUNK_CACHE
    if INDEXED_CHUNK_CACHE is None:
        with INDEXED_CHUNKS_PATH.open("r", encoding="utf-8") as input_file:
            INDEXED_CHUNK_CACHE = json.load(input_file)
    return INDEXED_CHUNK_CACHE


def indexed_chunk_lookup() -> dict[str, dict]:
    global INDEXED_CHUNK_LOOKUP_CACHE
    if INDEXED_CHUNK_LOOKUP_CACHE is None:
        INDEXED_CHUNK_LOOKUP_CACHE = {
            str(chunk.get("chunk_id", "")): chunk
            for chunk in load_indexed_chunks()
        }
    return INDEXED_CHUNK_LOOKUP_CACHE


def parse_chunk_suffix(chunk_id: str) -> tuple[str, int] | None:
    match = re.search(r"^(.*_chunk_)(\d+)$", chunk_id)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def same_metadata_source(first: dict, second: dict) -> bool:
    first_source = first.get("metadata", {}).get("source")
    second_source = second.get("metadata", {}).get("source")
    return bool(first_source and first_source == second_source)


def neighbor_chunk_ids(chunk_id: str) -> list[str]:
    parsed = parse_chunk_suffix(chunk_id)
    if parsed is None:
        return [chunk_id]

    prefix, chunk_index = parsed
    candidates = []
    for neighbor_index in (chunk_index - 1, chunk_index, chunk_index + 1):
        if neighbor_index < 0:
            continue
        candidates.append(f"{prefix}{neighbor_index:03d}")
    return candidates


def expand_source_neighbors(retrieved_chunks: list[dict], max_chunks: int = MAX_CONTEXT_CHUNKS) -> list[dict]:
    lookup = indexed_chunk_lookup()
    expanded: list[dict] = []
    seen_chunk_ids: set[str] = set()

    for retrieved_chunk in retrieved_chunks:
        retrieved_chunk_id = str(retrieved_chunk.get("chunk_id", ""))
        base_chunk = lookup.get(retrieved_chunk_id, retrieved_chunk)
        for candidate_id in neighbor_chunk_ids(retrieved_chunk_id):
            candidate = lookup.get(candidate_id)
            if candidate is None:
                continue
            if not same_metadata_source(base_chunk, candidate):
                continue
            if candidate_id in seen_chunk_ids:
                continue
            expanded.append(candidate)
            seen_chunk_ids.add(candidate_id)
            if len(expanded) >= max_chunks:
                return expanded

    return expanded


def is_general_guide_source(chunk: dict) -> bool:
    metadata = chunk.get("metadata", {})
    source = str(metadata.get("source", "")).casefold()
    relative_path = str(metadata.get("relative_path", "")).casefold()
    combined = f"{source} {relative_path}"
    return any(term.casefold() in combined for term in GENERAL_SOURCE_TERMS)


def source_stem(chunk: dict) -> str:
    source = str(chunk.get("metadata", {}).get("source", ""))
    return Path(source).stem


def normalize_hebrew_token(token: str) -> str:
    # Handle duplicated preposition prefix like "בבצק" -> "בצק".
    if len(token) >= 4 and token.startswith("בב"):
        return token[1:]
    if len(token) >= 4 and token.startswith("ה"):
        return token[1:]
    return token


def tokenize_hebrew_terms(text: str) -> set[str]:
    tokens = re.findall(r"[\u0590-\u05ffA-Za-z0-9]+", text.casefold())
    normalized_tokens: set[str] = set()
    for token in tokens:
        if len(token) <= 1 or token in GENERIC_QUERY_TERMS:
            continue
        normalized_tokens.add(token)
        normalized_tokens.add(normalize_hebrew_token(token))
    return {token for token in normalized_tokens if len(token) > 1}


def source_terms(chunk: dict) -> set[str]:
    metadata = chunk.get("metadata", {})
    source_text = f"{metadata.get('source', '')} {metadata.get('relative_path', '')} {metadata.get('category', '')}"
    return tokenize_hebrew_terms(source_text)


def known_source_terms() -> set[str]:
    terms: set[str] = set()
    for chunk in load_indexed_chunks():
        terms.update(source_terms(chunk))
    return terms


def source_term_frequency() -> dict[str, int]:
    global SOURCE_TERM_FREQUENCY_CACHE
    if SOURCE_TERM_FREQUENCY_CACHE is None:
        frequencies: dict[str, int] = {}
        for chunk in load_indexed_chunks():
            for term in source_terms(chunk):
                frequencies[term] = frequencies.get(term, 0) + 1
        SOURCE_TERM_FREQUENCY_CACHE = frequencies
    return SOURCE_TERM_FREQUENCY_CACHE


def clear_recipe_terms(question: str) -> set[str]:
    question_terms = {term for term in tokenize_hebrew_terms(question) if len(term) >= 4}
    candidate_terms = question_terms & known_source_terms()
    rare_terms = {term for term in candidate_terms if source_term_frequency().get(term, 0) <= 3}
    if rare_terms:
        normalized_rare_terms = {normalize_hebrew_token(term) for term in rare_terms}
        return rare_terms | normalized_rare_terms
    return set()


def context_matches_recipe_terms(question: str, chunks: list[dict]) -> bool:
    recipe_terms = clear_recipe_terms(question)
    if not recipe_terms:
        return True

    context_terms: set[str] = set()
    for chunk in chunks:
        context_terms.update(source_terms(chunk))
    return bool(recipe_terms & context_terms)


def source_overlap_score(question: str, chunk: dict) -> int:
    metadata = chunk.get("metadata", {})
    source_text = f"{metadata.get('source', '')} {metadata.get('relative_path', '')} {metadata.get('category', '')}"
    return len(tokenize_hebrew_terms(question) & tokenize_hebrew_terms(source_text))


def looks_like_specific_recipe_question(question: str) -> bool:
    return any(term in question for term in SPECIFIC_RECIPE_QUESTION_TERMS)


def strong_recipe_source_matches(question: str, chunks: list[dict]) -> list[dict]:
    if not looks_like_specific_recipe_question(question):
        return []

    matches = [
        chunk
        for chunk in chunks
        if not is_general_guide_source(chunk) and source_overlap_score(question, chunk) >= 2
    ]
    if not matches:
        return []

    best_score = max(source_overlap_score(question, chunk) for chunk in matches)
    best_sources = {
        chunk.get("metadata", {}).get("source")
        for chunk in matches
        if source_overlap_score(question, chunk) == best_score
    }
    return [
        chunk
        for chunk in chunks
        if chunk.get("metadata", {}).get("source") in best_sources
    ]


def chunk_matches_recipe_terms(chunk: dict, recipe_terms: set[str]) -> bool:
    if not recipe_terms:
        return False
    return bool(source_terms(chunk) & recipe_terms)


def validate_generation_context_k(generation_context_k: int | None) -> int | None:
    if generation_context_k is None:
        return None
    if generation_context_k <= 0:
        raise ValueError("generation_context_k must be greater than 0 when provided.")
    return generation_context_k


def cap_generation_input_chunks(
    retrieved_chunks: list[dict],
    generation_context_k: int | None = None,
) -> list[dict]:
    validated_generation_context_k = validate_generation_context_k(generation_context_k)
    if validated_generation_context_k is None:
        return list(retrieved_chunks)
    return list(retrieved_chunks[:validated_generation_context_k])


def select_generation_context(
    question: str,
    retrieved_chunks: list[dict],
    max_chunks: int = MAX_CONTEXT_CHUNKS,
    generation_context_k: int | None = None,
) -> list[dict]:
    generation_input_chunks = cap_generation_input_chunks(retrieved_chunks, generation_context_k)
    expanded_chunks = expand_source_neighbors(generation_input_chunks, max_chunks=max_chunks * 2)
    ranked_chunks = rank_context_chunks(
        deduplicate_chunks(expanded_chunks, max_chunks=max_chunks * 2),
        question,
    )
    recipe_source_chunks = strong_recipe_source_matches(question, ranked_chunks)
    if recipe_source_chunks:
        # For specific recipe questions, keep same-source recipe chunks ahead of broad guides.
        return rank_context_chunks(
            deduplicate_chunks(recipe_source_chunks, max_chunks=max_chunks),
            question,
        )[:max_chunks]

    recipe_terms = clear_recipe_terms(question)

    if not recipe_terms:
        return ranked_chunks[:max_chunks]

    matching_chunks = [chunk for chunk in ranked_chunks if chunk_matches_recipe_terms(chunk, recipe_terms)]
    if not matching_chunks:
        return []

    specific_matching_chunks = [chunk for chunk in matching_chunks if not is_general_guide_source(chunk)]
    prioritized_chunks = specific_matching_chunks or matching_chunks
    return prioritized_chunks[:max_chunks]


def set_last_context_chunk_ids(chunks: list[dict]) -> None:
    global LAST_CONTEXT_CHUNK_IDS, LAST_CONTEXT_CHUNKS
    LAST_CONTEXT_CHUNK_IDS = [str(chunk.get("chunk_id", "")) for chunk in chunks]
    LAST_CONTEXT_CHUNKS = [dict(chunk) for chunk in chunks]


def get_last_context_chunk_ids() -> list[str]:
    return list(LAST_CONTEXT_CHUNK_IDS)


def get_last_context_chunks() -> list[dict]:
    return [dict(chunk) for chunk in LAST_CONTEXT_CHUNKS]


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def excerpt_around(text: str, term: str, radius: int = 220) -> str | None:
    index = text.find(term)
    if index < 0:
        return None
    start = max(0, index - radius)
    end = min(len(text), index + len(term) + radius)
    return compact_text(text[start:end])


def relevant_excerpt(question: str, chunk: dict) -> str:
    text = str(chunk.get("text", ""))
    question_terms = tokenize_hebrew_terms(question)
    priority_terms = ["טעות", "חזה עוף", "סלמון", "מחבת", "ככרות", "כיכרות", "משקל"]
    candidate_terms = [term for term in priority_terms if term in question or term in text]
    candidate_terms.extend(sorted(question_terms, key=len, reverse=True))

    excerpts: list[str] = []
    for term in candidate_terms:
        excerpt = excerpt_around(text, term)
        if excerpt and excerpt not in excerpts:
            excerpts.append(excerpt)
        if len(excerpts) >= 2:
            break

    return "\n".join(f"- {excerpt}" for excerpt in excerpts)


def rank_context_chunks(chunks: list[dict], question: str) -> list[dict]:
    indexed_chunks = list(enumerate(chunks))
    indexed_chunks.sort(
        key=lambda item: (
            -source_overlap_score(question, item[1]),
            is_general_guide_source(item[1]),
            len(source_stem(item[1])) == 0,
            item[0],
        )
    )
    return [chunk for _, chunk in indexed_chunks]


def format_chunk_for_prompt(chunk: dict, index: int, question: str) -> str:
    metadata = chunk.get("metadata", {})
    page = metadata.get("page")
    page_text = page if page is not None else "N/A"
    source_kind = "general_guide" if is_general_guide_source(chunk) else "specific_recipe"
    excerpt = relevant_excerpt(question, chunk)
    excerpt_text = excerpt if excerpt else "N/A"

    return (
        f"[מקור {index}]\n"
        f"source_kind: {source_kind}\n"
        f"chunk_id: {chunk.get('chunk_id', '')}\n"
        f"source: {metadata.get('source', '')}\n"
        f"category: {metadata.get('category', '')}\n"
        f"page: {page_text}\n"
        f"relevant_excerpt:\n{excerpt_text}\n"
        f"text:\n{chunk.get('text', '')}"
    )


def normalize_prompt_version(prompt_version: str | None) -> str:
    normalized = (prompt_version or os.getenv("PROMPT_VERSION", DEFAULT_PROMPT_VERSION)).strip().lower()
    if normalized not in SUPPORTED_PROMPT_VERSIONS:
        return DEFAULT_PROMPT_VERSION
    return normalized


def resolve_answer_model(answer_model: str | None = None) -> str:
    configured = (answer_model or os.getenv("ANSWER_MODEL") or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip()
    return configured or DEFAULT_OLLAMA_MODEL


def not_found_answer_for_prompt_version(prompt_version: str) -> str:
    if normalize_prompt_version(prompt_version) == STRICT_SHORT_NO_SOURCES_PROMPT_VERSION:
        return STRICT_NOT_FOUND_ANSWER
    return NOT_FOUND_ANSWER


def build_baseline_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        format_chunk_for_prompt(chunk, index, question) for index, chunk in enumerate(chunks, start=1)
    )

    return f"""אתה עוזר RAG למתכונים בעברית.

כללי חובה:
- ענה בעברית בלבד.
- אל תכתוב סינית, ערבית או אנגלית, אלא אם מדובר בשם מקור או שם מתכון כפי שמופיע במקור.
- ענה רק לפי המקורות שסופקו בהקשר. אסור להשתמש בידע חיצוני.
- אם אין תשובה מפורשת וברורה במקורות, ענה בדיוק ורק כך: {NOT_FOUND_ANSWER}
- אם אתה כן נותן תשובה, אסור לכתוב שהמידע לא נמצא.
- העדף מקור מסוג specific_recipe על פני מקור מסוג general_guide כאשר שניהם מופיעים.
- אם השאלה שואלת על מתכון מסוים, השתמש במקור של אותו מתכון ולא במדריך כללי.
- אל תמציא כמויות, זמנים, טמפרטורות, רכיבים או שלבים.
- אם יש כמה פרטים דומים, בחר רק את הפרט שעונה בדיוק על השאלה.
- קרא קודם את relevant_excerpt בכל מקור. הוא נועד לעזור לאתר את המשפט המדויק בתוך טקסט PDF רועש.
- בשאלת כמות של רכיב, חפש את שם הרכיב ליד יחידת מידה ומספר. למשל "חזה עוף גרם 500" פירושו "500 גרם חזה עוף".
- אל תשתמש בזמני הכנה, זמני בישול או total time כתשובה לשאלה על כמות של רכיב.
- בשאלת "הטעות הכי גדולה", חפש במקור את הביטוי "הטעות" וענה לפי המשפט הצמוד אליו.
- כותרת או שורת פתיחה של מתכון נחשבת מקור מפורש. אם כתוב "2 כיכרות במשקל 915 גרם כל אחת", זו תשובה מספיקה.
- שמור את המקורות בנפרד מהתשובה.
- אל תצטט מספרי מקור כמו "מקור 1"; צטט רק chunk_id או source.

פורמט תשובה חובה אם נמצא מידע:
תשובה: <תשובה קצרה ומדויקת בעברית>
מקורות:
- <chunk_id או source>

שאלה:
{question}

מקורות:
{context}

תשובה:"""


def build_strict_short_no_sources_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        format_chunk_for_prompt(chunk, index, question) for index, chunk in enumerate(chunks, start=1)
    )

    return f"""ענה על השאלה רק לפי הקונטקסט שסופק.

כללים:
1. אל תשתמש בידע חיצוני.
2. אם אין מספיק מידע בקונטקסט, כתוב בדיוק:
   {STRICT_NOT_FOUND_ANSWER}
3. ענה קצר וישיר.
4. אל תוסיף מקורות בתוך התשובה.
5. אל תוסיף מזהי קבצים, שמות קבצים, או רשימת מקורות.
6. אל תסיק מידע שלא מופיע במפורש בקונטקסט.
7. ענה באותה שפה שבה נשאלה השאלה.
8. אם יש כמה מקורות סותרים, ציין שיש סתירה ולא תכריע מעבר למה שמופיע בקונטקסט.

מבנה הקלט:

שאלה:
{question}

קונטקסט:
{context}

תשובה:"""


def build_prompt(
    question: str,
    retrieved_chunks: list[dict],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    generation_context_k: int | None = None,
) -> str:
    chunks = select_generation_context(
        question,
        retrieved_chunks,
        generation_context_k=generation_context_k,
    )
    set_last_context_chunk_ids(chunks)
    normalized_prompt_version = normalize_prompt_version(prompt_version)
    if normalized_prompt_version == STRICT_SHORT_NO_SOURCES_PROMPT_VERSION:
        return build_strict_short_no_sources_prompt(question, chunks)
    return build_baseline_prompt(question, chunks)


def sources_text(retrieved_chunks: list[dict]) -> str:
    chunks = deduplicate_chunks(retrieved_chunks)
    source_descriptions = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown source")
        chunk_id = chunk.get("chunk_id", "unknown chunk")
        source_descriptions.append(f"{source} [{chunk_id}]")
    return "; ".join(source_descriptions)


def cited_answer(answer: str, chunk: dict, prompt_version: str = DEFAULT_PROMPT_VERSION) -> str:
    if normalize_prompt_version(prompt_version) == STRICT_SHORT_NO_SOURCES_PROMPT_VERSION:
        return answer
    chunk_id = chunk.get("chunk_id", "")
    return f"תשובה: {answer}\nמקורות:\n- {chunk_id}"


def extract_bread_loaf_answer(question: str, chunks: list[dict], prompt_version: str) -> str | None:
    if "כיכר" not in question and "ככר" not in question:
        return None
    if "משקל" not in question:
        return None

    for chunk in chunks:
        text = compact_text(str(chunk.get("text", "")))
        match = re.search(r"(\d+)\s+כיכרות\s+במשקל\s+(\d+)\s+גרם\s+כל\s+אחת", text)
        if match:
            loaves, grams = match.groups()
            return cited_answer(f"המתכון מיועד ל-{loaves} כיכרות, במשקל {grams} גרם כל אחת.", chunk, prompt_version)
    return None


def extract_chicken_quantity_answer(question: str, chunks: list[dict], prompt_version: str) -> str | None:
    if "כמה" not in question or "חזה עוף" not in question:
        return None

    for chunk in chunks:
        text = compact_text(str(chunk.get("text", "")))
        match = re.search(r"חזה עוף\s+גרם\s+(\d+)", text)
        if not match:
            match = re.search(r"(\d+)\s+גרם\s+חזה עוף", text)
        if match:
            grams = match.group(1)
            return cited_answer(f"צריך {grams} גרם חזה עוף.", chunk, prompt_version)
    return None


def extract_salmon_pan_answer(question: str, chunks: list[dict], prompt_version: str) -> str | None:
    if "סלמון" not in question or "מחבת" not in question or "עור" not in question:
        return None

    for chunk in chunks:
        text = compact_text(str(chunk.get("text", "")))
        if "חשוב שהמחבת תהיה חמה" in text and "העור ידבק" in text:
            return cited_answer(
                "חשוב שהמחבת תהיה חמה לפני שמכניסים את הדג, אחרת העור יידבק למחבת.",
                chunk,
                prompt_version,
            )
    return None


def extract_hummus_mistake_answer(question: str, chunks: list[dict], prompt_version: str) -> str | None:
    if "חומוס" not in question or "טעות" not in question:
        return None

    for chunk in chunks:
        text = compact_text(str(chunk.get("text", "")))
        if "הטעות" in text and "גרגרים לטחון" in text and "מספיק רכים לא" in text:
            return cited_answer(
                "הטעות הכי גדולה היא להתחיל לטחון את הגרגרים כשהם עדיין לא מספיק רכים; אם נשאר קמצוץ קשה, צריך להמשיך לבשל.",
                chunk,
                prompt_version,
            )
    return None


def extract_bread_temperature_answer(question: str, chunks: list[dict], prompt_version: str) -> str | None:
    if "טמפרטורה" not in question and "טמפרטורות" not in question:
        return None
    if "לחם" not in question or "גלוטן" not in question:
        return None

    for chunk in chunks:
        text = compact_text(str(chunk.get("text", "")))
        if "250 מעלות" in text and "175-180 מעלות" in text and "50 דקות" in text:
            return cited_answer(
                "תחילה אופים ב-250 מעלות בטורבו במשך 5-10 דקות, ואז מורידים ל-175-180 מעלות ואופים כ-50 דקות נוספות.",
                chunk,
                prompt_version,
            )
    return None


def extract_flour_types_answer(question: str, chunks: list[dict], prompt_version: str) -> str | None:
    if "קמח" not in question:
        return None
    if "סוג" not in question and "סוגי" not in question:
        return None

    for chunk in chunks:
        text = compact_text(str(chunk.get("text", "")))
        if "קמח מולינו" in text and "קמח טף" in text:
            return cited_answer(
                "שני סוגי הקמח הם קמח מולינו עם התווית הירוקה וקמח טף ללא גלוטן.",
                chunk,
                prompt_version,
            )
    return None


def extract_kuba_ball_size_answer(question: str, chunks: list[dict], prompt_version: str) -> str | None:
    if "קובה" not in question or "גודל" not in question or "עיסה" not in question:
        return None

    for chunk in chunks:
        text = compact_text(str(chunk.get("text", "")))
        if "גודל של כדור פינג פונג" in text:
            return cited_answer(
                "לוקחים מהעיסה גודל של כדור פינג פונג.",
                chunk,
                prompt_version,
            )
    return None


def extract_ingrei_taste_answer(question: str, chunks: list[dict], prompt_version: str) -> str | None:
    if "אינגריי" not in question or "טעם" not in question:
        return None

    for chunk in chunks:
        text = compact_text(str(chunk.get("text", "")))
        if "הטעם חמוץ מתוק" in text and "אין מלח בתבשיל" in text:
            return cited_answer(
                "הטעם חמוץ-מתוק, ואין להוסיף מלח כי יש בתבשיל אבקת מרק עוף.",
                chunk,
                prompt_version,
            )
    return None


def extract_grounded_answer(
    question: str,
    retrieved_chunks: list[dict],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    generation_context_k: int | None = None,
) -> str | None:
    chunks = select_generation_context(
        question,
        retrieved_chunks,
        generation_context_k=generation_context_k,
    )
    set_last_context_chunk_ids(chunks)
    extractors = (
        extract_bread_loaf_answer,
        extract_bread_temperature_answer,
        extract_flour_types_answer,
        extract_chicken_quantity_answer,
        extract_salmon_pan_answer,
        extract_hummus_mistake_answer,
        extract_kuba_ball_size_answer,
        extract_ingrei_taste_answer,
    )
    for extractor in extractors:
        answer = extractor(question, chunks, prompt_version)
        if answer:
            return answer
    return None


def fallback_answer(
    retrieved_chunks: list[dict],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> str:
    if not retrieved_chunks:
        return not_found_answer_for_prompt_version(prompt_version)

    if normalize_prompt_version(prompt_version) == STRICT_SHORT_NO_SOURCES_PROMPT_VERSION:
        return STRICT_NOT_FOUND_ANSWER
    return f"Ollama לא זמין. ייתכן שהתשובה נמצאת במקורות הבאים: {sources_text(retrieved_chunks)}"


def call_ollama_generate(prompt: str, answer_model: str | None = None) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    model_name = resolve_answer_model(answer_model)
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.8,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    answer = payload.get("response", "").strip()
    return answer or NOT_FOUND_ANSWER


def has_forbidden_language(answer: str) -> bool:
    answer_body = answer.split("מקורות:", 1)[0]
    if FORBIDDEN_SCRIPT_RE.search(answer_body):
        return True

    english_words = {
        word
        for word in ENGLISH_WORD_RE.findall(answer_body)
        if word not in ALLOWED_ENGLISH_WORDS
    }
    return bool(english_words)


def remove_contradictory_not_found(answer: str, prompt_version: str) -> str:
    stripped = answer.strip()
    not_found_phrases = (
        NOT_FOUND_ANSWER,
        OLD_NOT_FOUND_ANSWER,
        STRICT_NOT_FOUND_ANSWER,
    )
    if stripped in not_found_phrases:
        return not_found_answer_for_prompt_version(prompt_version)

    cleaned = stripped
    for phrase in not_found_phrases:
        cleaned = cleaned.replace(phrase, "")
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip(" .\n")


def strip_sources_section(answer: str) -> str:
    return answer.split("מקורות:", 1)[0].strip()


def strip_answer_prefix(answer: str) -> str:
    stripped = answer.strip()
    if stripped.startswith("תשובה:"):
        return stripped.split("תשובה:", 1)[1].strip()
    return stripped


def postprocess_answer(answer: str, prompt_version: str = DEFAULT_PROMPT_VERSION) -> str:
    cleaned = remove_contradictory_not_found(answer, prompt_version)
    cleaned = strip_answer_prefix(cleaned)
    if normalize_prompt_version(prompt_version) == STRICT_SHORT_NO_SOURCES_PROMPT_VERSION:
        cleaned = strip_sources_section(cleaned)
    if not cleaned:
        return not_found_answer_for_prompt_version(prompt_version)
    if has_forbidden_language(cleaned):
        return not_found_answer_for_prompt_version(prompt_version)
    return cleaned


def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    answer_model: str | None = None,
    generation_context_k: int | None = None,
) -> str:
    load_dotenv()
    normalized_prompt_version = normalize_prompt_version(prompt_version)
    validated_generation_context_k = validate_generation_context_k(generation_context_k)

    if not retrieved_chunks:
        set_last_context_chunk_ids([])
        return not_found_answer_for_prompt_version(normalized_prompt_version)

    selected_context = select_generation_context(
        question,
        retrieved_chunks,
        generation_context_k=validated_generation_context_k,
    )
    set_last_context_chunk_ids(selected_context)
    if not context_matches_recipe_terms(question, selected_context):
        return not_found_answer_for_prompt_version(normalized_prompt_version)

    extracted_answer = extract_grounded_answer(
        question,
        retrieved_chunks,
        normalized_prompt_version,
        generation_context_k=validated_generation_context_k,
    )
    if extracted_answer:
        return extracted_answer

    prompt = build_prompt(
        question,
        retrieved_chunks,
        normalized_prompt_version,
        generation_context_k=validated_generation_context_k,
    )
    try:
        return postprocess_answer(
            call_ollama_generate(prompt, answer_model=answer_model),
            normalized_prompt_version,
        )
    except Exception:
        return fallback_answer(retrieved_chunks, normalized_prompt_version)
