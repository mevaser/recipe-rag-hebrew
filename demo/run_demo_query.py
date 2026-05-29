from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env_utils import load_project_env  # noqa: E402
from eval_answer_backends import (  # noqa: E402
    DEFAULT_LOCAL_ANSWER_BACKEND,
    DEFAULT_OPENAI_ANSWER_MODEL,
    SUPPORTED_ANSWER_BACKENDS,
    generate_strict_answer,
)
from generation import (  # noqa: E402
    STRICT_SHORT_NO_SOURCES_PROMPT_VERSION,
    generate_answer,
    get_last_context_chunks,
)
from hybrid_retrieval import retrieve_hybrid  # noqa: E402


DEFAULT_LOCAL_MODEL = "qwen2.5:7b-instruct"
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 50
DEFAULT_RRF_K = 30
DEFAULT_DENSE_WEIGHT = 0.5
DEFAULT_BM25_WEIGHT = 2.0
DEFAULT_HTML_OUTPUT_PATH = PROJECT_ROOT / "demo" / "demo_output.html"
DEFAULT_QUESTIONS = [
    "מה כמות הקמח, המים והשמרים בבצק הפיצה?",
    "כמה חזה עוף צריך למתכון חזה עוף וירקות בקרם קוקוס?",
]


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short Hebrew RAG demo query with the current hybrid baseline.")
    parser.add_argument("--question", help="Optional single Hebrew question to run.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of retrieved chunks to show.")
    parser.add_argument(
        "--answer-backend",
        choices=sorted(SUPPORTED_ANSWER_BACKENDS),
        default=DEFAULT_LOCAL_ANSWER_BACKEND,
        help="Answer backend to use.",
    )
    parser.add_argument(
        "--answer-model",
        default=DEFAULT_LOCAL_MODEL,
        help=f"Answer model to use. For OpenAI the default is {DEFAULT_OPENAI_ANSWER_MODEL}.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show retrieval and generation-context diagnostics.",
    )
    return parser.parse_args()


def source_identifier(chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    source = str(metadata.get("source", "")).strip()
    chunk_id = str(chunk.get("chunk_id", "")).strip()
    if source and chunk_id:
        return f"{source} [{chunk_id}]"
    return source or chunk_id


def render_demo_html(results: list[dict]) -> str:
    sections: list[str] = []
    for result in results:
        question_html = html.escape(str(result["question"]))
        answer_html = html.escape(str(result["answer"]))
        sources_html = "\n".join(
            f"<li><div dir=\"ltr\">{html.escape(source)}</div></li>"
            for source in result["sources"]
        )
        sections.append(
            "\n".join(
                [
                    "<section class=\"demo-block\">",
                    "  <h2>Question</h2>",
                    f"  <div class=\"question\" dir=\"rtl\">{question_html}</div>",
                    "  <h2>Retrieved Sources</h2>",
                    "  <ul>",
                    sources_html or "    <li><div dir=\"ltr\">No sources returned.</div></li>",
                    "  </ul>",
                    "  <h2>Answer</h2>",
                    f"  <div class=\"answer\" dir=\"rtl\">{answer_html}</div>",
                    "</section>",
                ]
            )
        )

    return "\n".join(
        [
            "<!DOCTYPE html>",
            "<html lang=\"he\" dir=\"rtl\">",
            "<head>",
            "  <meta charset=\"utf-8\">",
            "  <title>Hebrew Recipe RAG Demo Output</title>",
            "  <style>",
            "    body { font-family: Arial, sans-serif; margin: 32px; line-height: 1.6; background: #fafafa; color: #111; }",
            "    h1 { margin-bottom: 24px; }",
            "    h2 { margin-top: 0; font-size: 1.05rem; }",
            "    .demo-block { background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 20px; margin-bottom: 24px; }",
            "    .question, .answer { font-size: 1.05rem; padding: 10px 12px; background: #f5f5f5; border-radius: 8px; }",
            "    ul { margin-top: 8px; padding-right: 20px; }",
            "    li { margin-bottom: 8px; }",
            "    pre { white-space: pre-wrap; word-break: break-word; }",
            "  </style>",
            "</head>",
            "<body>",
            "  <h1>Hebrew Recipe RAG Demo Output</h1>",
            *sections,
            "</body>",
            "</html>",
        ]
    )


def write_demo_html(results: list[dict], output_path: Path = DEFAULT_HTML_OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_demo_html(results), encoding="utf-8")


def retrieve_demo_chunks(question: str, top_k: int) -> list[dict]:
    return retrieve_hybrid(
        question,
        k=top_k,
        candidate_k=max(DEFAULT_CANDIDATE_K, top_k),
        rrf_k=DEFAULT_RRF_K,
        dense_weight=DEFAULT_DENSE_WEIGHT,
        bm25_weight=DEFAULT_BM25_WEIGHT,
    )


def generate_demo_answer(
    question: str,
    retrieved_chunks: list[dict],
    answer_backend: str,
    answer_model: str,
) -> str:
    if answer_backend == DEFAULT_LOCAL_ANSWER_BACKEND:
        return generate_answer(
            question,
            retrieved_chunks,
            prompt_version=STRICT_SHORT_NO_SOURCES_PROMPT_VERSION,
            answer_model=answer_model,
        )

    contexts = [str(chunk.get("text", "")) for chunk in retrieved_chunks]
    return generate_strict_answer(
        question,
        contexts,
        answer_backend=answer_backend,
        answer_model=answer_model,
    )


def print_demo_result(question: str, retrieved_chunks: list[dict], answer: str) -> None:
    print("=" * 80)
    print(f"Question: {question}")
    print("Top retrieved sources:")
    for index, chunk in enumerate(retrieved_chunks, start=1):
        print(f"{index}. {source_identifier(chunk)}")
    print("Final answer:")
    print(answer)
    print()


def chunk_contains_pizza_values(chunk: dict) -> bool:
    text = str(chunk.get("text", ""))
    return "320" in text and "188" in text and "0.65" in text


def print_debug_info(retrieved_chunks: list[dict]) -> None:
    generation_chunks = get_last_context_chunks()
    print("[debug] retrieval_count:", len(retrieved_chunks))
    print("[debug] generation_context_count:", len(generation_chunks))
    print("[debug] top_sources:")
    for index, chunk in enumerate(retrieved_chunks[:5], start=1):
        print(f"  {index}. {source_identifier(chunk)}")
    print("[debug] pizza_values_in_retrieval:", any(chunk_contains_pizza_values(chunk) for chunk in retrieved_chunks))
    print(
        "[debug] pizza_values_in_generation_context:",
        any(chunk_contains_pizza_values(chunk) for chunk in generation_chunks),
    )
    print()


def main() -> None:
    configure_stdout()
    load_project_env(__file__)
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")

    questions = [args.question] if args.question else list(DEFAULT_QUESTIONS)
    html_results: list[dict] = []

    for question in questions:
        retrieved_chunks = retrieve_demo_chunks(question, args.top_k)
        answer = generate_demo_answer(
            question,
            retrieved_chunks,
            answer_backend=args.answer_backend,
            answer_model=args.answer_model,
        )
        if args.debug:
            print_debug_info(retrieved_chunks)
        print_demo_result(question, retrieved_chunks, answer)
        html_results.append(
            {
                "question": question,
                "answer": answer,
                "sources": [source_identifier(chunk) for chunk in retrieved_chunks],
            }
        )

    write_demo_html(html_results)


if __name__ == "__main__":
    main()
