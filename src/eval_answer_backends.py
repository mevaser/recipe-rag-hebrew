from __future__ import annotations

import os
import re
from typing import Sequence

from openai import OpenAI

from generation import STRICT_NOT_FOUND_ANSWER, call_ollama_generate


DEFAULT_LOCAL_ANSWER_BACKEND = "local"
DEFAULT_OPENAI_ANSWER_MODEL = "gpt-4.1-mini"
SUPPORTED_ANSWER_BACKENDS = {DEFAULT_LOCAL_ANSWER_BACKEND, "openai"}


def sanitize_model_name(model_name: str) -> str:
    lowered = model_name.strip().lower()
    sanitized = re.sub(r"[^a-z0-9._-]+", "_", lowered).strip("_")
    return sanitized or "model"


def resolve_openai_answer_model(answer_model: str | None) -> str:
    configured = (answer_model or os.getenv("OPENAI_ANSWER_MODEL") or DEFAULT_OPENAI_ANSWER_MODEL).strip()
    return configured or DEFAULT_OPENAI_ANSWER_MODEL


def validate_generation_context_k(generation_context_k: int | None) -> int | None:
    if generation_context_k is None:
        return None
    if generation_context_k <= 0:
        raise ValueError("generation_context_k must be greater than 0 when provided.")
    return generation_context_k


def limit_contexts_for_generation(
    contexts: Sequence[str],
    generation_context_k: int | None = None,
) -> list[str]:
    validated_generation_context_k = validate_generation_context_k(generation_context_k)
    context_list = [str(context) for context in contexts]
    if validated_generation_context_k is None:
        return context_list
    return context_list[:validated_generation_context_k]


def require_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required when --answer-backend openai is used."
        )
    return api_key


def build_strict_context_prompt(question: str, contexts: Sequence[str]) -> str:
    context_blocks = []
    for index, context_text in enumerate(contexts, start=1):
        context_blocks.append(f"[Context {index}]\n{context_text}")
    combined_context = "\n\n---\n\n".join(context_blocks)

    return (
        "You are a Hebrew recipe RAG assistant.\n\n"
        "Rules:\n"
        "- Answer only from the provided contexts.\n"
        "- Answer in Hebrew only.\n"
        "- Give a short, direct answer.\n"
        "- Do not include sources.\n"
        "- Do not include explanations unless the question asks why.\n"
        "- Prefer explicit values and instructions from the most relevant context.\n"
        f"- If the answer is not found in the contexts, answer exactly: {STRICT_NOT_FOUND_ANSWER}\n"
        f"- If the answer exists in the context, do not say: {STRICT_NOT_FOUND_ANSWER}\n"
        "- Do not invent missing facts.\n\n"
        f"Question:\n{question}\n\n"
        f"Contexts:\n{combined_context}\n\n"
        "Return only the final answer text in Hebrew."
    )


def call_openai_generate(prompt: str, answer_model: str | None = None) -> str:
    api_key = require_openai_api_key()
    model_name = resolve_openai_answer_model(answer_model)
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    answer = response.choices[0].message.content or ""
    return answer.strip() or STRICT_NOT_FOUND_ANSWER


def generate_strict_answer(
    question: str,
    contexts: Sequence[str],
    answer_backend: str = DEFAULT_LOCAL_ANSWER_BACKEND,
    answer_model: str | None = None,
    generation_context_k: int | None = None,
) -> str:
    selected_contexts = limit_contexts_for_generation(contexts, generation_context_k)
    if not selected_contexts:
        return STRICT_NOT_FOUND_ANSWER

    prompt = build_strict_context_prompt(question, selected_contexts)
    if answer_backend == "openai":
        return call_openai_generate(prompt, answer_model=answer_model)
    return call_ollama_generate(prompt, answer_model=answer_model).strip() or STRICT_NOT_FOUND_ANSWER
