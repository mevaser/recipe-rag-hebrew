from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import inspect
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env_utils import print_openai_api_key_status  # noqa: E402
from rag_system import answer  # noqa: E402


DEFAULT_GOLD_PATH = PROJECT_ROOT / "eval" / "gold_set.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval"
DEFAULT_LIMIT = 10
DEFAULT_PROVIDER = "ollama"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LOCAL_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_METRICS = "faithfulness,answer_relevancy,context_precision,context_recall"
DEFAULT_EVAL_PROMPT_VERSION = "strict_short_no_sources"
SUPPORTED_METRICS = {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}


class RagasEvalError(RuntimeError):
    """Raised when optional RAGAS evaluation cannot run."""


@dataclass
class MetricDefinition:
    public_name: str
    metric: Any


@dataclass
class EvaluatorModels:
    provider: str
    model: str
    llm: Any
    embeddings: Any
    ollama_base_url: str | None = None
    ollama_json_mode: bool = False
    ollama_num_ctx: int | None = None
    ollama_num_predict: int | None = None


@dataclass
class EvaluationRunConfig:
    gold_path: Path
    start: int
    limit: int
    output_dir: Path
    output_stem: str
    output_csv: Path | None
    output_json: Path | None
    metrics: str
    llm_provider: str
    llm_model: str
    embedding_model: str
    prompt_version: str
    answer_model: str
    generation_context_k: int | None


@dataclass
class EvaluationArtifacts:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    models: EvaluatorModels
    metrics: list[MetricDefinition]
    csv_output_path: Path
    json_output_path: Path


def resolve_answer_model_name(answer_model: str | None) -> str:
    configured = (answer_model or os.getenv("ANSWER_MODEL") or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip()
    return configured or DEFAULT_OLLAMA_MODEL


def sanitize_model_name(model_name: str) -> str:
    lowered = model_name.strip().lower()
    sanitized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return sanitized or "default_model"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    env_provider = normalized_provider(os.getenv("RAGAS_LLM_PROVIDER", DEFAULT_PROVIDER))
    default_model = DEFAULT_OLLAMA_MODEL if env_provider == "ollama" else DEFAULT_OPENAI_MODEL
    default_embedding_model = (
        DEFAULT_LOCAL_EMBEDDING_MODEL if env_provider == "ollama" else DEFAULT_OPENAI_EMBEDDING_MODEL
    )

    parser = argparse.ArgumentParser(description="Run optional RAGAS evaluation for the Hebrew Recipe RAG system.")
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Gold set JSONL path. Defaults to eval/gold_set.jsonl.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Number of gold examples to skip before evaluation. Defaults to 0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of gold examples to evaluate. Defaults to 10.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for ragas_results.csv and ragas_results.json. Defaults to eval.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional explicit CSV output path. Defaults to <output-dir>/ragas_results.csv.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional explicit JSON output path. Defaults to <output-dir>/ragas_results.json.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=DEFAULT_METRICS,
        help="Comma-separated metric names. Supported: faithfulness, answer_relevancy, context_precision, context_recall.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=env_provider,
        help="Evaluator provider. Defaults to RAGAS_LLM_PROVIDER or ollama.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.getenv("RAGAS_LLM_MODEL", default_model),
        help="Evaluator model. Defaults to RAGAS_LLM_MODEL or provider-specific default.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=os.getenv("RAGAS_EMBEDDING_MODEL", default_embedding_model),
        help="Embedding model. Defaults to RAGAS_EMBEDDING_MODEL or provider-specific default.",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=os.getenv("PROMPT_VERSION", DEFAULT_EVAL_PROMPT_VERSION),
        help="Answer-generation prompt version. Defaults to PROMPT_VERSION or strict_short_no_sources.",
    )
    parser.add_argument(
        "--answer-model",
        type=str,
        default=os.getenv("ANSWER_MODEL", os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)),
        help="Answer-generation model name. Defaults to ANSWER_MODEL, OLLAMA_MODEL, or qwen2.5:7b-instruct.",
    )
    parser.add_argument(
        "--generation-context-k",
        type=int,
        default=None,
        help="Optional cap on how many retrieved contexts are passed into answer generation.",
    )
    parser.add_argument(
        "--preview-results",
        action="store_true",
        help="Print question, generated answer, and retrieved contexts for each evaluated row.",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)


def normalized_provider(provider: str) -> str:
    return provider.strip().lower()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
            validate_gold_row(row, line_number)
            rows.append(row)
    return rows


def validate_gold_row(row: dict[str, Any], line_number: int) -> None:
    required_fields = {
        "question": str,
        "reference_answer": str,
    }
    for field, expected_type in required_fields.items():
        if field not in row:
            raise ValueError(f"Missing field '{field}' on gold set line {line_number}.")
        if not isinstance(row[field], expected_type):
            raise ValueError(f"Field '{field}' on gold set line {line_number} has the wrong type.")


def selected_examples(gold_rows: list[dict[str, Any]], start: int, limit: int) -> list[dict[str, Any]]:
    if start < 0:
        raise ValueError("--start must be greater than or equal to 0.")
    if limit <= 0:
        raise ValueError("--limit must be greater than 0.")
    return gold_rows[start : start + limit]


def build_records(
    examples: list[dict[str, Any]],
    start_index: int = 0,
    prompt_version: str = DEFAULT_EVAL_PROMPT_VERSION,
    answer_model: str = DEFAULT_OLLAMA_MODEL,
    generation_context_k: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, example in enumerate(examples, start=start_index + 1):
        question = example["question"]
        result = run_answer(
            question,
            index,
            prompt_version=prompt_version,
            answer_model=answer_model,
            generation_context_k=generation_context_k,
        )
        retrieved_chunks = result["retrieved_chunks"]
        generation_chunks = result.get("generation_chunks") or retrieved_chunks
        contexts = [
            str(chunk.get("text", "")).strip()
            for chunk in generation_chunks
            if str(chunk.get("text", "")).strip()
        ]
        records.append(
            {
                "question_number": index,
                "category": example.get("category", ""),
                "question": question,
                "answer": result["answer"],
                "response": result["answer"],
                "contexts": contexts,
                "retrieved_contexts": contexts,
                "reference": example["reference_answer"],
                "ground_truth": example["reference_answer"],
                "expected_chunk_ids": example.get("must_cite_chunk_ids", []),
                "retrieved_chunk_ids": [
                    str(chunk.get("chunk_id", ""))
                    for chunk in generation_chunks
                    if str(chunk.get("chunk_id", "")).strip()
                ],
                "sources": result.get("sources", []),
            }
        )
    return records


def run_answer(
    question: str,
    question_number: int,
    prompt_version: str = DEFAULT_EVAL_PROMPT_VERSION,
    answer_model: str = DEFAULT_OLLAMA_MODEL,
    generation_context_k: int | None = None,
) -> dict[str, Any]:
    try:
        result = answer(
            question,
            prompt_version=prompt_version,
            answer_model=answer_model,
            generation_context_k=generation_context_k,
        )
    except Exception as exc:
        raise RagasEvalError(f"answer() failed on question {question_number}: {exc}") from exc

    if not isinstance(result, dict):
        raise RagasEvalError(f"answer() returned {type(result).__name__}; expected dict.")

    retrieved_chunks = result.get("retrieved_chunks", [])
    generation_chunks = result.get("generation_chunks", [])
    if not isinstance(retrieved_chunks, list):
        raise RagasEvalError("answer() returned a non-list retrieved_chunks value.")
    if not isinstance(generation_chunks, list):
        raise RagasEvalError("answer() returned a non-list generation_chunks value.")

    return {
        "answer": str(result.get("answer", "")).strip(),
        "retrieved_chunks": retrieved_chunks,
        "generation_chunks": generation_chunks,
        "sources": result.get("sources", []),
    }


def require_package(package_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(package_name) is None:
        raise RagasEvalError(f"Missing optional dependency '{package_name}'. {install_hint}")


def build_evaluator_models(provider: str, model: str, embedding_model: str) -> EvaluatorModels:
    provider = normalized_provider(provider)
    require_package("ragas", "Install RAGAS dependencies before running this optional evaluator.")

    if provider == "ollama":
        return build_ollama_models(model=model, embedding_model=embedding_model)
    if provider == "openai":
        return build_openai_models(model=model, embedding_model=embedding_model)
    raise RagasEvalError("Unsupported RAGAS_LLM_PROVIDER. Use 'ollama' or 'openai'.")


def build_ollama_models(model: str, embedding_model: str) -> EvaluatorModels:
    if os.getenv("OPENAI_API_KEY", "").strip():
        print("Ignoring OPENAI_API_KEY because RAGAS_LLM_PROVIDER=ollama")

    base_url = os.getenv("RAGAS_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip() or DEFAULT_OLLAMA_BASE_URL
    llm = create_ollama_llm(model=model, base_url=base_url)
    embeddings = create_local_embeddings(model_name=embedding_model)
    return EvaluatorModels(
        provider="ollama",
        model=model,
        llm=llm,
        embeddings=embeddings,
        ollama_base_url=base_url,
        ollama_json_mode=True,
        ollama_num_ctx=8192,
        ollama_num_predict=4096,
    )


def create_ollama_llm(model: str, base_url: str) -> Any:
    kwargs = {
        "model": model,
        "base_url": base_url,
        "temperature": 0,
        "format": "json",
        "num_ctx": 8192,
        "num_predict": 4096,
    }

    try:
        from langchain_ollama import OllamaLLM

        return OllamaLLM(**supported_kwargs(OllamaLLM, kwargs))
    except ImportError:
        warn("langchain_ollama is unavailable; falling back to langchain_community Ollama.")
    except Exception as exc:
        warn(f"Failed to initialize langchain_ollama.OllamaLLM: {exc}. Falling back to langchain_community Ollama.")

    try:
        from langchain_community.llms import Ollama
    except ImportError as exc:
        raise RagasEvalError("Ollama provider requires langchain_ollama or langchain_community.") from exc
    return Ollama(**supported_kwargs(Ollama, kwargs))


def supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def create_local_embeddings(model_name: str) -> Any:
    model_kwargs = {"local_files_only": True}
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RagasEvalError(
                "Local Ollama evaluation requires langchain_huggingface or langchain_community embeddings."
            ) from exc
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    ensure_embedding_interface(embeddings)
    return embeddings


def ensure_embedding_interface(embeddings: Any) -> None:
    missing = [
        method_name
        for method_name in ("embed_query", "embed_documents")
        if not hasattr(embeddings, method_name)
    ]
    if missing:
        raise RagasEvalError(
            "Embeddings object is incompatible with RAGAS answer_relevancy. "
            f"Missing methods: {', '.join(missing)}."
        )


def build_openai_models(model: str, embedding_model: str) -> EvaluatorModels:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RagasEvalError("OPENAI_API_KEY is required when RAGAS_LLM_PROVIDER=openai.")

    require_package("openai", "Install openai before running provider=openai.")
    from openai import OpenAI

    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    client = OpenAI(api_key=api_key, base_url=base_url)
    llm = llm_factory(model, provider="openai", client=client)
    embeddings = embedding_factory("openai", model=embedding_model, client=client)
    return EvaluatorModels(
        provider="openai",
        model=model,
        llm=llm,
        embeddings=embeddings,
    )


def parse_metric_names(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise RagasEvalError("--metrics must include at least one metric name.")
    return list(dict.fromkeys(names))


def load_metric_definitions(requested_names: list[str], llm: Any, embeddings: Any) -> list[MetricDefinition]:
    from ragas.metrics.base import Metric

    definitions: list[MetricDefinition] = []
    specs = metric_candidate_specs()
    for name in requested_names:
        if name not in SUPPORTED_METRICS:
            warn(f"Skipping unknown metric '{name}'.")
            continue
        metric = resolve_metric(name, specs[name], Metric)
        if metric is None:
            warn(f"Skipping unavailable RAGAS metric '{name}' in this installed version.")
            continue
        assign_metric_models(metric, llm=llm, embeddings=embeddings)
        definitions.append(MetricDefinition(public_name=name, metric=metric))

    if not definitions:
        raise RagasEvalError("No requested RAGAS metrics are available in this environment.")
    return definitions


def metric_candidate_specs() -> dict[str, list[tuple[str, str]]]:
    return {
        "faithfulness": [
            ("ragas.metrics._faithfulness", "faithfulness"),
            ("ragas.metrics.collections", "Faithfulness"),
            ("ragas.metrics", "faithfulness"),
            ("ragas.metrics", "Faithfulness"),
        ],
        "answer_relevancy": [
            ("ragas.metrics._answer_relevance", "answer_relevancy"),
            ("ragas.metrics.collections", "AnswerRelevancy"),
            ("ragas.metrics", "answer_relevancy"),
            ("ragas.metrics", "AnswerRelevancy"),
            ("ragas.metrics._answer_relevance", "response_relevancy"),
            ("ragas.metrics", "ResponseRelevancy"),
        ],
        "context_precision": [
            ("ragas.metrics._context_precision", "context_precision"),
            ("ragas.metrics.collections", "ContextPrecision"),
            ("ragas.metrics", "context_precision"),
            ("ragas.metrics", "ContextPrecision"),
        ],
        "context_recall": [
            ("ragas.metrics._context_recall", "context_recall"),
            ("ragas.metrics.collections", "ContextRecall"),
            ("ragas.metrics", "context_recall"),
            ("ragas.metrics", "ContextRecall"),
        ],
    }


def resolve_metric(name: str, candidates: list[tuple[str, str]], metric_base_class: type) -> Any | None:
    for module_name, attribute_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if not hasattr(module, attribute_name):
            continue
        metric = initialized_metric(getattr(module, attribute_name), metric_base_class)
        if metric is not None:
            return metric
    return None


def initialized_metric(candidate: Any, metric_base_class: type) -> Any | None:
    if isinstance(candidate, metric_base_class):
        return candidate
    if inspect.isclass(candidate):
        try:
            metric = candidate()
        except Exception:
            return None
        return metric if isinstance(metric, metric_base_class) else None
    if callable(candidate):
        try:
            metric = candidate()
        except Exception:
            return None
        return metric if isinstance(metric, metric_base_class) else None
    return None


def assign_metric_models(metric: Any, llm: Any, embeddings: Any) -> None:
    if hasattr(metric, "llm"):
        metric.llm = llm
    if hasattr(metric, "embeddings"):
        metric.embeddings = embeddings


def build_eval_dataset(records: list[dict[str, Any]]) -> tuple[Any, dict[str, str] | None]:
    try:
        from ragas import EvaluationDataset, SingleTurnSample

        samples = [
            SingleTurnSample(
                user_input=record["question"],
                response=record["response"],
                retrieved_contexts=record["retrieved_contexts"],
                reference=record["reference"],
            )
            for record in records
        ]
        return EvaluationDataset(samples=samples), None
    except ImportError:
        pass

    from datasets import Dataset

    dataset = Dataset.from_dict(
        {
            "question": [record["question"] for record in records],
            "answer": [record["answer"] for record in records],
            "contexts": [record["contexts"] for record in records],
            "ground_truth": [record["ground_truth"] for record in records],
        }
    )
    return dataset, None


def run_ragas(records: list[dict[str, Any]], models: EvaluatorModels, metrics: list[MetricDefinition]) -> list[dict[str, Any]]:
    from ragas import evaluate

    dataset, column_map = build_eval_dataset(records)
    metric_objects = [definition.metric for definition in metrics]
    kwargs = {
        "metrics": metric_objects,
        "llm": models.llm,
        "embeddings": models.embeddings,
    }
    if column_map is not None:
        kwargs["column_map"] = column_map

    result = evaluate(dataset, **kwargs)
    return result_rows(result)


def result_rows(ragas_result: Any) -> list[dict[str, Any]]:
    if hasattr(ragas_result, "to_pandas"):
        return ragas_result.to_pandas().to_dict(orient="records")
    if hasattr(ragas_result, "scores") and isinstance(ragas_result.scores, list):
        return [dict(row) for row in ragas_result.scores]
    if isinstance(ragas_result, list):
        return [dict(row) for row in ragas_result]
    if isinstance(ragas_result, dict):
        return [dict(ragas_result)]
    raise RagasEvalError("RAGAS returned an unsupported result format.")


def merge_records_with_scores(
    records: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    metric_names: list[str],
) -> list[dict[str, Any]]:
    if len(records) != len(score_rows):
        raise RagasEvalError(f"RAGAS returned {len(score_rows)} score rows for {len(records)} input records.")

    merged: list[dict[str, Any]] = []
    for record, scores in zip(records, score_rows):
        row = {
            "question_number": record["question_number"],
            "category": record["category"],
            "question": record["question"],
            "answer": record["answer"],
            "reference": record["reference"],
            "contexts": record["contexts"],
            "retrieved_contexts": record["retrieved_contexts"],
            "retrieved_context_count": len(record["retrieved_contexts"]),
            "expected_chunk_ids": record["expected_chunk_ids"],
            "retrieved_chunk_ids": record["retrieved_chunk_ids"],
            "sources": record["sources"],
        }
        for metric_name in metric_names:
            row[metric_name] = metric_value(scores, metric_name)
        merged.append(row)
    return merged


def metric_value(scores: dict[str, Any], metric_name: str) -> Any:
    aliases = [
        metric_name,
        metric_name.replace("answer_", "response_"),
        metric_name.replace("response_", "answer_"),
    ]
    for alias in aliases:
        if alias in scores:
            return scores[alias]
    return None


def summarize_scores(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    averages: dict[str, float | None] = {}
    for metric_name in metric_names:
        values = [
            float(row[metric_name])
            for row in rows
            if isinstance(row.get(metric_name), (int, float)) and not math.isnan(float(row[metric_name]))
        ]
        if rows and not values:
            warn(f"Metric {metric_name} returned only nan values, likely due to local evaluator JSON-format issues.")
        averages[metric_name] = sum(values) / len(values) if values else None
    return {
        "examples_evaluated": len(rows),
        "metric_averages": averages,
    }


def save_json_report(
    output_path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    config: EvaluationRunConfig,
    models: EvaluatorModels,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_config": {
            "gold_path": str(config.gold_path.resolve()),
            "start": config.start,
            "limit": config.limit,
            "output_dir": str(config.output_dir.resolve()),
            "output_stem": config.output_stem,
            "output_csv": str(config.output_csv.resolve()) if config.output_csv else None,
            "output_json": str(config.output_json.resolve()) if config.output_json else None,
            "llm_provider": models.provider,
            "llm_model": models.model,
            "ollama_base_url": models.ollama_base_url,
            "embedding_model": config.embedding_model,
            "metrics": config.metrics,
            "prompt_version": config.prompt_version,
            "answer_model": config.answer_model,
            "generation_context_k": config.generation_context_k,
        },
        "summary": summary,
        "results": rows,
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def save_csv_report(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [flatten_row(row) for row in rows]
    fieldnames = list(flattened[0].keys()) if flattened else []
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, list):
            flattened[key] = serialize_contexts_for_csv(value)
        else:
            flattened[key] = value
    return flattened


def serialize_contexts_for_csv(value: list[Any]) -> str:
    """Serialize list values for CSV output without escaping Hebrew text."""
    return json.dumps(value, ensure_ascii=False)


def output_stem_for_prompt_version(
    prompt_version: str,
    answer_model: str,
    generation_context_k: int | None = None,
) -> str:
    normalized_prompt = prompt_version.strip().lower()
    normalized_model = sanitize_model_name(answer_model)
    if normalized_prompt == "baseline":
        base_stem = "ragas_results"
    else:
        base_stem = f"ragas_results_{normalized_prompt}_{normalized_model}"
    if generation_context_k is not None:
        return f"{base_stem}_genctx{generation_context_k}"
    return base_stem


def print_startup_diagnostics(
    config: EvaluationRunConfig,
    models: EvaluatorModels,
    metrics: list[MetricDefinition],
    csv_output_path: Path,
    json_output_path: Path,
) -> None:
    print("RAGAS evaluation diagnostics")
    print("============================")
    print(f"provider: {models.provider}")
    print(f"model: {models.model}")
    print(f"prompt_version: {config.prompt_version}")
    print(f"answer_model: {config.answer_model}")
    print(
        "generation_context_k: "
        + ("all retrieved contexts" if config.generation_context_k is None else str(config.generation_context_k))
    )
    print(f"start: {config.start}")
    print(f"limit: {config.limit}")
    if models.ollama_base_url:
        print(f"base_url: {models.ollama_base_url}")
        print(f"ollama_json_mode: {'true' if models.ollama_json_mode else 'false'}")
        if models.ollama_num_ctx is not None:
            print(f"num_ctx: {models.ollama_num_ctx}")
        if models.ollama_num_predict is not None:
            print(f"num_predict: {models.ollama_num_predict}")
    print(f"embeddings class: {models.embeddings.__class__.__name__}")
    print(f"selected metrics: {', '.join(definition.public_name for definition in metrics)}")
    print(f"CSV output: {csv_output_path}")
    print(f"JSON output: {json_output_path}")
    print()


def build_run_config(args: argparse.Namespace) -> EvaluationRunConfig:
    resolved_answer_model = resolve_answer_model_name(args.answer_model)
    return EvaluationRunConfig(
        gold_path=args.gold_path.resolve(),
        start=args.start,
        limit=args.limit,
        output_dir=args.output_dir.resolve(),
        output_stem=output_stem_for_prompt_version(
            args.prompt_version,
            resolved_answer_model,
            args.generation_context_k,
        ),
        output_csv=args.output_csv.resolve() if args.output_csv else None,
        output_json=args.output_json.resolve() if args.output_json else None,
        metrics=args.metrics,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        prompt_version=args.prompt_version,
        answer_model=resolved_answer_model,
        generation_context_k=args.generation_context_k,
    )


def output_paths(
    output_dir: Path,
    output_stem: str,
    output_csv: Path | None = None,
    output_json: Path | None = None,
) -> tuple[Path, Path]:
    if output_csv is not None and output_json is not None:
        return output_csv.resolve(), output_json.resolve()

    resolved_output_dir = output_dir.resolve()
    return resolved_output_dir / f"{output_stem}.csv", resolved_output_dir / f"{output_stem}.json"


def execute_evaluation(config: EvaluationRunConfig, print_diagnostics: bool = True) -> EvaluationArtifacts:
    csv_output_path, json_output_path = output_paths(
        config.output_dir,
        config.output_stem,
        output_csv=config.output_csv,
        output_json=config.output_json,
    )
    try:
        examples = selected_examples(load_jsonl(config.gold_path), config.start, config.limit)
    except ValueError as exc:
        raise RagasEvalError(str(exc)) from exc
    if not examples:
        raise RagasEvalError(
            f"No examples found in {config.gold_path} for start={config.start}, limit={config.limit}."
        )

    records = build_records(
        examples,
        start_index=config.start,
        prompt_version=config.prompt_version,
        answer_model=config.answer_model,
        generation_context_k=config.generation_context_k,
    )
    models = build_evaluator_models(config.llm_provider, config.llm_model, config.embedding_model)
    metric_names = parse_metric_names(config.metrics)
    metrics = load_metric_definitions(metric_names, llm=models.llm, embeddings=models.embeddings)
    if print_diagnostics:
        print_startup_diagnostics(config, models, metrics, csv_output_path, json_output_path)

    score_rows = run_ragas(records, models, metrics)
    used_metric_names = [definition.public_name for definition in metrics]
    merged_rows = merge_records_with_scores(records, score_rows, used_metric_names)
    summary = summarize_scores(merged_rows, used_metric_names)
    save_json_report(json_output_path, merged_rows, summary, config, models)
    save_csv_report(csv_output_path, merged_rows)
    return EvaluationArtifacts(
        rows=merged_rows,
        summary=summary,
        models=models,
        metrics=metrics,
        csv_output_path=csv_output_path,
        json_output_path=json_output_path,
    )


def main() -> None:
    configure_stdout()
    print_openai_api_key_status(__file__)
    args = parse_args()
    config = build_run_config(args)

    try:
        artifacts = execute_evaluation(config)
    except RagasEvalError as exc:
        print(f"RAGAS evaluation could not run: {exc}", file=sys.stderr)
        sys.exit(1)

    print("RAGAS evaluation complete")
    print("=========================")
    print(f"Examples evaluated: {artifacts.summary['examples_evaluated']}")
    for metric_name, average in artifacts.summary["metric_averages"].items():
        print(f"{metric_name}: {'unavailable' if average is None else f'{average:.4f}'}")
    print(f"CSV saved to: {artifacts.csv_output_path}")
    print(f"JSON saved to: {artifacts.json_output_path}")
    if args.preview_results:
        print()
        print("Preview results")
        print("===============")
        print(f"answer_model: {config.answer_model}")
        print(f"prompt_version: {config.prompt_version}")
        print(
            "generation_context_k: "
            + ("all retrieved contexts" if config.generation_context_k is None else str(config.generation_context_k))
        )
        print()
        for row in artifacts.rows:
            print(f"question: {row.get('question', '')}")
            print(f"generated answer: {row.get('answer', '')}")
            print("retrieved contexts:")
            for index, context in enumerate(row.get("retrieved_contexts", []), start=1):
                print(f"Context {index}:")
                print(context)
            print()


if __name__ == "__main__":
    main()
