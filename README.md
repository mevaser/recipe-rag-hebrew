# Hebrew Recipe RAG Assistant

Custom Retrieval-Augmented Generation project for a Hebrew recipe corpus. The system is implemented manually in Python and supports Hebrew PDF/DOCX recipe documents, dense and lexical retrieval, hybrid ranking, grounded answer generation, and a structured evaluation workflow.

## Project Overview

This project answers Hebrew recipe questions from a local corpus. The main design goal is to keep the RAG pipeline transparent and debuggable rather than hiding it behind a framework wrapper.

The current work split the problem into separate layers:

- retrieval quality
- generation quality
- evaluator quality

That separation became important because low answer-evaluation scores were not caused by retrieval alone.

## Setup

```bash
pip install -r requirements.txt
```

## Data and Preprocessing Pipeline

Place recipe documents under:

```text
data/raw
```

Supported files:

- `.docx`
- `.pdf`

Ignored for now:

- `.doc`
- images
- videos
- OCR-only documents

Run the preprocessing pipeline:

```bash
python src/scan_corpus.py
python src/load_documents.py
python src/deduplicate_documents.py
python src/create_chunks.py
python src/build_index.py
python src/build_bm25_index.py
```

Key generated artifacts:

```text
data/processed/recipes_inventory.csv
data/processed/documents.json
data/processed/documents_dedup.json
data/processed/chunks.json
data/processed/index.faiss
data/processed/indexed_chunks.json
data/processed/bm25_index.pkl
```

## Retrieval Pipeline

The retrieval stack evolved through several controlled experiments:

| Retrieval setup | Hit@5 | Hit@20 | MRR |
| --- | ---: | ---: | ---: |
| Dense FAISS | 68% | 82% | 0.4857 |
| BM25 | 78% | 88% | 0.7155 |
| Weighted hybrid before metadata | 82% | 88% | 0.7248 |
| Metadata-aware weighted hybrid | 92% | 94% | 0.8412 |

Current best retrieval configuration:

- dense retrieval with `intfloat/multilingual-e5-small`
- BM25 lexical retrieval
- weighted hybrid fusion
- metadata-aware indexing through `indexed_text`
- optional lightweight reranking / boosting for diagnostic ranking improvements

Recommended retrieval command:

```bash
python eval/run_eval.py --mode hybrid --candidate-k 50 --rrf-k 30 --dense-weight 0.5 --bm25-weight 2.0
```

Current best retrieval metrics:

```text
Hit@1 = 78%
Hit@3 = 90%
Hit@5 = 92%
Hit@10 = 94%
Hit@20 = 94%
MRR = 0.8412
```

## Evaluation Methodology

The evaluation workflow is layered:

1. Retrieval was evaluated first on `eval/gold_set.jsonl`.
2. RAGAS was used as an automatic answer-quality signal.
3. Manual human review was added because Hebrew/RTL answer evaluation may be imperfect.
4. Small frozen-context experiments were used to isolate generation, context-ordering, and retrieval-side issues.

This project does not treat RAGAS as the only truth signal for Hebrew.

## RAGAS Results

Baseline RAGAS results:

```text
faithfulness = 0.3945
answer_relevancy = 0.5708
context_precision = 0.7278
context_recall = 0.8980
```

Interpretation:

- `context_recall` and `context_precision` were reasonably strong
- `faithfulness` and `answer_relevancy` were much weaker
- this suggested that answer failures were not caused by retrieval alone
- Hebrew evaluator limitations were also plausible

## Manual Human Review

Human vs RAGAS manual review was added over a selected subset of problematic and control rows.

Summary:

```text
reviewed rows = 15
average RAGAS faithfulness = 0.1429
average human faithfulness = 0.2692
average RAGAS answer relevancy = 0.2266
average human answer relevancy = 0.2000
answer correct but RAGAS faithfulness < 0.6 = 3
context contains answer but generated answer is incorrect = 7
context does not contain answer = 2
```

Main findings:

- some low RAGAS faithfulness rows were actually correct by human inspection
- several failures happened even when the answer was already present in the retrieved context
- some failures were true `missing_context` retrieval misses
- some failures were caused by context ordering, noisy contexts, or garbled Hebrew/RTL text

Important error categories that emerged:

- `missing_context`
- `prompt_problem`
- `correct_context_low_rank`
- `answer_not_direct`
- `answer_format_issue`
- `evaluator_hebrew_issue`

## Strict Prompt Experiment

The first controlled follow-up experiment tested a stricter answer prompt on frozen reviewed rows, without rerunning retrieval or RAGAS.

Strict prompt rules:

- answer only from provided contexts
- short direct answer
- no sources section
- no extra explanation unless the question asks why
- exact fallback when the answer is unsupported

Results:

```text
total rows = 6
strict_answer_correct:
  yes = 3
  partial = 1
  no = 2
strict_answer_better_than_original:
  yes = 4
  partial = 1
  no = 1
improvement rate = 0.8333
```

Conclusion:

- strict prompt significantly improved generation failures
- the answer format should become the default evaluation format
- low RAGAS scores were partly caused by avoidable generation behavior, not only by retrieval
- `strict_short_no_sources` is now the default prompt version for evaluation scripts
- the old prompt is still available through `--prompt-version baseline`
- evaluation scripts now auto-load the project `.env` file without overriding shell variables
- an optional OpenAI GPT backend is available for controlled comparison against the local answer model

## Context Ordering Experiment

The second controlled follow-up experiment tested different frozen context variants on the remaining problematic rows:

- `current_top_5`
- `top_1_only`
- `top_3_only`
- `oracle_expected_chunk_if_available`

Reviewed results:

```text
total reviewed rows = 35
manual_answer_correct:
  no = 27
  partial = 6
  yes = 2
manual_variant_better:
  no = 26
  yes = 5
  partial = 4
oracle unavailable questions:
  Q4, Q5, Q10, Q35, Q46
oracle available but still not fully correct:
  Q29, Q31, Q41
top_1_only better than current_top_5:
  Q31, Q41
top_3_only better than current_top_5:
  Q31, Q50
current_top_5 already correct:
  Q50
```

Interpretation:

- `missing_context` rows stayed unresolved when the expected chunk was not retrieved at all
- `top_1_only` is risky and should not become the default
- `top_3_only` looks more promising than `top_1_only`
- some oracle-available rows still failed, which points to Hebrew/RTL extraction or local model extraction limitations, not only ranking

## Current Conclusions

1. Retrieval improved substantially after the metadata-aware hybrid retrieval baseline.
2. Low RAGAS faithfulness and answer relevancy were not caused by retrieval alone.
3. Manual review and frozen-context experiments showed that the main remaining bottleneck is Hebrew/RTL answer generation quality, especially for the local model.
4. `strict_short_no_sources` is the default answer format for evaluation because it improved reviewed generation failures.
5. `generation_context_k` remains configurable, but `generation_context_k=3` is not part of the default local setup because it did not materially improve the local model.
6. Lightweight rerank was useful diagnostically, but no tuned configuration was stable enough to enable by default.
7. The final system candidate is intentionally conservative and is based on controlled evidence rather than on a single metric.

## Current Recommended Evaluation Setup

- retrieval: metadata-aware hybrid retrieval baseline
- prompt: `strict_short_no_sources`
- answer backend: local `qwen2.5:7b-instruct` via Ollama for offline/local demo
- optional higher-quality backend: `GPT-4.1-mini`
- `generation_context_k`: configurable, but not default
- lightweight rerank: diagnostic/configurable only, disabled by default

## Final Submission State

The project is packaged around a stable final candidate:

- retrieval: metadata-aware hybrid retrieval baseline
- prompt: `strict_short_no_sources`
- local backend: `qwen2.5:7b-instruct` via Ollama for offline/local demo
- optional higher-quality backend: `GPT-4.1-mini`
- `generation_context_k`: configurable, but not default
- lightweight rerank: diagnostic/configurable only, disabled by default

Further work remains optional and is documented in the final report and status documents as future improvements rather than as required unfinished tasks.

## How to Reproduce Evaluation Artifacts

Note: this section is optional historical/reproducibility material. It is not required for running the live demo or understanding the final submitted system.

Evaluation default prompt note:

- evaluation scripts now default to `strict_short_no_sources`
- the old prompt remains available with `--prompt-version baseline`
- this change is based on the strict prompt experiment
- evaluation scripts auto-load the project `.env` file
- the local answer backend remains the default backend
- the optional OpenAI backend is for controlled frozen-context comparison only
- `generation_context_k` can cap how many retrieved contexts are passed into answer generation without changing retrieval itself

Verify `.env` loading safely:

```bash
python -c "from src.env_utils import load_project_env; load_project_env(); import os; k=os.getenv('OPENAI_API_KEY'); print('OPENAI_API_KEY exists:', bool(k)); print('prefix:', k[:7] if k else None); print('length:', len(k) if k else 0)"
```

Lightweight smoke test:

```bash
python eval/run_ragas_eval.py --start 35 --limit 1 --metrics faithfulness --prompt-version strict_short_no_sources --preview-results
python eval/run_ragas_eval.py --start 35 --limit 1 --metrics faithfulness --generation-context-k 3 --preview-results
```

Retrieval evaluation:

```bash
python eval/run_eval.py --mode hybrid --candidate-k 50 --rrf-k 30 --dense-weight 0.5 --bm25-weight 2.0
```

Human vs RAGAS review artifacts:

```bash
python eval/create_human_ragas_review_set.py
python eval/fill_human_ragas_review_from_labels.py
python eval/summarize_human_ragas_review.py --input eval\human_vs_ragas_review_set_filled.csv --output eval\human_vs_ragas_review_summary_filled.csv
```

Strict prompt review artifacts:

```bash
python eval/run_strict_prompt_on_review_subset.py
python eval/fill_strict_prompt_review_from_labels.py
python eval/summarize_strict_prompt_review.py
```

Context ordering experiment artifacts:

```bash
python eval\build_context_ordering_experiment.py
python eval\run_strict_prompt_context_ordering_experiment.py --answer-backend local --answer-model qwen2.5:7b-instruct
python eval\run_strict_prompt_context_ordering_experiment.py --answer-backend openai --answer-model gpt-4.1-mini
python eval\fill_context_ordering_review_from_labels.py
python eval\summarize_context_ordering_review.py
```

Focused local vs OpenAI comparison:

```bash
python eval/compare_local_vs_openai_generation.py --openai-model gpt-4.1-mini
python eval\fill_openai_comparison_review_from_labels.py
python eval\summarize_openai_comparison_review.py
```

Frozen-context `generation_context_k` comparison:

```bash
python eval\test_generation_context_k.py --answer-backend local --answer-model qwen2.5:7b-instruct --generation-context-k 3
python eval\test_generation_context_k.py --answer-backend openai --answer-model gpt-4.1-mini --generation-context-k 3
python eval\fill_generation_context_k_review_from_labels.py
python eval\summarize_generation_context_k_review.py
```

Lightweight rerank diagnostic:

```bash
python eval\test_lightweight_rerank.py
```

Filled comparison outputs:

- `eval/local_vs_openai_generation_comparison_filled.csv`
- `eval/local_vs_openai_generation_comparison_summary_filled.csv`
- `eval/openai_backend_comparison_findings.md`
- `eval/generation_context_k_comparison_filled.csv`
- `eval/generation_context_k_comparison_summary_filled.csv`
- `eval/generation_context_k_findings.md`
- `eval/lightweight_rerank_comparison.csv`
- `eval/lightweight_rerank_findings.md`

Additional artifact helpers:

```bash
python eval\fill_human_ragas_review_from_labels.py
python eval\summarize_human_ragas_review.py
python eval\fill_strict_prompt_review_from_labels.py
python eval\summarize_strict_prompt_review.py
python eval\build_context_ordering_experiment.py
python eval\run_strict_prompt_context_ordering_experiment.py
python eval\fill_context_ordering_review_from_labels.py
python eval\summarize_context_ordering_review.py
```

## Project Files

Key documentation and artifact summaries:

- `report/final_report.md`
- `report/presentation_script.md`
- `report/presentation_bullets.md`
- `report/live_demo_plan.md`
- `report/final_report_outline.md`
- `report/report.md`
- `docs/current_rag_status.md`
- `docs/remaining_tasks.md`
- `eval/human_vs_ragas_findings_template.md`
- `eval/strict_prompt_review_findings.md`
- `eval/context_ordering_experiment_plan.md`
- `eval/context_ordering_review_findings.md`

## Live Demo

For the presentation demo, see:

- `report/live_demo_plan.md`
- `demo/live_demo_notebook.ipynb`
- `demo/run_demo_query.py`

Recommended usage:

- Open `demo/live_demo_notebook.ipynb` in VS Code.
- Run the demo cell.
- CLI equivalent command:
  `python demo\run_demo_query.py --question "מה כמות הקמח, המים והשמרים בבצק הפיצה?"`
- The notebook cell explicitly switches to the project root before calling `demo/run_demo_query.py`.
- Running `demo/run_demo_query.py` also creates `demo/demo_output.html` as a readable HTML backup.
- If Hebrew terminal output is not readable, display `demo/demo_output.html` inside the notebook or open it in the browser.
- The demo does not require running RAGAS, OpenAI, full evaluation, index rebuild, or chunk rebuild.
