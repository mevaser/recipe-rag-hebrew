# Remaining Tasks

Completed / in progress:

- `strict_short_no_sources` is now the default evaluation prompt.
- Project `.env` loading was added for evaluation scripts.
- Optional GPT API answer backend support was added for controlled comparison.
- Initial local-vs-OpenAI comparison on frozen contexts was completed.
- `generation_context_k` support was added.
- Initial local `generation_context_k=3` comparison was completed.
- Lightweight metadata/source/BM25 reranking was evaluated as a diagnostic option.
- Lightweight rerank weight tuning and guardrails were completed.
- Lightweight rerank tuning was completed and no safe default config was found.
- Final report structure preparation was completed.
- Final report draft was completed.

Remaining:

1. Decide whether to use GPT API for final evaluation or only as analysis.
2. Optional: evaluate a neural reranker.
3. Add Hebrew/RTL text normalization.
4. Run a small final RAGAS subset only if needed.
5. Prepare the final report/presentation.

Artifact note:
Evaluation artifacts were organized using inventory and archive scripts. No files are permanently deleted; old experimental files are moved to `eval/_archive`.

Evaluation prompt note:
`strict_short_no_sources` is now the default prompt version for evaluation scripts. The old prompt is still available through `--prompt-version baseline`.

Environment note:
Evaluation scripts now load the project `.env` file automatically. No files are permanently deleted; old experimental files are moved to `eval/_archive`.
