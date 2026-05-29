# Current RAG Status

## Short Summary

The RAG system has strong retrieval metrics after metadata-aware hybrid retrieval, but answer quality still depends on prompt quality, context ordering, and model extraction ability in Hebrew.

## Current Quality by Layer

- Retrieval: strong overall, but `missing_context` remains for specific rows.
- Generation: improved significantly with the strict prompt.
- Context ordering: `generation_context_k` remains configurable, but limiting the local model to `top_3` did not materially improve answer quality.
- Model extraction: the local model still struggles on some noisy Hebrew/RTL contexts.
- RAGAS: useful as a proxy metric, but not sufficient alone for Hebrew answer evaluation.
- Environment loading: evaluation scripts now load the project `.env` file automatically without overriding shell variables.

## Post-Evaluation Status

- metadata-aware hybrid retrieval baseline
- `strict_short_no_sources` prompt
- local `qwen2.5:7b-instruct` via Ollama for offline/local demo
- optional `GPT-4.1-mini` backend for higher answer quality

Evaluation note:

- `strict_short_no_sources` is now the default prompt version for evaluation scripts
- the old evaluation prompt is still available through `--prompt-version baseline`
- this change was made after the strict prompt experiment improved most reviewed generation failures
- evaluation scripts can now use an optional OpenAI answer backend for controlled frozen-context comparison
- the local backend remains the default backend
- the GPT comparison path is meant to test whether some failures come from local Hebrew extraction weakness rather than retrieval alone
- `generation_context_k` was added as a configurable evaluation option
- the `generation_context_k=3` experiment did not materially improve the local model, so the option remains configurable but is not part of the default local setup
- this does not change retrieval itself

## Known Remaining Issues

- `Q4`, `Q5`, `Q10`, `Q35`, `Q46`: expected chunk exists, but it ranked too low in the evaluated retrieval path
- `Q29`, `Q31`, `Q41`: oracle available but the local model still failed or only partially succeeded
- `Q50`: `top_3` and `current_top_5` worked, but `top_1` failed

## Local vs OpenAI Generation Comparison

GPT-4.1-mini was tested on frozen contexts with the same strict prompt.

- OpenAI improved `8/12` rows.
- OpenAI partially improved `2/12` rows.
- OpenAI did not improve `2/12` rows.
- The strongest improvements were `Q31` and `Q41`.
- This supports the conclusion that the local model is one bottleneck for Hebrew/RTL answer extraction.
- GPT does not solve `missing_context` cases because it cannot recover chunks that were never retrieved.

## Generation Context K Experiment

`generation_context_k=3` was tested with the local model.

- It produced `0 yes`, `1 partial`, and `15 no`.
- The result suggests that reducing contexts alone does not fix local model Hebrew/RTL extraction issues.
- The option remains configurable, but it is not part of the default local setup.

## Missing Context Investigation

`Q4`, `Q5`, `Q10`, `Q35`, and `Q46` were inspected as apparent missing-context failures. The diagnostic showed that their expected chunks do exist in processed chunks, but they rank too low in the evaluated retrieval path. That shifts the next fix from chunk coverage to retrieval ranking.

## Lightweight Reranking Candidate

The missing-context investigation showed that the expected chunks exist but are ranked too low. Lightweight reranking / boosting was tested as a retrieval-side improvement candidate. It is configurable and does not change embeddings or chunking.

## Lightweight Rerank Tuning

The initial rerank pass improved some rows, but it also worsened `Q41`. Tuning confirmed that no safe default configuration was found, so rerank remains diagnostic-only and disabled by default.

## Final Candidate Status

The current recommended candidate is:

- retrieval: metadata-aware hybrid retrieval baseline
- prompt: `strict_short_no_sources`
- answer backend: local `qwen2.5:7b-instruct` via Ollama for offline/local demo, with `GPT-4.1-mini` as an optional higher-quality backend
- `generation_context_k`: configurable, but not the default for the local model
- lightweight rerank: disabled by default because the tuned configurations did not pass the Q41/Q50 safety guardrails

## Optional Future Directions

1. Decide whether to use GPT API as a final high-quality backend or only as an analysis backend.
2. Optionally evaluate a neural reranker.
3. Improve Hebrew/RTL text normalization.
4. Run a small final RAGAS subset only if needed.
5. Prepare final presentation delivery materials as needed.
