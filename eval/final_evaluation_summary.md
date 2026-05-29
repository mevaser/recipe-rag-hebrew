# Final Evaluation Summary

## Goal
Summarize the controlled evaluation work done for the Hebrew recipe RAG project.

## Final Recommended Candidate
- Retrieval: current hybrid baseline
- Prompt: `strict_short_no_sources`
- Answer backend:
  - local `qwen2.5:7b-instruct` for offline/local demo
  - `GPT-4.1-mini` as optional high-quality backend
- `generation_context_k`: configurable, not default for the local model
- lightweight rerank: configurable / diagnostic only, not default

## Key Results
- RAGAS baseline metrics:
  - faithfulness: `0.3945`
  - answer_relevancy: `0.5708`
  - context_precision: `0.7278`
  - context_recall: `0.8980`
- Human vs RAGAS review:
  - reviewed rows: `15`
  - answer correct but RAGAS faithfulness < 0.6: `3`
  - context contains answer but generated answer is incorrect: `7`
  - context does not contain answer: `2`
- Strict prompt subset:
  - `strict_answer_correct`: `yes=3`, `partial=1`, `no=2`
  - `strict_answer_better_than_original`: `yes=4`, `partial=1`, `no=1`
- Local vs OpenAI:
  - `yes=8`, `partial=2`, `no=2`
- `generation_context_k=3`:
  - `yes=0`, `partial=1`, `no=15`
- Missing-context investigation:
  - `expected_chunk_exists_but_ranked_low=5`
- Lightweight rerank tuning:
  - best config: `none`
  - rerank remains disabled by default

## Interpretation
Retrieval is relatively strong, especially after the hybrid baseline improvements. The main remaining weakness is generation quality in Hebrew/RTL with the local model. `GPT-4.1-mini` significantly improves frozen-context answer extraction, which shows that part of the quality gap comes from answer extraction rather than retrieval alone. Reranking is still a reasonable future direction, but the lightweight version was not stable enough to enable by default because it improved some rows while still harming control behavior. RAGAS remains useful as a proxy metric, but it should be interpreted carefully for Hebrew, especially alongside local evaluator JSON-format issues.

## Final Recommendation
- Keep the baseline hybrid retrieval.
- Keep `strict_short_no_sources` as the default prompt.
- Keep `GPT-4.1-mini` as an optional high-quality backend.
- Keep local `qwen2.5:7b-instruct` as the offline/local option with documented limitations.
- Keep lightweight rerank disabled unless explicitly used for diagnostics.

## Future Work
1. Neural reranker evaluation.
2. Hebrew/RTL normalization.
3. Better answer extraction model for Hebrew.
4. More robust RAGAS/evaluator setup.
5. Larger human evaluation set.
6. Metadata-aware retrieval improvements.
