# Lightweight Rerank Tuning Findings

## Goal
Tune the rerank layer without changing embeddings, chunking, or the index.

## Why Tuning Was Needed
The first rerank improved Q5, Q10, and Q46 but worsened Q41, so the weights need guardrail-based tuning.

## Configurations Tested
- Config A: `bm25_only`
- Config B: `bm25_keyword`
- Config C: `conservative_source`
- Config D: `current_default`
- Config E: `no_rerank_baseline`

## Summary Table
| config_name | hit_at_1 | hit_at_3 | hit_at_5 | improved_count | worsened_count | avg_rank_delta | worsened_questions | q41_worsened | q50_rank_delta | recommended |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_only | 7 | 8 | 9 | 3 | 2 | 0.3333 | Q41,Q50 | yes | -1 | no |
| bm25_keyword | 7 | 9 | 9 | 3 | 1 | 0.4444 | Q41 | yes | 0 | no |
| conservative_source | 7 | 9 | 9 | 3 | 1 | 0.4444 | Q41 | yes | 0 | no |
| current_default | 7 | 9 | 9 | 3 | 1 | 0.4444 | Q41 | yes | 0 | no |
| no_rerank_baseline | 5 | 8 | 9 | 0 | 0 | 0.0 |  | no | 0 | no |

## Recommendation
No tested configuration was safe enough. Keep lightweight rerank diagnostic-only for now.

## Next Steps
- If a safe config exists, test it on a small generation subset.
- If no safe config exists, consider a neural reranker later.
- Continue with Hebrew/RTL normalization and the GPT backend decision.
