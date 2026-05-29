# Lightweight Rerank Findings

## Goal
Improve ranking for cases where expected chunks exist but are ranked too low.

## Method
No embeddings, chunking, or index rebuild was changed. The diagnostic compared baseline hybrid retrieval against an optional lightweight rerank layer that adds BM25/source/keyword boosts after hybrid fusion.

## Questions Tested
Q4, Q5, Q10, Q35, Q46 plus controls Q29, Q31, Q41, Q50.

## Results
- baseline Hit@5: 9/9
- reranked Hit@5: 9/9
- improved questions: Q5, Q10, Q46
- worsened questions: Q41

## Interpretation
The lightweight rerank layer shows whether simple BM25/source/keyword boosts can improve expected chunk ranking without changing embeddings or chunking.

## Next Steps
- If rerank improves Hit@5 without hurting controls, consider enabling it for evaluation.
- If rerank hurts controls, tune weights or keep it diagnostic only.
- If rerank is insufficient, consider neural reranker.
