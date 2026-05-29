# Strict Prompt Review Findings

## Goal

Test whether a stricter generation prompt improves cases where the retrieved context contains the answer but the original answer was incorrect.

## Manual Results

- total rows: `6`
- `strict_answer_correct`:
  - `yes = 3`
  - `partial = 1`
  - `no = 2`
- `strict_answer_better_than_original`:
  - `yes = 4`
  - `partial = 1`
  - `no = 1`
- improvement rate: `0.8333`

Improved rows:

- correct after strict prompt: `Q24`, `Q25`, `Q27`
- improved but still not fully solved: `Q31`, `Q41`

Still unresolved:

- `Q5`
- `Q31`

## Interpretation

The strict prompt improved most generation failures, especially when the answer was explicit in the first retrieved context. It reduced answer-format noise and forced more direct extraction from context. However, it did not solve all cases. Remaining failures likely require context ordering, top-k reduction, reranking, missing-chunk inspection, or better Hebrew/RTL extraction.

## Recommendation

1. Make `strict_short_no_sources` the default answer format for evaluation.
2. Do not run full RAGAS yet.
3. Continue with context-ordering and retrieval-side debugging on the remaining problematic rows.
