# OpenAI Backend Comparison Findings

## Goal
Compare the local model against GPT-4.1-mini using the same frozen contexts and the same strict prompt.

## Method
No RAGAS was used and no retrieval was rerun. The comparison used frozen context variants from:
`eval/context_ordering_experiment_dataset.csv`

## Reviewed Questions
- Q29
- Q31
- Q41
- Q50

## Results Summary
- total reviewed rows: 12
- OpenAI better:
  - yes = 8
  - partial = 2
  - no = 2

## Main Findings
1. OpenAI improved cases where the local model returned "not enough information" despite relevant context.
2. OpenAI produced cleaner Hebrew answers for noisy RTL/mixed-language contexts.
3. OpenAI improved Q31 strongly by extracting 200°C and about 20 minutes.
4. OpenAI improved Q41 by extracting the health/aesthetic reasons more clearly.
5. Q50 shows that when the task is simple and context is clean, both models can answer correctly.
6. In Q50 oracle, OpenAI corrected the local model's wrong yeast quantity.

## Interpretation
The comparison suggests that part of the remaining quality gap is caused by local model answer extraction limitations, especially in Hebrew/RTL contexts. Retrieval is not the only issue.

## Impact on Next Steps
Recommended next steps:
1. Keep local as default for cost/control unless project quality requires GPT.
2. Use GPT API as an optional high-quality backend.
3. Continue testing generation_context_k=3.
4. Investigate missing_context rows separately because GPT cannot fix missing retrieved chunks.
5. Consider Hebrew/RTL text normalization.
