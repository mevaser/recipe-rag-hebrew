# Generation Context K Findings

## Goal
Test whether limiting answer generation to the top 3 contexts improves answer quality while keeping retrieval unchanged.

## Method
No retrieval was rerun.
No RAGAS was used.
Frozen contexts from `eval/context_ordering_experiment_dataset.csv` were used.
Backend: local
Model: qwen2.5:7b-instruct
Compared full contexts vs generation_context_k=3.

## Results Summary
- total reviewed rows: 16
- genctx3_better_than_full:
  - partial = 1
  - no = 15
  - yes = 0

## Main Findings
1. generation_context_k=3 did not materially improve the local model.
2. Q31 current_top_5 showed a partial improvement because genctx3 avoided a distracting unrelated context.
3. Most rows were unchanged.
4. Q41 top_3_only became worse because the local model produced mixed-language artifacts.
5. Q50 shows that if the relevant context is wrong or the model extracts the wrong number, genctx3 does not fix it.
6. The result supports the conclusion that local Hebrew/RTL answer extraction is a major bottleneck.

## Interpretation
generation_context_k=3 may still be useful as a configuration option, especially with a stronger model, but it should not be treated as a complete fix for local model failures.

## Impact on Next Steps
Recommended next steps:
1. Keep generation_context_k configurable.
2. Do not make generation_context_k=3 the default for the local model yet.
3. Test generation_context_k=3 with OpenAI only if needed.
4. Focus next on missing_context investigation and metadata/source boosting.
5. Consider Hebrew/RTL normalization.
