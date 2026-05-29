# Final Report Review Notes

## Review Goal
Review `report/final_report.md` for clarity, consistency, and course-submission quality without changing the technical conclusions or introducing new results.

## Changes Made
- Tightened wording to improve academic and professional tone.
- Reduced repetition across sections discussing evaluation logic and conclusions.
- Made the final recommended candidate more explicit in the overview and recommendation sections.
- Improved consistency of terms such as "hybrid baseline," "local backend," "optional high-quality backend," and "lightweight rerank."
- Kept all reported numbers aligned with the documented evaluation artifacts already present in the project.

## Numeric Consistency Check
The report remains consistent with the documented project artifacts for:

- retrieval comparison results
- baseline RAGAS metrics
- manual human review counts
- strict prompt experiment results
- local vs OpenAI frozen-context comparison
- `generation_context_k=3` results
- missing-context investigation summary
- lightweight rerank tuning conclusion

## Potential Issues Found
- Some source findings files contain minor encoding artifacts in terminal display, but the final report itself was kept in clean English ASCII punctuation where needed.
- The human review subset is intentionally small, so the report correctly presents its findings as diagnostic rather than fully conclusive.
- The optional GPT backend is clearly stronger in the frozen-context comparison, but the report appropriately avoids overstating it as the only acceptable final backend.

## Overall Assessment
The report is now suitable as a concise final project submission draft. It presents a coherent engineering story: strong retrieval development, careful diagnosis of answer-quality failures, controlled experiments, and a justified final recommended system candidate.
