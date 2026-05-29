# Hebrew Recipe RAG Evaluation Report

This file is the docs copy of the main evaluation report.

The primary detailed report is:

- `report/report.md`

Use this docs entrypoint when you want the evaluation report to remain visible alongside:

- `docs/current_rag_status.md`
- `docs/remaining_tasks.md`

Current high-level conclusions:

- metadata-aware weighted hybrid retrieval is the best retrieval setup so far
- strict prompt improved generation quality substantially
- remaining issues are mainly missing chunks, context ordering, and Hebrew/RTL extraction quality
- a small controlled subset should be used for future RAGAS reruns after fixes
