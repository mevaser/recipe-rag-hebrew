# Presentation Bullets

- Project goal:
  Build a Hebrew recipe RAG system that answers questions from PDF and DOCX recipe documents.

- Main challenge:
  Hebrew/RTL extraction noise, many similar recipes, and answer-generation failures even when the right context is retrieved.

- Architecture:
  Documents -> chunks -> embeddings -> dense retrieval + BM25 -> hybrid retrieval -> strict prompt -> answer generation -> evaluation.

- Main evaluation results:
  Metadata-aware hybrid retrieval reached Hit@5 92%, Hit@20 94%, MRR 0.8412.
  RAGAS context recall was high at 0.8980, but generation-oriented metrics were weaker.

- What failed:
  Local generation often failed on Hebrew/RTL extraction.
  `generation_context_k=3` did not materially help.
  Lightweight rerank improved some rows but was not stable enough.

- What worked:
  Metadata-aware hybrid retrieval improved retrieval strongly.
  `strict_short_no_sources` improved reviewed failures.
  GPT-4.1-mini improved 8/12 frozen-context rows and partially improved 2/12 more.

- Final system candidate:
  Metadata-aware hybrid retrieval baseline, `strict_short_no_sources`, local qwen2.5 backend for offline demo, optional GPT-4.1-mini backend, rerank disabled by default.

- Rerank status:
  Lightweight rerank was useful diagnostically but disabled by default because no tuned config was stable enough.

- Future work:
  Neural reranker, Hebrew/RTL normalization, stronger Hebrew answer model, larger human review, stronger evaluator setup.
