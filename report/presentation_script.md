# Presentation Script

## Opening
Hello, my project is a Hebrew recipe RAG system. The goal was to build a system that answers recipe questions from a local collection of PDF and DOCX recipe documents, while keeping the pipeline transparent and easy to evaluate.

## Problem
This problem is challenging for three main reasons. First, the documents are in Hebrew, so the system has to deal with right-to-left text and noisy extraction from PDFs. Second, the corpus contains many similar recipes, which makes retrieval ranking important. Third, even when the correct context is retrieved, answer generation can still fail if the model does not extract the relevant detail clearly.

## Architecture
The pipeline is straightforward. Documents are loaded, cleaned, and split into chunks. The chunks are embedded and indexed for dense retrieval. In parallel, BM25 provides lexical retrieval. These two signals are combined in a hybrid retriever. The retrieved contexts are passed into a strict prompt, and the model generates an answer only from those contexts. The system is then evaluated with retrieval metrics, RAGAS, manual review, and frozen-context experiments.

## Evaluation Story
The evaluation process was iterative and diagnostic. I started with a RAGAS baseline to get automatic answer-quality metrics. Then I added manual human review because Hebrew evaluation can be imperfect. After that, I ran several controlled experiments: a stricter prompt, a local-versus-GPT frozen-context comparison, a generation-context-size experiment, a missing-context investigation, and a lightweight reranking experiment.

## Key Results
On the retrieval side, the metadata-aware hybrid retrieval baseline reached Hit@5 of 92 percent, Hit@20 of 94 percent, and MRR of 0.8412. RAGAS context recall was high at 0.8980, but the generation-oriented metrics were weaker, especially faithfulness at 0.3945. In the frozen-context comparison, GPT-4.1-mini improved 8 out of 12 rows and partially improved 2 more, which showed that generation quality is a major bottleneck for the local model. The generation_context_k=3 experiment did not materially help the local model, and lightweight rerank was tested but not enabled by default because it was not stable enough.

## Final Candidate
The final recommended candidate keeps the metadata-aware hybrid retrieval baseline and uses the strict_short_no_sources prompt. For answer generation, the local qwen2.5 backend remains the default offline demo option, while GPT-4.1-mini is retained as an optional higher-quality backend. Lightweight rerank remains disabled by default and generation_context_k stays configurable, but not part of the default local setup.

## Conclusion
The main lesson from this project is that careful diagnosis matters. Instead of changing the architecture blindly, I used RAGAS, manual review, and controlled experiments to separate retrieval issues from generation issues. The final design decisions were based on evidence, not intuition, and that produced a clearer and more justified final system candidate.
