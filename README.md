# Hebrew Recipe RAG Assistant

Custom Retrieval-Augmented Generation project for a Hebrew recipe corpus.

This project currently includes the project skeleton, corpus scanner, document loader, deduplication, chunk creation, dense and BM25 indexing, dense/BM25/hybrid retrieval, grounded answer generation with a local Ollama model, and retrieval plus manual answer evaluation.

## Setup

```bash
pip install -r requirements.txt
```

## Add Documents

Place recipe documents under:

```text
data/raw
```

The current scanner supports:

- `.docx`
- `.pdf`

Unsupported files are ignored for now, including `.doc`, images, videos, and temporary Word files.

## Run Scanner

```bash
python src/scan_corpus.py
```

The scanner recursively scans `data/raw`, prints a summary, and writes:

```text
data/processed/recipes_inventory.csv
```

## Load Documents

```bash
python src/load_documents.py
```

The loader extracts text from supported files in `data/raw` and writes:

```text
data/processed/documents.json
```

## Create Chunks

```bash
python src/create_chunks.py
```

Optional chunking commands:

```bash
python src/create_chunks.py --strategy full_document
python src/create_chunks.py --strategy fixed_size --chunk-size 300 --overlap 50
```

The chunking script reads `data/processed/documents_dedup.json` by default and writes:

```text
data/processed/chunks.json
```

## Deduplicate Documents

```bash
python src/deduplicate_documents.py
```

Optional threshold:

```bash
python src/deduplicate_documents.py --threshold 0.92
```

The deduplication step reads `data/processed/documents.json` and writes:

```text
data/processed/documents_dedup.json
data/processed/dedup_report.json
```

Recommended pipeline after document loading:

```bash
python src/load_documents.py
python src/deduplicate_documents.py
python src/create_chunks.py
python src/build_index.py
python src/build_bm25_index.py
python eval/run_eval.py --mode hybrid --candidate-k 50 --rrf-k 30 --dense-weight 0.5 --bm25-weight 2.0
```

## Build Index

```bash
python src/build_index.py
```

Optional index command:

```bash
python src/build_index.py --chunks-path data/processed/chunks.json --index-path data/processed/index.faiss --indexed-chunks-path data/processed/indexed_chunks.json --model-name intfloat/multilingual-e5-small --batch-size 32 --min-words 5
```

The index builder embeds chunks with `intfloat/multilingual-e5-small`, writes the FAISS index, and writes the filtered chunks used by the index:

```text
data/processed/index.faiss
data/processed/indexed_chunks.json
```

## Build BM25 Index

```bash
python src/build_bm25_index.py
```

The BM25 index is built over the same filtered chunk file used by FAISS:

```text
data/processed/indexed_chunks.json
data/processed/bm25_index.pkl
```

## Test Retrieval

```bash
python src/test_retrieval.py
```

Example retrieval commands:

```bash
python src/test_retrieval.py --query "איזה מתכון מתאים ללחם ללא גלוטן?" --k 5
python src/test_retrieval.py --query "איך מכינים קובה?" --k 5
```

Retrieval depends on:

```text
data/processed/index.faiss
data/processed/indexed_chunks.json
```

## Inspect Chunks For Gold Set

Use this helper to find real chunk IDs while manually building `eval/gold_set.jsonl`:

```bash
python src/inspect_chunks.py --limit 20
python src/inspect_chunks.py --category "קובה סלק" --limit 10
python src/inspect_chunks.py --contains "תפוחי אדמה" --limit 10
python src/inspect_chunks.py --source "לחם ללא גלוטן" --limit 10
```

Save selected previews to JSON:

```bash
python src/inspect_chunks.py --contains "קובה" --limit 5 --output eval/debug/inspect_kuba.json
```

## Run Retrieval Evaluation

Evaluate `eval/gold_set.jsonl` with dense retrieval by default:

```bash
python eval/run_eval.py
python eval/run_eval.py --gold-path eval/gold_set.jsonl --mode dense
```

Compare dense, BM25, and hybrid retrieval:

```bash
python eval/run_eval.py --mode dense
python eval/run_eval.py --mode bm25
python eval/run_eval.py --mode hybrid
python eval/run_eval.py --mode hybrid --candidate-k 30 --rrf-k 60
python eval/run_eval.py --mode hybrid --candidate-k 50 --rrf-k 60 --dense-weight 0.5 --bm25-weight 2.0
```

Current recommended retrieval configuration:

```bash
python eval/run_eval.py --mode hybrid --candidate-k 50 --rrf-k 30 --dense-weight 0.5 --bm25-weight 2.0
```

Current recommended retrieval metrics:

```text
Hit@1: 78%
Hit@3: 90%
Hit@5: 92%
Hit@10: 94%
Hit@20: 94%
MRR: 0.8412
```

Detailed results are saved to:

```text
eval/eval_results.json
eval/misses_at_5.json
eval/misses_at_10.json
eval/misses_at_20.json
```

Strict Hit@K remains the primary retrieval metric: a result is correct only when the exact expected `chunk_id` appears in the retrieved top K. The evaluator also reports source Hit@K and neighbor Hit@K as diagnostic metrics. Source Hit@K checks whether retrieval found the same source document, and neighbor Hit@K checks whether retrieval found the same source within one adjacent chunk. These diagnostic metrics help identify chunk-boundary or overly strict gold-label issues without changing the main score.

Run selected gold-set questions through the full answer pipeline for manual answer-quality inspection:

```bash
python eval/manual_answer_eval.py --question-numbers 1,6,7,9,13,14,20,30,40,49
```

The manual answer helper uses the current recommended hybrid retrieval configuration by default and writes:

```text
eval/manual_answer_eval.json
eval/manual_answer_eval.md
```

Latest answer-generation status:

- Retrieval improved after metadata-aware indexing: Hybrid Hit@5 is `92%`, Hybrid Hit@20 is `94%`, and Hybrid MRR is `0.8412`.
- Manual answer evaluation improved from `correct 0 / partial 3 / incorrect 7` before the generation and context fixes to `correct 8 / partial 2 / incorrect 0` after the fixes.
- Generation now uses a Hebrew-only grounded prompt, source-neighbor context expansion, recipe-specific context filtering, and a strict fallback when the answer is not supported by the retrieved context.

## Experiment History

### Summary Table

| Stage | Change | Hit@5 | Hit@20 | MRR | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Dense baseline after gold/dedup fixes | Dense FAISS | 68% | 82% | 0.4857 | ranking and recall limitations |
| BM25 | BM25 over text | 78% | 88% | 0.7155 | strong lexical matching |
| Weighted hybrid before metadata | Dense + BM25 + weighted RRF | 82% | 88% | 0.7248 | best pre-metadata setup |
| Metadata-aware weighted hybrid | `indexed_text` + hybrid | 92% | 94% | 0.8412 | current best |

### Development Log

1. Initial dense retrieval baseline
   - Embedding model: `intfloat/multilingual-e5-small`
   - Retrieval stack: FAISS dense retrieval over chunk text
   - Early result after dedup and initial gold fixes: Hit@5 was around `66%`, Hit@10 was around `78%`
   - Main issue: relevant chunks often appeared below top 5, and process questions were weak

2. Ranking diagnostics
   - Added Hit@1, Hit@3, Hit@5, Hit@10, and Hit@20
   - Added MRR and average first relevant rank
   - Main finding: dense retrieval often found the right material by top 10, but ranking quality was weak
   - Hit@20 around `80%` to `82%` showed that some real recall failures still remained

3. Failure analysis
   - Added `eval/analyze_misses.py`
   - Inspected `misses_at_20` rather than only aggregate scores
   - Found a mix of strict chunk-ID issues and real retrieval failures
   - Important examples:
     - `אינגריי` was not retrieved
     - the bread recipe was confused with general bread and gluten guides
     - some expected chunks were adjacent to retrieved chunks rather than exact matches

4. Gold set cleanup
   - Fixed weak or invalid questions in `eval/gold_set.jsonl`
   - Repaired JSONL formatting issues
   - Cleaned the `eval` folder structure:
     - active files stay in `eval/`
     - archived drafts and candidate files moved under `eval/archive`
     - inspect and debug outputs moved under `eval/debug`

5. Hybrid retrieval
   - Added BM25 retrieval with `rank-bm25`
   - Added dense + BM25 fusion with Reciprocal Rank Fusion
   - Result: BM25 outperformed dense on exact recipe names, ingredients, quantities, and other lexical questions
   - Hybrid improved over dense, but the first untuned version still needed parameter work

6. Weighted hybrid tuning
   - Added `dense_weight` and `bm25_weight`
   - Best configuration before metadata-aware indexing:
     - `python eval/run_eval.py --mode hybrid --candidate-k 50 --rrf-k 30 --dense-weight 0.5 --bm25-weight 2.0`
   - Result before metadata-aware indexing:
     - Hit@5: `82%`
     - Hit@20: `88%`
     - MRR: `0.7248`

7. Source and neighbor diagnostic metrics
   - Added strict, source-aware, and neighbor-aware Hit@K
   - Main finding: strict Hit@20 was lower than source and neighbor Hit@20
   - This showed that many misses were chunk-boundary problems rather than full retrieval failure
   - Source and neighbor metrics made it clear that retrieval was often close to the right answer even when strict chunk-ID scoring said miss

8. Manual answer evaluation
   - Added `eval/manual_answer_eval.py`
   - Initial answer quality on 10 selected questions:
     - correct: `0`
     - partial: `3`
     - incorrect: `7`
   - Main generation problems:
     - hallucination and wrong-context usage
     - contradictory fallback behavior
     - Chinese, Arabic, and English leakage
     - answering from general guides instead of the specific recipe

9. Generation prompt and context fixes
   - Improved the grounded Hebrew-only prompt
   - Added strict fallback:
     - `לא מצאתי את המידע במקורות שנשלפו.`
   - Added source-neighbor context expansion
   - Added recipe-specific context filtering
   - Prevented unrelated sources such as hamburger and bolognese from dominating when the question explicitly named a recipe

10. Metadata-aware indexing
    - Diagnostics showed that rare recipe names such as `אינגריי` appeared in `metadata.source` and `metadata.category`, but not in `chunk["text"]`
    - Added `indexed_text` composed from:
      - source filename
      - category
      - original chunk text
    - Dense and BM25 now index `indexed_text` when available, while answer generation still uses the original `text`
    - Final retrieval metrics:
      - Hybrid Hit@1: `78%`
      - Hybrid Hit@3: `90%`
      - Hybrid Hit@5: `92%`
      - Hybrid Hit@10: `94%`
      - Hybrid Hit@20: `94%`
      - MRR: `0.8412`
      - `misses_at_20`: `3`

11. Final manual answer evaluation
    - After the generation and context fixes:
      - correct: `8`
      - partial: `2`
      - incorrect: `0`
    - Remaining partials:
      - Q6 and Q7 still fall back on gluten-free bread proofing questions
    - Known limitation:
      - general bread and gluten guide documents can still compete with the specific bread recipe
      - a future improvement would be stronger recipe-title matching or section-aware chunking

## Key Lessons

- Dense retrieval alone was not enough for Hebrew recipe QA.
- BM25 helped with exact recipe names, ingredients, quantities, and cooking times.
- Metadata was critical when recipe names existed in filenames and categories but not in chunk text.
- Strict chunk-ID metrics understated retrieval quality when answers spanned neighboring chunks.
- Answer generation had to be evaluated separately from retrieval.
- Safe fallback was better than hallucinating unsupported answers.

## Draft Gold Questions

Use Ollama to draft candidate gold-set questions from selected chunks. This writes a draft file only; manually review candidates before copying approved entries into `eval/gold_set.jsonl`.
The script asks for practical Hebrew recipe questions and rejects weak candidates such as empty answers, very short questions, invalid categories, excessive English, or broken text.

```bash
python src/draft_gold_questions.py --contains "קובה" --limit 10 --questions-per-chunk 2
python src/draft_gold_questions.py --contains "לחם ללא גלוטן" --limit 5 --questions-per-chunk 2
python src/draft_gold_questions.py --contains "חומוס" --limit 10 --questions-per-chunk 2
python src/draft_gold_questions.py --contains "פיצה" --limit 10 --questions-per-chunk 2
```

By default, the output file is overwritten. Use `--append` to add candidates to an existing JSONL file:

```bash
python src/draft_gold_questions.py --contains "חומוס" --limit 5 --questions-per-chunk 2 --output eval/archive/gold_candidates_kuba.jsonl --append
```

Draft candidates are saved to:

```text
eval/archive/gold_candidates_*.jsonl
```

Merge all topic-specific candidate files into one review set:

```bash
python src/merge_gold_candidates.py
```

The merger reads `eval/archive/gold_candidates_*.jsonl`, removes exact duplicates, and writes:
Archived candidate and review files are kept under `eval/archive` to keep the active evaluation root clean.

```text
eval/archive/gold_candidates_all.jsonl
eval/archive/gold_candidates_review.csv
eval/archive/gold_candidates_review.xlsx
```

Excel users should open `eval/archive/gold_candidates_review.xlsx` for manual review. The CSV is also written as UTF-8 with BOM (`utf-8-sig`) so Excel can detect Hebrew text more reliably.

## Ask A Question

Install and pull the local Ollama model:

```bash
ollama pull qwen2.5:7b-instruct
```

Ask with the default question:

```bash
python src/ask.py
```

Example question commands:

```bash
python src/ask.py --question "איך מכינים קובה?"
python src/ask.py --question "איזה מתכון מתאים ללחם ללא גלוטן?"
```

Optional `.env` file:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b-instruct
```

If Ollama is not running or the model is unavailable, the pipeline still runs and returns a grounded placeholder answer listing the retrieved sources instead of inventing an answer.
