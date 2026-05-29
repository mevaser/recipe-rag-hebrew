# Hebrew Recipe RAG Evaluation Report

## 1. Goal

The goal of this project is to build and evaluate a Hebrew Retrieval-Augmented Generation system over local recipe documents. The system must retrieve relevant passages from Hebrew PDF and DOCX recipe sources and answer user questions about ingredients, quantities, temperatures, and preparation steps.

The evaluation goal is not only to measure final answer quality, but also to separate:

- retrieval quality
- generation quality
- evaluator quality

That separation became necessary because automatic scores alone did not explain all observed failures.

## 2. Retrieval Evaluation

Retrieval was evaluated first, independently from answer generation. The project moved through four main retrieval configurations:

| Retrieval setup | Hit@5 | Hit@20 | MRR |
| --- | ---: | ---: | ---: |
| Dense FAISS | 68% | 82% | 0.4857 |
| BM25 | 78% | 88% | 0.7155 |
| Weighted hybrid before metadata | 82% | 88% | 0.7248 |
| Metadata-aware weighted hybrid | 92% | 94% | 0.8412 |

Interpretation:

- dense retrieval alone was not strong enough for recipe-specific lexical questions
- BM25 improved exact lexical matching
- hybrid retrieval improved over dense and BM25 alone
- metadata-aware weighted hybrid became the strongest setup because recipe names and source metadata often contain critical signals that do not appear directly in the chunk text

Current best retrieval configuration:

```bash
python eval/run_eval.py --mode hybrid --candidate-k 50 --rrf-k 30 --dense-weight 0.5 --bm25-weight 2.0
```

This is the current best retrieval candidate for the project.

## 3. RAGAS Evaluation

Baseline RAGAS results:

```text
faithfulness = 0.3945
answer_relevancy = 0.5708
context_precision = 0.7278
context_recall = 0.8980
```

Interpretation:

- `context_precision` and `context_recall` suggested that retrieval was often reasonably good
- `faithfulness` and `answer_relevancy` were much weaker
- therefore, the bottleneck was not retrieval alone
- Hebrew and RTL evaluation quality may also have affected RAGAS scoring

These results motivated a manual human review layer rather than relying only on automatic evaluation.

## 4. Human vs RAGAS Review

Manual review was added because RAGAS may not be fully reliable for Hebrew recipe answers, especially when contexts are messy, partially garbled, or contain RTL extraction artifacts.

Manual review summary:

```text
reviewed rows = 15
average RAGAS faithfulness = 0.1429
average human faithfulness = 0.2692
average RAGAS answer relevancy = 0.2266
average human answer relevancy = 0.2000
answer correct but RAGAS faithfulness < 0.6 = 3
context contains answer but generated answer is incorrect = 7
context does not contain answer = 2
```

Main findings:

- some answers were correct by human inspection even when RAGAS faithfulness was low
- several failures occurred even when the answer already existed in the retrieved contexts
- some rows were true `missing_context` retrieval misses
- some failures came from prompt behavior, context ordering, or extraction noise rather than retrieval itself

Important failure categories:

- `missing_context`
- `prompt_problem`
- `correct_context_low_rank`
- `answer_not_direct`
- `answer_format_issue`
- `evaluator_hebrew_issue`

## 5. Strict Prompt Experiment

The next controlled experiment tested a stricter answer prompt on six frozen problematic rows, without rerunning retrieval and without RAGAS.

The strict prompt required:

- answer only from the provided contexts
- short direct answer
- no sources section
- no extra explanation unless the question asks why
- exact fallback when information is missing

Results:

```text
total rows = 6
strict_answer_correct:
  yes = 3
  partial = 1
  no = 2
strict_answer_better_than_original:
  yes = 4
  partial = 1
  no = 1
improvement rate = 0.8333
```

Interpretation:

- strict prompt significantly improved generation failures
- answer formatting and prompt discipline mattered a lot
- low answer quality was not only a retrieval problem

This experiment strongly suggests that `strict_short_no_sources` should become the default answer format for evaluation.

## 6. Context Ordering Experiment

The next experiment tested whether the remaining failures were caused by noisy contexts, bad ranking, or truly missing chunks.

Each target row was tested with four context variants:

- `current_top_5`
- `top_1_only`
- `top_3_only`
- `oracle_expected_chunk_if_available`

Reviewed results:

```text
total reviewed rows = 35
manual_answer_correct:
  no = 27
  partial = 6
  yes = 2
manual_variant_better:
  no = 26
  yes = 5
  partial = 4
oracle unavailable questions:
  Q4, Q5, Q10, Q35, Q46
oracle available but still not fully correct:
  Q29, Q31, Q41
top_1_only better than current_top_5:
  Q31, Q41
top_3_only better than current_top_5:
  Q31, Q50
current_top_5 already correct:
  Q50
```

Interpretation:

- `missing_context` rows remained unresolved when the expected chunk was not retrieved at all
- `top_1_only` can sometimes help, but it is risky and should not become the default
- `top_3_only` looks more promising than `top_1_only`
- some oracle-available rows still produced partial or bad answers, which suggests Hebrew/RTL text extraction or model extraction issues beyond ranking alone

## 7. Error Analysis

The current error profile is mixed rather than single-cause.

### missing_context

Rows such as `Q4`, `Q5`, `Q10`, `Q35`, and `Q46` still fail because the expected chunk was not retrieved at all. These are retrieval-side problems.

### prompt_problem

Some rows failed even when the answer was already in the context because the model answered too loosely, hallucinated, or fell back incorrectly.

### correct_context_low_rank

Rows such as `Q29` and `Q50` show that the answer can exist in retrieved contexts but appear too low or get drowned out by noise.

### answer_not_direct

Some answers used related information but did not directly answer the exact question being asked.

### answer_format_issue

Including a sources section or extra formatting can hurt evaluation even when the answer is basically correct.

### evaluator_hebrew_issue

Some human-correct answers still received poor RAGAS faithfulness scores, which suggests evaluator limitations on Hebrew or noisy RTL contexts.

### Hebrew/RTL text extraction issues

Some contexts and generated outputs show garbling, mixed-language artifacts, or noisy extraction. This is especially visible in rows like `Q31` and `Q41`, where the right facts are present but the answer is not extracted cleanly.

## 8. Conclusions

The project currently has strong retrieval quality under the metadata-aware weighted hybrid setup. However, final answer quality still depends heavily on:

- prompt discipline
- context ordering
- local model extraction quality on Hebrew/RTL text

The strongest current conclusions are:

1. retrieval improved significantly after metadata-aware weighted hybrid retrieval
2. low RAGAS faithfulness and answer relevancy were not caused by retrieval alone
3. strict prompt improved most generation failures
4. `top_1` should not become the default generation context size
5. `top_3` is a more promising next candidate for generation context size
6. RAGAS is useful as a proxy metric, but not sufficient alone for Hebrew answer evaluation

## 9. Next Steps

Priority next steps:

1. make `strict_short_no_sources` the default for evaluation
2. compare local model vs GPT API backend on frozen contexts
3. test `generation_context_k=3`
4. investigate `missing_context` rows `Q4`, `Q5`, `Q10`, `Q35`, `Q46`
5. add metadata/source boosting or lightweight reranking
6. normalize Hebrew/RTL text to reduce garbled chunks
7. rerun only a small RAGAS subset after controlled fixes

Overall, the current system is already strong on retrieval, but the remaining quality gains are likely to come from generation control, cleaner context selection, and better handling of noisy Hebrew/RTL text.
