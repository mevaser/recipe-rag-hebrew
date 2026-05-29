# Hebrew Recipe RAG System - Final Project Report

## 1. Project Overview
This project implemented a Hebrew Retrieval-Augmented Generation (RAG) system for answering recipe questions from a local document collection. The main goal was to build a transparent and debuggable pipeline that could retrieve relevant recipe content and produce grounded answers in Hebrew. A central engineering objective was to separate retrieval problems from generation problems instead of treating the system as a single black box.

The work therefore focused not only on improving retrieval quality, but also on explaining why answer quality sometimes remained weak even when relevant context had already been retrieved. This led to a structured evaluation process based on measurement, diagnosis, and controlled follow-up experiments. The final recommended candidate keeps the strongest validated metadata-aware hybrid retrieval baseline, uses the `strict_short_no_sources` prompt, retains the local model for offline demonstration, and treats `GPT-4.1-mini` as an optional higher-quality backend.

## 2. Dataset and Document Processing
The corpus consists of Hebrew recipe documents stored as PDF and DOCX files. These documents include recipes, preparation instructions, ingredient lists, and cooking notes from multiple sources and formats. Because the corpus is in Hebrew, the system had to preserve right-to-left text and source information throughout processing.

The preprocessing pipeline scans the corpus, loads supported documents, cleans text conservatively, and splits content into processed chunks. Each chunk retains metadata such as source name, relative path, file type, category, and other document-level information when available. The processed artifacts include document inventories, cleaned document files, chunk files, and retrieval indexes.

Hebrew and RTL content create additional challenges compared with English-only corpora. Text extraction can introduce noisy formatting, mixed-language artifacts, or directionality issues. These issues can affect both retrieval ranking and answer generation, especially when the local generation model must extract short factual answers from messy context.

## 3. System Architecture
The system was implemented as a manual Python RAG pipeline rather than as a framework wrapper. The architecture includes the following stages:

- document loading from PDF and DOCX files
- cleaning and normalization while preserving Hebrew content
- chunk creation and metadata attachment
- embedding and FAISS indexing for dense retrieval
- BM25 indexing for lexical retrieval
- hybrid retrieval that combines dense and lexical evidence
- prompt construction over retrieved contexts
- answer generation using a strict grounded prompt
- evaluation through retrieval metrics, RAGAS, manual review, and frozen-context experiments

Dense retrieval is based on multilingual sentence embeddings, while BM25 provides a lexical retrieval path that is often helpful for exact recipe wording and source-specific matching. Hybrid retrieval combines both signals and became the strongest retrieval baseline in the project. Generation then operates over the retrieved contexts and is expected to answer only from those contexts.

## 4. Retrieval Development
The retrieval pipeline evolved through several stages. The initial baseline used dense FAISS retrieval only. That baseline worked reasonably well, but lexical matching was often stronger for recipe questions that depended on exact wording, ingredient terms, or source-specific phrasing.

BM25 was then evaluated as a lexical alternative and outperformed dense retrieval on the project's gold retrieval set. Next, a weighted hybrid approach combined dense and BM25 evidence. Finally, metadata-aware indexing improved the representation of source and document information, which further strengthened the hybrid setup.

The final documented retrieval results were:

| Retrieval Setup | Hit@5 | Hit@20 | MRR |
| --- | ---: | ---: | ---: |
| Dense FAISS | 68% | 82% | 0.4857 |
| BM25 | 78% | 88% | 0.7155 |
| Weighted hybrid | 82% | 88% | 0.7248 |
| Metadata-aware weighted hybrid | 92% | 94% | 0.8412 |

These results show that retrieval became relatively strong after hybridization and metadata-aware improvements. Later experiments confirmed that many remaining answer-quality failures were not caused by missing data alone.

## 5. Evaluation Methodology
Evaluation was performed in multiple layers because a single metric was not sufficient to diagnose system behavior.

First, retrieval quality was measured with a gold set of questions and expected chunks. This made it possible to verify whether the correct evidence was retrieved before analyzing answer generation. Second, RAGAS was used to measure automatic answer-quality metrics such as faithfulness and answer relevancy. Third, a manual human review set was created to inspect cases where RAGAS and human judgment might disagree, especially for Hebrew answers.

Additional frozen-context experiments were used to isolate generation from retrieval. In those experiments, the retrieved contexts were held fixed while prompts, backends, or context subsets were changed. This made it possible to test whether failures came from retrieval, prompt design, context ordering, or answer extraction quality. This layered methodology was important because the system's weaknesses turned out to be distributed across multiple components rather than concentrated in a single stage.

## 6. RAGAS Results
The baseline RAGAS results were:

- faithfulness: 0.3945
- answer_relevancy: 0.5708
- context_precision: 0.7278
- context_recall: 0.8980

These numbers suggested an important pattern. The retrieval-oriented metrics, especially context recall and context precision, were reasonably strong. In contrast, the generation-oriented metrics, especially faithfulness, were much weaker. This gap indicated that the system's remaining problems could not be explained by retrieval quality alone. It also raised the possibility that the evaluator itself was imperfect for Hebrew, especially when combined with local-model JSON-format issues in evaluator outputs.

## 7. Manual Review and Diagnosis
To better understand the RAGAS results, a manual review was performed on a selected subset of rows.

The documented outcomes were:

- reviewed rows: 15
- average RAGAS faithfulness: 0.1429
- average human faithfulness: 0.2692
- average RAGAS answer relevancy: 0.2266
- average human answer relevancy: 0.2000
- answer correct but RAGAS faithfulness < 0.6: 3
- context contains answer but generated answer is incorrect: 7
- context does not contain answer: 2

This manual review was important for two reasons. First, it showed that some low RAGAS faithfulness scores were false negatives from a human perspective. Second, it showed that several failures were genuine generation failures: in multiple rows, the retrieved context already contained the correct answer, but the generated answer still failed to extract it correctly. The review therefore identified both evaluator limitations and real local-generation weaknesses.

## 8. Prompt Improvement
One of the first controlled interventions was a stricter generation prompt called `strict_short_no_sources`. Its purpose was to reduce answer noise and force the model to answer only from the provided contexts, without adding a sources section or extra explanations unless necessary.

On the reviewed strict-prompt subset, the results were:

- `strict_answer_correct`: yes=3, partial=1, no=2
- `strict_answer_better_than_original`: yes=4, partial=1, no=1

This experiment showed that prompt design mattered substantially. A cleaner and stricter format improved multiple generation failures without changing retrieval. As a result, `strict_short_no_sources` became the default prompt for evaluation flows. This was an important engineering result: some answer-quality problems were fixable through prompt control alone.

## 9. Local vs OpenAI Generation Comparison
A focused frozen-context comparison was then used to compare the local model with `GPT-4.1-mini` on exactly the same retrieved contexts and the same strict prompt.

The results were:

- GPT-4.1-mini improved 8/12 rows
- GPT-4.1-mini partially improved 2/12 rows
- GPT-4.1-mini did not improve 2/12 rows

Because retrieval was held fixed, this experiment isolated answer generation from retrieval quality. The results strongly suggested that generation quality, especially Hebrew answer extraction from noisy RTL contexts, is a major bottleneck for the local `qwen2.5:7b-instruct` backend. The local model remains useful for an offline or fully local demo, but it has documented limitations in extracting concise and accurate Hebrew answers from difficult contexts. `GPT-4.1-mini` performed significantly better in this controlled setting and is therefore retained as an optional higher-quality backend.

## 10. generation_context_k Experiment
Another hypothesis was that generation might improve if the model saw fewer contexts, reducing distraction from irrelevant lower-ranked results. This was tested using `generation_context_k=3`.

The documented outcomes were:

- yes=0
- partial=1
- no=15

This experiment showed that limiting answer generation to the top three contexts did not materially fix the local model. While one row showed a partial improvement, most rows were unchanged, and some remained garbled or incorrect. The conclusion was that reducing the number of contexts alone is not enough to solve local Hebrew/RTL answer extraction failures.

## 11. Missing Context Investigation
Five rows were investigated as possible missing-context retrieval failures:

- Q4
- Q5
- Q10
- Q35
- Q46

The investigation showed:

- `expected_chunk_exists_but_ranked_low = 5`

This means the expected chunks were present in the processed chunk collection and were not missing from the indexed data. The real issue was rank ordering. In other words, retrieval coverage existed, but the evaluated retrieval path did not always rank the expected chunk high enough in the most useful position. This distinction matters because it shifts the problem from data coverage to ranking behavior.

## 12. Lightweight Reranking
Because the expected chunks existed but were sometimes ranked too low, a lightweight reranking / boosting layer was tested. The idea was to use small explainable boosts based on BM25 ranking, source overlap, and keyword overlap without changing embeddings, chunking, or the index.

The first rerank attempt improved Q5, Q10, and Q46, but it also worsened Q41. A tuning phase then tested multiple rerank configurations with guardrails on control questions. The final result was:

- baseline Hit@5: 9/9
- reranked Hit@5: 9/9
- best config: none

No tested configuration was safe enough to enable by default. Although reranking showed diagnostic value, it also introduced instability on control questions. For this reason, lightweight rerank remains configurable and diagnostic-only, but disabled by default in the final recommendation.

## 13. Final Recommended Candidate
The final recommended system candidate is:

- Retrieval: metadata-aware hybrid retrieval baseline
- Prompt: `strict_short_no_sources`
- Local backend: `qwen2.5:7b-instruct` via Ollama for offline/local demo
- Optional high-quality backend: `GPT-4.1-mini`
- `generation_context_k`: configurable, not default for the local model
- lightweight rerank: diagnostic/configurable only, disabled by default

This recommendation is based on the strongest combination of measured retrieval quality, prompt improvements, and controlled backend comparisons. It preserves the strongest validated retrieval baseline, adopts the prompt that clearly improved generation behavior, and keeps more experimental ranking interventions disabled unless explicitly used for analysis.

In short, the final candidate is deliberately conservative: it keeps the validated metadata-aware hybrid retrieval baseline, uses the prompt that improved reviewed failures, retains the local backend for offline use, and treats both GPT and reranking as optional tools rather than default assumptions.

## 14. Limitations
Several limitations remain. Hebrew and RTL generation is still challenging, especially for noisy or partially corrupted extracted contexts. The local model has clear limitations on answer extraction quality in Hebrew. The evaluator setup also has its own issues: RAGAS is informative, but not always reliable enough to serve as the only judgment signal in this setting, and local evaluator JSON behavior can be unstable.

Additional limitations include the relatively small manual review sample, the existence of similar or duplicate recipe documents in the corpus, and the fact that ranking improvements can help some problematic rows while harming control questions. These limitations justify the project's emphasis on controlled experiments rather than broad unverified changes.

## 15. Future Work / Optional Improvements
The current system is complete for the scope of this project and includes a stable final candidate based on controlled experiments. The following directions are optional improvements that could further improve quality in future versions:

- neural reranker evaluation
- Hebrew/RTL normalization
- a stronger Hebrew answer model
- a larger human evaluation set
- a more robust evaluator setup
- additional metadata-aware retrieval improvements

These directions follow directly from the diagnosis already performed. Retrieval is already relatively strong, so future work should focus on safer ranking improvements, stronger answer extraction, and more robust Hebrew-specific evaluation.

## 16. Conclusion
This project demonstrates a full engineering workflow for building and evaluating a Hebrew recipe RAG system. Rather than relying only on headline metrics, the work used layered measurement, manual diagnosis, and controlled experiments to separate retrieval issues from generation issues. The final recommendation is therefore not just a configuration choice, but the result of a justified process: retrieval was strengthened with hybrid methods, prompt behavior was improved with a stricter format, local model limitations were exposed through frozen-context comparison, and unstable reranking ideas were kept out of the default system. The result is a final candidate that is evidence-based, transparent, and suitable for further improvement.
