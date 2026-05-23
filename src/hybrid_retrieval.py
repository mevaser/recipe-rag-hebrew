from __future__ import annotations

from bm25_retrieval import retrieve_bm25
from retrieval import retrieve as retrieve_dense


def reciprocal_rank_fusion(
    ranked_results: list[list[dict]],
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[dict]:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0.")
    if weights is None:
        weights = [1.0] * len(ranked_results)
    if len(weights) != len(ranked_results):
        raise ValueError("weights length must match ranked_results length.")

    fused_scores: dict[str, float] = {}
    result_lookup: dict[str, dict] = {}

    for results, source_weight in zip(ranked_results, weights):
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                float(source_weight) * (1.0 / (rrf_k + rank))
            )
            result_lookup.setdefault(chunk_id, result)

    fused_results: list[dict] = []
    for chunk_id, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True):
        result = dict(result_lookup[chunk_id])
        result["score"] = float(score)
        fused_results.append(result)
    return fused_results


def retrieve_hybrid(
    query: str,
    k: int = 5,
    candidate_k: int = 50,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[dict]:
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than 0.")

    retrieval_depth = max(k, candidate_k)
    dense_results = retrieve_dense(query, k=retrieval_depth)
    bm25_results = retrieve_bm25(query, k=retrieval_depth)
    fused_results = reciprocal_rank_fusion(
        [dense_results, bm25_results],
        rrf_k=rrf_k,
        weights=[dense_weight, bm25_weight],
    )
    return fused_results[:k]
