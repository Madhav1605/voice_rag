from sentence_transformers import CrossEncoder

from src.config import (
    RERANKER_MODEL,
    RERANK_TOP_K,
    FINAL_TOP_K,
    CONFIDENCE_TOP_SCORE,
    CONFIDENCE_AVG_SCORE
)

_reranker = None
def get_reranker():
    """Load reranker once."""
    global _reranker
    if _reranker is None:
        # _reranker = CrossEncoder(RERANKER_MODEL)
        _reranker = CrossEncoder(
            RERANKER_MODEL,
            local_files_only=True
        )
    return _reranker

# CONFIDENCE CHECK
def confidence_check(reranked_results):
    """
    Check retrieval confidence.

    Matches the architecture exactly: relevant if EITHER
    condition holds (OR), not both (AND).

    These are raw MiniLM cross-encoder logits (not sigmoid-
    normalized), so CONFIDENCE_TOP_SCORE/CONFIDENCE_AVG_SCORE
    in config.py must stay tuned to this model's actual score
    range - re-check the printed "Rerank scores" line whenever
    the reranker model changes.
    """
    if not reranked_results:
        return False
    top_score = (reranked_results[0]["score"])
    avg_score = (
        sum(item["score"] for item in reranked_results) / len(reranked_results)
    )
    return (
        top_score >= CONFIDENCE_TOP_SCORE
        or
        avg_score >= CONFIDENCE_AVG_SCORE
    )

# RERANK RESULTS
def rerank_results(query,candidate_chunks):
    """
    Cross-encoder reranking + confidence evaluation.
    Always returns the top reranked chunks together with
    confidence information. The caller decides how to use
    the confidence flag.
    """
    if not candidate_chunks:
        return {
            "chunks": [],
            "confidence": False,
            "top_score": None,
            "avg_score": None
        }
    reranker = (get_reranker())
    candidate_chunks = (candidate_chunks[:RERANK_TOP_K])
    pairs = []
    for chunk in candidate_chunks:
        pairs.append(
            (
                query,
                chunk["text"]
            )
        )
    scores = reranker.predict(pairs,show_progress_bar=False)
    reranked_results = []
    for chunk, score in zip(candidate_chunks,scores):
        reranked_results.append(
            {
                "chunk": chunk,
                "score": float(score)
            }
        )
    reranked_results.sort(
        key=lambda item:item["score"],reverse=True
    )
    # Printed every time so thresholds in config.py can be
    # calibrated against the real distribution of scores
    # instead of guessed - this was missing in the last run,
    # which made the 2.00/1.00 thresholds untraceable.
    print(
        f"Rerank scores: "
        f"{[round(item['score'],3) for item in reranked_results]}"
    )
    is_confident = confidence_check(reranked_results)
    if not is_confident:
        print("Low confidence retrieval")
    final_chunks = [
        item["chunk"]
        for item in reranked_results[:FINAL_TOP_K]
    ]
    if is_confident:
        print(f"Reranked {len(final_chunks)} chunks")
    else:
        print(f"Returning {len(final_chunks)} low-confidence chunks")
    return {
        "chunks": final_chunks,
        "confidence": is_confident,
        "top_score": reranked_results[0]["score"],
        "avg_score": (
            sum(item["score"] for item in reranked_results) / len(reranked_results)
        )
    }