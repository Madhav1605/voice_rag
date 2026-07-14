import numpy as np
from src.config import (
    VECTOR_K,
    BM25_K,
    RRF_K,
    MMR_K,
    MMR_LAMBDA
)
from src.vectorstore import (get_vectorstore,get_embedding_model,load_bm25_index,get_embeddings_by_ids)

# SEMANTIC SEARCH
def semantic_search(query):
    """Search Chroma vector store."""
    vectorstore = get_vectorstore()
    results = (
        vectorstore.similarity_search_with_score(
            query=query,
            k=VECTOR_K
        )
    )
    documents = []
    for rank, (document, score) in enumerate(results,start=1):
        documents.append(
            {
                "document": document,
                "rank": rank,
                "score": score,
                "source": "vector"
            }
        )
    return documents

# BM25 SEARCH
def bm25_search(query):
    """Search BM25 index."""
    bm25_data = (load_bm25_index())
    bm25 = (bm25_data["bm25"])
    chunks = (bm25_data["chunks"])
    tokenized_query = (query.lower().split())
    scores = (bm25.get_scores(tokenized_query))
    ranked_indices = (np.argsort(scores)[::-1])[:BM25_K]
    documents = []
    for rank, index in enumerate(ranked_indices,start=1):
        chunk = chunks[index]
        documents.append(
            {
                "chunk": chunk,
                "rank": rank,
                "score": float(scores[index]),
                "source": "bm25"
            }
        )
    return documents

# RRF FUSION
def rrf_fusion(vector_results,bm25_results):
    """Reciprocal Rank Fusion."""
    fused_scores = {}
    # Vector Results
    for result in vector_results:
        chunk_id = (result["document"].metadata["chunk_id"])
        if chunk_id not in fused_scores:
            fused_scores[chunk_id] = {
                "score": 0.0,
                "result": result
            }
        fused_scores[chunk_id]["score"] += (1 /(RRF_K+ result["rank"]))
    # BM25 Results
    for result in bm25_results:
        chunk_id = (result["chunk"]["chunk_id"])
        if chunk_id not in fused_scores:
            fused_scores[chunk_id] = {
                "score": 0.0,
                "result": result
            }
        fused_scores[chunk_id]["score"] += (1 /(RRF_K+ result["rank"]))
    fused_results = sorted(
        fused_scores.values(),
        key=lambda item:
            item["score"],
        reverse=True
    )
    return fused_results

# COSINE SIMILARITY
def cosine_similarity(vector_a,vector_b):
    """Cosine similarity."""
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return (np.dot(vector_a, vector_b) / (norm_a * norm_b))

# CHUNK PAYLOAD
def build_chunk_payload(result):
    """
    Normalize a fused result (which may have come from
    either the vector path or the BM25 path) into one
    consistent chunk dict shape.
    """
    if "document" in result:
        document = result["document"]
        image_refs = (document.metadata.get("image_refs",""))
        table_refs = (document.metadata.get("table_refs",""))
        return {
            "chunk_id":document.metadata["chunk_id"],
            "parent_chunk_id":document.metadata.get("parent_chunk_id"),
            "doc_id":document.metadata["doc_id"],
            "file_name":document.metadata["file_name"],
            "title":document.metadata["title"],
            "page_numbers":document.metadata.get("page_numbers"),
            "image_refs":(
                image_refs.split(",") if image_refs else []
            ),
            "table_refs":(
                table_refs.split(",") if table_refs else []
            ),
            "text":document.page_content
        }
    return result["chunk"]

# MMR DEDUPLICATION
def mmr_deduplication(query, fused_results):
    """Remove highly similar chunks using embeddings stored in Chroma."""

    # ---------------------------------------------------------
    # Query embedding
    # ---------------------------------------------------------
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(query)

    # ---------------------------------------------------------
    # Build payloads
    # ---------------------------------------------------------
    candidates = []

    for item in fused_results:

        payload = build_chunk_payload(item["result"])

        candidates.append(
            {
                "payload": payload,
                "chunk_id": payload["chunk_id"]
            }
        )

    # ---------------------------------------------------------
    # Fetch embeddings from Chroma
    # ---------------------------------------------------------
    embedding_map = get_embeddings_by_ids(
        [
            candidate["chunk_id"]
            for candidate in candidates
        ]
    )

    # ---------------------------------------------------------
    # Attach embeddings
    # ---------------------------------------------------------
    for candidate in candidates:

        candidate["embedding"] = (
            embedding_map[candidate["chunk_id"]]
        )

    # ---------------------------------------------------------
    # MMR
    # ---------------------------------------------------------
    selected = []

    while candidates and len(selected) < MMR_K:

        best_candidate = None
        best_score = float("-inf")

        for candidate in candidates:

            relevance = cosine_similarity(
                query_embedding,
                candidate["embedding"]
            )

            diversity = 0

            if selected:

                diversity = max(
                    cosine_similarity(
                        candidate["embedding"],
                        chosen["embedding"]
                    )
                    for chosen in selected
                )

            score = (
                (MMR_LAMBDA * relevance)
                -
                ((1 - MMR_LAMBDA) * diversity)
            )

            if score > best_score:

                best_score = score
                best_candidate = candidate

        selected.append(best_candidate)
        candidates.remove(best_candidate)

    return [
        item["payload"]
        for item in selected
    ]

# MAIN RETRIEVAL
def retrieve_content(query):
    """
    Hybrid retrieval:
    Vector Search + BM25 Search -> RRF -> MMR -> candidate chunks.

    Returns candidates only - no confidence check, no reranking,
    no model routing. Pass this output straight into
    reranker.rerank_results().
    """
    vector_results = (semantic_search(query))
    bm25_results = (bm25_search(query))
    fused_results = (rrf_fusion(vector_results,bm25_results))
    candidate_chunks = (mmr_deduplication(query,fused_results))
    print(
        f"Retrieved "
        f"{len(candidate_chunks)} "
        f"candidate chunks"
    )
    return candidate_chunks