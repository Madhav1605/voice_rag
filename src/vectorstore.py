import os
import pickle

from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import (
    DB_DIR,
    BM25_INDEX_FILE,
    EMBEDDING_MODEL,
    COLLECTION_NAME
)

# EMBEDDING MODEL (CACHED)
# Previously this reloaded the HuggingFace model from disk
# on every single call - and retriever.py was calling it
# twice per query (once via get_vectorstore, once directly
# for MMR). Caching it here means it loads once per process,
# not once per query.
_embedding_model = None
def get_embedding_model():
    """Load embedding model once, reuse afterwards."""
    global _embedding_model
    if _embedding_model is None:
        # _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "local_files_only": True
            }
        )
    return _embedding_model

# VECTOR STORE (CACHED)
_vectorstore = None
def get_vectorstore():
    """Load/Create Chroma collection once, reuse afterwards."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = (get_embedding_model())
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
    return _vectorstore

def get_embeddings_by_ids(chunk_ids):
    """
    Return stored embeddings for the given chunk ids.
    """
    vectorstore = get_vectorstore()

    result = vectorstore._collection.get(
        ids=chunk_ids,
        include=["embeddings"]
    )

    return {
        chunk_id: embedding
        for chunk_id, embedding in zip(
            result["ids"],
            result["embeddings"]
        )
    }

# BM25 SAVE
def save_bm25_index(bm25,all_chunks):
    """
    Save BM25 index and the full,
    cumulative chunk list it was built from.
    """
    with open(BM25_INDEX_FILE,"wb") as file:
        pickle.dump(
            {
                "bm25": bm25,
                "chunks": all_chunks
            },
            file
        )

# BM25 LOAD
def load_bm25_index():
    """Load BM25 index."""
    with open(BM25_INDEX_FILE,"rb") as file:
        return pickle.load(file)

# LOAD EXISTING CHUNKS (IF ANY)
def load_existing_bm25_chunks():
    """
    Return chunks already indexed by BM25 from a
    previous ingestion run, or an empty list if
    this is the first document ever ingested.
    """
    if not os.path.exists(BM25_INDEX_FILE):
        return []
    try:
        existing = (load_bm25_index())
        return existing.get("chunks",[])
    except Exception as error:
        print(f"Could not load existing BM25 index: {error}")
        return []

# BUILD BM25
def build_bm25_index(enriched_chunks):
    """
    Build BM25 from enriched chunks.
    Merges with chunks from previously ingested
    documents instead of overwriting them, so
    keyword search keeps working across every
    document ever ingested, not just the latest one.
    """
    existing_chunks = (load_existing_bm25_chunks())
    existing_ids = {
        chunk["chunk_id"]
        for chunk in existing_chunks
    }
    new_chunks = [
        chunk for chunk in enriched_chunks
        if chunk["chunk_id"] not in existing_ids
    ]
    all_chunks = (existing_chunks + new_chunks)
    tokenized_corpus = []
    for chunk in all_chunks:
        tokenized_corpus.append(chunk["text"].lower().split())
    bm25 = BM25Okapi(tokenized_corpus)
    save_bm25_index(bm25,all_chunks)
    print(
        f"BM25 indexed "
        f"{len(all_chunks)} chunks total "
        f"({len(new_chunks)} new)"
    )

# STORE CHUNKS
def store_chunks(enriched_chunks):
    """
    Store chunks in Chroma
    and rebuild BM25 (merged).
    """
    vectorstore = (get_vectorstore())
    documents = []
    for chunk in enriched_chunks:
        metadata = {
            "chunk_id":chunk["chunk_id"],
            "doc_id":chunk["doc_id"],
            "file_name":chunk["file_name"],
            "title":chunk["title"],
            "page_numbers":
                str(
                    chunk.get(
                        "page_numbers",
                        ""
                    )
                ),
            "image_refs":
                ",".join(
                    chunk[
                        "image_refs"
                    ]
                ),
            "table_refs":
                ",".join(
                    chunk[
                        "table_refs"
                    ]
                )
        }
        if (chunk["parent_chunk_id"] is not None):
            metadata["parent_chunk_id"] = (chunk["parent_chunk_id"])
        document = Document(
            page_content=chunk["text"],
            metadata=metadata
        )
        documents.append(document)
    ids = [
        chunk["chunk_id"]
        for chunk in enriched_chunks
    ]
    vectorstore.add_documents(
        documents,
        ids=ids
    )
    print(
        f"Stored "
        f"{len(documents)} "
        f"chunks in Chroma"
    )
    build_bm25_index(enriched_chunks)
    return vectorstore