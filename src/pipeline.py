from src.partitioning import partition_document
from src.table_processor import extract_table_documents
from src.image_processor import extract_image_documents
from src.chunking import create_chunks
from src.enrichment import enrich_chunks
from src.vectorstore import store_chunks
from src.reference_store import add_image_references

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".webp"
}

# STANDALONE IMAGE CHUNK
def build_standalone_image_chunk(image_documents,doc_id,file_name,file_path):
    """
    Build a single enriched chunk for a standalone
    image file. Skips enrich_chunks entirely - there is
    no surrounding text to regex-match a "Figure N"
    caption against, so image_refs is attached directly
    instead of relying on caption detection.
    """
    add_image_references(image_documents)
    image_document = image_documents[0]
    image_id = (image_document.metadata["image_id"])
    ocr_text = (image_document.metadata.get("ocr_text",""))
    return [
        {
            "chunk_id": f"{doc_id}_chunk_1",
            "parent_chunk_id": None,
            "file_path": file_path,
            "doc_id": doc_id,
            "file_name": file_name,
            "page_numbers": [1],
            "title": file_name,
            "text": ocr_text or f"Image file: {file_name}",
            "image_refs": [image_id],
            "table_refs": []
        }
    ]

# INGEST SINGLE DOCUMENT
def ingest_document(file_path):
    """
    Run the full ingestion pipeline on one document:
    partition -> duplicate check -> image/table processing
    -> chunk -> enrich -> store.

    Returns a summary dict. Callers (ingest.py) should
    check result["status"] - "skipped" means the file was
    already ingested and nothing else happened.
    """
    partition_result = (partition_document(file_path))

    if partition_result["duplicate"]:
        print(
            f"Skipping (already ingested): "
            f"{partition_result['file_name']}"
        )
        return {
            "status":"skipped",
            "doc_id":partition_result["doc_id"],
            "file_name":partition_result["file_name"]
        }

    elements = (partition_result["elements"])
    doc_id = (partition_result["doc_id"])
    file_name = (partition_result["file_name"])
    file_path_resolved = (partition_result["file_path"])
    file_type = (partition_result["file_type"])

    image_documents = (
        extract_image_documents(elements,file_type,doc_id)
    )

    if file_type in IMAGE_EXTENSIONS:
        table_documents = []
        enriched_chunks = (
            build_standalone_image_chunk(
                image_documents,
                doc_id,
                file_name,
                file_path_resolved
            )
        )
    else:
        table_documents = (extract_table_documents(elements,doc_id))
        # chunks = (create_chunks(elements,doc_id))
        
        # for code 3
        chunks = create_chunks(elements=elements,doc_id=doc_id,file_type=file_type)

        enriched_chunks = (
            enrich_chunks(
                chunks,
                image_documents,
                table_documents,
                doc_id,
                file_name,
                file_path_resolved
            )
        )

    store_chunks(enriched_chunks)

    print(f"Ingestion complete: {file_name}")
    return {
        "status":"ingested",
        "doc_id":doc_id,
        "file_name":file_name,
        "chunks":len(enriched_chunks),
        "tables":len(table_documents),
        "images":len(image_documents)
    }

# INGEST MULTIPLE DOCUMENTS
def ingest_documents(file_paths):
    """
    Run ingestion across multiple files, one at a time.
    A failure on one file does not stop the rest -
    each result is recorded independently.
    """
    results = []
    for file_path in file_paths:
        try:
            result = (ingest_document(file_path))
        except Exception as error:
            print(f"Ingestion failed for {file_path}: {error}")
            result = {
                "status":"failed",
                "file_name":file_path,
                "error":str(error)
            }
        results.append(result)
    return results