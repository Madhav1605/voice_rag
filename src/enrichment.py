import re
from src.reference_store import (
    add_image_references,
    add_table_references
)

# FIGURE REFERENCES
def extract_figure_references(text):
    """Extract Figure references."""
    matches = re.findall(
        r"Figure\s+(\d+)",
        text,
        flags=re.IGNORECASE
    )
    return list({int(match) for match in matches})

# TABLE REFERENCES
def extract_table_references(text):
    """Extract Table references."""
    matches = re.findall(
        r"Table\s+(\d+)",
        text,
        flags=re.IGNORECASE
    )

    return list({int(match) for match in matches})

# IMAGE LOOKUP
def build_image_lookup(image_documents):
    """Build:
    figure_number -> image document
    """
    lookup = {}
    for image in image_documents:
        figure_number = image.metadata.get("figure_number")
        if figure_number is not None:
            lookup[figure_number] = image
    return lookup

# TABLE LOOKUP
def build_table_lookup(table_documents):
    """Build:
    table_number -> table document
    """
    lookup = {}
    for table in table_documents:
        table_number = table.metadata.get("table_number")
        if table_number is not None:
            lookup[table_number] = table
    return lookup

# CHUNK ENRICHMENT
def enrich_chunks(chunks,image_documents,table_documents,doc_id,file_name,file_path):
    """
    Enrich chunks using only
    referenced figures/tables.
    Also persists every image/table this document
    produced into the reference store, so they can
    be fetched by ID at query time.
    """
    if image_documents:
        add_image_references(image_documents)
    if table_documents:
        add_table_references(table_documents)

    enriched_chunks = []
    image_lookup = build_image_lookup(image_documents)
    table_lookup = build_table_lookup(table_documents)
    for chunk in chunks:
        chunk_text = chunk["text"]
        image_refs = []
        table_refs = []
        related_ocr = []
        related_tables = []

        # Figure References
        figure_numbers = (
            extract_figure_references(
                chunk_text
            )
        )
        for figure_number in figure_numbers:
            image = image_lookup.get(figure_number)
            if not image:
                continue
            image_refs.append(image.metadata["image_id"])
            ocr_text = image.metadata.get("ocr_text","")
            if ocr_text.strip():
                related_ocr.append(ocr_text)

        # Table References
        table_numbers = (extract_table_references(chunk_text))
        for table_number in table_numbers:
            table = table_lookup.get(table_number)
            if not table:
                continue
            table_refs.append(table.metadata["table_id"])
            table_text = table.metadata.get("table_text","")
            if table_text.strip():
                related_tables.append(table_text)

        # Text Enrichment
        enriched_text = chunk_text
        if related_ocr:
            enriched_text += (
                "\n\nRELATED IMAGE OCR:\n"
                + "\n".join(
                    related_ocr
                )
            )
        if related_tables:
            enriched_text += (
                "\n\nRELATED TABLE CONTENT:\n"
                + "\n".join(
                    related_tables
                )
            )
        enriched_chunks.append(
            {
                "chunk_id":chunk["chunk_id"],
                "parent_chunk_id":chunk["parent_chunk_id"],
                "file_path":file_path,
                "doc_id":doc_id,
                "file_name":file_name,
                "page_numbers":chunk.get("page_numbers",[]),
                "title":chunk["title"],
                "text":enriched_text,
                "image_refs":image_refs,
                "table_refs":table_refs
            }
        )
    print(
        f"Created "
        f"{len(enriched_chunks)} "
        f"enriched chunks"
    )
    return enriched_chunks