import os
import json

from src.config import REFERENCE_STORE_FILE

# LOAD STORE
def load_reference_store():
    """
    Load the persistent reference store.
    Returns empty structure if it does not exist yet.
    """
    if not os.path.exists(REFERENCE_STORE_FILE):
        return {
            "images": {},
            "tables": {}
        }
    with open(REFERENCE_STORE_FILE,"r",encoding="utf-8") as file:
        return json.load(file)

# SAVE STORE
def save_reference_store(store):
    """Persist the reference store to disk."""
    with open(REFERENCE_STORE_FILE,"w",encoding="utf-8") as file:
        json.dump(store,file,indent=4,ensure_ascii=False)

# ADD IMAGES
def add_image_references(image_documents):
    """
    Merge image documents into the reference store.
    Keyed by image_id, exactly what chunk image_refs
    point to at query time.
    """
    store = (load_reference_store())
    for image in image_documents:
        image_id = image.metadata["image_id"]
        store["images"][image_id] = {
            "ocr_text": image.metadata.get("ocr_text",""),
            "image_path": image.metadata.get("image_path"),
            "image_base64": image.metadata.get("image_base64"),
            "page_number": image.metadata.get("page_number")
        }
    save_reference_store(store)
    print(
        f"Reference store: "
        f"{len(image_documents)} images saved"
    )

# ADD TABLES
def add_table_references(table_documents):
    """
    Merge table documents into the reference store.
    Keyed by table_id, exactly what chunk table_refs
    point to at query time.
    """
    store = (load_reference_store())
    for table in table_documents:
        table_id = table.metadata["table_id"]
        store["tables"][table_id] = {
            "table_html": table.metadata.get("table_html",""),
            "table_text": table.metadata.get("table_text",""),
            "page_number": table.metadata.get("page_number")
        }
    save_reference_store(store)
    print(
        f"Reference store: "
        f"{len(table_documents)} tables saved"
    )

# FETCH IMAGE
def get_image(image_id):
    """Fetch a single image record by ID."""
    store = (load_reference_store())
    return store["images"].get(image_id)

# FETCH TABLE
def get_table(table_id):
    """Fetch a single table record by ID."""
    store = (load_reference_store())
    return store["tables"].get(table_id)

# FETCH MANY
def fetch_references(image_refs,table_refs):
    """
    Fetch full image/table content for a
    retrieved chunk's image_refs/table_refs.
    """
    store = (load_reference_store())
    images = [
        store["images"][image_id]
        for image_id in image_refs
        if image_id in store["images"]
    ]
    tables = [
        store["tables"][table_id]
        for table_id in table_refs
        if table_id in store["tables"]
    ]
    return {
        "images":images,
        "tables":tables
    }