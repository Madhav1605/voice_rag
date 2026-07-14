import os
import json
import base64
import hashlib

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.documents.elements import NarrativeText

from src.config import (
    IMAGES_DIR,
    SUPPORTED_EXTENSIONS,
    HASH_ALGORITHM,
    DOC_ID_PREFIX,
    DOCUMENT_REGISTRY_FILE
)

# FILE HASH
def get_file_hash(file_path):
    """Generate file hash.
       Used for duplicate detection - same content
       always produces the same hash, regardless of
       how many times the file is re-ingested.
    """
    hasher = hashlib.new(HASH_ALGORITHM)
    with open(file_path, "rb") as file:
        while chunk := file.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# DOC ID
def generate_doc_id(file_hash):
    """
    Create a stable document id from the file hash.
    Same file content -> same doc_id, every single time.
    This is what makes image filenames deterministic
    (see save_extracted_images below) instead of
    depending on id(elements), which changes every run.
    """
    return (
        f"{DOC_ID_PREFIX}_"
        f"{file_hash[:12]}"
    )

# FILE EXTENSION
def get_file_extension(file_path):
    """Return lowercase file extension."""
    return os.path.splitext(file_path)[1].lower()

# DOCUMENT REGISTRY (DUPLICATE CHECK)
def load_document_registry():
    """
    Load the registry of already-ingested files,
    keyed by file_hash. Returns empty dict if this
    is the first document ever ingested.
    """
    if not os.path.exists(DOCUMENT_REGISTRY_FILE):
        return {}
    with open(DOCUMENT_REGISTRY_FILE,"r",encoding="utf-8") as file:
        return json.load(file)

def save_document_registry(registry):
    """Persist the document registry to disk."""
    with open(DOCUMENT_REGISTRY_FILE,"w",encoding="utf-8") as file:
        json.dump(registry,file,indent=4,ensure_ascii=False)

def is_duplicate_file(file_hash):
    """
    Check whether this exact file content has
    already been ingested. This is the actual
    duplicate-skip check the architecture calls
    for right after the file hash is generated -
    previously this never ran, so every re-run
    reprocessed the file and resaved its images
    under a new random prefix.
    """
    registry = load_document_registry()
    return file_hash in registry

def register_document(file_hash,doc_id,file_name):
    """Record a successfully ingested file."""
    registry = load_document_registry()
    registry[file_hash] = {
        "doc_id":doc_id,
        "file_name":file_name
    }
    save_document_registry(registry)

# IMAGE EXTRACTION
def save_extracted_images(elements,doc_id):
    """Save images extracted from any element-based source
       (PDF, DOCX, PPTX). Adds image_path into metadata.

       Filenames are prefixed with doc_id (derived from the
       file's content hash), not id(elements) - id() is a
       Python memory address and changes every run even for
       the exact same file, which is why duplicate images
       were piling up. doc_id is deterministic, so re-running
       on the same file always produces the same filenames.
    """
    image_counter = 1
    for element in elements:
        if getattr(element, "category", "").lower() != "image":
            continue
        try:
            image_base64 = getattr(
                element.metadata,
                "image_base64",
                None
            )
            if not image_base64:
                continue
            image_path = os.path.join(
                IMAGES_DIR,
                f"image_{doc_id}_{image_counter}.jpg"
            )
            with open(image_path, "wb") as file:
                file.write(base64.b64decode(image_base64))
            element.metadata.image_path = image_path
            image_counter += 1
        except Exception as error:
            print(f"Image save failed: {error}")

# PDF
def partition_pdf_document(file_path,doc_id):
    """Partition PDF using Unstructured."""
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["image"],
        extract_image_block_to_payload=True
    )
    save_extracted_images(elements,doc_id)
    return elements

# DOCX
def partition_docx_document(file_path,doc_id):
    """Partition DOCX file.
       extract_image_block_types is passed so embedded
       images are pulled out the same way as PDFs.
    """
    elements = partition_docx(
        filename=file_path,
        extract_image_block_types=["image"],
        extract_image_block_to_payload=True
    )
    save_extracted_images(elements,doc_id)
    return elements

# PPTX
def partition_pptx_document(file_path,doc_id):
    """Partition PowerPoint file.
       Same image extraction as DOCX/PDF.
    """
    elements = partition_pptx(
        filename=file_path,
        extract_image_block_types=["image"],
        extract_image_block_to_payload=True
    )
    save_extracted_images(elements,doc_id)
    return elements

# TXT
def partition_txt_document(file_path):
    """
    Load TXT file and convert
    into Unstructured element.
    """
    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:
        text = file.read()
    return [NarrativeText(text=text)]

# IMAGE
def partition_image_document(file_path):
    """Image files are processed later
       by image_processor.py.
       Return image path for now.
    """
    return file_path

# UNIVERSAL PARTITIONER
def partition_document(file_path):
    """
    Universal document loader.
    Supports:
    PDF
    DOCX
    PPTX
    TXT
    Images

    Performs the duplicate-file check FIRST, before any
    partitioning work happens. If the file's content hash
    is already in the registry, ingestion is skipped
    entirely and elements is returned as None - callers
    must check result["duplicate"] before using "elements".
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )
    extension = get_file_extension(file_path)
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    file_hash = get_file_hash(file_path)
    doc_id = generate_doc_id(file_hash)
    file_name = os.path.basename(file_path)

    if is_duplicate_file(file_hash):
        print(
            f"Duplicate file detected, skipping ingestion: "
            f"{file_name}"
        )
        return {
            "duplicate": True,
            "elements": None,
            "file_name": file_name,
            "file_path": file_path,
            "file_hash": file_hash,
            "file_type": extension,
            "doc_id": doc_id
        }

    print(f"Processing: {file_name}")
    if extension == ".pdf":
        elements = partition_pdf_document(file_path,doc_id)
    elif extension == ".docx":
        elements = partition_docx_document(file_path,doc_id)
    elif extension == ".pptx":
        elements = partition_pptx_document(file_path,doc_id)
    elif extension == ".txt":
        elements = partition_txt_document(file_path)
    else:
        elements = partition_image_document(file_path)

    register_document(file_hash,doc_id,file_name)

    return {
        "duplicate": False,
        "elements": elements,
        "file_name": file_name,
        "file_path": file_path,
        "file_hash": file_hash,
        "file_type": extension,
        "doc_id": doc_id
    }