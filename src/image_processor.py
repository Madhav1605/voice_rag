import base64
import pytesseract

from PIL import Image
from langchain_core.documents import Document
from src.config import (
    IMG_ID_PREFIX,
    TESSERACT_PATH
)

# OPTIONAL TESSERACT PATH (WINDOWS)
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = (TESSERACT_PATH)

# BASE64 ENCODING
def image_to_base64(image_path):
    """
    Convert image into base64 string.
    Used later when sending images to Groq.
    """
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

# OCR
def extract_ocr_text(image_path):
    """
    Extract text from image using Tesseract OCR.
    """
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image).strip()
    except Exception as error:
        print(f"OCR failed: {error}")
        return ""

# EXTRACTED IMAGES (PDF / DOCX / PPTX)
def process_extracted_images(elements,doc_id):
    """
    Process images extracted from any
    element-based document (PDF, DOCX, PPTX).
    image_id is prefixed with doc_id so IDs
    never collide across documents.
    """
    image_documents = []
    image_counter = 1
    for element in elements:
        if getattr(element,"category","").lower() != "image":
            continue
        image_path = getattr(element.metadata,"image_path",None)
        if not image_path:
            continue
        ocr_text = extract_ocr_text(image_path)
        page_number = getattr(
            element.metadata,
            "page_number",
            None
        )
        image_document = Document(
            page_content=ocr_text,
            metadata={
                "image_id": f"{doc_id}_{IMG_ID_PREFIX}_{image_counter}",
                "figure_number": image_counter,
                "image_path": image_path,
                "image_base64": image_to_base64(image_path),
                "page_number": page_number,
                "ocr_text": ocr_text
            }
        )
        image_documents.append(image_document)
        image_counter += 1
    return image_documents

# STANDALONE IMAGE FILE
def process_image_file(image_path,doc_id):
    """
    Process image uploaded directly
    by the user.
    """
    ocr_text = extract_ocr_text(image_path)
    image_document = Document(
        page_content=ocr_text,
        metadata={
            "image_id":f"{doc_id}_{IMG_ID_PREFIX}_1",
            "figure_number": 1,
            "image_path":image_path,
            "image_base64":image_to_base64(image_path),
            "page_number":1,
            "ocr_text":ocr_text
        }
    )
    return [image_document]

# MAIN ENTRY
def extract_image_documents(elements,file_type,doc_id):
    """Universal image processor."""
    print("Processing images...")
    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".webp"
    }
    if file_type in image_extensions:
        image_documents = (process_image_file(elements,doc_id))
    else:
        image_documents = (process_extracted_images(elements,doc_id))
    print(
        f"Processed "
        f"{len(image_documents)} "
        f"images"
    )
    return image_documents