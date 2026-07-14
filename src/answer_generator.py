import re
import ollama
from groq import Groq

from src.config import (
    GROQ_API_KEY,
    OLLAMA_MODEL,
    GROQ_VISION_MODEL,
    GROQ_OCR_MIN_CHARS
)
from src.reference_store import fetch_references
from src.model_router import route_model

MAX_CONTEXT_CHUNKS = 4
MAX_CHUNK_CHARS = 800
MAX_TABLES = 2
MAX_TABLE_HTML_CHARS = 600
MAX_IMAGES = 2
OLLAMA_NUM_CTX = 4096

SYSTEM_PROMPT = (
    "You are a document question-answering assistant.\n"
    "Answer ONLY from the provided document context.\n"
    "Do not use outside knowledge.\n"
    "Never reveal your reasoning.\n"
    "Never output <think> tags or internal thoughts.\n"
    "If the answer is not present in the provided context, reply exactly:\n"
    "\"The answer is not available in the provided documents.\"\n"
    "Be concise, accurate, and mention page numbers naturally when available."
)

# GROQ CLIENT (CACHED)
_groq_client = None
def get_groq_client():
    """Load Groq client once, reuse afterwards."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

# FETCH REFERENCES FOR ALL CHUNKS
def fetch_chunk_references(chunks):
    """Collect every image/table referenced across all retrieved chunks."""
    all_images = []
    all_tables = []
    for chunk in chunks:
        refs = fetch_references(
            chunk.get("image_refs",[]),
            chunk.get("table_refs",[])
        )
        all_images.extend(refs["images"])
        all_tables.extend(refs["tables"])
    return {
        "images":all_images,
        "tables":all_tables
    }

# BUILD CITATIONS
def build_citations(chunks):
    """Build a deduplicated source list for the final answer."""
    seen = set()
    citations = []
    for chunk in chunks:
        key = (chunk.get("file_name"),str(chunk.get("page_numbers")))
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "file_name":chunk.get("file_name"),
                "page_numbers":chunk.get("page_numbers")
            }
        )
    return citations

# CLEAN MODEL OUTPUT
def strip_thinking(answer):
    # Remove complete <think>...</think> blocks
    answer = re.sub(r"<think>.*?</think>","",answer,flags=re.DOTALL)
    # Also strip incomplete block if model truncated mid-thought
    answer = re.sub(r"<think>.*","",answer,flags=re.DOTALL)
    return answer.strip()

# CONTEXT ASSEMBLY
def build_context_text(query,chunks,references):
    """
    Build the shared text portion of the prompt.
    Hard caps on chunk count, chunk length, table count, and
    image count to stay under Groq's free-tier token limit.
    """
    context_parts = [f"Question: {query}\n"]
    context_parts.append("RETRIEVED CONTEXT:\n")
    for chunk in chunks[:MAX_CONTEXT_CHUNKS]:
        chunk_text = (chunk.get("text","")[:MAX_CHUNK_CHARS])
        context_parts.append(
            f"[Source: {chunk.get('file_name')}, "
            f"Page(s): {chunk.get('page_numbers')}]\n"
            f"{chunk_text}\n"
        )
    if references["tables"]:
        context_parts.append("\nFULL TABLE DATA:\n")
        seen = set()
        table_count = 0
        for table in references["tables"]:
            if table_count >= MAX_TABLES:
                break
            html = (table.get("table_html") or table.get("table_text",""))
            if not html or html in seen:
                continue
            seen.add(html)
            context_parts.append(html[:MAX_TABLE_HTML_CHARS] + "\n")
            table_count += 1
    if references["images"]:
        context_parts.append("\nIMAGE OCR TEXT:\n")
        seen = set()
        for image in references["images"][:MAX_IMAGES]:
            ocr_text = (image.get("ocr_text","")).strip()
            if not ocr_text or ocr_text in seen:
                continue
            seen.add(ocr_text)
            context_parts.append(ocr_text + "\n")
    context_parts.append(
        "\nInstructions:\n"
        "- Answer only from the provided context.\n"
        "- Do not use outside knowledge.\n"
        "- If the answer is not present, reply exactly:\n"
        "\"The answer is not available in the provided documents.\"\n"
        "- When possible, mention the relevant page numbers naturally."
    )
    return "\n".join(context_parts)

# OLLAMA CALL
def call_ollama(prompt_text,temperature=None):
    """
    Call local Ollama model.
    temperature param exposed so query rewriting (if re-enabled)
    can force deterministic output with temperature=0.
    """
    options = {"num_ctx":OLLAMA_NUM_CTX}
    if temperature is not None:
        options["temperature"] = temperature
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":prompt_text}
        ],
        options=options
    )
    return strip_thinking(response["message"]["content"])

# GROQ CALL
def call_groq(prompt_text,references):
    """
    Call Groq vision model.

    Images are only attached when OCR text is insufficient
    (below GROQ_OCR_MIN_CHARS). If OCR already captured
    enough text, the raw image is not sent - this is the
    primary way we avoid burning the 200k daily token
    budget on queries that don't actually need vision.
    """
    client = (get_groq_client())
    content = [{"type":"text","text":prompt_text}]
    seen = set()
    image_count = 0
    for image in references["images"]:
        if image_count >= MAX_IMAGES:
            break
        image_base64 = (image.get("image_base64"))
        if not image_base64 or image_base64 in seen:
            continue
        # Only attach if OCR text is genuinely insufficient
        ocr_text = (image.get("ocr_text","")).strip()
        if len(ocr_text) >= GROQ_OCR_MIN_CHARS:
            continue
        seen.add(image_base64)
        content.append(
            {
                "type":"image_url",
                "image_url":{
                    "url":f"data:image/jpeg;base64,{image_base64}"
                }
            }
        )
        image_count += 1
    response = client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":content}
        ],
        temperature=0
    )
    return strip_thinking(response.choices[0].message.content)

# MAIN ENTRY
def generate_final_answer(chunks,query,override=None):
    """
    fetch references -> assemble context -> route model
    -> generate -> fallback if primary fails
    -> error message if both fail.
    """
    if not chunks:
        return {
            "answer":"Not found in the provided documents.",
            "citations":[],
            "model_used":None
        }
    references = (fetch_chunk_references(chunks))
    prompt_text = (build_context_text(query,chunks,references))
    citations = (build_citations(chunks))
    routing = (route_model(chunks,override=override))
    print(routing)
    primary_provider = (routing["provider"])
    try:
        if primary_provider == "groq":
            print("Prompt chars:", len(prompt_text))
            print("Estimated prompt tokens:", len(prompt_text) // 4)
            print("Images:", len(references["images"]))
            print("Tables:", len(references["tables"]))
            answer = (call_groq(prompt_text,references))
        else:
            answer = (call_ollama(prompt_text))
        return {
            "answer":answer,
            "citations":citations,
            "model_used":routing["model"]
        }
    except Exception as primary_error:
        print(f"{primary_provider} failed: {primary_error}")
    fallback_provider = ("ollama" if primary_provider == "groq" else "groq")
    try:
        if fallback_provider == "groq":
            print("Prompt chars:", len(prompt_text))
            print("Estimated prompt tokens:", len(prompt_text) // 4)
            print("Images:", len(references["images"]))
            print("Tables:", len(references["tables"]))
            answer = (call_groq(prompt_text,references))
            fallback_model = GROQ_VISION_MODEL
        else:
            answer = (call_ollama(prompt_text))
            fallback_model = OLLAMA_MODEL
        return {
            "answer":answer,
            "citations":citations,
            "model_used":fallback_model
        }
    except Exception as fallback_error:
        print(f"{fallback_provider} failed: {fallback_error}")
    return {
        "answer":"Unable to generate answer, please retry.",
        "citations":[],
        "model_used":None
    }

