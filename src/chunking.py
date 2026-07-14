import re
import math

from unstructured.chunking.title import chunk_by_title
from src.config import (
    CHUNK_SIZE,
    NEW_AFTER_N_CHARS,
    COMBINE_TEXT_UNDER_N_CHARS,
    SUBCHUNK_SIZE,
    CHUNK_ID_PREFIX
)

SUBSECTION_PATTERN = re.compile(
    r"(?m)^\s*(\d+(?:\.\d+)+\s+[A-Za-z].*)$"
)

def extract_chunk_title(chunk):
    """
    Extract first title from
    original elements.
    """
    for element in chunk.metadata.orig_elements:

        if element.category == "Title":
            return element.text.strip()

    return ""

# TITLE CHUNKING
def create_chunk_by_title(elements):
    """
    Create title-based chunks
    using Unstructured.
    """
    print("Creating chunks...")
    chunks = chunk_by_title(
        elements,
        max_characters=CHUNK_SIZE,
        new_after_n_chars=NEW_AFTER_N_CHARS,
        combine_text_under_n_chars=COMBINE_TEXT_UNDER_N_CHARS
    )
    print(f"Created {len(chunks)} chunks")
    return chunks

# SUBSECTION SPLIT
def split_by_subsection_titles(chunk_text):
    """
    Try splitting on numbered subsection headings
    (5.1, 5.2, 6.3...). Returns None if no such
    headings are found in the text, so the caller
    can fall back to even character splitting.

    Each piece keeps its own heading, so children
    end up titled "5.1 Training Data and Batching"
    instead of inheriting the parent's title.
    """
    matches = list(SUBSECTION_PATTERN.finditer(chunk_text))
    if not matches:
        return None

    pieces = []

    # Text before the first heading (if any) stays with
    # the parent section, no separate heading of its own.
    if matches[0].start() > 0:
        preamble = chunk_text[:matches[0].start()].strip()
        if preamble:
            pieces.append({"heading":None,"text":preamble})

    for index, match in enumerate(matches):
        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(chunk_text)
        )
        heading = match.group(1).strip()
        text = chunk_text[start:end].strip()
        pieces.append({"heading":heading,"text":text})

    return pieces

# EVEN SPLIT (FALLBACK)
def split_evenly(chunk_text):
    """
    Split text into evenly sized pieces.
    Used only when no subsection headings are found -
    e.g. plain documents without numbered sections.
    """
    text_length = len(chunk_text)
    num_children = math.ceil(text_length / SUBCHUNK_SIZE)
    split_size = math.ceil(text_length / num_children)
    pieces = []
    for start in range(0,text_length,split_size):
        pieces.append(
            {
                "heading":None,
                "text":chunk_text[start:start + split_size]
            }
        )
    return pieces

# LARGE CHUNK SPLIT (HYBRID)
def split_large_chunk(chunk_text):
    """
    Split a large chunk into children.

    1. Try splitting on numbered subsection headings first -
       keeps each child as a complete, coherent subsection
       instead of cutting mid-sentence.
    2. If a subsection piece is itself still too large,
       split that piece evenly (recursive fallback).
    3. If no subsection headings exist at all (plain
       documents, no numbering), fall back to a pure
       even split across the whole chunk.
    """
    subsection_pieces = split_by_subsection_titles(chunk_text)

    if subsection_pieces is None:
        return split_evenly(chunk_text)

    final_pieces = []
    for piece in subsection_pieces:
        if len(piece["text"]) <= SUBCHUNK_SIZE:
            final_pieces.append(piece)
            continue
        # Subsection itself too large - fall back to
        # even split for just this piece, keep its heading
        # on every fragment so all of them stay identifiable
        # as part of that subsection (not just the first).
        sub_pieces = split_evenly(piece["text"])
        for sub_piece in sub_pieces:
            sub_piece["heading"] = piece["heading"]
            final_pieces.append(sub_piece)

    return final_pieces

def get_chunk_pages(chunk):
    """Return all pages covered by the chunk."""
    pages = set()

    for element in chunk.metadata.orig_elements:
        page = getattr(
            element.metadata,
            "page_number",
            None
        )

        if page is not None:
            pages.add(page)

    return sorted(pages)

# for code 3
from unstructured.documents.elements import NarrativeText
from src.config import (
    CHUNK_SIZE,
    NEW_AFTER_N_CHARS,
    COMBINE_TEXT_UNDER_N_CHARS,
    SUBCHUNK_SIZE,
    CHUNK_ID_PREFIX,
    TXT_CHUNK_SIZE,
    TXT_CHUNK_OVERLAP
)

# TXT CHUNKING
def create_txt_chunks(elements, doc_id):
    """
    Create overlapping chunks for TXT documents.

    TXT files have no titles, so chunk_by_title is not useful.
    Instead, split into fixed-size overlapping chunks.
    """

    print("Creating TXT chunks...")

    text = ""

    for element in elements:
        if isinstance(element, NarrativeText):
            text += element.text + "\n"

    final_chunks = []

    start = 0
    chunk_counter = 1

    while start < len(text):

        end = min(
            start + TXT_CHUNK_SIZE,
            len(text)
        )

        final_chunks.append(
            {
                "chunk_id": f"{doc_id}_{CHUNK_ID_PREFIX}_{chunk_counter}",
                "parent_chunk_id": None,
                "page_numbers": [],
                "title": "",
                "text": text[start:end]
            }
        )

        chunk_counter += 1

        if end >= len(text):
            break

        start += (
            TXT_CHUNK_SIZE
            - TXT_CHUNK_OVERLAP
        )

    print(f"Created {len(final_chunks)} TXT chunks")

    return final_chunks


# PARENT CHILD STRUCTURE
def create_parent_child_chunks(chunks,doc_id):
    """
    Create parent-child chunk mapping.
    """
    final_chunks = []
    parent_counter = 1
    last_known_title = ""
    for chunk in chunks:
        chunk_title = extract_chunk_title(chunk)
        if not chunk_title:
            # No Title element found for this section -
            # inherit the previous chunk's title instead
            # of leaving citations with a blank section name.
            chunk_title = last_known_title
        else:
            last_known_title = chunk_title
        chunk_pages = get_chunk_pages(chunk)
        chunk_text = chunk.text
        parent_chunk_id = (f"{doc_id}_{CHUNK_ID_PREFIX}_{parent_counter}")

        # Small chunk - stored as its own standalone chunk
        if len(chunk_text) <= SUBCHUNK_SIZE:
            final_chunks.append(
                {
                    "chunk_id":parent_chunk_id,
                    "parent_chunk_id":None,
                    "page_numbers": chunk_pages,
                    "title": chunk_title,
                    "text":chunk_text
                }
            )
            parent_counter += 1
            continue

        # Large chunk - split into dot-numbered children
        subchunks = split_large_chunk(chunk_text)
        for child_index, subchunk in enumerate(subchunks,start=1):
            final_chunks.append(
                {
                    "chunk_id": f"{parent_chunk_id}.{child_index}",
                    "parent_chunk_id":parent_chunk_id,
                    "page_numbers": chunk_pages,
                    "title": subchunk["heading"] or chunk_title,
                    "text":subchunk["text"]
                }
            )
        parent_counter += 1
    print(
        f"Created "
        f"{len(final_chunks)} "
        f"final chunks"
    )
    return final_chunks

def create_chunks(
    elements,
    doc_id,
    file_type
):
    """
    Complete chunking pipeline.

    Structured documents (PDF, DOCX, PPTX)
    use title-based parent-child chunking.

    TXT documents use fixed-size overlapping
    chunking because they contain no titles.
    """

    if file_type == ".txt":
        return create_txt_chunks(
            elements,
            doc_id
        )

    chunks = create_chunk_by_title(elements)

    return create_parent_child_chunks(
        chunks,
        doc_id
    )
