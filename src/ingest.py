import sys
import os

from src.pipeline import ingest_document
from src.config import DOCUMENTS_DIR

def main():
    """
    Ingest a single document.
    Usage: python ingest.py path/to/file.pdf
    If no path is given, prompts for one.
    Relative paths are resolved against DOCUMENTS_DIR.
    """
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = (input("Enter path to document: ")).strip()

    if not os.path.isabs(file_path):
        candidate = os.path.join(DOCUMENTS_DIR,file_path)
        if os.path.exists(candidate):
            file_path = candidate

    result = (ingest_document(file_path))
    print(result)

if __name__ == "__main__":
    main()