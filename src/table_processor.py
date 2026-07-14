from langchain_core.documents import Document
from src.config import TBL_ID_PREFIX

def extract_table_documents(elements,doc_id):
    """Create table documents.
       table_id is prefixed with doc_id so IDs
       never collide across documents.
    """

    print("Processing tables...")
    table_documents = []
    table_counter = 1
    for element in elements:
        if getattr(element,"category","").lower() == "table":
            try:
                table_html = (getattr(element.metadata,"text_as_html",""))
                table_text = (getattr(element,"text",""))
                table_document = (
                    Document(
                        page_content=table_text,
                        metadata={
                            "table_id":f"{doc_id}_{TBL_ID_PREFIX}_{table_counter}",
                            "table_number":table_counter,
                            "page_number":
                                getattr(
                                element.metadata,
                                "page_number",
                                None
                            ),
                            "table_html":table_html,
                            "table_text":table_text
                        }
                    )
                )
                table_documents.append(table_document)
                table_counter += 1

            except Exception as e:
                print(f"Table processing failed: {e}")

    print(f"Processed "f"{len(table_documents)} "f"tables")
    return table_documents