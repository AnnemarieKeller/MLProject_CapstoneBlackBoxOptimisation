
from typing import List
from pypdf import PdfReader

class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

def split_text(text: str, chunk_size: int = 1000, overlap: int = 0) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 1000, overlap: int = 100) -> List[Document]:
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    chunks = split_text(full_text, chunk_size=chunk_size, overlap=overlap)

    # Wrap chunks in Document objects
    return [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in chunks]