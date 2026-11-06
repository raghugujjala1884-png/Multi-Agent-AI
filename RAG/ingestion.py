from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_FOLDER = Path("rag/docs")


def load_pdfs():
    documents = []

    for pdf_path in PDF_FOLDER.glob("*.pdf"):

        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if text and text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_number
                        }
                    )
                )

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    print("Loading PDFs...")

    documents = load_pdfs()

    print(f"Pages loaded: {len(documents)}")

    print("\nSplitting documents...")

    chunks = split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    print("\nFirst chunk:")
    print("=" * 60)
    print(chunks[0].page_content[:1000])

    print("\nMetadata:")
    print(chunks[0].metadata)