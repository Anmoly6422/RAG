from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_all_pdfs(pdf_directory: str) -> List[Document]:
    """
    Load all PDF files from a directory recursively.

    Args:
        pdf_directory: Path to the directory containing PDFs.

    Returns:
        List of LangChain Document objects.
    """

    pdf_dir = Path(pdf_directory)

    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"PDF directory not found: {pdf_dir}"
        )

    pdf_files = list(pdf_dir.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    all_documents = []

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")

        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            all_documents.extend(documents)

            print(
                f"Loaded {len(documents)} pages "
                f"from {pdf_file.name}"
            )

        except Exception as e:
            print(
                f"Error processing {pdf_file.name}: {e}"
            )

    print(
        f"Total pages loaded: {len(all_documents)}"
    )

    return all_documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into smaller chunks.

    Args:
        documents: List of LangChain documents.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of document chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks "
        f"from {len(documents)} pages"
    )

    return chunks