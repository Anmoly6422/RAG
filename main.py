import sys
import argparse
from pathlib import Path

# Add src to Python path for seamless importing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag.data_loader import load_all_pdfs, split_documents
from rag.embedding import EmbeddingManager
from rag.vectorstore import VectorStore
from rag.search import RAGRetriever
from rag.llm import GeminiLLM
from rag.rag import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline for PDF Question Answering")
    parser.add_argument(
        "query",
        nargs="?",
        default="Explain about ExpoMind?",
        help="Question to ask the RAG pipeline"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of documents into vector store"
    )
    parser.add_argument(
        "--pdf_dir",
        default="data/pdf",
        help="Path to directory containing PDF documents"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of relevant chunks to retrieve"
    )
    parser.add_argument(
        "--model",
        default="gemini-3.8-flash",
        help="Gemini model name (e.g., gemini-3.8-flash, gemini-2.5-flash, gemini-2.0-flash)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Starting RAG Pipeline...")
    print("=" * 60)

    # 1. Initialize Vector Store
    vector_store = VectorStore()

    # 2. Check if we need to index or re-index documents
    if args.reindex or vector_store.count() == 0:
        print("\n--- Document Ingestion & Vector Indexing ---")
        documents = load_all_pdfs(args.pdf_dir)

        if not documents:
            print(f"[!] Warning: No PDF documents found in '{args.pdf_dir}'!")
            return

        # Split documents into smaller chunks for optimal context retrieval
        chunks = split_documents(documents, chunk_size=800, chunk_overlap=150)

        # Create embedding manager
        embedding_manager = EmbeddingManager()

        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)

        # Clear existing entries if re-indexing
        if args.reindex and vector_store.count() > 0:
            vector_store.clear_store()

        vector_store.add_documents(chunks, embeddings)
    else:
        print(f"\n[+] Using existing Vector Store ({vector_store.count()} indexed chunks).")
        embedding_manager = EmbeddingManager()

    # 3. Create Retriever
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager
    )

    # 4. Create Gemini LLM
    print("\n--- Initializing Gemini LLM ---")
    llm = GeminiLLM(model_name=args.model)

    # 5. Create RAG pipeline
    rag = RAGPipeline(
        retriever=retriever,
        llm=llm
    )

    # 6. Ask Question
    query = args.query
    print(f"\n[?] Question: {query}")

    result = rag.generate(
        query=query,
        top_k=args.top_k
    )

    # 7. Print Output
    print("\n" + "=" * 60)
    print("Answer:")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("Sources:")
    print("=" * 60)
    if result["sources"]:
        for src in result["sources"]:
            if isinstance(src, dict):
                print(f" - {src.get('display', src.get('file'))}")
            else:
                print(f" - {src}")
    else:
        print("No sources returned.")
    print("=" * 60)


if __name__ == "__main__":
    main()