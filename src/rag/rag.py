from typing import Dict, Any

from .search import RAGRetriever
from .llm import GeminiLLM


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline.
    """

    def __init__(
        self,
        retriever: RAGRetriever,
        llm: GeminiLLM
    ):
        self.retriever = retriever
        self.llm = llm

    def generate(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks and generate
        an answer using Gemini.
        """

        # Retrieve relevant chunks
        retrieved_docs = self.retriever.retrieve(
            query=query,
            top_k=top_k
        )

        if not retrieved_docs:
            return {
                "answer": (
                    "I could not find relevant "
                    "information in the documents."
                ),
                "sources": []
            }

        # Build context
        context = "\n\n".join(
            document["content"]
            for document in retrieved_docs
        )

        # Create prompt
        prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY the
provided context.

If the answer is not available in the
context, say:

"I don't have enough information in
the provided documents."

Do not make up information.

Context:
{context}

Question:
{query}

Answer:
"""

        # Generate answer
        answer = self.llm.generate(prompt)

        # Format sources with metadata
        sources = []
        for doc in retrieved_docs:
            meta = doc.get("metadata", {})
            source_file = meta.get("source", doc.get("id"))
            page_info = f" (Page {meta.get('page') + 1})" if "page" in meta else ""
            score_info = f" [score: {doc.get('relevance_score', 0):.2f}]"
            sources.append({
                "id": doc.get("id"),
                "file": source_file,
                "page": meta.get("page"),
                "score": doc.get("relevance_score"),
                "display": f"{source_file}{page_info}{score_info}"
            })

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs": retrieved_docs
        }