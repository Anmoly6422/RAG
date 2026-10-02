from typing import List, Dict, Any


class RAGRetriever:
    """
    Retrieve relevant document chunks
    from the vector store.
    """

    def __init__(
        self,
        vector_store,
        embedding_manager
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks
        for a query.
        """

        query_embedding = (
            self.embedding_manager
            .generate_embeddings([query])[0]
        )

        results = self.vector_store.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved_documents = []

        if not results.get("documents") or not results["documents"][0]:
            return retrieved_documents

        for i in range(
            len(results["documents"][0])
        ):
            distance = results["distances"][0][i]
            metadata = results["metadatas"][0][i] if results.get("metadatas") and results["metadatas"][0] else {}

            relevance_score = (
                1 / (1 + distance)
            )

            retrieved_documents.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": metadata,
                "distance": distance,
                "relevance_score": relevance_score
            })

        return retrieved_documents