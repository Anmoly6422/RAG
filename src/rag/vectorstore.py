import os
import uuid
from typing import List

import chromadb
import numpy as np
from langchain_core.documents import Document


class VectorStore:
    """
    Manage document embeddings using ChromaDB.
    """

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "data/vector_store"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.client = None
        self.collection = None

        self._initialize_store()

    def _initialize_store(self):
        """Initialize the ChromaDB vector store."""

        os.makedirs(
            self.persist_directory,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description":
                    "PDF document embeddings"
                }
            )
        )

        print(
            f"Vector store initialized: "
            f"{self.collection_name}"
        )

        print(
            f"Existing documents: "
            f"{self.collection.count()}"
        )

    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray
    ):
        """Add documents and embeddings to ChromaDB."""

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents and embeddings "
                "must be the same."
            )

        ids = []
        metadatas = []
        texts = []
        embedding_list = []

        for i, (document, embedding) in enumerate(
            zip(documents, embeddings)
        ):
            document_id = (
                f"doc_{uuid.uuid4().hex[:8]}_{i}"
            )

            ids.append(document_id)

            metadata = {}
            for k, v in document.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    metadata[k] = v
                else:
                    metadata[k] = str(v)

            metadata["doc_index"] = i
            metadata["content_length"] = len(document.page_content)

            metadatas.append(metadata)

            texts.append(
                document.page_content
            )

            embedding_list.append(
                embedding.tolist()
            )

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embedding_list
        )

        print(
            f"Added {len(documents)} documents "
            f"to ChromaDB."
        )

        print(
            f"Total documents: "
            f"{self.collection.count()}"
        )

    def clear_store(self):
        """Clear all documents from the collection."""
        if self.client and self.collection_name:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings"}
            )
            print(f"Vector store '{self.collection_name}' cleared.")

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count() if self.collection else 0