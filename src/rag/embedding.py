from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """
    Generate vector embeddings using
    SentenceTransformer.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.model_name = model_name
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load the embedding model."""

        print(
            f"Loading embedding model: "
            f"{self.model_name}"
        )

        self.model = SentenceTransformer(
            self.model_name
        )

        print(
            "Model loaded successfully."
        )

        dim_func = getattr(self.model, "get_embedding_dimension", getattr(self.model, "get_sentence_embedding_dimension", None))
        dim = dim_func() if dim_func else 384
        print(f"Embedding dimension: {dim}")

    def generate_embeddings(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        """

        if self.model is None:
            raise ValueError(
                "Embedding model is not loaded."
            )

        print(
            f"Generating embeddings for "
            f"{len(texts)} texts..."
        )

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        print(
            f"Generated embeddings with shape: "
            f"{embeddings.shape}"
        )

        return embeddings

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension."""

        if self.model is None:
            raise ValueError(
                "Embedding model is not loaded."
            )

        dim_func = getattr(self.model, "get_embedding_dimension", getattr(self.model, "get_sentence_embedding_dimension", None))
        return dim_func() if dim_func else 384