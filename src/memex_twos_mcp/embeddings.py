"""
Embedding generation for semantic search using sentence-transformers.

Uses all-MiniLM-L6-v2 model (384 dimensions, ~90MB) for fast, high-quality
embeddings. Gracefully degrades if model unavailable.
"""

import warnings
from typing import List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingGenerator:
    """
    Generate semantic embeddings for text using sentence-transformers.

    Provides graceful degradation if sentence-transformers not installed.
    All embeddings are L2-normalized for cosine similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.

        Args:
            model_name: HuggingFace model identifier (default: all-MiniLM-L6-v2)

        Attributes:
            available: bool indicating if model loaded successfully
            model: SentenceTransformer instance or None
            model_name: Name of the loaded model
            dimension: Embedding dimension (384 for default model)
        """
        self.model_name = model_name
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.model: Optional[SentenceTransformer] = None
        self.available = False

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            warnings.warn(
                "sentence-transformers not installed. Embeddings disabled. "
                "Install with: pip install sentence-transformers"
            )
            return

        try:
            # Suppress tokenizer warnings during model download
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.model = SentenceTransformer(model_name)
            self.available = True
        except Exception as e:
            warnings.warn(f"Failed to load embedding model {model_name}: {e}")
            self.available = False

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text string to a normalized embedding vector.

        Args:
            text: Text to encode

        Returns:
            L2-normalized embedding vector (shape: [384])

        Raises:
            RuntimeError: If model not available
        """
        if not self.available or self.model is None:
            raise RuntimeError("Embedding model not available")

        # encode() returns normalized embeddings by default
        embedding = self.model.encode(text, normalize_embeddings=True)
        return np.array(embedding, dtype=np.float32)

    def encode_batch(
        self, texts: List[str], batch_size: int = 64, show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a batch of texts to normalized embedding vectors.

        Processes in batches to avoid memory issues with large datasets.

        Args:
            texts: List of text strings to encode
            batch_size: Number of texts to encode per batch (default: 64)
            show_progress: Show progress bar (default: True)

        Returns:
            L2-normalized embedding matrix (shape: [N, 384])

        Raises:
            RuntimeError: If model not available
        """
        if not self.available or self.model is None:
            raise RuntimeError("Embedding model not available")

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
            convert_to_numpy=True,
        )

        return np.array(embeddings, dtype=np.float32)
