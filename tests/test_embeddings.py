"""Tests for embedding generation module."""

import pytest
import numpy as np
from memex_twos_mcp.embeddings import EmbeddingGenerator


def test_embedding_generation():
    """Test that embeddings are generated with correct dimensions."""
    gen = EmbeddingGenerator()

    if not gen.available:
        pytest.skip("Embedding model not available")

    embedding = gen.encode_single("test text")
    assert embedding.shape == (384,)
    assert np.allclose(np.linalg.norm(embedding), 1.0)  # L2 normalized


def test_embedding_determinism():
    """Test that same input produces same embedding."""
    gen = EmbeddingGenerator()

    if not gen.available:
        pytest.skip("Embedding model not available")

    emb1 = gen.encode_single("test text")
    emb2 = gen.encode_single("test text")

    assert np.allclose(emb1, emb2)


def test_batch_encoding():
    """Test batch encoding produces correct shape."""
    gen = EmbeddingGenerator()

    if not gen.available:
        pytest.skip("Embedding model not available")

    texts = ["text 1", "text 2", "text 3"]
    embeddings = gen.encode_batch(texts, show_progress=False)

    assert embeddings.shape == (3, 384)


def test_graceful_degradation():
    """Test that generator handles unavailable model gracefully."""
    gen = EmbeddingGenerator(model_name="nonexistent-model-12345")

    assert gen.available is False
    assert gen.model is None

    # Should raise RuntimeError when trying to encode
    with pytest.raises(RuntimeError):
        gen.encode_single("test")


def test_empty_batch():
    """Test that empty batch returns empty array."""
    gen = EmbeddingGenerator()

    if not gen.available:
        pytest.skip("Embedding model not available")

    embeddings = gen.encode_batch([], show_progress=False)
    assert embeddings.shape == (0, 384)
