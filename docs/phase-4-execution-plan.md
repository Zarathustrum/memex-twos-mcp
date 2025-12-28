# Phase 4: Hybrid Lexical + Semantic Search - Execution Plan

**Status:** Not Started
**Complexity:** LARGE
**Estimated Effort:** 2-3 days
**Prerequisites:** Phases 1-3 completed (BM25 ranking, two-phase retrieval, caching)

---

## Goal

Implement hybrid search combining lexical (BM25) and semantic (vector) search with Reciprocal Rank Fusion (RRF) to improve recall for semantic queries like "moving house" → "relocating", "new apartment".

**Expected Impact:** 40-60% better recall for semantic queries while maintaining precision.

---

## Repo Reality

**Current State:**
- BM25 lexical search implemented and tested (Phase 1)
- Two-phase retrieval pattern in place (Phase 2)
- Connection pooling and caching operational (Phase 3)
- SQLite + FTS5 architecture
- ~10K things in typical database, targeting 100K-500K scale

**Known Constraints:**
- Local-first (no cloud dependencies)
- Open-source components only
- SQLite-based (no Postgres/DuckDB)
- Must work offline
- ~5MB maximum footprint for extensions

**Codebase Structure:**
```
src/memex_twos_mcp/
  database.py          # 540 lines - add hybrid search methods
  cache.py             # 138 lines - cache embeddings queries
  server.py            # 337 lines - add hybrid search MCP tool

schema/
  schema.sql           # 144 lines - add embeddings table

scripts/
  load_to_sqlite.py    # 487 lines - add embedding generation during load

tests/
  test_database.py     # 469 lines - add hybrid search tests
```

---

## Non-Negotiable Constraints

1. **Local-first:** All processing must happen locally
2. **Open-source:** Use sentence-transformers (Apache 2.0) + sqlite-vec (MIT)
3. **Offline-capable:** Model must be downloadable and cached locally
4. **Lightweight:** Embedding model ≤100MB, sqlite-vec ≤5MB
5. **Optional:** Must have graceful fallback if embeddings disabled
6. **Deterministic:** Given same input, produce same embeddings (for testing)
7. **Incremental-ready:** Design with Phase 5 in mind (incremental embedding updates)

---

## Deliverables

### 1. Schema Changes

**File:** `schema/schema.sql`

Add embeddings table:
```sql
-- Embedding storage (384-dim vectors from all-MiniLM-L6-v2)
CREATE TABLE IF NOT EXISTS thing_embeddings (
    thing_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,       -- 384-dim float32 vector (~1.5KB per thing)
    model_version TEXT NOT NULL,   -- 'all-MiniLM-L6-v2'
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_embeddings_thing ON thing_embeddings(thing_id);

-- Track embedding model metadata
INSERT OR REPLACE INTO metadata (key, value)
VALUES ('embedding_model', 'all-MiniLM-L6-v2');

INSERT OR REPLACE INTO metadata (key, value)
VALUES ('embedding_dimension', '384');
```

**Migration Script:** `scripts/migrate_add_embeddings.py`
- Add embeddings table to existing databases
- Backfill embeddings for all existing things
- Track migration in metadata table

### 2. Embedding Generation Module

**File:** `src/memex_twos_mcp/embeddings.py` (NEW)

```python
"""
Embedding generation using sentence-transformers.

Uses all-MiniLM-L6-v2 model (90MB, 384 dimensions):
- Fast inference (~1000 things/sec on CPU)
- Good semantic quality
- Widely used baseline
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
import os

class EmbeddingGenerator:
    """Local embedding generation for semantic search."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: Optional[str] = None):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name
            cache_dir: Directory to cache model (default: ~/.cache/torch/sentence_transformers/)

        Side effects:
            Downloads model on first use (~90MB)
        """
        self.model_name = model_name
        self.dimension = 384

        # Set cache directory if specified
        if cache_dir:
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir

        try:
            self.model = SentenceTransformer(model_name)
            self.available = True
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.model = None
            self.available = False

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings in batches.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            numpy array of shape (len(texts), 384)
        """
        if not self.available:
            raise RuntimeError("Embedding model not available")

        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text (for query-time)."""
        if not self.available:
            raise RuntimeError("Embedding model not available")

        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
```

**Requirements:**
- Graceful degradation if model download fails
- Progress bars for batch encoding
- L2 normalization for cosine similarity
- Thread-safe (model can be shared)

### 3. Vector Search Integration

**Use sqlite-vec for vector similarity search:**

```bash
pip install sqlite-vec
```

**Database Integration:**
```python
# In database.py initialization
import sqlite_vec

def _init_vector_search(self):
    """Initialize sqlite-vec extension."""
    conn = self._get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Create virtual table for vector search
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_index USING vec0(
            thing_id TEXT PRIMARY KEY,
            embedding float[384]
        )
    """)
```

### 4. Hybrid Search Implementation

**File:** `src/memex_twos_mcp/database.py`

Add method:
```python
def hybrid_search(
    self,
    query: str,
    limit: int = 50,
    lexical_weight: float = 0.5,
    semantic_weight: float = 0.5,
    rrf_k: int = 60,
    enable_semantic: bool = True
) -> List[Dict[str, Any]]:
    """
    Hybrid search combining BM25 (lexical) + vector similarity (semantic).

    Uses Reciprocal Rank Fusion (RRF) to merge rankings:
        RRF_score(doc) = sum over all rankings ( weight / (k + rank) )

    Args:
        query: Search query text
        limit: Number of results to return
        lexical_weight: Weight for BM25 scores (0-1)
        semantic_weight: Weight for vector similarity (0-1)
        rrf_k: RRF constant (default 60 per literature)
        enable_semantic: If False, fall back to lexical-only

    Returns:
        List of candidate dictionaries with hybrid_score field
    """

    # Phase 1: BM25 lexical search
    lexical_results = self.search_candidates(query, limit=limit*2)

    # Phase 2: Vector semantic search (if enabled)
    if enable_semantic and self.embeddings_enabled:
        try:
            semantic_results = self._vector_search(query, limit=limit*2)
        except Exception as e:
            print(f"Semantic search failed, falling back to lexical: {e}")
            semantic_results = []
    else:
        semantic_results = []

    # Phase 3: Reciprocal Rank Fusion
    thing_scores: Dict[str, float] = {}
    thing_data: Dict[str, Dict] = {}

    # Add lexical scores
    for rank, result in enumerate(lexical_results, start=1):
        thing_id = result['id']
        thing_scores[thing_id] = lexical_weight / (rrf_k + rank)
        thing_data[thing_id] = result

    # Add semantic scores
    for rank, result in enumerate(semantic_results, start=1):
        thing_id = result['id']
        thing_scores[thing_id] = thing_scores.get(thing_id, 0) + (
            semantic_weight / (rrf_k + rank)
        )
        if thing_id not in thing_data:
            thing_data[thing_id] = result

    # Sort by combined score and return top N
    ranked = sorted(thing_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

    results = []
    for thing_id, hybrid_score in ranked:
        result = thing_data[thing_id].copy()
        result['hybrid_score'] = hybrid_score
        results.append(result)

    return results

def _vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
    """
    Vector similarity search using cosine similarity.

    Args:
        query: Search query text
        limit: Number of results

    Returns:
        List of candidates with cosine_similarity scores
    """
    # Generate query embedding
    query_embedding = self.embedding_gen.encode_single(query)

    # Search using sqlite-vec
    conn = self._get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            v.thing_id,
            vec_distance_cosine(v.embedding, ?) AS distance
        FROM vec_index v
        ORDER BY distance ASC
        LIMIT ?
    """, (query_embedding.tobytes(), limit))

    # Fetch thing metadata for results
    results = []
    for row in cursor.fetchall():
        thing_id, distance = row
        thing = self.get_thing_by_id(thing_id)
        if thing:
            # Convert distance to similarity (cosine distance = 1 - similarity)
            thing['cosine_similarity'] = 1 - distance
            results.append(thing)

    return results
```

**Critical Requirements:**
- Graceful fallback if semantic search unavailable
- Cache hybrid search results (add to QueryCache)
- Return both lexical and semantic scores for debugging
- Deterministic ordering (break ties consistently)

### 5. Embedding Generation During Load

**File:** `scripts/load_to_sqlite.py`

Add function:
```python
def generate_embeddings(
    conn: sqlite3.Connection,
    tasks: list,
    batch_size: int = 64,
    show_progress: bool = True
) -> None:
    """
    Generate and store embeddings for all things.

    Args:
        conn: Open SQLite connection
        tasks: List of thing dictionaries
        batch_size: Batch size for encoding
        show_progress: Show progress bar
    """
    from memex_twos_mcp.embeddings import EmbeddingGenerator

    print(f"Generating embeddings for {len(tasks)} things...")

    # Check if user wants embeddings (environment variable)
    if os.getenv('MEMEX_DISABLE_EMBEDDINGS') == '1':
        print("Embeddings disabled via MEMEX_DISABLE_EMBEDDINGS=1")
        return

    try:
        embedding_gen = EmbeddingGenerator()
    except Exception as e:
        print(f"Warning: Could not initialize embeddings: {e}")
        print("Skipping embedding generation. Set MEMEX_DISABLE_EMBEDDINGS=1 to silence this warning.")
        return

    # Prepare texts (use content field)
    texts = [task.get('content', '') for task in tasks]
    thing_ids = [task['id'] for task in tasks]

    # Generate embeddings in batches
    embeddings = embedding_gen.encode_batch(
        texts,
        batch_size=batch_size,
        show_progress=show_progress
    )

    # Store in database
    cursor = conn.cursor()
    for thing_id, embedding in zip(thing_ids, embeddings):
        # Serialize as float32 bytes
        embedding_blob = embedding.astype(np.float32).tobytes()

        cursor.execute("""
            INSERT INTO thing_embeddings (thing_id, embedding, model_version)
            VALUES (?, ?, ?)
        """, (thing_id, embedding_blob, embedding_gen.model_name))

        # Also insert into vec_index for fast search
        cursor.execute("""
            INSERT INTO vec_index (thing_id, embedding)
            VALUES (?, ?)
        """, (thing_id, embedding))

    conn.commit()
    print(f"Generated and stored {len(embeddings)} embeddings")

# Add to main() after load_tags()
if enable_embeddings:
    generate_embeddings(conn, tasks)
```

### 6. New MCP Tool

**File:** `src/memex_twos_mcp/server.py`

Add tool:
```python
Tool(
    name="hybrid_search",
    description=(
        "Hybrid search combining lexical (BM25) and semantic (vector) search. "
        "Better for conceptual queries like 'moving house', 'health issues'. "
        "Returns results ranked by combined lexical + semantic relevance. "
        "Falls back to lexical-only if embeddings unavailable."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (natural language or keywords)"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default: 50)",
                "default": 50
            },
            "lexical_weight": {
                "type": "number",
                "description": "Weight for BM25 scores (0-1, default: 0.5)",
                "default": 0.5
            },
            "semantic_weight": {
                "type": "number",
                "description": "Weight for semantic scores (0-1, default: 0.5)",
                "default": 0.5
            }
        },
        "required": ["query"]
    }
)
```

---

## Testing Requirements

### Unit Tests

**File:** `tests/test_embeddings.py` (NEW)

```python
def test_embedding_generation():
    """Test that embeddings are generated with correct dimensions."""
    gen = EmbeddingGenerator()

    embedding = gen.encode_single("test text")
    assert embedding.shape == (384,)
    assert np.allclose(np.linalg.norm(embedding), 1.0)  # L2 normalized

def test_embedding_determinism():
    """Test that same input produces same embedding."""
    gen = EmbeddingGenerator()

    emb1 = gen.encode_single("test text")
    emb2 = gen.encode_single("test text")

    assert np.allclose(emb1, emb2)

def test_batch_encoding():
    """Test batch encoding produces correct shape."""
    gen = EmbeddingGenerator()

    texts = ["text 1", "text 2", "text 3"]
    embeddings = gen.encode_batch(texts, show_progress=False)

    assert embeddings.shape == (3, 384)
```

**File:** `tests/test_hybrid_search.py` (NEW)

```python
def test_hybrid_search_basic(tmp_path):
    """Test basic hybrid search functionality."""
    # Create test DB with embeddings
    db = _create_test_db_with_embeddings(tmp_path)

    results = db.hybrid_search("doctor appointment", limit=10)

    assert len(results) > 0
    assert 'hybrid_score' in results[0]
    assert 'id' in results[0]

def test_hybrid_search_semantic_similarity(tmp_path):
    """Test that hybrid search finds semantically similar results."""
    db = _create_test_db_with_embeddings(tmp_path)

    # Add things with semantic similarity but different words
    # "relocating to new apartment" should match "moving house"

    results = db.hybrid_search("moving house", limit=10)

    # Verify semantically similar results appear
    result_ids = [r['id'] for r in results]
    assert 'thing_relocating' in result_ids  # Semantic match

def test_hybrid_search_fallback(tmp_path):
    """Test fallback to lexical-only if embeddings disabled."""
    db = _create_test_db_without_embeddings(tmp_path)

    # Should not crash, should fall back to lexical
    results = db.hybrid_search("doctor", limit=10, enable_semantic=False)

    assert len(results) > 0

def test_rrf_fusion_correctness():
    """Test that RRF correctly merges rankings."""
    # Mock test with known rankings
    lexical = [('doc1', 1), ('doc2', 2), ('doc3', 3)]
    semantic = [('doc3', 1), ('doc1', 2), ('doc4', 3)]

    # doc1 and doc3 should rank higher (appear in both)
    # doc2 and doc4 should rank lower (appear in one)

    # ... implement RRF and verify
```

### Integration Tests

```python
def test_full_workflow_with_embeddings(tmp_path):
    """Test complete workflow: load → embed → hybrid search."""
    # 1. Load data
    # 2. Generate embeddings
    # 3. Run hybrid search
    # 4. Verify results contain hybrid scores
```

---

## Performance Targets

### Embedding Generation
- **10K things:** <30 seconds on CPU
- **100K things:** <5 minutes on CPU
- Memory usage: <2GB peak

### Hybrid Search Query
- **10K things:** <200ms median
- **100K things:** <500ms median
- Cache hit: <10ms (same as lexical)

### Storage
- **Embeddings:** ~1.5KB per thing (384 × 4 bytes)
- **10K things:** ~15MB embeddings
- **100K things:** ~150MB embeddings

---

## Configuration

**Environment Variables:**

```bash
# Disable embeddings entirely (fall back to lexical-only)
export MEMEX_DISABLE_EMBEDDINGS=1

# Custom model cache directory
export MEMEX_EMBEDDING_CACHE_DIR=/path/to/cache

# Custom embedding model (advanced users)
export MEMEX_EMBEDDING_MODEL=all-mpnet-base-v2
```

---

## Dependencies to Add

**File:** `pyproject.toml`

```toml
[project]
dependencies = [
    # ... existing ...
    "sentence-transformers>=2.0.0",  # Embeddings (~90MB model download)
    "sqlite-vec>=0.1.0",             # Vector search extension
    "numpy>=1.24.0",                 # Vector operations
]
```

---

## Migration Path

**For Existing Installations:**

```bash
# Option 1: Rebuild database with embeddings
rm data/processed/twos.db
python scripts/load_to_sqlite.py data/processed/twos_data_cleaned.json

# Option 2: Migrate existing database
python scripts/migrate_add_embeddings.py data/processed/twos.db
```

**Migration script should:**
1. Add embeddings table
2. Generate embeddings for all existing things
3. Show progress bar
4. Allow resume on failure

---

## Quality Bar

### Must Have
- ✅ Hybrid search returns better semantic results than lexical-only
- ✅ Graceful fallback if embeddings disabled
- ✅ All tests passing
- ✅ No performance regression for lexical-only search
- ✅ Deterministic embeddings (same input → same output)
- ✅ Works offline after initial model download

### Should Have
- ✅ <200ms hybrid search latency for 100K things
- ✅ Progress bars for embedding generation
- ✅ Cache hybrid search results
- ✅ Migration script for existing databases

### Nice to Have
- Configurable embedding models
- GPU acceleration if available
- Embedding cache warming on startup

---

## Success Criteria

**Phase 4 is complete when:**

1. `hybrid_search()` method works and is tested
2. Embeddings generated during `load_to_sqlite.py`
3. New `hybrid_search` MCP tool exposed
4. All tests passing (unit + integration)
5. Documentation updated (README, implementation notes)
6. Performance benchmarks meet targets
7. Graceful degradation verified
8. Migration script works on existing databases

**Deliverables:**
- Working hybrid search with RRF
- sqlite-vec integration
- Embedding generation pipeline
- Comprehensive tests
- Migration guide

---

## Common Pitfalls to Avoid

1. **Large model downloads:** Use all-MiniLM-L6-v2 (90MB), not all-mpnet-base-v2 (420MB)
2. **Memory leaks:** Batch encoding, not all-at-once
3. **Non-deterministic results:** Ensure consistent L2 normalization
4. **Slow queries:** Use sqlite-vec index, not brute-force cosine
5. **Missing fallback:** Always handle case where embeddings unavailable
6. **Cache pollution:** Don't cache full embeddings, only results
7. **Phase 5 conflict:** Design embedding updates to be incremental-friendly

---

## Notes for Implementation

- Start with embedding generation module (easiest to test in isolation)
- Add sqlite-vec integration next (can test with dummy embeddings)
- Implement hybrid search last (depends on previous two)
- Test semantic quality manually before automated tests
- Use small test datasets (100 things) for fast iteration
- Profile memory usage during batch encoding

---

**End of Phase 4 Execution Plan**
