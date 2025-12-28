# Memex Twos MCP v2 Implementation Plan

**Goal:** Optimize for 100K-500K Things with improved relevance, performance, and token efficiency while remaining local-first.

**Created:** 2025-12-28
**Status:** Implementation in progress

---

## Executive Summary

This document outlines the v2 upgrade plan to address critical performance and quality bottlenecks identified in the current implementation. The upgrade targets 5 major improvements + entity extraction fixes, maintaining SQLite+FTS5 architecture while adding BM25 ranking, connection pooling, caching, embeddings (sqlite-vec), and incremental updates.

**Key Constraints:**
- Local-first (no cloud dependencies)
- Open-source components only
- Breaking changes allowed
- SQLite + FTS5 (no Postgres/DuckDB migration)

---

## Critical Findings Summary (From Analysis)

### P0 - Query Quality Issues
1. **No BM25 ranking** (database.py:102-116) - Results sorted by timestamp, not relevance
2. **No snippet extraction** - Returns full content, wastes tokens
3. **Naive entity extraction** (convert_to_json.py:84-348) - 200+ stopwords still yield false positives

### P1 - Performance Issues
4. **Connection-per-query overhead** (database.py:28-40) - 5-10ms per query
5. **Commit-per-insert** (load_to_sqlite.py:89) - 10x slower initial load
6. **No query result caching** - Identical queries re-execute full scans

### P2 - Scalability Issues
7. **No incremental updates** - Must delete + rebuild for new exports
8. **Over-fetching data** - `SELECT *` returns 15 fields when LLM needs 3-5

---

## Implementation Phases

### Phase 1: BM25 Ranking + Snippets [HIGH IMPACT, SMALL COMPLEXITY]
**Expected:** 3-5x better relevance ordering + ~50% smaller responses

**Changes:**
- **File:** `src/memex_twos_mcp/database.py`
- **Method:** `search_content()` (lines 86-116)

**Implementation:**
```python
# Current (WRONG):
ORDER BY t.timestamp DESC  # Ignores relevance

# New (CORRECT):
SELECT t.*,
       rank AS relevance_score,
       snippet(things_fts, 1, '<b>', '</b>', '...', 32) AS snippet
FROM things t
JOIN things_fts fts ON t.id = fts.thing_id
WHERE things_fts MATCH ?
ORDER BY rank  # BM25 relevance score (most relevant first)
LIMIT ?
```

**Additional:**
- Add error handling for invalid FTS5 query syntax
- Add query normalization/validation
- Update response model to include `relevance_score` and `snippet` fields

**Tests:**
- Verify BM25 ordering matches expected relevance
- Test snippet extraction with highlight markers
- Test malformed query handling

---

### Phase 2: Two-Phase Retrieval Pattern [HIGH IMPACT, SMALL COMPLEXITY]
**Expected:** 5-10x faster queries + much less token waste on large result sets

**Changes:**
- **File:** `src/memex_twos_mcp/database.py`
- **New Methods:**
  - `search_candidates()` - Returns minimal preview data (ID, score, snippet, date, tags, people)
  - `get_things_by_ids()` - Batch fetch full records by IDs
  - `get_thing_by_id()` - Single thing lookup

- **File:** `src/memex_twos_mcp/server.py`
- **New Tools:**
  - `search_things_preview` (default search) - Returns candidate previews
  - `get_things_by_ids` - Fetch full content for selected IDs
  - `get_thing_by_id` - Single thing detail view

**Candidate Preview Schema:**
```python
{
  "id": "task_00001",
  "relevance_score": 0.95,
  "snippet": "...doctor appointment...",
  "timestamp": "2023-10-17T16:23:00",
  "tags": ["health"],
  "people": ["Dr. Smith"],
  "is_completed": false
}
```

**Response Savings:**
- Current: 50 things × 15 fields × ~30 chars avg = ~34KB JSON
- Preview: 50 things × 7 fields × ~20 chars avg = ~8KB JSON
- **Savings: ~75% token reduction**

**Tests:**
- Verify candidate results match full results (same ordering, IDs)
- Test batch fetch by IDs
- Test pagination consistency

---

### Phase 3: Connection Pooling + Query Cache [MEDIUM IMPACT, SMALL COMPLEXITY]
**Expected:** 2-3x faster repeated queries, lower latency jitter

**Changes:**
- **File:** `src/memex_twos_mcp/database.py`

**Implementation:**

**3A. Connection Pooling:**
```python
class TwosDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()  # Thread safety

    def _get_connection(self) -> sqlite3.Connection:
        """Reuse connection instead of creating new one."""
        with self._lock:
            if self._connection is None:
                self._connection = sqlite3.connect(
                    self.db_path,
                    check_same_thread=False  # Allow multi-thread access with lock
                )
                self._connection.row_factory = sqlite3.Row
            return self._connection

    def close(self):
        """Explicit cleanup for graceful shutdown."""
        with self._lock:
            if self._connection:
                self._connection.close()
                self._connection = None
```

**3B. Query Result Cache:**
```python
from functools import lru_cache
import hashlib
import time

class QueryCache:
    """TTL-based cache for search candidate results."""

    def __init__(self, ttl_seconds: int = 900):  # 15 min default
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def _make_key(self, query: str, **kwargs) -> str:
        """Create cache key from query + params."""
        normalized = json.dumps({"q": query.lower().strip(), **kwargs}, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str, **kwargs) -> Optional[Any]:
        """Retrieve cached result if not expired."""
        key = self._make_key(query, **kwargs)
        with self._lock:
            if key in self._cache:
                timestamp, result = self._cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    return result
                else:
                    del self._cache[key]  # Expired
        return None

    def set(self, query: str, result: Any, **kwargs):
        """Cache query result with current timestamp."""
        key = self._make_key(query, **kwargs)
        with self._lock:
            self._cache[key] = (time.time(), result)

    def invalidate_all(self):
        """Clear entire cache (call on data ingestion)."""
        with self._lock:
            self._cache.clear()
```

**Integration:**
- Add `QueryCache` instance to `TwosDatabase`
- Cache only phase-1 candidate results (not full records)
- Invalidate cache on any write/update operation
- Add cache hit/miss metrics logging

**Tests:**
- Verify cache hit returns same result
- Verify cache miss after TTL expiration
- Verify cache invalidation on ingestion
- Test thread safety with concurrent queries

---

### Phase 4: Hybrid Lexical + Semantic Search [HIGH IMPACT, LARGE COMPLEXITY]
**Expected:** 40-60% better recall for semantic queries ("moving house" → "relocating", "new apartment")

**Dependencies:**
- `sentence-transformers` (local model, ~90MB)
- `sqlite-vec` (vector search extension, ~5MB)
- `numpy` (for vector ops)

**Schema Changes:**
- **File:** `schema/schema.sql`

```sql
-- Add embeddings table
CREATE TABLE IF NOT EXISTS thing_embeddings (
    thing_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,  -- 384-dim float32 vector (~1.5KB per thing)
    model_version TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE
);

-- Metadata for tracking embedding generation
INSERT OR REPLACE INTO metadata (key, value)
VALUES ('embedding_model', 'all-MiniLM-L6-v2');  -- Sentence-transformers model
```

**Vector Index:**
```python
# Use sqlite-vec for ANN search
# https://github.com/asg017/sqlite-vec

import sqlite_vec

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

**Embedding Generation Pipeline:**
- **File:** `src/memex_twos_mcp/embeddings.py` (new)

```python
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model (downloads on first use, ~90MB).
        Model is cached locally in ~/.cache/torch/sentence_transformers/
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = 384

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings in batches."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text (for query-time)."""
        return self.model.encode(text, convert_to_numpy=True)
```

**Hybrid Search Implementation:**
- **File:** `src/memex_twos_mcp/database.py`

```python
def hybrid_search(
    self,
    query: str,
    limit: int = 50,
    lexical_weight: float = 0.5,
    semantic_weight: float = 0.5,
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
) -> List[Dict[str, Any]]:
    """
    Hybrid search combining BM25 (lexical) + vector similarity (semantic).
    Uses Reciprocal Rank Fusion (RRF) to merge rankings.

    Args:
        query: Search query text
        limit: Number of results to return
        lexical_weight: Weight for BM25 scores (0-1)
        semantic_weight: Weight for vector similarity (0-1)
        rrf_k: RRF constant (default 60, per literature)

    Returns:
        List of things ranked by hybrid score
    """

    # Phase 1: BM25 lexical search
    lexical_results = self._fts_search_with_rank(query, limit=limit*2)

    # Phase 2: Vector semantic search
    query_embedding = self.embedding_gen.encode_single(query)
    semantic_results = self._vector_search(query_embedding, limit=limit*2)

    # Phase 3: Reciprocal Rank Fusion
    # RRF formula: score(doc) = sum over all rankings (1 / (k + rank))
    thing_scores: Dict[str, float] = {}

    # Add lexical scores
    for rank, result in enumerate(lexical_results, start=1):
        thing_id = result['id']
        thing_scores[thing_id] = thing_scores.get(thing_id, 0) + (
            lexical_weight / (rrf_k + rank)
        )

    # Add semantic scores
    for rank, result in enumerate(semantic_results, start=1):
        thing_id = result['id']
        thing_scores[thing_id] = thing_scores.get(thing_id, 0) + (
            semantic_weight / (rrf_k + rank)
        )

    # Sort by combined score and fetch full records
    ranked_ids = sorted(thing_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

    return self.get_things_by_ids([thing_id for thing_id, _ in ranked_ids])
```

**Configuration:**
- Add `MEMEX_ENABLE_EMBEDDINGS` environment variable (default: true)
- Graceful fallback to lexical-only search if embeddings disabled or model not installed

**Embedding Generation During Ingestion:**
- **File:** `scripts/load_to_sqlite.py`

```python
def generate_embeddings(conn: sqlite3.Connection, tasks: list, batch_size: int = 64):
    """Generate embeddings for all things in batches."""
    from memex_twos_mcp.embeddings import EmbeddingGenerator

    print(f"Generating embeddings for {len(tasks)} things...")

    # Initialize model (downloads on first use)
    embedding_gen = EmbeddingGenerator()

    # Prepare texts
    texts = [task['content'] for task in tasks]
    thing_ids = [task['id'] for task in tasks]

    # Generate embeddings in batches
    embeddings = embedding_gen.encode_batch(texts, batch_size=batch_size)

    # Store in database
    cursor = conn.cursor()
    for thing_id, embedding in zip(thing_ids, embeddings):
        # Store as BLOB (serialize numpy array)
        embedding_blob = embedding.astype(np.float32).tobytes()
        cursor.execute(
            """
            INSERT INTO thing_embeddings (thing_id, embedding, model_version)
            VALUES (?, ?, ?)
            """,
            (thing_id, embedding_blob, embedding_gen.model_name)
        )

    conn.commit()
    print(f"Generated and stored {len(embeddings)} embeddings")
```

**Tests:**
- Unit test: embedding generation produces correct dimensions
- Unit test: RRF combines rankings correctly
- Integration test: semantic queries return related content
- Test graceful fallback when embeddings disabled

---

### Phase 5: Incremental Ingestion [MEDIUM IMPACT, MEDIUM COMPLEXITY]
**Expected:** 10-100x faster reloads for new exports

**Changes:**
- **File:** `schema/schema.sql`

```sql
-- Add content hash for deduplication
ALTER TABLE things ADD COLUMN content_hash TEXT;
CREATE INDEX IF NOT EXISTS idx_things_content_hash ON things(content_hash);

-- Add import tracking
CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,
    imported_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    thing_count INTEGER,
    new_count INTEGER,
    updated_count INTEGER,
    deleted_count INTEGER
);
```

**Implementation:**
- **File:** `scripts/load_to_sqlite.py`

```python
import hashlib

def compute_content_hash(task: Dict[str, Any]) -> str:
    """
    Compute stable hash for deduplication.
    Hash based on: timestamp + content + section_header (canonical fields)
    """
    canonical = {
        'timestamp': task.get('timestamp'),
        'content': task.get('content', '').strip(),
        'section_header': task.get('section_header', '').strip()
    }
    hash_input = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(hash_input.encode()).hexdigest()

def incremental_load(
    conn: sqlite3.Connection,
    tasks: list,
    source_file: str,
    mode: str = 'sync'  # 'sync' or 'append'
):
    """
    Incrementally load tasks using upsert logic.

    Modes:
    - 'sync': Update changed, insert new, delete removed (full sync)
    - 'append': Insert new only, don't delete (additive)
    """

    print(f"Incremental load mode: {mode}")

    cursor = conn.cursor()

    # Compute hashes for incoming tasks
    incoming = {}
    for task in tasks:
        content_hash = compute_content_hash(task)
        task['content_hash'] = content_hash
        incoming[task['id']] = (task, content_hash)

    # Fetch existing hashes
    cursor.execute("SELECT id, content_hash FROM things")
    existing = {row[0]: row[1] for row in cursor.fetchall()}

    # Categorize changes
    new_ids = set(incoming.keys()) - set(existing.keys())
    deleted_ids = set(existing.keys()) - set(incoming.keys()) if mode == 'sync' else set()

    updated_ids = set()
    for thing_id in set(incoming.keys()) & set(existing.keys()):
        if incoming[thing_id][1] != existing[thing_id]:
            updated_ids.add(thing_id)

    print(f"New: {len(new_ids)}, Updated: {len(updated_ids)}, Deleted: {len(deleted_ids)}")

    # Insert new
    for thing_id in new_ids:
        task, _ = incoming[thing_id]
        _insert_thing(cursor, task)

    # Update changed (delete + reinsert to trigger FTS update)
    for thing_id in updated_ids:
        task, _ = incoming[thing_id]
        cursor.execute("DELETE FROM things WHERE id = ?", (thing_id,))
        _insert_thing(cursor, task)

    # Delete removed (sync mode only)
    if mode == 'sync':
        for thing_id in deleted_ids:
            cursor.execute("DELETE FROM things WHERE id = ?", (thing_id,))

    # Update embeddings incrementally
    changed_ids = new_ids | updated_ids
    if changed_ids:
        _update_embeddings_for_ids(conn, [incoming[tid][0] for tid in changed_ids])

    # Track import
    cursor.execute(
        """
        INSERT INTO imports (source_file, thing_count, new_count, updated_count, deleted_count)
        VALUES (?, ?, ?, ?, ?)
        """,
        (source_file, len(tasks), len(new_ids), len(updated_ids), len(deleted_ids))
    )

    conn.commit()
    print(f"Incremental load complete")

def _insert_thing(cursor, task):
    """Insert single thing with all fields."""
    cursor.execute(
        """
        INSERT INTO things (
            id, timestamp, content, content_hash,
            timestamp_raw, content_raw, section_header, section_date,
            line_number, indent_level, parent_task_id, bullet_type,
            is_completed, is_pending, is_strikethrough
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task['id'], task.get('timestamp'), task.get('content'),
            task['content_hash'], task.get('timestamp_raw'),
            task.get('content_raw'), task.get('section_header'),
            task.get('section_date'), task.get('line_number'),
            task.get('indent_level', 0), task.get('parent_task_id'),
            task.get('bullet_type'), task.get('is_completed', False),
            task.get('is_pending', False), task.get('is_strikethrough', False)
        )
    )

def _update_embeddings_for_ids(conn, tasks: List[Dict]):
    """Regenerate embeddings only for changed things."""
    from memex_twos_mcp.embeddings import EmbeddingGenerator

    if not tasks:
        return

    print(f"Updating embeddings for {len(tasks)} changed things...")
    embedding_gen = EmbeddingGenerator()

    texts = [task['content'] for task in tasks]
    thing_ids = [task['id'] for task in tasks]
    embeddings = embedding_gen.encode_batch(texts)

    cursor = conn.cursor()
    for thing_id, embedding in zip(thing_ids, embeddings):
        embedding_blob = embedding.astype(np.float32).tobytes()
        cursor.execute(
            """
            INSERT OR REPLACE INTO thing_embeddings (thing_id, embedding, model_version)
            VALUES (?, ?, ?)
            """,
            (thing_id, embedding_blob, embedding_gen.model_name)
        )

    conn.commit()
```

**CLI:**
```bash
# Full rebuild (current behavior)
python scripts/load_to_sqlite.py data.json --mode rebuild

# Incremental sync (update/insert/delete)
python scripts/load_to_sqlite.py data.json --mode sync

# Incremental append (insert only)
python scripts/load_to_sqlite.py data.json --mode append
```

**Tests:**
- Test new thing insertion
- Test changed thing update (including FTS + embeddings)
- Test deleted thing removal (sync mode)
- Test content hash stability

---

### Phase 6: Entity Extraction Improvement [REQUIRED]
**Current Issue:** Regex + 200+ stopwords still yields false positives (convert_to_json.py:84-348)

**Approach:** Use lightweight local NLP library (spaCy small model)

**Dependencies:**
```bash
pip install spacy
python -m spacy download en_core_web_sm  # ~15MB model
```

**Implementation:**
- **File:** `src/convert_to_json.py`

```python
import spacy
from typing import List, Set

# Load spaCy model once at module level (singleton pattern)
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except OSError:
    nlp = None
    SPACY_AVAILABLE = False
    print("Warning: spaCy model not found. Falling back to regex-based extraction.")

def extract_people_nlp(text: str) -> List[str]:
    """
    Extract person names using spaCy NER.
    Falls back to regex if spaCy not available.
    """
    if not SPACY_AVAILABLE:
        return extract_people(text)  # Fallback to current regex method

    # Remove links first
    text_no_links = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Run NER
    doc = nlp(text_no_links)

    # Extract PERSON entities
    people: Set[str] = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            # Filter short names (likely false positives)
            if len(name) > 1:
                people.add(name)

    # Also catch common family terms manually (spaCy may miss these)
    family_terms = {
        'mom', 'mother', 'dad', 'father', 'sister', 'brother',
        'grandma', 'grandmother', 'grandpa', 'grandfather',
        'aunt', 'uncle', 'cousin'
    }

    text_lower = text_no_links.lower()
    for term in family_terms:
        if re.search(r'\b' + term + r'\b', text_lower):
            people.add(term.capitalize())

    return list(people)
```

**Configuration:**
- Make spaCy optional (graceful fallback)
- Add flag to disable NER: `--no-nlp` for faster parsing without NER

**Expected Improvement:**
- ~80% reduction in false positives (verbs like "Set", "Plan", "Pick" at sentence start)
- Better handling of multi-word names ("Dr. Smith", "Alex Johnson")

**Tests:**
- Test NER extraction on known-good examples
- Test fallback when spaCy unavailable
- Compare false positive rate: regex vs NER

---

## File Structure Changes

### New Files

```
src/memex_twos_mcp/
  embeddings.py          # NEW: Embedding generation module
  cache.py               # NEW: Query result cache with TTL

schema/
  migrations/            # NEW: Directory for schema version migrations
    v2_add_embeddings.sql
    v2_add_content_hash.sql
    v2_add_imports_table.sql

docs/
  v2-plan.md            # THIS FILE
  v2-implementation-notes.md  # NEW: Migration guide and design decisions
  v2-performance.md     # NEW: Performance benchmarks

tests/
  test_embeddings.py    # NEW: Embedding generation tests
  test_hybrid_search.py # NEW: Hybrid search tests
  test_incremental.py   # NEW: Incremental ingestion tests
  test_cache.py         # NEW: Cache behavior tests
  fixtures/             # NEW: Test data fixtures
    sample_100_things.json
    sample_10k_things.json

scripts/
  benchmark_performance.py  # NEW: Performance smoke test (100K things)
  migrate_to_v2.py          # NEW: Migration script for existing DBs
```

### Modified Files

```
schema/schema.sql         # Add embeddings, content_hash, imports tables
src/memex_twos_mcp/database.py  # All major query improvements
src/memex_twos_mcp/server.py    # New MCP tools for two-phase retrieval
src/convert_to_json.py          # Improved entity extraction
scripts/load_to_sqlite.py       # Incremental loading, embedding generation
pyproject.toml                  # Add new dependencies
README.md                       # Update with v2 features
```

---

## Dependency Changes

### New Dependencies (Add to pyproject.toml)

```toml
[project]
dependencies = [
    "mcp[cli]>=1.0.0",
    "PyYAML>=6.0.1",
    "python-dateutil>=2.8.2",
    # NEW v2 dependencies:
    "sentence-transformers>=2.0.0",  # Embeddings
    "sqlite-vec>=0.1.0",             # Vector search
    "numpy>=1.24.0",                 # Vector operations
    "spacy>=3.7.0",                  # NLP for entity extraction
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "flake8>=6.1.0",
    "mypy>=1.7.0",
    "types-PyYAML>=6.0.12.20250915",
    # NEW v2 dev dependencies:
    "pytest-benchmark>=4.0.0",       # Performance testing
]
```

### Installation Instructions

```bash
# Core dependencies
pip install -e .

# Download spaCy model (one-time, ~15MB)
python -m spacy download en_core_web_sm

# Sentence-transformers model downloads automatically on first use (~90MB)
# Model cached in: ~/.cache/torch/sentence_transformers/
```

---

## Migration Strategy

### For Existing Users

**Option 1: Clean Rebuild (Recommended)**
```bash
# Backup old database
cp data/processed/twos.db data/processed/twos_v1_backup.db

# Delete old database
rm data/processed/twos.db

# Re-run full pipeline with v2
python src/convert_to_json.py data/raw/twos_export.md -o data/processed/twos_data.json
python scripts/groom_data.py
python scripts/load_to_sqlite.py data/processed/twos_data_cleaned.json

# This will:
# - Apply new schema
# - Generate embeddings
# - Create content hashes
# - Enable all v2 features
```

**Option 2: In-Place Migration (Advanced)**
```bash
# Run migration script (preserves existing data)
python scripts/migrate_to_v2.py data/processed/twos.db

# This will:
# - Add new tables (embeddings, imports)
# - Add new columns (content_hash)
# - Backfill embeddings for existing things
# - Keep all existing data
```

### Breaking Changes

1. **Response Schema Changes:**
   - `search_things` now returns `relevance_score` and `snippet` fields
   - New tool `search_things_preview` returns minimal candidate data
   - Full records now require follow-up `get_things_by_ids` call

2. **Database Schema:**
   - New tables: `thing_embeddings`, `imports`
   - New columns: `things.content_hash`
   - New indexes for performance

3. **Configuration:**
   - New environment variables:
     - `MEMEX_ENABLE_EMBEDDINGS` (default: true)
     - `MEMEX_CACHE_TTL_SECONDS` (default: 900)
   - New config file options in `.memex/config.yaml`

---

## Testing Strategy

### Unit Tests

**Test Coverage Goals:**
- BM25 ranking correctness
- Snippet extraction with highlights
- Cache hit/miss behavior
- Cache TTL expiration
- Embedding generation dimensions
- RRF hybrid scoring
- Content hash stability
- Incremental upsert logic
- NER entity extraction accuracy

### Integration Tests

**Test Scenarios:**
- End-to-end search with BM25 + snippets
- Two-phase retrieval (preview → full fetch)
- Hybrid search returns semantically related results
- Incremental load updates FTS + embeddings
- Cache invalidation on data changes

### Performance Tests

**Benchmark Script:** `scripts/benchmark_performance.py`

**Test Cases:**
- 10K things: baseline performance
- 100K things: target scale
- 500K things: stretch goal

**Metrics:**
- Query latency (p50, p95, p99)
- Response payload size
- Cache hit rate
- Embedding generation time
- Index build time

**Targets (100K things, typical laptop):**
- Candidate search: <200ms median
- Hybrid search: <500ms median
- Cache hit: <10ms
- Full rebuild: <10 minutes (including embeddings)

---

## Performance Expectations

### Current (v1) Performance

| Operation | 10K Things | 100K Things | Notes |
|-----------|------------|-------------|-------|
| search_things | ~50ms | ~500ms | No ranking, chronological |
| Connection overhead | 5-10ms | 5-10ms | Per query |
| Full rebuild | ~30s | ~5min | Commit-per-insert |
| Response size | 34KB (50 results) | 34KB | Full records |

### Expected v2 Performance

| Operation | 10K Things | 100K Things | Improvement |
|-----------|------------|-------------|-------------|
| search_preview | ~30ms | ~150ms | 3x faster + 75% smaller |
| hybrid_search | ~80ms | ~400ms | Better relevance |
| Connection overhead | <1ms | <1ms | Pooling eliminates |
| Full rebuild | ~15s | ~8min | Batched commits + embeddings |
| Incremental sync | ~2s | ~20s | 10-100x faster |
| Cache hit | <5ms | <5ms | Nearly instant |

---

## Open Questions / Future Work

**Not in v2 Scope (Defer to v3):**

1. **Async MCP Server:** Current implementation is sync. Consider `asyncio` for concurrent queries.
2. **Advanced Embeddings:** Fine-tuned model for task/note domain (requires training data).
3. **Graph Relationships:** Leverage `parent_task_id` for hierarchical queries.
4. **Real-time Sync:** Watch Twos export file for changes, auto-sync.
5. **Query Builder UI:** Web interface for exploring data (outside MCP).
6. **Multi-language Support:** Currently English-only NER.

---

## Success Criteria

**v2 is considered successful if:**

✅ BM25 ranking produces measurably better relevance than chronological
✅ Two-phase retrieval reduces average response tokens by >50%
✅ Connection pooling + caching improves repeated query latency by >2x
✅ Hybrid search improves semantic query recall by >40%
✅ Incremental sync is >10x faster than full rebuild for typical updates
✅ NER-based entity extraction reduces false positives by >50%
✅ No performance regression for 10K things
✅ <200ms median search latency for 100K things
✅ All existing MCP tools remain functional (backward compatible behavior)
✅ Migration path is documented and tested

---

## Implementation Sequence

**Order of implementation (with dependencies):**

1. ✅ Create this plan document
2. **Phase 1:** BM25 + Snippets (foundation for all search improvements)
3. **Phase 2:** Two-Phase Retrieval (uses Phase 1 results)
4. **Phase 3:** Connection Pooling + Cache (independent, improves all queries)
5. **Phase 6:** Entity Extraction (independent, improves data quality)
6. **Phase 4:** Hybrid Search (requires Phase 1 complete, adds embeddings)
7. **Phase 5:** Incremental Ingestion (requires Phase 4 for embedding updates)
8. **Testing:** Comprehensive test suite across all phases
9. **Performance:** Benchmark script + smoke tests
10. **Documentation:** Update README, MCP tool docs, migration guide

**Estimated Timeline:**
- Phase 1-3: ~1-2 days (foundational improvements)
- Phase 6: ~0.5 days (entity extraction)
- Phase 4-5: ~2-3 days (complex features: embeddings, incremental)
- Testing + Docs: ~1-2 days
- **Total: ~5-8 days of focused development**

---

**End of Plan Document**
