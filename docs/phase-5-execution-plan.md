# Phase 5: Incremental Ingestion - Execution Plan

**Status:** Not Started
**Complexity:** MEDIUM
**Estimated Effort:** 1-2 days (1.5-2 days if Phase 4 completed)
**Prerequisites:** Phases 1-3 completed. Phase 4 optional but affects implementation.

---

## Goal

Implement incremental ingestion with content-hash based deduplication and upsert logic to enable 10-100x faster reloads when re-importing Twos exports. Support both sync mode (update/insert/delete) and append mode (insert only).

**Expected Impact:** 10-100x faster reloads for typical export updates (100 new things vs 10K total).

---

## Repo Reality

**Current State:**
- Full rebuild required for any data changes
- `load_to_sqlite.py` deletes database and recreates from scratch
- No change tracking or stable IDs
- FTS5 triggers auto-update on changes (good!)
- If Phase 4 complete: embeddings exist and need incremental updates

**Current Load Performance:**
- 10K things: ~3 seconds (after Phase 3 batching fix)
- 100K things: ~30 seconds
- Problem: Even adding 100 new things requires full reload

**Desired State:**
- Incremental import detects: new, changed, deleted things
- Only regenerates FTS/embeddings for changed things
- Tracks import runs for audit trail
- Supports both sync (delete removed) and append (keep all) modes

**Codebase Structure:**
```
scripts/
  load_to_sqlite.py     # 487 lines - add incremental mode

schema/
  schema.sql            # 144 lines - add content_hash, imports table

src/memex_twos_mcp/
  database.py           # 540 lines - add invalidate_cache on changes

tests/
  test_incremental.py   # NEW - test incremental ingestion
```

---

## Non-Negotiable Constraints

1. **Backward compatible:** Existing `load_to_sqlite.py` usage must still work
2. **Deterministic hashing:** Same content → same hash (for reliable change detection)
3. **Atomic operations:** Partial imports must be safe (transactional)
4. **Audit trail:** Track what changed in each import
5. **FTS sync:** FTS5 must stay in sync with things table
6. **Phase 4 compatibility:** If embeddings exist, update them incrementally
7. **No data loss:** Sync mode must be opt-in (default: append mode is safer)

---

## Deliverables

### 1. Schema Changes

**File:** `schema/schema.sql`

```sql
-- Add content hash column for change detection
ALTER TABLE things ADD COLUMN content_hash TEXT;
CREATE INDEX IF NOT EXISTS idx_things_content_hash ON things(content_hash);

-- Track import runs for audit trail
CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,
    json_file TEXT,
    mode TEXT NOT NULL,              -- 'rebuild', 'sync', 'append'
    imported_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    thing_count INTEGER,             -- Total things in import file
    new_count INTEGER,               -- New things inserted
    updated_count INTEGER,           -- Changed things updated
    deleted_count INTEGER,           -- Things deleted (sync mode only)
    duration_seconds REAL
);

-- Add metadata for tracking incremental state
INSERT OR REPLACE INTO metadata (key, value)
VALUES ('last_incremental_import', NULL);
```

**Migration Script:** `scripts/migrate_add_incremental.py`
- Add content_hash column to existing databases
- Backfill hashes for all existing things
- Create imports table
- Safe for existing installations

### 2. Content Hash Algorithm

**Stable hash based on canonical fields:**

```python
import hashlib
import json
from typing import Dict, Any

def compute_content_hash(task: Dict[str, Any]) -> str:
    """
    Compute stable content hash for change detection.

    Hash includes: timestamp, content, section_header (canonical fields only).
    Excludes: line_number, content_raw (formatting/position don't matter).

    Args:
        task: Thing dictionary

    Returns:
        SHA256 hex digest (64 chars)
    """
    # Use only canonical fields that matter for content identity
    canonical = {
        'timestamp': task.get('timestamp'),
        'content': task.get('content', '').strip(),
        'section_header': task.get('section_header', '').strip(),
        # Note: Not including tags/people (they're junction tables, updated separately)
    }

    # Sort keys for deterministic JSON
    hash_input = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(hash_input.encode()).hexdigest()

def verify_hash_stability():
    """Test that hash is deterministic."""
    task = {
        'timestamp': '2024-01-01T10:00:00',
        'content': 'Test task',
        'section_header': 'Mon, Jan 1',
        'line_number': 42  # Should not affect hash
    }

    hash1 = compute_content_hash(task)
    hash2 = compute_content_hash(task)
    assert hash1 == hash2, "Hash must be deterministic"
```

**Critical Requirements:**
- Deterministic (same input → same hash)
- Fast to compute (<1ms per thing)
- Exclude formatting/position fields
- Stable across parser versions

### 3. Incremental Load Implementation

**File:** `scripts/load_to_sqlite.py`

Add function:
```python
def incremental_load(
    conn: sqlite3.Connection,
    tasks: list,
    source_file: str,
    json_file: str,
    mode: str = 'append'
) -> Dict[str, int]:
    """
    Incrementally load tasks using upsert logic.

    Modes:
    - 'append': Insert new only, don't delete (safest, default)
    - 'sync': Update changed, insert new, delete removed (full sync)
    - 'rebuild': Delete all and rebuild (current behavior)

    Args:
        conn: Open SQLite connection
        tasks: List of thing dictionaries
        source_file: Path to original Twos export
        json_file: Path to JSON file
        mode: Import mode ('append', 'sync', 'rebuild')

    Returns:
        Stats dict: {new_count, updated_count, deleted_count, duration_seconds}
    """
    import time
    start_time = time.time()

    print(f"Incremental load mode: {mode}")

    cursor = conn.cursor()

    # Step 1: Compute hashes for incoming tasks
    incoming = {}
    for task in tasks:
        content_hash = compute_content_hash(task)
        task['content_hash'] = content_hash
        incoming[task['id']] = (task, content_hash)

    # Step 2: Fetch existing hashes
    cursor.execute("SELECT id, content_hash FROM things")
    existing = {row[0]: row[1] for row in cursor.fetchall()}

    # Step 3: Categorize changes
    new_ids = set(incoming.keys()) - set(existing.keys())
    deleted_ids = set(existing.keys()) - set(incoming.keys()) if mode == 'sync' else set()

    updated_ids = set()
    for thing_id in set(incoming.keys()) & set(existing.keys()):
        if incoming[thing_id][1] != existing[thing_id]:
            updated_ids.add(thing_id)

    print(f"Changes detected: {len(new_ids)} new, {len(updated_ids)} updated, {len(deleted_ids)} deleted")

    # Step 4: Apply changes

    # Insert new things
    if new_ids:
        print(f"Inserting {len(new_ids)} new things...")
        for thing_id in new_ids:
            task, _ = incoming[thing_id]
            _insert_thing(cursor, task)

    # Update changed things (delete + reinsert to trigger FTS update)
    if updated_ids:
        print(f"Updating {len(updated_ids)} changed things...")
        for thing_id in updated_ids:
            task, _ = incoming[thing_id]
            # Delete old version (cascades to FTS via trigger)
            cursor.execute("DELETE FROM things WHERE id = ?", (thing_id,))
            # Insert new version
            _insert_thing(cursor, task)

    # Delete removed things (sync mode only)
    if mode == 'sync' and deleted_ids:
        print(f"Deleting {len(deleted_ids)} removed things...")
        for thing_id in deleted_ids:
            cursor.execute("DELETE FROM things WHERE id = ?", (thing_id,))

    conn.commit()

    # Step 5: Update related entities (people, tags, links)
    # Only for new/updated things
    changed_things = [incoming[tid][0] for tid in (new_ids | updated_ids)]
    if changed_things:
        _update_people(conn, changed_things)
        _update_tags(conn, changed_things)
        _update_links(conn, changed_things)

    # Step 6: Update embeddings incrementally (if Phase 4 exists)
    if embeddings_enabled() and changed_things:
        _update_embeddings_incremental(conn, changed_things)

    # Step 7: Record import run
    duration = time.time() - start_time
    cursor.execute("""
        INSERT INTO imports (
            source_file, json_file, mode, thing_count,
            new_count, updated_count, deleted_count, duration_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        source_file, json_file, mode, len(tasks),
        len(new_ids), len(updated_ids), len(deleted_ids), duration
    ))

    cursor.execute("""
        INSERT OR REPLACE INTO metadata (key, value)
        VALUES ('last_incremental_import', ?)
    """, (datetime.now().isoformat(),))

    conn.commit()

    print(f"Incremental import completed in {duration:.2f}s")

    return {
        'new_count': len(new_ids),
        'updated_count': len(updated_ids),
        'deleted_count': len(deleted_ids),
        'duration_seconds': duration
    }

def _insert_thing(cursor, task):
    """Insert single thing with all fields including content_hash."""
    cursor.execute("""
        INSERT INTO things (
            id, timestamp, content, content_hash,
            timestamp_raw, content_raw, section_header, section_date,
            line_number, indent_level, parent_task_id, bullet_type,
            is_completed, is_pending, is_strikethrough
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        task['id'],
        task.get('timestamp'),
        task.get('content'),
        task['content_hash'],  # NEW
        task.get('timestamp_raw'),
        task.get('content_raw'),
        task.get('section_header'),
        task.get('section_date'),
        task.get('line_number'),
        task.get('indent_level', 0),
        task.get('parent_task_id'),
        task.get('bullet_type'),
        task.get('is_completed', False),
        task.get('is_pending', False),
        task.get('is_strikethrough', False)
    ))

def _update_embeddings_incremental(conn, tasks: List[Dict]):
    """
    Regenerate embeddings only for changed things.

    Only runs if Phase 4 (embeddings) is implemented.
    """
    if not embeddings_available():
        return

    print(f"Updating embeddings for {len(tasks)} changed things...")

    from memex_twos_mcp.embeddings import EmbeddingGenerator

    embedding_gen = EmbeddingGenerator()
    texts = [task['content'] for task in tasks]
    thing_ids = [task['id'] for task in tasks]

    # Generate new embeddings
    embeddings = embedding_gen.encode_batch(texts, show_progress=True)

    cursor = conn.cursor()
    for thing_id, embedding in zip(thing_ids, embeddings):
        embedding_blob = embedding.astype(np.float32).tobytes()

        # Upsert (replace if exists)
        cursor.execute("""
            INSERT OR REPLACE INTO thing_embeddings (thing_id, embedding, model_version)
            VALUES (?, ?, ?)
        """, (thing_id, embedding_blob, embedding_gen.model_name))

    conn.commit()
    print(f"Updated {len(embeddings)} embeddings")
```

### 4. CLI Updates

**File:** `scripts/load_to_sqlite.py`

Update argument parser:
```python
parser.add_argument(
    '--mode',
    choices=['rebuild', 'sync', 'append'],
    default='rebuild',
    help=(
        "Import mode: "
        "'rebuild' (delete all, full reload), "
        "'sync' (update changed, delete removed), "
        "'append' (insert new only, safest)"
    )
)

parser.add_argument(
    '--incremental',
    action='store_true',
    help="Enable incremental mode (same as --mode=append)"
)
```

**Usage:**
```bash
# Full rebuild (current behavior)
python scripts/load_to_sqlite.py data.json

# Incremental append (insert new only)
python scripts/load_to_sqlite.py data.json --mode=append

# Incremental sync (update/insert/delete)
python scripts/load_to_sqlite.py data.json --mode=sync

# Shorthand for append mode
python scripts/load_to_sqlite.py data.json --incremental
```

### 5. Cache Invalidation

**File:** `src/memex_twos_mcp/database.py`

Add method:
```python
def invalidate_cache_on_import(self):
    """
    Invalidate query cache after data import.

    Call this from MCP server if you add a "reimport" tool.
    """
    self.cache.invalidate_all()
    print("Query cache invalidated due to data import")
```

**Note:** MCP server process needs restart after incremental import for cache to clear. Future: add MCP tool to trigger reimport + cache invalidation.

---

## Testing Requirements

### Unit Tests

**File:** `tests/test_incremental.py` (NEW)

```python
def test_compute_content_hash_deterministic():
    """Test hash stability."""
    task = {
        'timestamp': '2024-01-01T10:00:00',
        'content': 'Test task',
        'section_header': 'Mon, Jan 1',
        'line_number': 42
    }

    hash1 = compute_content_hash(task)
    hash2 = compute_content_hash(task)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex

def test_incremental_insert_new(tmp_path):
    """Test inserting new things in incremental mode."""
    db_path = tmp_path / "twos.db"

    # Initial load
    tasks1 = [
        {'id': 'task_001', 'timestamp': '2024-01-01T10:00:00', 'content': 'Task 1'},
        {'id': 'task_002', 'timestamp': '2024-01-02T10:00:00', 'content': 'Task 2'},
    ]
    _load_db(db_path, tasks1, mode='rebuild')

    # Incremental load with new task
    tasks2 = [
        {'id': 'task_001', 'timestamp': '2024-01-01T10:00:00', 'content': 'Task 1'},
        {'id': 'task_002', 'timestamp': '2024-01-02T10:00:00', 'content': 'Task 2'},
        {'id': 'task_003', 'timestamp': '2024-01-03T10:00:00', 'content': 'Task 3'},
    ]
    stats = _load_db(db_path, tasks2, mode='append')

    assert stats['new_count'] == 1
    assert stats['updated_count'] == 0
    assert stats['deleted_count'] == 0

    # Verify database has 3 things
    db = TwosDatabase(db_path)
    count = db.get_stats()['total_things']
    assert count == 3

def test_incremental_update_changed(tmp_path):
    """Test updating changed things."""
    db_path = tmp_path / "twos.db"

    # Initial load
    tasks1 = [
        {'id': 'task_001', 'timestamp': '2024-01-01T10:00:00', 'content': 'Original content'},
    ]
    _load_db(db_path, tasks1, mode='rebuild')

    # Change content
    tasks2 = [
        {'id': 'task_001', 'timestamp': '2024-01-01T10:00:00', 'content': 'Updated content'},
    ]
    stats = _load_db(db_path, tasks2, mode='sync')

    assert stats['new_count'] == 0
    assert stats['updated_count'] == 1
    assert stats['deleted_count'] == 0

    # Verify content updated
    db = TwosDatabase(db_path)
    thing = db.get_thing_by_id('task_001')
    assert thing['content'] == 'Updated content'

def test_incremental_delete_removed_sync_mode(tmp_path):
    """Test deleting removed things in sync mode."""
    db_path = tmp_path / "twos.db"

    # Initial load with 3 things
    tasks1 = [
        {'id': 'task_001', 'content': 'Task 1'},
        {'id': 'task_002', 'content': 'Task 2'},
        {'id': 'task_003', 'content': 'Task 3'},
    ]
    _load_db(db_path, tasks1, mode='rebuild')

    # Incremental load with only 2 things (task_003 removed)
    tasks2 = [
        {'id': 'task_001', 'content': 'Task 1'},
        {'id': 'task_002', 'content': 'Task 2'},
    ]
    stats = _load_db(db_path, tasks2, mode='sync')

    assert stats['deleted_count'] == 1

    # Verify task_003 is gone
    db = TwosDatabase(db_path)
    assert db.get_thing_by_id('task_003') is None

def test_incremental_preserve_removed_append_mode(tmp_path):
    """Test that append mode keeps removed things."""
    db_path = tmp_path / "twos.db"

    # Initial load
    tasks1 = [
        {'id': 'task_001', 'content': 'Task 1'},
        {'id': 'task_002', 'content': 'Task 2'},
    ]
    _load_db(db_path, tasks1, mode='rebuild')

    # Incremental append with only task_001 (task_002 "removed" from export)
    tasks2 = [
        {'id': 'task_001', 'content': 'Task 1'},
    ]
    stats = _load_db(db_path, tasks2, mode='append')

    assert stats['deleted_count'] == 0

    # Verify task_002 still exists
    db = TwosDatabase(db_path)
    assert db.get_thing_by_id('task_002') is not None

def test_fts_stays_in_sync(tmp_path):
    """Test that FTS index updates with incremental changes."""
    db_path = tmp_path / "twos.db"

    # Initial load
    tasks1 = [
        {'id': 'task_001', 'content': 'doctor appointment'},
    ]
    _load_db(db_path, tasks1, mode='rebuild')

    # Update content
    tasks2 = [
        {'id': 'task_001', 'content': 'dentist appointment'},  # Changed
    ]
    _load_db(db_path, tasks2, mode='sync')

    # Search should find new content, not old
    db = TwosDatabase(db_path)
    results = db.search_content('dentist')
    assert len(results) == 1

    results = db.search_content('doctor')
    assert len(results) == 0  # Old content gone from FTS

def test_import_audit_trail(tmp_path):
    """Test that import runs are tracked."""
    db_path = tmp_path / "twos.db"

    tasks = [{'id': 'task_001', 'content': 'Task 1'}]
    _load_db(db_path, tasks, mode='rebuild', source_file='test.md')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM imports ORDER BY id DESC LIMIT 1")
    import_record = cursor.fetchone()

    assert import_record is not None
    assert 'test.md' in str(import_record)  # Source file tracked
```

### Integration Tests

```python
def test_incremental_with_embeddings(tmp_path):
    """Test that embeddings update incrementally (if Phase 4 exists)."""
    # Skip if embeddings not available
    if not embeddings_available():
        pytest.skip("Phase 4 not implemented")

    # Initial load
    # Change thing content
    # Verify embedding updated for changed thing only
```

---

## Performance Targets

### Incremental Import Performance

| Scenario | Full Rebuild | Incremental | Speedup |
|----------|--------------|-------------|---------|
| Add 100 new (10K total) | ~3s | ~0.3s | 10x |
| Update 100 (10K total) | ~3s | ~0.5s | 6x |
| Add 1000 new (100K total) | ~30s | ~3s | 10x |

### Memory Usage
- Should not increase significantly vs full rebuild
- Hash computation: O(n) time, O(1) space per thing

---

## Migration Path

**For Existing Installations:**

```bash
# Option 1: Migrate existing database
python scripts/migrate_add_incremental.py data/processed/twos.db

# Option 2: Rebuild with new schema
rm data/processed/twos.db
python scripts/load_to_sqlite.py data/processed/twos_data.json
```

**Migration script:**
```python
# scripts/migrate_add_incremental.py
def migrate_add_incremental(db_path: Path):
    """Add incremental ingestion support to existing database."""
    conn = sqlite3.connect(db_path)

    # Add content_hash column
    conn.execute("ALTER TABLE things ADD COLUMN content_hash TEXT")

    # Backfill hashes
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, content, section_header FROM things")

    for row in cursor.fetchall():
        thing_id, timestamp, content, section_header = row
        task = {
            'timestamp': timestamp,
            'content': content,
            'section_header': section_header
        }
        content_hash = compute_content_hash(task)

        conn.execute(
            "UPDATE things SET content_hash = ? WHERE id = ?",
            (content_hash, thing_id)
        )

    # Create imports table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS imports (...)
    """)

    conn.commit()
    print("Migration complete")
```

---

## Quality Bar

### Must Have
- ✅ Incremental mode correctly identifies new/changed/deleted things
- ✅ FTS5 stays in sync with changes
- ✅ Content hash is deterministic
- ✅ All tests passing
- ✅ Backward compatible (rebuild mode still works)
- ✅ Import audit trail functional

### Should Have
- ✅ 10x faster for typical incremental updates
- ✅ Embeddings update incrementally (if Phase 4 exists)
- ✅ Cache invalidation on import
- ✅ Migration script for existing databases

### Nice to Have
- Resume on failure (transaction safety)
- Conflict resolution strategy
- MCP tool for triggering reimport

---

## Success Criteria

**Phase 5 is complete when:**

1. `incremental_load()` function works correctly
2. Content hash computation is deterministic
3. All three modes work (rebuild, sync, append)
4. FTS5 stays in sync with changes
5. Import audit trail tracks all runs
6. All tests passing (unit + integration)
7. Migration script works on existing databases
8. Documentation updated

**Deliverables:**
- Working incremental ingestion
- Content hash algorithm
- Import audit trail
- Comprehensive tests
- Migration guide

---

## Common Pitfalls to Avoid

1. **Non-deterministic hashing:** Use sorted JSON, exclude volatile fields
2. **FTS desync:** Test that FTS updates correctly on delete+reinsert
3. **Partial transactions:** Ensure atomicity (commit at end, or use savepoints)
4. **Embedding orphans:** Clean up old embeddings when things deleted
5. **Cache staleness:** Document that MCP server needs restart after import
6. **Default mode:** Make 'rebuild' default to avoid breaking existing workflows
7. **Hash collision:** Use SHA256 (64 char hex), not MD5 or shorter hashes

---

## Notes for Implementation

- Start with content hash algorithm (unit test in isolation)
- Add schema changes next (test migration script)
- Implement incremental_load() function
- Test FTS sync thoroughly (common failure point)
- Add CLI arguments last
- Test on real data (export from Twos app)
- Profile performance with 10K and 100K datasets

---

**Dependencies:**

**If Phase 4 NOT implemented:**
- Simpler: no embedding updates
- Faster: skip embedding generation

**If Phase 4 IS implemented:**
- Must update embeddings incrementally
- Add `_update_embeddings_incremental()` function
- Test embedding sync

---

**End of Phase 5 Execution Plan**
