# Claude Context: memex-twos-mcp

## Project Overview

Model Context Protocol (MCP) server for analyzing personal thing data from Twos app exports. Primary goals are learning MCP fundamentals, data engineering, and enabling LLM-powered life pattern analysis.

## Architecture & Data Flow

```
memex-twos-data repo (raw exports)
    ↓
src/convert_to_json.py (parser)
    ↓
data/processed/twos_data.json (intermediate)
    ↓
LLM grooming review
    ↓
SQLite database
    ↓
MCP server
    ↓
Claude queries and analysis
```

## Implemented Features

- ✅ Project structure and documentation
- ✅ JSON converter with rich metadata extraction
- ✅ LLM-assisted data grooming (identify duplicates, normalize entities, suggest schema improvements)
- ✅ BM25 Full-Text Search
- ✅ Two-Phase Retrieval with Caching
- ✅ Hybrid Search (semantic + lexical with RRF)
- ✅ Incremental Ingestion (content-hash based change detection)
- ✅ Entity Extraction (spaCy NER for people extraction)
- ✅ **List Semantics** (section-based queries, date/topic lists, proper "list" understanding)
- ✅ **TimePacks** (day/week/month rollups for "what happened last week?" queries)
- ✅ **ThreadPacks** (active tag/person thread indices for activity tracking)
- ✅ **MonthlySummaries** (LLM-powered semantic summaries with suggested questions)

## File Structure

```
src/
  convert_to_json.py          # Markdown → JSON parser
  memex_twos_mcp/             # MCP server package
    tools/                    # MCP tool implementations
    resources/                # MCP resource definitions

data/
  raw/                        # Gitignored - source data
  processed/                  # Gitignored - JSON/SQLite outputs

tests/                        # Test suite
```

## Code Conventions

**Python**:
- Type hints for function signatures
- Docstrings for modules, classes, public functions
- Descriptive variable names (no abbreviations unless obvious)
- Follow PEP 8

**Git/Commits**: Follow the ⚠️ section at the top of this file before any git operations

## Data Model

### JSON Schema (current)

Each thing object contains:
```json
{
  "id": "task_00001",
  "line_number": 2,
  "timestamp": "2023-10-17T16:23:00",          // ISO format
  "timestamp_raw": "10/17/23 4:23 pm",
  "section_header": "Sun, Oct 8, 2023",         // Original Day grouping
  "section_date": "10/27/23 9:14 pm",
  "content": "cleaned text",
  "content_raw": "• original line with timestamp",
  "indent_level": 0,                            // Tabs for hierarchy
  "parent_task_id": "task_00042",               // Null if top-level
  "bullet_type": "bullet|checkbox_done|checkbox_pending|dash",
  "is_completed": false,
  "is_pending": false,
  "is_strikethrough": false,
  "links": [{"text": "...", "url": "..."}],
  "tags": ["journal", "dinner"],                // Extracted from #tag#
  "people_mentioned": ["Alex", "Pat"]          // Simple heuristic
}
```

**Known Issues**:
- Strikethrough detection simple (single dash pattern)
- No semantic understanding of thing relationships
- Date/time parsing limited to specific formats

**People Extraction**:
- **With spaCy NER** (recommended): High accuracy (~90% precision), low false positives
- **Without spaCy** (fallback): Regex-based extraction, prone to false positives (verbs, months)
- Install NER: `pip install -e ".[ner]" && python -m spacy download en_core_web_sm`

## Running the Converter

```bash
# With NER (recommended, requires spaCy)
python3 src/convert_to_json.py data/raw/input.md -o data/processed/output.json --pretty

# Without NER (fallback, no extra dependencies)
python3 src/convert_to_json.py data/raw/input.md -o data/processed/output.json --pretty --no-ner
```

## SQLite Schema

Current tables:
- `things` (core thing data, includes `content_hash`, `item_type`, `list_id`, `is_substantive`)
- `lists` (section/header metadata with type, dates, boundaries, counts)
- `thing_lists` (many-to-many thing-list relationships with position)
- `people` (extracted and normalized)
- `tags` (normalized taxonomy)
- `links` (URLs with metadata)
- `thing_people` (thing-person relationships)
- `thing_tags` (thing-tag relationships)
- `things_fts` (FTS5 search)
- `thing_embeddings` (semantic embeddings)
- `vec_index` (vector similarity search)
- `imports` (import audit trail: mode, counts, duration)
- `metadata` (versioning and stats, includes incremental import tracking)
- **`rollups` + `rollup_evidence`** (day/week/month time-based rollups - Phase 7)
- **`month_summaries` + `month_summary_evidence`** (LLM semantic summaries - Phase 8)
- **`threads` + `thread_evidence` + `threads_fts`** (tag/person activity indices - Phase 9)

## MCP Server Design

**Resources**:
- `twos://database/stats` - Database statistics
- `twos://database/people` - People list
- `twos://database/tags` - Tag list

**Tools**:
- `query_things_by_date(start_date, end_date, filters)` - Timestamp-based queries
- `search_things(query)` - Keyword search for exact matches (BM25)
- `semantic_search(query)` - ⭐ Semantic + keyword search - for conceptual queries
- `get_person_things(person_name)` - Things mentioning a person
- `get_tag_things(tag_name)` - Things with a tag
- `get_things_stats()` - Database statistics
- **⭐ `get_list_by_date(date)` - Get ALL items on a date's list**
- **`get_list_by_name(name, list_type)` - Get ALL items on a named list**
- **`list_all_lists(list_type, limit)` - Get all lists with stats**
- **`search_within_list(query, list_id/date/name)` - Search within a list**
- **⭐ `get_timepack(rollup_id)` - Get day/week/month rollup** (Phase 7)
- **`list_timepacks(kind, limit)` - List time-based rollups** (Phase 7)
- **⭐ `get_month_summary(month_id, offset)` - Get LLM semantic summary** (Phase 8)
- **`list_threads(status, kind, limit)` - List active/stale threads** (Phase 9)
- **`get_thread(thread_id)` - Get tag or person thread details** (Phase 9)
- **`get_thread_highlights(thread_id, limit)` - Get thread highlights** (Phase 9)

**Prompts**:
- Life narrative generation templates
- Analysis prompt templates

## Hybrid Search

**Overview:**
Combines lexical (BM25) and semantic (vector) search using Reciprocal Rank Fusion (RRF) for better conceptual query matching.

**Use Cases:**
- **Lexical search** (search_things): Exact keyword matching - "doctor appointment" finds "doctor" and "appointment"
- **Hybrid search** (hybrid_search): Semantic understanding - "medical checkup" finds "doctor", "dentist", "physician", etc.

**Dependencies:**
- sentence-transformers (embedding model)
- sqlite-vec (vector similarity search)
- numpy (vector operations)

**Installation:**
```bash
pip install sentence-transformers sqlite-vec numpy

# Download embedding model (~90MB)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Usage:**
```bash
# Load data with embeddings (default)
python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned.json

# Disable embeddings (faster loading, lexical-only)
MEMEX_DISABLE_EMBEDDINGS=1 python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned.json

# Migrate existing database
python3 scripts/migrate_add_embeddings.py data/processed/twos.db
```

**How It Works:**
1. Generates 384-dimensional embeddings for all thing content using all-MiniLM-L6-v2
2. Stores embeddings in `thing_embeddings` table and `vec_index` virtual table
3. During hybrid_search:
   - Phase 1: BM25 lexical search for keyword matches
   - Phase 2: Vector similarity search for semantic matches
   - Phase 3: Reciprocal Rank Fusion (RRF) merges rankings with configurable weights

**Performance:**
- Embedding generation: ~10K things in <30 seconds (CPU)
- Hybrid search: <200ms median (10K things)
- Storage overhead: ~1.5KB per thing (~15MB for 10K things)

**Graceful Degradation:**
- If embeddings unavailable: Falls back to lexical-only search
- If sqlite-vec unavailable: Hybrid search disabled, lexical search still works
- All existing tools continue to work without embeddings

## Incremental Ingestion

**Overview:**
Content-hash based change detection for 10-100x faster database updates when adding or modifying small numbers of things.

**Problem Solved:**
- Before: Adding 100 new things to 10K database = full rebuild (~3s)
- After: Adding 100 new things = incremental update (~0.3s, 10x faster)

**Implementation:**
- Content hashing: SHA256 of (timestamp + content + section_header)
- Change detection: Compare incoming vs existing hashes
- Upsert logic: Insert new, update changed, optionally delete removed
- Embedding preservation: Only regenerate embeddings for changed things
- Audit trail: All imports tracked in `imports` table

**Usage:**
```bash
# First load (full rebuild)
python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned.json

# Subsequent updates (incremental modes)
python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned.json --mode=append  # Insert new only
python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned.json --incremental  # Shorthand for append
python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned.json --mode=sync    # Insert new, update changed, delete removed

# Migrate existing database
python3 scripts/migrate_add_incremental.py data/processed/twos.db
```

**Three Modes:**
1. **rebuild** (default): Delete all, full reload - use for first import or major schema changes
2. **append**: Insert new only, safest for incremental additions (recommended for daily updates)
3. **sync**: Insert new, update changed, delete removed - use when export file is authoritative

**Schema Changes:**
- Added `content_hash TEXT` column to `things` table
- Added `imports` table for audit trail (mode, counts, duration, timestamp)
- Added metadata tracking for last incremental import

**Performance:**
- Add 100 new things (10K total): ~0.3s (10x faster than rebuild)
- Update 100 things (10K total): ~0.5s (6x faster than rebuild)
- Embedding generation: Only for changed things (not entire dataset)

**Key Features:**
- **Deterministic hashing**: Same content always produces same hash
- **FTS synchronization**: Delete+reinsert triggers FTS update automatically
- **Embedding preservation**: Embeddings only regenerated for changed things
- **Import audit trail**: Every import run tracked with stats (new/updated/deleted counts, duration)
- **Backward compatible**: Default mode is 'rebuild' (existing behavior)

**Migration:**
For existing databases without incremental support:
```bash
python3 scripts/migrate_add_incremental.py data/processed/twos.db
```
- Adds `content_hash` column
- Backfills hashes for existing things
- Creates `imports` table
- Safe to run on existing databases (idempotent)

## List Semantics (Phase 6)

**Overview:**
Explicit metadata for section/header-based list groupings. Fixes the mismatch between "things timestamped on a date" vs "all things on that date's list."

**Problem Solved:**
- Before: User asks "what's on my list for today?" → returns only timestamped items (misses most things)
- After: User asks "what's on my list for today?" → returns ALL items under that day's section header

**Implementation:**
- **`lists` table**: Catalog of all sections with type (date/topic), boundaries, counts
- **`item_type` column**: Classify things as content/divider/header/metadata
- **`list_id` column**: Denormalized FK to lists for fast queries
- **`is_substantive` column**: Cached flag for filtering noise
- **`thing_lists` junction**: Many-to-many relationships with position

**Schema Changes:**
```sql
CREATE TABLE lists (
    list_id TEXT PRIMARY KEY,              -- 'date_2025-12-30', 'topic_tech-projects_5678'
    list_type TEXT NOT NULL,               -- 'date' | 'topic' | 'category' | 'metadata'
    list_name TEXT NOT NULL,               -- Normalized: '2025-12-30' | 'Tech Projects'
    list_name_raw TEXT NOT NULL,           -- Original: 'Mon, Dec 30, 2025'
    list_date TEXT,                        -- ISO date for type='date', NULL otherwise
    start_line INTEGER NOT NULL,           -- First line of section
    end_line INTEGER NOT NULL,             -- Last line of section
    item_count INTEGER,                    -- Total things in range
    substantive_count INTEGER,             -- Things with item_type='content'
    created_at TEXT NOT NULL
);

ALTER TABLE things ADD COLUMN item_type TEXT DEFAULT 'content';
ALTER TABLE things ADD COLUMN list_id TEXT;
ALTER TABLE things ADD COLUMN is_substantive BOOLEAN DEFAULT 1;
```

**Usage (New Data):**
```bash
# Step 1: Convert export to JSON
python3 src/convert_to_json.py data/raw/twos_export.md -o data/processed/twos_data.json

# Step 2: Build list metadata
python3 scripts/build_list_metadata.py data/processed/twos_data.json

# Step 3: Load to SQLite (automatically loads list metadata)
python3 scripts/load_to_sqlite.py data/processed/twos_data_with_lists.json
```

**Usage (Existing Database):**
```bash
# Migrate existing database
python3 scripts/migrate_add_list_semantics.py data/processed/twos.db
```

**New MCP Tools:**
- **get_list_by_date(date)** - Get ALL items on a date's list (e.g., "what's on my list for today?")
- **get_list_by_name(name)** - Get ALL items on a topic list (e.g., "what's on Tech Projects?")
- **list_all_lists()** - Get all lists with stats
- **search_within_list(query, list_id/date/name)** - Search within a specific list

**Example Queries:**
```sql
-- What's on my list for Dec 30, 2025?
SELECT * FROM things WHERE list_id = 'date_2025-12-30' AND item_type = 'content';

-- What's on my Tech Projects list?
SELECT t.* FROM things t
JOIN lists l ON t.list_id = l.list_id
WHERE l.list_name = 'Tech Projects' AND t.item_type = 'content';

-- All date-based lists in December 2025
SELECT * FROM lists
WHERE list_type = 'date' AND list_date BETWEEN '2025-12-01' AND '2025-12-31';
```

**Key Features:**
- **Deterministic list_id**: Same content always produces same ID across imports
- **Item classification**: Filters out dividers/headers automatically
- **Date + topic lists**: Handles both daily lists and arbitrary named sections
- **Incremental-compatible**: Works with incremental import modes
- **Fast queries**: Denormalized list_id enables single-table lookups

**Performance:**
- List metadata build: ~10K things in <5 seconds
- Query by date: <50ms (10K things)
- Migration (existing DB): ~10 seconds (10K things)

## TimePacks: Time-Based Rollups (Phase 7)

**Overview:**
Precomputed day/week/month rollups for fast "what happened last week/month?" queries with minimal token usage.

**Problem Solved:**
- Before: "What happened last month?" → query all things, process 300+ items, high token cost
- After: "What happened last month?" → 1 query, compact pack format, <100 tokens

**Implementation:**
- **Builder:** `scripts/build_timepacks.py`
- **Tables:** `rollups` (rollup packs), `rollup_evidence` (highlight/evidence links)
- **Pack Format:** TP1 (pipe-delimited, ~800 chars max)
- **Highlight Scoring:** Deterministic weighted formula (40% recency, 30% entity density, 20% length, 10% status)
- **Incremental:** SHA256 src_hash of (thing_id + content_hash) for change detection

**TP1 Pack Format:**
```
TP1|k=<d|w|m>|s=<start_date>|e=<end_date>|n=<total>|cx=<completed>|pn=<pending>|st=<strikethrough>|tg=<tag:count,...>|pp=<person:count,...>|kw=<word,word,...>|hi=<thing_id~label;...>
```

**Usage:**
```bash
# Build rollups for last 12 months (default: day, week, month)
python3 scripts/build_timepacks.py --db data/processed/twos.db

# Build specific kinds only
python3 scripts/build_timepacks.py --db data/processed/twos.db --kinds d,w

# Force rebuild (ignore incremental)
python3 scripts/build_timepacks.py --db data/processed/twos.db --force
```

**MCP Tools:**
- `get_timepack(rollup_id)` - Get specific rollup (e.g., `d:2025-12-30`, `w:2025-12-22`, `m:2025-12`)
- `list_timepacks(kind, limit)` - List recent rollups by kind

**Performance:**
- Build time: ~3-5 seconds for 10K things (730 days, 104 weeks, 24 months)
- Storage: ~860 rollup rows for 2 years of data
- Query time: <50ms for single rollup fetch

## MonthlySummaries: LLM Semantic Framing (Phase 8)

**Overview:**
LLM-powered semantic summaries that provide "system prompt lite" context for monthly exploration.

**Purpose:**
- **TimePacks (Phase 7):** Mechanical facts - "what happened" (counts, tags, keywords)
- **MonthlySummaries (Phase 8):** Semantic framing - "so what" (themes, insights, suggested questions)

**Problem Solved:**
When user asks "What happened this month?", provide semantic context that guides LLM exploration:
- Identifies themes (e.g., "work_planning", "health_care")
- Suggests relevant follow-up questions
- Anchors all insights to specific thing IDs (no hallucinations)

**Implementation:**
- **Builder:** `scripts/build_month_summaries.py`
- **Tables:** `month_summaries` (packs + suggested_questions JSON), `month_summary_evidence`
- **LLM:** Invokes Claude via Claude Code CLI
- **Pack Format:** MS1 (pipe-delimited, ~1200 chars max)
- **Questions:** Separate JSON column with anchors and rationale

**MS1 Pack Format:**
```
MS1|m=<YYYY-MM>|n=<total>|tg=<tag:count,...>|pp=<person:count,...>|th=<theme@thing_id,...;...>|hi=<thing_id~label;...>|nq=<question_count>
```

**Suggested Questions JSON:**
```json
{
  "questions": [
    {
      "rank": 1,
      "text": "What progress on Q4 planning this month?",
      "anchors": ["task_08190", "task_08155"],
      "thread_id": "thr:tag:work",
      "rationale": "High work activity, multiple planning items"
    }
  ]
}
```

**Usage:**
```bash
# Build summaries for last 12 months (requires Claude Code CLI)
python3 scripts/build_month_summaries.py --db data/processed/twos.db

# Build last 24 months
python3 scripts/build_month_summaries.py --db data/processed/twos.db --months 24

# Dry run (show what would be built, no LLM calls)
python3 scripts/build_month_summaries.py --db data/processed/twos.db --dry-run
```

**MCP Tools:**
- `get_month_summary(month_id, offset)` - Get summary (offset=0 for current month, 1 for last month, etc.)
- `list_month_summaries(limit)` - List recent summaries

**Validation:**
- All thing_ids in themes/highlights must be from provided candidates
- All themes must cite at least 2 thing_ids
- Questions must be <100 chars and anchored to evidence
- Invalid LLM responses trigger retry (max 2 attempts)

**Performance:**
- Build time: ~5-10 seconds per month (LLM API calls)
- 12 months: ~1-2 minutes total
- Storage: ~1KB per month summary (12-24 rows)

## ThreadPacks: Active Thread Indices (Phase 9)

**Overview:**
Deterministic activity indices for tags and people. Answers "What's active with Alice?" or "Show me recent #work threads" with no embeddings required.

**Problem Solved:**
- Before: "What's active with Alice?" → query all things, filter by person, manual recency analysis
- After: "What's active with Alice?" → 1 query, pre-indexed thread with highlights and activity stats

**Implementation:**
- **Builder:** `scripts/build_threads.py`
- **Tables:** `threads` (thread packs), `thread_evidence`, `threads_fts` (search)
- **Thread Types:** Single-tag threads (e.g., `thr:tag:work`), single-person threads (e.g., `thr:person:alice`)
- **Active Window:** Configurable (default 90 days)
- **Status:** `active` (activity in last 90d), `stale` (no recent activity), `archived` (manually archived)

**TH1 Pack Format:**
```
TH1|id=<thread_id>|st=<active|stale>|last=<last_ts>|n=<total>|a90=<count_90d>|kw=<word,word,...>|hi=<thing_id~label;...>
```

**Usage:**
```bash
# Build all threads (default: tag + person, 90-day active window)
python3 scripts/build_threads.py --db data/processed/twos.db

# Custom active window (60 days)
python3 scripts/build_threads.py --db data/processed/twos.db --active-days 60

# Build only tag threads
python3 scripts/build_threads.py --db data/processed/twos.db --kinds tag

# Force rebuild
python3 scripts/build_threads.py --db data/processed/twos.db --force
```

**MCP Tools:**
- `list_threads(status, kind, limit)` - List threads (filter by active/stale, tag/person)
- `get_thread(thread_id)` - Get thread details
- `get_thread_highlights(thread_id, limit)` - Get highlight things
- (Implicit) `search_threads(query)` via threads_fts

**Highlight Scoring:**
For threads, scoring focuses on recency within the active window:
- 50% recency (exponential decay from now)
- 30% entity density (tags + people)
- 20% content length

**Incremental Rebuild:**
- Collects affected threads by checking thing_tags/thing_people changes
- Recomputes src_hash only for affected threads
- Skips unchanged threads (fast updates)

**Performance:**
- Build time: ~1-3 seconds for 10K things with 50 tags + 100 people
- Storage: ~150 thread rows typical (varies by tag/person count)
- Query time: <50ms for thread fetch, <100ms for FTS search

**Future Enhancements (Deferred):**
- Multi-tag intersection threads (e.g., `thr:tag:work+urgent`) - currently computed at query time
- Thread archival with archived_at timestamp

## Unified Builder Orchestrator (build_all.py)

**Overview:**
Coordinates execution of all three derived index builders with dependency management and error handling.

**Usage:**
```bash
# Run default builders (timepacks + threads, NO LLM)
python3 scripts/build_all.py --db data/processed/twos.db

# Include LLM-powered monthly summaries
python3 scripts/build_all.py --db data/processed/twos.db --with-llm

# Run specific builders only
python3 scripts/build_all.py --db data/processed/twos.db --builders timepacks,threads

# Force rebuild all
python3 scripts/build_all.py --db data/processed/twos.db --force
```

**Execution Order:**
1. TimePacks (no dependencies)
2. ThreadPacks (no dependencies, runs after TimePacks for simplicity)
3. MonthlySummaries (depends on TimePacks existing)

**Error Handling:**
- If TimePacks fails → skip MonthlySummaries, but still run ThreadPacks
- If ThreadPacks fails → continue to MonthlySummaries (if requested)
- All errors logged, partial success = exit code 1

**Integration:**
Called by `setup_wizard.py` after database load (opt-in via prompt).

## Development Workflow

1. Make changes in `<path-to-project>/memex-twos-mcp`
2. Test locally
3. Commit with proper message format
4. Push to GitHub
5. Data changes sourced from `<path-to-project>/memex-twos-data`

## LLM Micro-corrections

Short reminders to avoid common retry loops during development. Scan this section before starting work.

### Database Operations
- SQLite: Delete database file completely before reload (`rm *.db && load`), `--force` doesn't truncate existing data
- `get_stats()` returns: `total_things`, `total_people`, `total_tags` (not `thing_count`, `person_count`)

### Environment & Dependencies
- Always `pip install -e .` after pulling changes or before testing imports
- MCP package must be installed before `from mcp.server import Server` works

### Testing & Validation
- Don't use `timeout` command (permission issues); use `python3 -c "..."` for quick validation
- Test module loading before running server: `python3 -c "from memex_twos_mcp.server import main"`

### Common Paths
- Database: `data/processed/twos.db`
- Export source: `data/raw/twos_export.md`
- Python path: `PYTHONPATH=src` when running server

### Quick Validation Chain
```bash
# After data refresh, run in order:

# Full rebuild (first time or major changes):
rm data/processed/twos.db  # Clean slate
python3 src/convert_to_json.py data/raw/twos_export.md -o data/processed/twos_data.json
python3 scripts/groom_data.py  # Auto-fix duplicates, generate cleaned file
python3 scripts/build_list_metadata.py data/processed/twos_data_cleaned.json  # Add list metadata
python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned_with_lists.json  # Use enhanced version
python3 scripts/build_all.py --db data/processed/twos.db  # Build derived indices (timepacks, threads)
# Optional: python3 scripts/build_all.py --db data/processed/twos.db --with-llm  # Include AI summaries
source .venv/bin/activate && python3 -c "from memex_twos_mcp.database import TwosDatabase; from pathlib import Path; print(TwosDatabase(Path('data/processed/twos.db')).get_stats())"

# Incremental update (daily updates, new data):
python3 src/convert_to_json.py data/raw/twos_export.md -o data/processed/twos_data.json
python3 scripts/groom_data.py
python3 scripts/build_list_metadata.py data/processed/twos_data_cleaned.json  # Add list metadata
python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned_with_lists.json --incremental  # 10x faster
python3 scripts/build_all.py --db data/processed/twos.db  # Rebuild derived indices incrementally
source .venv/bin/activate && python3 -c "from memex_twos_mcp.database import TwosDatabase; from pathlib import Path; print(TwosDatabase(Path('data/processed/twos.db')).get_stats())"

# Migrate existing database (if you already have a database without list metadata):
python3 scripts/migrate_add_list_semantics.py data/processed/twos.db
```

### Data Grooming Workflow

**Two-tier auto-fix + analysis system:**

**Python Auto-Fix (Fast, Free, Always Run):**
```bash
python3 scripts/groom_data.py
```
**What it does:**
- ✅ Removes exact duplicates (same timestamp within 1 minute)
- ✅ Fixes broken parent references (nulls orphaned links)
- ✅ Detects normalization opportunities (case variants)
- ✅ Flags ambiguous duplicates for review
- ✅ Writes cleaned data: `twos_data_cleaned.json`
- ✅ Generates reports: changes.md (what was fixed), python.md (analysis)

**Output:**
- `data/processed/twos_data_cleaned.json` - Use this for SQLite
- `docs/grooming-reports/YYYY-MM-DD-changes.md` - What was auto-fixed
- `docs/grooming-reports/YYYY-MM-DD-changes.json` - Machine-readable audit
- `docs/grooming-reports/YYYY-MM-DD-python.md` - Full analysis

**Tuning Flags:**
```bash
# More aggressive duplicate detection
python3 scripts/groom_data.py --duplicate-window 14

# Only catch same-day duplicates
python3 scripts/groom_data.py --duplicate-window 0

# Flag shorter content as unusual
python3 scripts/groom_data.py --long-content-threshold 1000

# Show more items in reports
python3 scripts/groom_data.py --report-limit 20
```

**AI Semantic Analysis (Uses Subscription Quota, Optional):**
```bash
python3 scripts/groom_data.py --ai-analysis
```
- Requires Claude Code CLI installed
- Uses subscription quota (rate limits apply)
- Analyzes the CLEANED data from Python pass
- Provides semantic insights: patterns, themes, schema recommendations
- Judgment on ambiguous duplicates flagged by Python
- Generates enhanced report: `docs/grooming-reports/YYYY-MM-DD-ai-analysis.md`

**When to use:**
- Python only: Every data refresh (auto-fixes mechanical issues)
- Python + AI: First-time analysis, before schema changes, quarterly deep reviews

**Example workflow:**
```bash
# 1. Convert export to JSON
python3 src/convert_to_json.py data/raw/twos_export.md -o data/processed/twos_data.json

# 2. Auto-fix and analyze
python3 scripts/groom_data.py

# 3. Review what was changed
cat docs/grooming-reports/$(date +%Y-%m-%d)-changes.md

# 4. Load CLEANED data to SQLite
python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned.json
```

### Anti-patterns
- ❌ Assume field names without checking code
- ❌ Use `--force` flags without understanding what they do
- ❌ Test imports without installing package first
- ✅ Delete-then-create instead of overwrite-in-place
- ✅ Check actual return values before using them

## Testing Strategy

- Unit tests for parsers and data extraction
- Integration tests for MCP tool implementations
- Validation against known data samples
- Manual testing with Claude for MCP functionality

## Dependencies

**Required:**
- `python-dateutil>=2.8.2` - Date parsing
- `mcp[cli]>=1.0.0` - Model Context Protocol
- `PyYAML>=6.0.1` - Configuration

**Optional:**
- `spacy>=3.7.0` + `en_core_web_sm` model - Named Entity Recognition for accurate people extraction (recommended)
  - Install: `pip install -e ".[ner]" && python -m spacy download en_core_web_sm`
  - Without this: Falls back to regex-based extraction (higher false positive rate)

**Development:**
- `pytest>=7.4.0` - Testing
- `black>=23.0.0` - Code formatting
- `mypy>=1.7.0` - Type checking

## Data Privacy

All personal data files are gitignored. Never commit:
- `data/raw/*.md`
- `data/processed/*.json`
- `*.db`, `*.sqlite*`

## Learning Resources

- MCP Protocol Docs: (to be added)
- Claude MCP Integration: (to be added)
- Twos app context: See memex-twos-data/README.md

## Notes for Future Claude Sessions

- Prefer simple, understandable implementations over clever optimizations
- The human's actual data is valuable - be careful with transformations, always preserve originals
- When adding features, consider: "Does this help answer questions about life patterns?"
- MCP server should be queryable, not just batch processing
- Human is interested in narrative generation, not just statistics
