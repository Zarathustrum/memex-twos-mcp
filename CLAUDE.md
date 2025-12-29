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

## Current Status

**Completed**:
- ✅ Project structure and documentation
- ✅ JSON converter with rich metadata extraction
- ✅ Initial data conversion (~10,000 things)
- ✅ Both repos on GitHub
- ✅ LLM-assisted data grooming (identify duplicates, normalize entities, suggest schema improvements)

**Next Steps**:
1. Add analysis tools (patterns, timelines, narratives)2. Add analysis tools (patterns, timelines, narratives)

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

**People Extraction** (Improved in Phase 6):
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
- `things` (core thing data)
- `people` (extracted and normalized)
- `tags` (normalized taxonomy)
- `links` (URLs with metadata)
- `thing_people` (thing-person relationships)
- `thing_tags` (thing-tag relationships)
- `things_fts` (FTS5 search)
- `thing_embeddings` (semantic embeddings, Phase 4)
- `vec_index` (vector similarity search, Phase 4)

## MCP Server Design (Planned)

**Resources**:
- `twos://database/stats` - Database statistics
- `twos://database/people` - People list
- `twos://database/tags` - Tag list

**Tools**:
- `query_things_by_date(start_date, end_date, filters)` - Basic queries
- `search_things(query)` - Keyword search for exact matches (BM25)
- `semantic_search(query)` - ⭐ Semantic + keyword search (Phase 4) - for conceptual queries
- `get_person_things(person_name)` - Things mentioning a person
- `get_tag_things(tag_name)` - Things with a tag
- `get_things_stats()` - Database statistics

**Prompts**:
- Life narrative generation templates
- Analysis prompt templates

## Hybrid Search (Phase 4)

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
rm data/processed/twos.db  # Clean slate
python3 src/convert_to_json.py data/raw/twos_export.md -o data/processed/twos_data.json
python3 scripts/groom_data.py  # Auto-fix duplicates, generate cleaned file
python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned.json  # Use cleaned version
source .venv/bin/activate && python3 -c "from memex_twos_mcp.database import TwosDatabase; from pathlib import Path; print(TwosDatabase(Path('data/processed/twos.db')).get_stats())"
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
