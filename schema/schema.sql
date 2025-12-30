-- SQLite Schema for Twos Task Database
-- Based on grooming analysis and FTS requirements
-- Version: 1.0

PRAGMA foreign_keys = ON;

-- ============================================================================
-- Core Tasks Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS things (
    id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    timestamp_raw TEXT,
    content TEXT NOT NULL,
    content_raw TEXT,
    content_hash TEXT,
    section_header TEXT,
    section_date TEXT,
    line_number INTEGER,
    indent_level INTEGER DEFAULT 0,
    parent_task_id TEXT,
    bullet_type TEXT,
    is_completed BOOLEAN DEFAULT 0,
    is_pending BOOLEAN DEFAULT 0,
    is_strikethrough BOOLEAN DEFAULT 0,
    item_type TEXT DEFAULT 'content',          -- 'content' | 'divider' | 'header' | 'metadata'
    list_id TEXT,                              -- Denormalized FK to lists.list_id
    is_substantive BOOLEAN DEFAULT 1,          -- Cached flag: item_type='content' AND length(content) > 3
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_task_id) REFERENCES things(id) ON DELETE SET NULL,
    FOREIGN KEY (list_id) REFERENCES lists(list_id) ON DELETE SET NULL
);

-- ============================================================================
-- People / Entities
-- ============================================================================

CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    normalized_name TEXT,
    category TEXT DEFAULT 'Uncategorized',
    notes TEXT
);

CREATE TABLE IF NOT EXISTS thing_people (
    thing_id TEXT,
    person_id INTEGER,
    PRIMARY KEY (thing_id, person_id),
    FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
);

-- ============================================================================
-- Tags
-- ============================================================================

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS thing_tags (
    thing_id TEXT,
    tag_id INTEGER,
    PRIMARY KEY (thing_id, tag_id),
    FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

-- ============================================================================
-- Links
-- ============================================================================

CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thing_id TEXT,
    link_text TEXT,
    url TEXT NOT NULL,
    FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE
);

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_things_timestamp ON things(timestamp);
CREATE INDEX IF NOT EXISTS idx_things_parent ON things(parent_task_id);
CREATE INDEX IF NOT EXISTS idx_things_completed ON things(is_completed);
CREATE INDEX IF NOT EXISTS idx_things_strikethrough ON things(is_strikethrough);
CREATE INDEX IF NOT EXISTS idx_things_section ON things(section_header);
CREATE INDEX IF NOT EXISTS idx_things_content_hash ON things(content_hash);
CREATE INDEX IF NOT EXISTS idx_things_list_id ON things(list_id);
CREATE INDEX IF NOT EXISTS idx_things_item_type ON things(item_type);
CREATE INDEX IF NOT EXISTS idx_things_substantive ON things(is_substantive) WHERE is_substantive = 1;

CREATE INDEX IF NOT EXISTS idx_people_name ON people(name);
CREATE INDEX IF NOT EXISTS idx_people_category ON people(category);

CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);

CREATE INDEX IF NOT EXISTS idx_thing_people_person ON thing_people(person_id);
CREATE INDEX IF NOT EXISTS idx_thing_tags_tag ON thing_tags(tag_id);

CREATE INDEX IF NOT EXISTS idx_links_thing ON links(thing_id);

-- ============================================================================
-- Lists (Section/Header Groupings)
-- ============================================================================

CREATE TABLE IF NOT EXISTS lists (
    list_id TEXT PRIMARY KEY,              -- e.g., 'date_2025-12-30', 'topic_tech-projects_5678'
    list_type TEXT NOT NULL,               -- 'date' | 'topic' | 'category' | 'metadata'
    list_name TEXT NOT NULL,               -- Normalized: '2025-12-30' | 'Tech Projects'
    list_name_raw TEXT NOT NULL,           -- Original: 'Mon, Dec 30, 2025' | 'Tech Projects'
    list_date TEXT,                        -- ISO date for type='date', NULL otherwise
    start_line INTEGER NOT NULL,           -- First line of section (inclusive)
    end_line INTEGER NOT NULL,             -- Last line of section (inclusive)
    item_count INTEGER DEFAULT 0,          -- Total things in range
    substantive_count INTEGER DEFAULT 0,   -- Things with item_type='content'
    created_at TEXT NOT NULL,              -- Import timestamp
    UNIQUE(list_name, start_line)          -- Prevent duplicates
);

CREATE INDEX IF NOT EXISTS idx_lists_type ON lists(list_type);
CREATE INDEX IF NOT EXISTS idx_lists_date ON lists(list_date) WHERE list_date IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_lists_name ON lists(list_name);
CREATE INDEX IF NOT EXISTS idx_lists_lines ON lists(start_line, end_line);

-- ============================================================================
-- Thing-List Relationships (Many-to-Many)
-- ============================================================================

CREATE TABLE IF NOT EXISTS thing_lists (
    thing_id TEXT NOT NULL,
    list_id TEXT NOT NULL,
    position_in_list INTEGER,             -- 0-indexed position within list
    PRIMARY KEY (thing_id, list_id),
    FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE,
    FOREIGN KEY (list_id) REFERENCES lists(list_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_thing_lists_list ON thing_lists(list_id, position_in_list);

-- ============================================================================
-- Full Text Search (FTS5)
-- Critical for unstructured content analysis
-- ============================================================================

CREATE VIRTUAL TABLE IF NOT EXISTS things_fts USING fts5(
    thing_id UNINDEXED,
    content,
    section_header,
    tokenize = 'porter unicode61'
);

-- ============================================================================
-- Triggers to Keep FTS in Sync
-- ============================================================================

CREATE TRIGGER IF NOT EXISTS things_ai AFTER INSERT ON things BEGIN
    INSERT INTO things_fts(thing_id, content, section_header)
    VALUES (new.id, new.content, new.section_header);
END;

CREATE TRIGGER IF NOT EXISTS things_ad AFTER DELETE ON things BEGIN
    DELETE FROM things_fts WHERE thing_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS things_au AFTER UPDATE ON things BEGIN
    DELETE FROM things_fts WHERE thing_id = old.id;
    INSERT INTO things_fts(thing_id, content, section_header)
    VALUES (new.id, new.content, new.section_header);
END;

-- ============================================================================
-- Embeddings for Semantic Search (Phase 4)
-- ============================================================================

CREATE TABLE IF NOT EXISTS thing_embeddings (
    thing_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model_version TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_embeddings_thing ON thing_embeddings(thing_id);

-- Note: vec_index virtual table is created at runtime by sqlite-vec
-- It requires loading the extension dynamically and cannot be created in static schema
-- See database.py _init_vector_search() for runtime initialization

-- ============================================================================
-- Metadata Table (for versioning and stats)
-- ============================================================================

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial metadata
INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', '1.0');
INSERT OR REPLACE INTO metadata (key, value) VALUES ('created_at', datetime('now'));

-- ============================================================================
-- Import Tracking (Phase 5: Incremental Ingestion)
-- ============================================================================

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

-- ============================================================================
-- TimePacks: Time-Based Rollups (Phase 7)
-- ============================================================================

CREATE TABLE IF NOT EXISTS rollups (
    rollup_id TEXT PRIMARY KEY,        -- 'd:2025-12-30' | 'w:2025-12-22' | 'm:2025-12'
    kind TEXT NOT NULL CHECK(kind IN ('d','w','m')),
    start_date TEXT NOT NULL,          -- ISO date
    end_date TEXT NOT NULL,            -- ISO date (inclusive)
    thing_count INTEGER NOT NULL,
    completed_count INTEGER NOT NULL,
    pending_count INTEGER NOT NULL,
    strikethrough_count INTEGER NOT NULL,

    pack_v INTEGER NOT NULL,           -- Pack format version (1)
    pack TEXT NOT NULL,                -- TP1 format (spec below)
    kw TEXT NOT NULL,                  -- Space-separated keywords for search

    src_hash TEXT NOT NULL,            -- Hash of (thing_id + content_hash) in window
    build_import_id INTEGER,           -- imports.id if available
    builder_v TEXT NOT NULL,           -- '1.0' or git sha
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),

    FOREIGN KEY (build_import_id) REFERENCES imports(id)
);

CREATE INDEX IF NOT EXISTS idx_rollups_kind_start ON rollups(kind, start_date);

CREATE TABLE IF NOT EXISTS rollup_evidence (
    rollup_id TEXT NOT NULL,
    thing_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('hi','ev')),  -- hi=highlight, ev=evidence
    rank INTEGER,
    PRIMARY KEY (rollup_id, thing_id, role),
    FOREIGN KEY (rollup_id) REFERENCES rollups(rollup_id) ON DELETE CASCADE,
    FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rollup_evidence_rollup_role ON rollup_evidence(rollup_id, role, rank);

-- ============================================================================
-- MonthlySummaries: LLM-Powered Semantic Framing (Phase 8)
-- ============================================================================

CREATE TABLE IF NOT EXISTS month_summaries (
    month_id TEXT PRIMARY KEY,         -- 'YYYY-MM' (e.g., '2025-12')
    start_date TEXT NOT NULL,          -- 'YYYY-MM-01'
    end_date TEXT NOT NULL,            -- Last day of month
    thing_count INTEGER NOT NULL,

    pack_v INTEGER NOT NULL,           -- Pack format version (1)
    pack TEXT NOT NULL,                -- MS1 format (spec below)
    suggested_questions TEXT,          -- JSON array of suggested questions

    src_hash TEXT NOT NULL,            -- Hash of (thing_id + content_hash) in month
    build_import_id INTEGER,           -- imports.id if available
    builder_v TEXT NOT NULL,           -- '1.0' or git sha

    llm_model TEXT,                    -- 'claude-sonnet-4-5' or NULL
    llm_conf REAL,                     -- Confidence score (0.0-1.0)
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),

    FOREIGN KEY (build_import_id) REFERENCES imports(id)
);

CREATE TABLE IF NOT EXISTS month_summary_evidence (
    month_id TEXT NOT NULL,
    thing_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('hi','ev')),  -- hi=highlight, ev=evidence
    rank INTEGER,
    PRIMARY KEY (month_id, thing_id, role),
    FOREIGN KEY (month_id) REFERENCES month_summaries(month_id) ON DELETE CASCADE,
    FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_month_summaries_start ON month_summaries(start_date);
CREATE INDEX IF NOT EXISTS idx_month_evidence_role ON month_summary_evidence(month_id, role, rank);

-- ============================================================================
-- ThreadPacks: Active Tag/Person Thread Indices (Phase 9)
-- ============================================================================

CREATE TABLE IF NOT EXISTS threads (
    thread_id TEXT PRIMARY KEY,        -- 'thr:tag:work' | 'thr:person:alice'
    kind TEXT NOT NULL CHECK(kind IN ('tag','person')),
    label TEXT NOT NULL,               -- Display: 'work' | 'Alice'
    label_norm TEXT NOT NULL,          -- Lowercase: 'work' | 'alice'

    start_ts TEXT,                     -- First thing timestamp
    last_ts TEXT,                      -- Most recent thing timestamp
    thing_count INTEGER NOT NULL,      -- Total things in thread
    thing_count_90d INTEGER NOT NULL,  -- Things in last 90 days
    status TEXT NOT NULL CHECK(status IN ('active','stale','archived')),
    archived_at TEXT,                  -- When thread was archived (NULL if active/stale)

    pack_v INTEGER NOT NULL,           -- Pack format version (1)
    pack TEXT NOT NULL,                -- TH1 format (spec below)
    kw TEXT NOT NULL,                  -- Space-separated keywords for search

    src_hash TEXT NOT NULL,            -- Hash of (thing_id + content_hash) in thread
    builder_v TEXT NOT NULL,           -- '1.0' or git sha
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_threads_status_last ON threads(status, last_ts DESC);
CREATE INDEX IF NOT EXISTS idx_threads_kind_label ON threads(kind, label_norm);

CREATE TABLE IF NOT EXISTS thread_evidence (
    thread_id TEXT NOT NULL,
    thing_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('hi','ev')),  -- hi=highlight, ev=evidence
    rank INTEGER,
    PRIMARY KEY (thread_id, thing_id, role),
    FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE,
    FOREIGN KEY (thing_id) REFERENCES things(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_thread_evidence_thread_role ON thread_evidence(thread_id, role, rank);

-- FTS for thread search
CREATE VIRTUAL TABLE IF NOT EXISTS threads_fts USING fts5(
    thread_id UNINDEXED,
    label,
    kw,
    tokenize='porter unicode61'
);
