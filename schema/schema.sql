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
    section_header TEXT,
    section_date TEXT,
    line_number INTEGER,
    indent_level INTEGER DEFAULT 0,
    parent_task_id TEXT,
    bullet_type TEXT,
    is_completed BOOLEAN DEFAULT 0,
    is_pending BOOLEAN DEFAULT 0,
    is_strikethrough BOOLEAN DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_task_id) REFERENCES things(id) ON DELETE SET NULL
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

CREATE INDEX IF NOT EXISTS idx_people_name ON people(name);
CREATE INDEX IF NOT EXISTS idx_people_category ON people(category);

CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);

CREATE INDEX IF NOT EXISTS idx_thing_people_person ON thing_people(person_id);
CREATE INDEX IF NOT EXISTS idx_thing_tags_tag ON thing_tags(tag_id);

CREATE INDEX IF NOT EXISTS idx_links_thing ON links(thing_id);

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
