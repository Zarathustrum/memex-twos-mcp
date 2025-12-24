# Database Schema

SQLite database schema for Twos task data.

## Tables

### Core Tables

**things**
- Primary table containing all thing data
- Fields: id, timestamp, content, section_header, completion status, etc.
- Self-referencing foreign key for parent-child relationships (currently unused)

**people**
- Normalized person/entity names extracted from things
- Fields: id, name, normalized_name, category, notes
- Typically contains a few thousand unique entities (varies by dataset)

**tags**
- Normalized hashtags from things
- Fields: id, name, description
- Typically contains a few dozen tags (varies by dataset)

**links**
- URLs extracted from task content
- Fields: id, task_id, link_text, url
- Contains links extracted from task content

### Junction Tables

**thing_people**
- Many-to-many relationship between things and people
- Many-to-many relationships between things and people

**thing_tags**
- Many-to-many relationship between things and tags
- Many-to-many relationships between things and tags

### Full Text Search

**things_fts** (FTS5 virtual table)
- Enables fast content search across thing content and headers
- Automatically kept in sync with things table via triggers
- Uses porter stemming and unicode61 tokenization

### Metadata

**metadata**
- Key-value store for database versioning and load information
- Tracks: schema_version, source_file, task_count, last_loaded

## Indexes

Performance indexes on:
- `things.timestamp` - date range queries
- `things.parent_task_id` - hierarchical navigation
- `things.is_completed`, `things.is_strikethrough` - filtering
- `people.name`, `people.category` - entity lookups
- `tags.name` - tag lookups
- Junction table foreign keys

## Usage

### Loading Data

```bash
python3 scripts/load_to_sqlite.py data/processed/twos_data.json
```

### Example Queries

**Find all things mentioning a person:**
```sql
SELECT t.* FROM things t
JOIN thing_people tp ON t.id = tp.thing_id
JOIN people p ON tp.person_id = p.id
WHERE p.name = 'Alice'
ORDER BY t.timestamp;
```

**Full-text search:**
```sql
SELECT t.* FROM things t
JOIN things_fts fts ON t.id = fts.thing_id
WHERE things_fts MATCH 'college OR university'
ORDER BY t.timestamp;
```

**Get all things with a specific tag:**
```sql
SELECT t.* FROM things t
JOIN thing_tags tt ON t.id = tt.thing_id
JOIN tags tag ON tt.tag_id = tag.id
WHERE tag.name = 'siri';
```

**Timeline of mentions by month:**
```sql
SELECT
    strftime('%Y-%m', timestamp) as month,
    COUNT(*) as mentions
FROM things t
JOIN thing_people tp ON t.id = tp.thing_id
JOIN people p ON tp.person_id = p.id
WHERE p.name = 'Alice'
GROUP BY month
ORDER BY month;
```

## Schema Version

Current version: 1.0

## Future Enhancements

- Populate parent-child task relationships
- Add entity categorization (person vs place vs project)
- Create materialized views for common queries
- Add spatial/temporal clustering tables
