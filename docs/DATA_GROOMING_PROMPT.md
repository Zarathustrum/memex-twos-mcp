# LLM Data Grooming Prompt

This prompt is used to analyze converted Twos JSON data for quality issues, normalization opportunities, and schema design recommendations.

## When to Use

- After converting new Twos exports to JSON
- Before loading data into SQLite
- When data quality issues are suspected
- Before major schema changes

## Prerequisites

1. Have the JSON file ready: `data/processed/twos_data.json`
2. Use in a fresh Claude conversation (large context window recommended)
3. Upload or paste the JSON data

---

## PROMPT START

I need help analyzing and grooming personal task data exported from the Twos productivity app. This data will be loaded into a SQLite database for an MCP server that enables life pattern analysis.

### Data Context

**Source**: Twos app markdown export converted to JSON
**Time Range**: Recent 2-year range
**Record Count**: ~10,000 things
**Purpose**: Personal knowledge management, life chronology, pattern detection

### JSON Schema

Each task object contains:
```json
{
  "id": "task_00001",
  "line_number": 2,
  "timestamp": "2023-10-17T16:23:00",
  "timestamp_raw": "10/17/23 4:23 pm",
  "section_header": "Sun, Oct 8, 2023",
  "section_date": "10/27/23 9:14 pm",
  "content": "cleaned task text",
  "content_raw": "• original line with all formatting",
  "indent_level": 0,
  "parent_task_id": "task_00042",
  "bullet_type": "bullet|checkbox_done|checkbox_pending|dash",
  "is_completed": false,
  "is_pending": false,
  "is_strikethrough": false,
  "links": [{"text": "display", "url": "https://..."}],
  "tags": ["journal", "dinner"],
  "people_mentioned": ["Alice", "Pat"]
}
```

### Analysis Tasks

Please analyze the data and provide recommendations for the following:

---

#### 1. Duplicate Detection (CAREFUL)

**Goal**: Identify true duplicates while preserving legitimate recurring things.

**Rules**:
- **Temporal distance matters**: "Call mom" appearing monthly is NOT a duplicate
- Only flag as duplicates if:
  - Same or very similar content AND
  - Timestamps within 7 days AND
  - Same parent_task_id (or both null)
- Recurring patterns are EXPECTED and VALUABLE - do not flag these

**Output Format**:
```
DUPLICATES FOUND: X clusters
- Cluster 1: task_00123, task_00456 (both "Buy milk" within 2 days)
- Cluster 2: task_00789, task_00790 (identical content, same timestamp)
```

---

#### 2. Entity Normalization

**Goal**: Identify inconsistent entity references that should be normalized.

**Analyze**:
- **People names**: Case inconsistencies (Alice vs alice), nicknames, typos
- **Places**: Location mentions with different spellings
- **Projects**: Related things that reference same project with different names
- **Tags**: Similar tags that could be consolidated (#dinner# vs #meal#)

**Output Format**:
```
NORMALIZATION RECOMMENDATIONS:

People:
- "Alice" (523), "alice" (12) → standardize to "Alice"
- "Mom" (45), "mom" (8), "Mother" (2) → standardize to "Mom"

Places:
- "Seattle" (89), "seattle" (5) → standardize to "Seattle"

Tags:
- #dinner# (234), #meal# (12) → suggest: keep separate or merge?

Projects:
- "house renovation", "renovation", "remodel" → appear related
```

**IMPORTANT**: Provide the normalization mapping, do NOT modify the source data.

---

#### 3. Data Quality Validation

**Goal**: Identify malformed, suspicious, or problematic entries.

**Check for**:
- Malformed timestamps (if timestamp is null but timestamp_raw exists)
- Missing critical fields (content, timestamp)
- Unusually long content (might indicate parsing errors)
- Broken link structures
- Parent_task_id references non-existent things (orphaned children)
- Suspicious patterns (repeated exact timestamps, unusual characters)

**Output Format**:
```
QUALITY ISSUES:

Parsing Errors (X found):
- task_00456: timestamp_raw exists but timestamp null
- task_01234: content unusually long (5000+ chars)

Broken References (X found):
- task_00789: parent_task_id="task_99999" (parent doesn't exist)

Data Anomalies (X found):
- 47 things have identical timestamp "2024-03-15T14:23:00" (suspicious?)
```

---

#### 4. SQLite Schema Recommendations

**Goal**: Design an efficient, normalized SQLite schema.

**Consider**:
- Should people, tags, places be separate tables with foreign keys?
- How to handle hierarchical relationships (parent/child things)?
- What indexes are needed for common queries?
- How to preserve original data while enabling normalization?

**Output Format**:
```sql
-- Suggested Schema

CREATE TABLE things (
  id TEXT PRIMARY KEY,
  timestamp DATETIME NOT NULL,
  content TEXT NOT NULL,
  section_header TEXT,
  indent_level INTEGER,
  parent_task_id TEXT,
  bullet_type TEXT,
  is_completed BOOLEAN,
  is_strikethrough BOOLEAN,
  -- ... other fields
  FOREIGN KEY (parent_task_id) REFERENCES things(id)
);

CREATE TABLE people (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL,
  normalized_name TEXT
);

CREATE TABLE thing_people (
  thing_id TEXT,
  person_id INTEGER,
  FOREIGN KEY (thing_id) REFERENCES things(id),
  FOREIGN KEY (person_id) REFERENCES people(id),
  PRIMARY KEY (thing_id, person_id)
);

-- Indexes for common queries
CREATE INDEX idx_things_timestamp ON things(timestamp);
CREATE INDEX idx_things_completed ON things(is_completed);
CREATE INDEX idx_thing_people_person ON thing_people(person_id);

-- Rationale:
-- 1. Normalized people to enable name consistency
-- 2. Junction table for many-to-many task-person relationships
-- 3. Indexes on likely query patterns (date range, completion status)
```

**Include reasoning** for major design decisions.

---

#### 5. Pattern & Category Detection

**Goal**: Identify natural themes, categories, and patterns for enrichment.

**Analyze**:
- Common themes (home, work, family, health, etc.)
- Project threads (related things across time)
- Temporal patterns (certain things on certain days/times)
- Activity clusters (busy periods vs quiet periods)

**Output Format**:
```
DETECTED PATTERNS:

Themes (top 10 by frequency):
1. Home/Renovation (1,247 things) - keywords: contractor, paint, install
2. Family/Alice (856 things) - keywords: Alice, college, school
3. Health/Appointments (234 things) - keywords: doctor, dentist, appointment

Project Threads:
- "House Renovation" (2023-10 to 2024-08): 340 related things
- "College Search" (2023-11 to 2024-05): 127 related things

Temporal Patterns:
- "Call mom" appears ~weekly on Sundays
- "Trash" appears weekly (Mondays)
- High task creation volume: 11pm-1am (night planning sessions)

RECOMMENDATION: Store these as separate enrichment data (new table: task_categories)
rather than modifying source records. Allows recomputation without data loss.
```

---

#### 6. Manual Cleanup Flags

**Goal**: Flag items needing human review or decision.

**Flag**:
- Ambiguous duplicates (human should decide)
- Unclear entity references
- Potential data loss (truncated content, special characters)
- Tasks that might be sensitive/should be filtered
- Anomalies that automated processes can't resolve

**Output Format**:
```
MANUAL REVIEW NEEDED (X items):

Ambiguous Duplicates (require human judgment):
- task_00234, task_00567: "Buy groceries" 5 days apart - recurring or duplicate?

Unclear References:
- task_01234: mentions "Alex" - is this the user (metadata) or another person?

Data Sensitivity:
- task_00456: contains potential password/credential
- task_00789: very personal medical information - flag for filtering?

Anomalies:
- task_03456: content is just "￼" (attachment placeholder?) - keep or remove?
```

---

### Output Summary

Please provide:

1. **Executive Summary**: High-level findings (3-5 bullets)
2. **Detailed Analysis**: Each section above with findings
3. **Recommended SQLite Schema**: Complete SQL with rationale
4. **Action Items**: Prioritized list of what to fix/implement
5. **Sample Normalization Mappings**: JSON/CSV format for automated processing

### Data Integrity Principle

**CRITICAL**: All recommendations should preserve original data. Normalization should be additive (new fields/tables) rather than destructive (modifying source). The raw JSON should remain the source of truth.

---

## PROMPT END

---

## Usage Notes

1. **Prepare Data**: Ensure `twos_data.json` is ready
2. **Upload**: Paste JSON or upload file to Claude
3. **Use Prompt**: Copy the prompt between "PROMPT START" and "PROMPT END"
4. **Review Output**: Carefully review recommendations
5. **Iterate**: May need follow-up questions for clarification
6. **Document**: Save the grooming report in `docs/grooming-reports/YYYY-MM-DD.md`

## Output Storage

Save grooming reports with date:
```
docs/
  grooming-reports/
    2025-01-15-initial-grooming.md
    2025-02-01-post-cleanup.md
```

This creates a history of data quality improvements over time.
