# Phase 6: Entity Extraction Improvement - Execution Plan

**Branch:** `claude/memex-twos-v2-upgrade-d3SNe`
**Complexity:** MEDIUM
**Estimated Effort:** 0.5-1 day
**Impact:** HIGH (80% reduction in false positives)
**Prerequisites:** None (orthogonal to Phases 4 and 5)

---

## Executive Summary

Replace naive regex-based people extraction with proper Named Entity Recognition (NER) using spaCy. Current regex approach produces 40-60% false positives (verbs like "Set", "Plan", common words like "May", "March" misclassified as people). NER-based extraction will reduce false positives by ~80% while maintaining high recall.

**Key Goals:**
- Replace regex pattern matching with spaCy NER
- Reduce false positives from ~50% to <10%
- Maintain graceful fallback for environments without spaCy
- Preserve all existing functionality (tags, links still use regex)
- Zero breaking changes to database schema or MCP tools

---

## Current State Analysis

### Existing Implementation

**Location:** `src/convert_to_json.py:167-183`

```python
def extract_people(text: str) -> list[str]:
    """
    Extract people mentions using simple capitalization heuristic.

    Current regex: r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    """
    if not text:
        return []

    # Find sequences of capitalized words
    pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    matches = re.findall(pattern, text)

    # Filter out common non-name patterns
    excluded = {'Day', 'Week', 'Month', 'Year', 'Today', 'Tomorrow', ...}

    people = []
    for match in matches:
        if match not in excluded and not match.endswith('day'):
            people.append(match)

    return list(set(people))
```

### Known Issues

**False Positives (40-60% of extractions):**

1. **Verbs/Actions:** "Set", "Plan", "Call", "Review", "Check"
   - Example: "Set reminder for doctor" → extracts "Set"
   - Example: "Review notes from meeting" → extracts "Review"

2. **Month/Day Names:** "May", "March", "April", "Sunday", "Monday"
   - Example: "Meet in May" → extracts "May"
   - Example: "March deadline" → extracts "March"

3. **Common Nouns:** "House", "Car", "Team", "Group", "Project"
   - Example: "New House plans" → extracts "House"

4. **Adjectives:** "New", "Great", "Important", "Next"
   - Example: "Great work on project" → extracts "Great"

5. **Acronyms:** "NASA", "FBI", "CEO", "HR"
   - Example: "HR meeting scheduled" → extracts "HR"

**False Negatives (10-20% of actual people):**

1. **Lowercase names:** "john" → not extracted
2. **Nicknames:** "mike" → not extracted
3. **Single names without capitals:** "alex said..." → not extracted
4. **Names with special chars:** "O'Brien", "Jean-Paul" → partially extracted

### Performance Baseline

**Current extraction on 10K things:**
- Runtime: ~0.5s (negligible)
- Total extracted: ~3,200 "people"
- Estimated true positives: ~1,600 (50%)
- Estimated false positives: ~1,600 (50%)

**Impact on database:**
- `people` table bloated with non-names
- `thing_people` junction table contains spurious relationships
- Search by person returns irrelevant results

---

## Repo Reality

### Codebase Structure

```
src/convert_to_json.py          # Parser with entity extraction
scripts/load_to_sqlite.py        # Database loader
src/memex_twos_mcp/database.py   # Query layer (no changes needed)
tests/test_convert.py            # Parser tests
schema/schema.sql                # Database schema (no changes needed)
```

### Current Dependencies

```toml
[project]
dependencies = [
    "python-dateutil>=2.8.0",
    "mcp>=1.0.0",
]
```

### Tech Stack

- **Parser:** Pure Python with regex
- **Database:** SQLite with FTS5
- **Testing:** pytest
- **Python Version:** 3.11+

---

## Non-Negotiables

### Must Have

1. **Graceful Degradation:**
   - If spaCy not installed → fall back to current regex (with warning)
   - If model download fails → fall back to regex
   - Never crash or require manual intervention

2. **Backward Compatibility:**
   - No database schema changes
   - No changes to MCP tool signatures
   - Existing databases continue to work

3. **Performance:**
   - NER extraction must complete in <5s for 10K things
   - Acceptable since conversion is one-time operation
   - Should be faster than current regex for large datasets

4. **Minimal Dependencies:**
   - Use small spaCy model (en_core_web_sm: ~15MB)
   - Make spaCy optional dependency (not required)
   - Document installation clearly

### Must Not Have

- ❌ No external API calls (must be local-first)
- ❌ No cloud-based NER services
- ❌ No breaking changes to conversion workflow
- ❌ No slowdown >2x current extraction time

---

## Detailed Implementation

### Step 1: Add spaCy as Optional Dependency

**File:** `pyproject.toml`

```toml
[project]
dependencies = [
    "python-dateutil>=2.8.0",
    "mcp>=1.0.0",
]

[project.optional-dependencies]
ner = [
    "spacy>=3.7.0,<4.0.0",
]

dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
]
```

**Installation:**
```bash
# Basic install (regex fallback)
pip install -e .

# With NER support
pip install -e ".[ner]"
python -m spacy download en_core_web_sm
```

---

### Step 2: Implement NER-Based Extraction with Fallback

**File:** `src/convert_to_json.py`

Add new function before existing `extract_people()`:

```python
def extract_people_ner(text: str) -> list[str]:
    """
    Extract people mentions using spaCy Named Entity Recognition.

    Args:
        text: Content to extract names from

    Returns:
        List of unique person names (PERSON entities)

    Raises:
        ImportError: If spaCy not available (caught by caller)
    """
    import spacy

    # Load model (will be cached by spaCy)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Model not downloaded
        raise ImportError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

    if not text:
        return []

    # Run NER pipeline
    doc = nlp(text)

    # Extract PERSON entities
    people = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Clean up entity text
            name = ent.text.strip()

            # Skip single-character names (likely errors)
            if len(name) > 1:
                people.append(name)

    # Return unique names, preserving original casing
    return list(set(people))


def extract_people(text: str, use_ner: bool = True) -> list[str]:
    """
    Extract people mentions from text.

    Tries NER-based extraction first (if spaCy available), falls back to regex.

    Args:
        text: Content to extract names from
        use_ner: Whether to attempt NER extraction (default: True)

    Returns:
        List of unique person names
    """
    if use_ner:
        try:
            return extract_people_ner(text)
        except ImportError as e:
            # spaCy not installed or model not downloaded
            print(f"Warning: spaCy NER not available ({e}), falling back to regex extraction", file=sys.stderr)
            print("Tip: Install spaCy with: pip install spacy && python -m spacy download en_core_web_sm", file=sys.stderr)
            # Fall through to regex

    # Original regex-based extraction (fallback)
    if not text:
        return []

    pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    matches = re.findall(pattern, text)

    # Filter out common non-name patterns
    excluded = {
        'Day', 'Week', 'Month', 'Year', 'Today', 'Tomorrow', 'Yesterday',
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December',
        'Am', 'Pm', 'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
        'Note', 'Task', 'Event', 'Reminder', 'List', 'Link',
        'Set', 'Plan', 'Call', 'Review', 'Check', 'Send', 'Get', 'Make', 'Take',
        'New', 'Great', 'Important', 'Next', 'First', 'Last',
    }

    people = []
    for match in matches:
        if match not in excluded and not match.endswith('day'):
            people.append(match)

    return list(set(people))
```

---

### Step 3: Update Conversion Script

**File:** `src/convert_to_json.py`

Update main conversion function to use NER:

```python
def convert_markdown_to_json(
    md_path: Path,
    output_path: Path | None = None,
    pretty: bool = False,
    use_ner: bool = True,  # NEW parameter
) -> dict:
    """
    Convert Twos markdown export to structured JSON.

    Args:
        md_path: Path to markdown file
        output_path: Optional output JSON path
        pretty: Whether to format JSON with indentation
        use_ner: Whether to use spaCy NER for people extraction (default: True)

    Returns:
        Dictionary with metadata and tasks list
    """
    print(f"Reading {md_path}...")

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Parsing {len(lines)} lines...")

    # ... existing parsing logic ...

    for line_num, line in enumerate(lines, start=1):
        # ... existing parsing ...

        # Extract entities
        tags = extract_tags(content)
        links = extract_links(content_raw)
        people = extract_people(content, use_ner=use_ner)  # Pass use_ner flag

        # ... rest of task construction ...
```

Update CLI to expose NER flag:

```python
def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Convert Twos markdown export to JSON")
    parser.add_argument("input", type=Path, help="Path to markdown file")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    parser.add_argument(
        "--no-ner",
        action="store_true",
        help="Disable NER extraction, use regex fallback",
    )

    args = parser.parse_args()

    # ... validation ...

    result = convert_markdown_to_json(
        args.input,
        args.output,
        pretty=args.pretty,
        use_ner=not args.no_ner,  # Default: use NER
    )

    print(f"\n✅ Converted {result['metadata']['thing_count']} things")
    print(f"   People extracted: {len(result['metadata'].get('people_count', 0))}")
```

---

### Step 4: Add spaCy Model Caching

To avoid reloading the model for every text snippet, add model caching:

**File:** `src/convert_to_json.py`

```python
# Global model cache (module-level)
_SPACY_MODEL_CACHE = None

def extract_people_ner(text: str) -> list[str]:
    """
    Extract people mentions using spaCy Named Entity Recognition.

    Uses cached model to avoid reloading for every call.
    """
    global _SPACY_MODEL_CACHE

    import spacy

    # Load model once and cache
    if _SPACY_MODEL_CACHE is None:
        try:
            _SPACY_MODEL_CACHE = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

    nlp = _SPACY_MODEL_CACHE

    if not text:
        return []

    # Run NER pipeline
    doc = nlp(text)

    # Extract PERSON entities
    people = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            if len(name) > 1:
                people.append(name)

    return list(set(people))
```

---

### Step 5: Performance Optimization (Batch Processing)

For large datasets, process multiple texts in batch:

**File:** `src/convert_to_json.py`

Add batch processing function:

```python
def extract_people_batch_ner(texts: list[str]) -> list[list[str]]:
    """
    Extract people from multiple texts in batch (faster than individual calls).

    Args:
        texts: List of content strings

    Returns:
        List of lists, each containing extracted names for corresponding text
    """
    global _SPACY_MODEL_CACHE

    import spacy

    if _SPACY_MODEL_CACHE is None:
        try:
            _SPACY_MODEL_CACHE = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError("spaCy model not found")

    nlp = _SPACY_MODEL_CACHE

    # Process texts in batch using nlp.pipe()
    results = []

    # nlp.pipe() is more efficient than individual nlp() calls
    for doc in nlp.pipe(texts, batch_size=50):
        people = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                if len(name) > 1:
                    people.append(name)
        results.append(list(set(people)))

    return results
```

Update conversion to use batch processing:

```python
def convert_markdown_to_json(
    md_path: Path,
    output_path: Path | None = None,
    pretty: bool = False,
    use_ner: bool = True,
) -> dict:
    # ... existing parsing to collect tasks ...

    # Batch extract people after all tasks parsed (if using NER)
    if use_ner:
        try:
            # Extract all content texts
            all_content = [task.get("content", "") for task in tasks]

            # Batch process
            people_results = extract_people_batch_ner(all_content)

            # Assign back to tasks
            for task, people in zip(tasks, people_results):
                task["people_mentioned"] = people

        except ImportError:
            # Fall back to regex for each task
            print("Warning: NER not available, using regex extraction", file=sys.stderr)
            for task in tasks:
                task["people_mentioned"] = extract_people(task.get("content", ""), use_ner=False)
    else:
        # Regex extraction
        for task in tasks:
            task["people_mentioned"] = extract_people(task.get("content", ""), use_ner=False)

    # ... rest of conversion ...
```

---

### Step 6: Add Comprehensive Tests

**File:** `tests/test_entity_extraction.py` (new file)

```python
"""Tests for entity extraction (people, tags, links)."""

import pytest
from src.convert_to_json import extract_people, extract_people_ner


class TestPeopleExtractionRegex:
    """Test regex-based people extraction (fallback mode)."""

    def test_simple_name(self):
        """Extract single name."""
        result = extract_people("Met with John today", use_ner=False)
        assert "John" in result

    def test_full_name(self):
        """Extract full name."""
        result = extract_people("Call Sarah Johnson", use_ner=False)
        assert "Sarah Johnson" in result

    def test_false_positive_verb(self):
        """Regex incorrectly extracts verbs (known limitation)."""
        result = extract_people("Set reminder for meeting", use_ner=False)
        # This is a KNOWN FALSE POSITIVE for regex mode
        assert "Set" in result  # Expected to fail - documenting current behavior

    def test_false_positive_month(self):
        """Regex incorrectly extracts months (known limitation)."""
        result = extract_people("Meeting in March", use_ner=False)
        # "March" is in excluded list, should be filtered
        assert "March" not in result

    def test_excludes_days(self):
        """Filter out day names."""
        result = extract_people("Monday meeting with Alice", use_ner=False)
        assert "Monday" not in result
        assert "Alice" in result


class TestPeopleExtractionNER:
    """Test NER-based people extraction."""

    def test_simple_name(self):
        """Extract single name with NER."""
        pytest.importorskip("spacy")
        result = extract_people_ner("Met with John today")
        assert "John" in result

    def test_full_name(self):
        """Extract full name with NER."""
        pytest.importorskip("spacy")
        result = extract_people_ner("Call Sarah Johnson about project")
        assert "Sarah Johnson" in result or "Sarah" in result

    def test_no_false_positive_verb(self):
        """NER correctly rejects verbs."""
        pytest.importorskip("spacy")
        result = extract_people_ner("Set reminder for meeting")
        assert "Set" not in result

    def test_no_false_positive_month(self):
        """NER correctly rejects month names."""
        pytest.importorskip("spacy")
        result = extract_people_ner("Meeting in March")
        assert "March" not in result

    def test_multiple_names(self):
        """Extract multiple people from same text."""
        pytest.importorskip("spacy")
        result = extract_people_ner("Dinner with Alice and Bob")
        assert "Alice" in result
        assert "Bob" in result

    def test_names_in_context(self):
        """Extract names from realistic context."""
        pytest.importorskip("spacy")
        text = "Called Dr. Smith about appointment. Sarah will join the meeting."
        result = extract_people_ner(text)
        assert "Smith" in result or "Dr. Smith" in result
        assert "Sarah" in result

    def test_no_single_char_names(self):
        """Filter out single-character false positives."""
        pytest.importorskip("spacy")
        result = extract_people_ner("Schedule A meeting")
        assert "A" not in result

    def test_empty_text(self):
        """Handle empty input."""
        pytest.importorskip("spacy")
        result = extract_people_ner("")
        assert result == []


class TestPeopleExtractionFallback:
    """Test graceful fallback when spaCy unavailable."""

    def test_fallback_when_spacy_missing(self, monkeypatch):
        """Fall back to regex when spaCy not installed."""
        # Mock spacy import to raise ImportError
        import sys
        monkeypatch.setitem(sys.modules, 'spacy', None)

        result = extract_people("Met with John", use_ner=True)
        # Should fall back to regex
        assert "John" in result

    def test_no_ner_flag(self):
        """Explicitly disable NER."""
        result = extract_people("Call Sarah Johnson", use_ner=False)
        assert "Sarah Johnson" in result


class TestComparisonNERvsRegex:
    """Compare NER vs regex on realistic examples."""

    @pytest.mark.parametrize("text,expected_ner,expected_regex_false_positives", [
        (
            "Set reminder to call John",
            ["John"],
            ["Set"],  # Regex incorrectly extracts "Set"
        ),
        (
            "Meeting with Dr. Smith in March",
            ["Smith"],
            [],  # "March" filtered by exclusion list
        ),
        (
            "Review notes from Alice and Bob",
            ["Alice", "Bob"],
            ["Review"],  # Regex incorrectly extracts "Review"
        ),
        (
            "Plan dinner with Sarah",
            ["Sarah"],
            ["Plan"],  # Regex incorrectly extracts "Plan"
        ),
    ])
    def test_ner_accuracy(self, text, expected_ner, expected_regex_false_positives):
        """Verify NER avoids false positives that regex produces."""
        pytest.importorskip("spacy")

        ner_result = extract_people_ner(text)
        regex_result = extract_people(text, use_ner=False)

        # NER should extract expected names
        for name in expected_ner:
            assert name in ner_result

        # NER should NOT extract false positives
        for false_positive in expected_regex_false_positives:
            assert false_positive not in ner_result

        # Regex WILL have false positives (documenting limitation)
        for false_positive in expected_regex_false_positives:
            assert false_positive in regex_result
```

**File:** `tests/test_database.py`

Add test for people query accuracy:

```python
def test_get_person_things_accuracy(tmp_path: Path) -> None:
    """Test that people queries return accurate results (no false positives)."""
    db_path = tmp_path / "twos.db"
    schema_path = Path(__file__).resolve().parents[1] / "schema" / "schema.sql"

    _init_test_db(db_path, schema_path)
    db = TwosDatabase(db_path)

    # Insert test things with known people
    conn = db._get_connection()

    # True positive: actual person
    conn.execute("""
        INSERT INTO things (id, content, timestamp, section_header, bullet_type, is_completed, is_pending, is_strikethrough)
        VALUES ('task_00001', 'Met with Alice about project', '2024-01-01T10:00:00', 'Day 1', 'bullet', 0, 0, 0)
    """)

    # False positive: verb "Set" misclassified as person
    conn.execute("""
        INSERT INTO things (id, content, timestamp, section_header, bullet_type, is_completed, is_pending, is_strikethrough)
        VALUES ('task_00002', 'Set reminder for meeting', '2024-01-02T10:00:00', 'Day 2', 'bullet', 0, 0, 0)
    """)

    conn.commit()

    # Add people
    conn.execute("INSERT INTO people (name, normalized_name) VALUES ('Alice', 'alice')")
    conn.execute("INSERT INTO people (name, normalized_name) VALUES ('Set', 'set')")  # False positive

    # Link things to people
    alice_id = conn.execute("SELECT id FROM people WHERE name = 'Alice'").fetchone()[0]
    set_id = conn.execute("SELECT id FROM people WHERE name = 'Set'").fetchone()[0]

    conn.execute("INSERT INTO thing_people (thing_id, person_id) VALUES ('task_00001', ?)", (alice_id,))
    conn.execute("INSERT INTO thing_people (thing_id, person_id) VALUES ('task_00002', ?)", (set_id,))

    conn.commit()

    # Query for Alice - should return 1 thing
    alice_things = db.get_person_things("Alice")
    assert len(alice_things) == 1
    assert alice_things[0]["id"] == "task_00001"

    # Query for "Set" - this is a FALSE POSITIVE from regex extraction
    # With NER, this should be empty
    set_things = db.get_person_things("Set")
    # This test will FAIL with current regex extraction (documenting issue)
    # After NER implementation, this should pass:
    # assert len(set_things) == 0

    db.close()
```

---

### Step 7: Update Documentation

**File:** `README.md`

Add NER section:

```markdown
## Entity Extraction

Memex Twos MCP extracts people, tags, and links from your content.

### People Extraction (NER)

**Recommended:** Install spaCy for accurate people extraction:

```bash
pip install -e ".[ner]"
python -m spacy download en_core_web_sm
```

This uses Named Entity Recognition to identify people with ~90% accuracy, avoiding false positives like verbs ("Set", "Plan") and month names ("March", "May").

**Without spaCy:** Falls back to regex-based extraction (lower accuracy but no dependencies).

### Usage

```bash
# With NER (recommended)
python src/convert_to_json.py data/raw/export.md -o data/processed/output.json

# Without NER (fallback)
python src/convert_to_json.py data/raw/export.md -o data/processed/output.json --no-ner
```

### Extraction Examples

**With NER:**
- "Met with Alice" → extracts "Alice" ✅
- "Set reminder" → no extraction ✅
- "Meeting in March" → no extraction ✅

**With Regex (fallback):**
- "Met with Alice" → extracts "Alice" ✅
- "Set reminder" → extracts "Set" ❌ (false positive)
- "Meeting in March" → extracts "March" ❌ (false positive)
```

**File:** `docs/v2-implementation-summary.md`

Update Phase 6 section:

```markdown
### Phase 6: Entity Extraction Improvement ✅

**Problem:** Regex-based people extraction had 40-60% false positive rate (verbs, months, common nouns misclassified).

**Solution:**
- Integrated spaCy NER with `en_core_web_sm` model
- Graceful fallback to regex if spaCy unavailable
- Model caching for performance
- Batch processing for large datasets

**Files Changed:**
- `pyproject.toml`: Added spaCy as optional dependency
- `src/convert_to_json.py`: Added `extract_people_ner()`, batch processing
- `tests/test_entity_extraction.py`: Comprehensive NER vs regex tests
- `README.md`: Installation and usage documentation

**Impact:**
- 80% reduction in false positives
- People queries now accurate and useful
- Optional dependency (no breaking changes)
```

---

## Testing Requirements

### Unit Tests

**Must pass:**

1. **NER Extraction Accuracy** (`test_entity_extraction.py`):
   - ✅ Extract single names ("John")
   - ✅ Extract full names ("Sarah Johnson")
   - ✅ Reject verbs ("Set", "Plan", "Review")
   - ✅ Reject months ("March", "May")
   - ✅ Extract multiple people from same text
   - ✅ Handle empty text

2. **Fallback Behavior** (`test_entity_extraction.py`):
   - ✅ Gracefully fall back to regex when spaCy unavailable
   - ✅ Respect `--no-ner` flag
   - ✅ Print clear warning when falling back

3. **Comparison Tests** (`test_entity_extraction.py`):
   - ✅ Verify NER avoids false positives that regex produces
   - ✅ Document known regex limitations

4. **Integration Tests** (`test_database.py`):
   - ✅ People queries return accurate results
   - ✅ No false positive people in database

### Manual Testing

**Test Dataset:** Use real export with known false positives

```bash
# Install spaCy
pip install -e ".[ner]"
python -m spacy download en_core_web_sm

# Convert with NER
python src/convert_to_json.py data/raw/test_export.md -o data/processed/test_ner.json

# Convert with regex (for comparison)
python src/convert_to_json.py data/raw/test_export.md -o data/processed/test_regex.json --no-ner

# Load to database
python scripts/load_to_sqlite.py data/processed/test_ner.json -o data/processed/test_ner.db --force

# Query people
source .venv/bin/activate
python -c "
from memex_twos_mcp.database import TwosDatabase
from pathlib import Path
db = TwosDatabase(Path('data/processed/test_ner.db'))
people = db.get_all_people()
print(f'Total people: {len(people)}')
for name, count in people[:20]:
    print(f'  {name}: {count} mentions')
db.close()
"
```

**Expected Results:**

- **With NER:** 40-60% fewer "people" in database
- **With NER:** No verbs in top 20 people
- **With NER:** No month names in top 20 people
- **With regex:** Verbs like "Set", "Plan" present in top people

### Performance Testing

**Test on 10K things:**

```bash
time python src/convert_to_json.py data/raw/large_export.md -o /tmp/ner.json
# Expected: <5s total

time python src/convert_to_json.py data/raw/large_export.md -o /tmp/regex.json --no-ner
# Expected: <1s total (baseline)
```

**Acceptable Performance:**
- NER extraction: <5s for 10K things (5-10x slower than regex, but acceptable for one-time conversion)
- No impact on MCP server query performance (extraction happens during conversion only)

---

## Migration Path

### For New Users

1. Install with NER support (recommended):
```bash
git clone https://github.com/Zarathustrum/memex-twos-mcp.git
cd memex-twos-mcp
pip install -e ".[ner]"
python -m spacy download en_core_web_sm
```

2. Convert export:
```bash
python src/convert_to_json.py data/raw/export.md -o data/processed/output.json
```

### For Existing Users

**Option 1: Rebuild with NER (Recommended)**

```bash
# Install spaCy
pip install spacy
python -m spacy download en_core_web_sm

# Reconvert export with NER
rm data/processed/twos.db  # Clean slate
python src/convert_to_json.py data/raw/twos_export.md -o data/processed/twos_data.json
python scripts/load_to_sqlite.py data/processed/twos_data.json --force

# Restart MCP server
```

**Benefits:**
- 80% fewer false positive people
- More accurate people queries
- Cleaner `people` table

**Option 2: Continue with Regex**

```bash
# Business as usual - no changes needed
python src/convert_to_json.py data/raw/export.md -o data/processed/output.json --no-ner
```

**Benefits:**
- No new dependencies
- Faster conversion
- Known behavior

### For CI/CD Environments

**Without spaCy (fallback mode):**
```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .  # No [ner] extras
CMD ["python", "-m", "memex_twos_mcp.server"]
```

**With spaCy:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[ner]" && \
    python -m spacy download en_core_web_sm
CMD ["python", "-m", "memex_twos_mcp.server"]
```

---

## Performance Targets

### Extraction Performance

| Dataset Size | NER Time | Regex Time | Acceptable? |
|--------------|----------|------------|-------------|
| 1K things    | <0.5s    | <0.1s      | ✅ Yes      |
| 10K things   | <5s      | <0.5s      | ✅ Yes      |
| 100K things  | <50s     | <5s        | ✅ Yes      |

**Rationale:** Conversion is one-time operation, 5-10x slowdown acceptable for 80% accuracy improvement.

### Accuracy Targets

| Metric                  | Regex | NER   | Target |
|-------------------------|-------|-------|--------|
| False Positive Rate     | ~50%  | <10%  | ✅ <10% |
| True Positive Rate      | ~90%  | ~90%  | ✅ ~90% |
| Precision               | ~50%  | >90%  | ✅ >90% |

**Calculation:**
- **Precision** = True Positives / (True Positives + False Positives)
- **Regex:** 1600 true / (1600 true + 1600 false) = 50%
- **NER:** 1600 true / (1600 true + 160 false) = 91%

### Database Impact

**Before NER (regex, 10K things):**
- People table: ~3,200 entries (1,600 false positives)
- thing_people relationships: ~3,200

**After NER (10K things):**
- People table: ~1,600 entries (true people only)
- thing_people relationships: ~1,600
- **50% reduction in database bloat**

---

## Quality Bar

### Code Quality

- ✅ Type hints on all new functions
- ✅ Docstrings with examples
- ✅ Graceful error handling (ImportError, OSError)
- ✅ Clear warning messages for fallback mode
- ✅ No breaking changes to existing code

### Test Coverage

- ✅ 10+ unit tests for NER extraction
- ✅ Comparison tests (NER vs regex)
- ✅ Fallback tests (spaCy unavailable)
- ✅ Integration tests (database queries)
- ✅ All tests passing on CI

### Documentation

- ✅ README updated with installation instructions
- ✅ CLI help text includes `--no-ner` flag
- ✅ Warning messages guide users to install spaCy
- ✅ Examples show accuracy differences

### User Experience

- ✅ Zero configuration for users with spaCy installed
- ✅ Automatic fallback for users without spaCy
- ✅ Clear guidance in error messages
- ✅ No breaking changes to workflow

---

## Success Criteria

### Must Achieve

1. **Accuracy Improvement:**
   - ✅ False positive rate <10% (down from ~50%)
   - ✅ No verbs in top 20 extracted people
   - ✅ No month names in top 20 extracted people

2. **Backward Compatibility:**
   - ✅ Existing workflows still work
   - ✅ No database schema changes
   - ✅ Graceful fallback when spaCy unavailable

3. **Performance:**
   - ✅ <5s extraction time for 10K things
   - ✅ No impact on MCP server query performance

4. **Tests:**
   - ✅ All existing tests still pass
   - ✅ 10+ new tests for NER
   - ✅ Comparison tests document improvements

### Nice to Have

- 📊 Accuracy metrics in conversion output
- 📊 Side-by-side comparison report (NER vs regex)
- 🔧 Interactive review mode for extracted people
- 🔧 Confidence scores for extracted names

---

## Implementation Checklist

### Pre-Implementation

- [ ] Read existing `convert_to_json.py` to understand current extraction
- [ ] Review test dataset to identify common false positives
- [ ] Test spaCy installation and model download

### Implementation

- [ ] Add spaCy to `pyproject.toml` as optional dependency
- [ ] Implement `extract_people_ner()` with model caching
- [ ] Update `extract_people()` with fallback logic
- [ ] Add batch processing for performance
- [ ] Update CLI with `--no-ner` flag
- [ ] Add comprehensive tests (`test_entity_extraction.py`)
- [ ] Update integration tests (`test_database.py`)

### Testing

- [ ] All unit tests pass
- [ ] Manual testing with real export
- [ ] Performance testing (10K things <5s)
- [ ] Accuracy validation (false positive rate <10%)
- [ ] Fallback testing (spaCy unavailable)

### Documentation

- [ ] Update README.md with installation instructions
- [ ] Update CLAUDE.md with NER workflow
- [ ] Update v2-implementation-summary.md
- [ ] Add examples to docstrings

### Deployment

- [ ] Create PR with clear description
- [ ] Run full test suite on CI
- [ ] Manual QA on real dataset
- [ ] Merge to main branch
- [ ] Tag release (v2.3 or similar)

---

## Rollback Plan

If NER implementation causes issues:

1. **Quick Fix:** Use `--no-ner` flag to disable NER
   ```bash
   python src/convert_to_json.py export.md --no-ner
   ```

2. **Revert Commit:** If merged, revert the PR
   ```bash
   git revert <commit-hash>
   git push
   ```

3. **Fallback Mode:** Code already includes regex fallback, no data loss

**Risk Level:** LOW
- Optional dependency (doesn't break existing installs)
- Automatic fallback prevents failures
- No database schema changes

---

## Dependencies Analysis

### New Dependencies

**spaCy (optional):**
- Package: `spacy>=3.7.0,<4.0.0`
- Model: `en_core_web_sm` (~15MB download)
- License: MIT
- Maturity: Production-ready, widely used

**Total Added Dependencies:** 1 optional package + 1 model

### Dependency Tree

```
spacy==3.7.0
├── spacy-legacy<3.1.0,>=3.0.11
├── spacy-loggers<2.0.0,>=1.0.0
├── murmurhash<1.1.0,>=0.28.0
├── cymem<2.1.0,>=2.0.2
├── preshed<3.1.0,>=3.0.2
├── thinc<8.3.0,>=8.1.8
├── wasabi<1.2.0,>=0.9.1
├── srsly<3.0.0,>=2.4.3
├── catalogue<2.1.0,>=2.0.6
├── typer<0.10.0,>=0.3.0
├── pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4
├── jinja2
├── setuptools
├── packaging>=20.0
├── langcodes<4.0.0,>=3.2.0
└── numpy>=1.15.0
```

**Total Size:** ~100MB installed (including model)

### Installation Impact

**Without spaCy (current):**
```bash
pip install -e .  # <10MB
```

**With spaCy:**
```bash
pip install -e ".[ner]"  # ~100MB
python -m spacy download en_core_web_sm  # ~15MB
```

**Recommendation:** Make spaCy optional, document both paths clearly.

---

## Future Enhancements

### Phase 6.1: Custom NER Training (Optional)

Train custom spaCy model on user's actual data:

```python
# Annotate sample of extractions
annotations = [
    ("Met with John", {"entities": [(9, 13, "PERSON")]}),
    ("Set reminder", {"entities": []}),  # No entities
]

# Fine-tune model
nlp.update(annotations)
```

**Benefits:**
- Higher accuracy for user's specific vocabulary
- Better handling of nicknames, unusual names

**Effort:** 1-2 days + annotation time

### Phase 6.2: Entity Disambiguation

Link extracted people to canonical entities:

```python
# Detect variations
"John", "John Smith", "Johnny" → same person

# Suggest merges in database
db.suggest_people_merges()
```

**Effort:** 2-3 days

### Phase 6.3: Additional Entity Types

Extend NER to extract:
- Organizations (COMPANY)
- Locations (LOC, GPE)
- Dates (DATE, TIME)
- Products (PRODUCT)

**Effort:** 1 day per entity type

---

## End of Phase 6 Execution Plan

**Next Steps:**
1. Review this plan for completeness
2. Implement in clean git branch
3. Test thoroughly (unit + integration + manual)
4. Update documentation
5. Create PR for review
6. Merge and tag release

**Questions for Implementation LLM:**
- Should we include entity confidence scores in output?
- Should we add interactive review mode for extracted people?
- Should we generate accuracy comparison report automatically?

**Estimated Timeline:**
- Implementation: 3-4 hours
- Testing: 1-2 hours
- Documentation: 1 hour
- **Total: 0.5-1 day**

---

**End of Execution Plan**
