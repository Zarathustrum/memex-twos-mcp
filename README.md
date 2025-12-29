# Memex Twos MCP

Transform your [Twos](https://www.twosapp.com/) app exports into a queryable knowledge base for Claude.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## What is this?

Memex Twos MCP is a [Model Context Protocol](https://modelcontextprotocol.io/) server that lets you query your personal "Thing" history from the Twos productivity app using natural language through Claude.

Instead of manually searching through thousands of things, ask Claude:
- "What things did I have about home renovation last summer?"
- "Show me everything I planned with Alice"
- "Find things tagged #work from January"

## Features

- Full-text search across all things (FTS5)
- Date-range queries for temporal analysis
- People and tag filtering
- Pattern analysis building blocks
- Automated data grooming (removes duplicates, fixes quality issues)
- Optional AI-powered semantic analysis via Claude Code
- Local-first privacy: your data stays on your machine
- Interactive setup wizard for beginners

## Quick Start

### 1. Export your Twos data

In the Twos app: Settings -> Export -> **Markdown with timestamps**

> **Important:** You must select "Markdown with timestamps" (not plain Markdown). The timestamps are required for the parser to extract dates and organize your data.

### 2. Run setup wizard

```bash
git clone https://github.com/yourusername/memex-twos-mcp.git
cd memex-twos-mcp
./setup_wizard.sh /path/to/Twos-Export.md
```

**That's it!** The wizard will:
- Create a virtual environment (if needed)
- Install dependencies
- Convert your export to JSON
- Clean and groom your data (remove duplicates, fix quality issues)
- Load cleaned data to SQLite
- Generate Claude Desktop configuration

**Optional:** Specify a custom Claude config location:
```bash
./setup_wizard.sh /path/to/Twos-Export.md /path/to/claude_desktop_config.json
```

**Alternative:** Run the Python wizard directly for interactive mode:
```bash
python scripts/setup_wizard.py
# Or with explicit paths:
python scripts/setup_wizard.py --export-file /path/to/Twos-Export.md --claude-config /path/to/claude_desktop_config.json
```

To override Claude config auto-detection globally, set `MEMEX_CLAUDE_CONFIG`.

### 3. Configure Claude

Restart Claude Desktop and try:
"What is in my task database?"

> **Windows Users:** Closing the Claude Desktop window sometimes doesn't exit the process. If MCP config doesn't load after "restarting," fully terminate Claude via Task Manager (Ctrl+Shift+Esc → find "Claude" → End Task), then restart the app.

## Requirements

### Essential
- Python 3.10 or higher
- Twos app with data to export
- Claude Desktop (for using the MCP server)

### Optional (for AI-powered features)
- [Claude Code CLI](https://code.claude.com) - Enables AI-powered data grooming and entity classification
- Active Claude subscription (if you have Claude Desktop, you already have this)

**Note:** AI features are optional. The MCP server works with basic grooming only.

### Installing Claude Code (Optional)

If you want to use AI-powered grooming and entity classification:

1. **Install Claude Code CLI:**
   ```bash
   npm install -g @anthropic/claude-code
   # or follow: https://code.claude.com/docs/quickstart
   ```

2. **Verify installation:**
   ```bash
   claude --version
   ```

3. **You're ready!** The scripts will automatically use Claude Code when you pass `--ai-analysis` or `--ai-classify` flags.

**When AI features are used:**
- Uses your Claude subscription quota (same as Claude Desktop)
- Processing time: 2-5 minutes for typical datasets
- Internet connection required
- All data stays local (Claude Code runs on your machine)

### Installing spaCy NER (Optional, Recommended)

For accurate people extraction with 80% fewer false positives:

```bash
pip install -e ".[ner]"
python -m spacy download en_core_web_sm
```

**Why use NER?**
- **With NER:** Accurately identifies real people names (~90% precision)
- **Without NER:** Uses regex fallback, prone to false positives (verbs like "Set", "Plan", months like "March", "May")

**Usage:**
```bash
# With NER (recommended, default)
python3 src/convert_to_json.py data/raw/twos_export.md -o data/processed/twos_data.json

# Without NER (faster but less accurate)
python3 src/convert_to_json.py data/raw/twos_export.md -o data/processed/twos_data.json --no-ner
```

**Examples:**
- "Met with Alice" → NER extracts "Alice" ✅ | Regex extracts "Alice" ✅
- "Set reminder" → NER extracts nothing ✅ | Regex extracts "Set" ❌
- "Meeting in March" → NER extracts nothing ✅ | Regex extracts "March" ❌

### Installing Semantic Search (Optional)

For hybrid search combining lexical (BM25) and semantic (vector) search:

```bash
pip install sentence-transformers sqlite-vec

# Download embedding model (~90MB)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Why use semantic search?**
- **Lexical-only:** Matches exact keywords - "doctor appointment" finds "doctor" and "appointment"
- **With semantic:** Understands concepts - "medical checkup" also finds "doctor appointment", "dentist visit"
- Better for conceptual queries like "moving house", "health issues", "work projects"

**Usage:**
```bash
# Load data with embeddings (default)
python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned.json

# Disable embeddings (faster loading)
MEMEX_DISABLE_EMBEDDINGS=1 python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned.json

# Migrate existing database to add embeddings
python3 scripts/migrate_add_embeddings.py data/processed/twos.db
```

**In Claude Desktop:**
```
# Lexical search (keyword-based)
"Search for doctor appointments"

# Hybrid search (semantic + keyword)
"Find health-related things from last year"  # Finds "doctor", "dentist", "checkup", etc.
```

**Performance:**
- Embedding generation: ~10K things in <30 seconds (CPU)
- Hybrid search: <200ms median (10K things)
- Storage: ~1.5KB per thing (~15MB for 10K things)

## Example Queries

- "Show me all things from last December"
- "What did I plan with Bob this year?"
- "Find things about doctor appointments"
- "Show me all things tagged #urgent"
- "What are my most common task themes?"
- "Give me a database health check"

## Quick Health Check

To verify your configuration is pointing at the right database:

```bash
python3 scripts/db_count.py
```

This prints the database path, total thing count, and any available load metadata.

## Manual Workflows

If you prefer step-by-step control over the setup wizard:

### Basic Workflow (No AI)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Convert Twos export to JSON
python3 src/convert_to_json.py data/raw/twos_export.md -o data/processed/twos_data.json

# 3. Groom data (remove duplicates, fix issues)
python3 scripts/groom_data.py

# 4. Load to SQLite
python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned.json

# 5. Configure Claude Desktop (add MCP server config)
# See MCP_SETUP.md for configuration details
```

### Advanced Workflow (With AI)

```bash
# 1-2. Same as basic (install deps, convert to JSON)

# 3. Groom data WITH AI semantic analysis
python3 scripts/groom_data.py --ai-analysis

# 4. Classify and normalize entities
python3 scripts/classify_entities.py --ai-classify --apply-mappings

# 5. Load normalized data to SQLite
python3 scripts/load_to_sqlite.py data/processed/twos_data_cleaned_normalized.json

# 6. Configure Claude Desktop
```

### Workflow Comparison

| Feature | Basic | Advanced (AI) |
|---------|-------|---------------|
| Duplicate removal | ✅ Exact duplicates | ✅ Exact + semantic |
| Broken reference fixes | ✅ | ✅ |
| Entity classification | ❌ | ✅ person/place/verb |
| Case normalization | ❌ | ✅ |
| Pattern detection | ❌ | ✅ |
| Query accuracy | Good | Excellent |
| Setup time | 2 min | 5-10 min |
| Claude Code required | No | Yes |

**Recommendation:** Start with basic workflow, upgrade to AI if you want better entity filtering.

## Documentation

- [Setup Guide](MCP_SETUP.md)
- [Data Grooming](docs/DATA_GROOMING_PROMPT.md)
- [Database Schema](schema/README.md)

## How It Works

```
Twos Export (MD)
    ↓
Parser (convert_to_json.py)
    ↓
JSON (twos_data.json)
    ↓
Data Grooming (groom_data.py)
    ↓
Cleaned JSON (twos_data_cleaned.json)
    ↓
Entity Classification (classify_entities.py) [optional]
    ↓
Normalized JSON (twos_data_cleaned_normalized.json)
    ↓
SQLite Database (load_to_sqlite.py)
    ↓
MCP Server (server.py)
    ↓
Claude Desktop
```

**Steps explained:**

1. **Parser** - Extracts things, people, tags, timestamps from Markdown
2. **Grooming** - Removes duplicates, fixes broken references, detects issues
3. **Entity Classification** (optional) - Classifies people vs places vs verbs, normalizes names
4. **SQLite** - Stores data with full-text search (FTS5), indexed queries
5. **MCP Server** - Exposes query tools for Claude (search, date ranges, filters)
6. **Claude Desktop** - Natural language interface to your task history

## Data Grooming

The setup wizard includes automated data cleaning that runs by default:

**Python Auto-Fix (Fast, Free):**
- Removes exact duplicates (same content within 1 minute)
- Fixes broken parent task references
- Detects normalization opportunities
- Generates detailed change reports

**Optional AI Analysis:**
- Semantic pattern detection and theme categorization
- Enhanced duplicate identification
- Schema improvement recommendations
- Uses Claude Code subscription quota (opt-in with `--ai-analysis`)

All changes are logged to `docs/grooming-reports/` and the original data is never modified. For manual grooming or tuning:

```bash
# Basic grooming (always recommended)
python scripts/groom_data.py data/processed/twos_data.json

# With AI semantic analysis
python scripts/groom_data.py data/processed/twos_data.json --ai-analysis

# Tuning options
python scripts/groom_data.py --duplicate-window 14  # More aggressive duplicate detection
```

See [DATA_GROOMING_PROMPT.md](docs/DATA_GROOMING_PROMPT.md) for details.

## Entity Classification

The parser extracts "people" from your tasks, but often misclassifies verbs, places, and common words as people (e.g., "New", "Put", "Seattle"). Entity classification fixes this:

### Quick Workflow

```bash
# 1. Extract entities and review summary
python scripts/classify_entities.py

# 2. AI classification (requires Claude Code)
python scripts/classify_entities.py --ai-classify

# 3. Apply classifications to create normalized data
python scripts/classify_entities.py --apply-mappings

# 4. Load normalized data
python scripts/load_to_sqlite.py data/processed/twos_data_cleaned_normalized.json
```

### What It Does

- **Extracts** all unique people/tags with frequency counts
- **Classifies** entities into: person, place, project, verb, other (AI or manual)
- **Normalizes** case variants (e.g., "Alice" and "alice" → "Alice")
- **Filters** non-person entities from people_mentioned field
- **Preserves** original data (creates new normalized file)

### Without AI (Skip Entity Classification)

If you don't have Claude Code, you can skip entity classification:

```bash
# Just load the cleaned data directly
python scripts/load_to_sqlite.py data/processed/twos_data_cleaned.json
```

The MCP server will work, but queries like "things with Alice" may include false matches (e.g., sentence-starting verbs misclassified as people).

**Note:** Entity classification is optional but recommended for better query accuracy. Manual CSV import/export is planned for a future release.

## Privacy & Security

- All data processing happens locally
- Nothing is uploaded to external servers
- MCP connection is local (stdin/stdout)

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [LICENSE](LICENSE).

## Support

- Report bugs: use the GitHub issue template
- Request features: use the GitHub issue template
- Read docs: [MCP_SETUP.md](MCP_SETUP.md)

# Memex Twos MCP - FAQ

## **Do I need to know how to code?**
Nope! There's a setup wizard that walks you through everything. You'll need to be comfortable running a few terminal commands, but it's pretty straightforward. If you can export from Twos and copy-paste commands, you're good.

## **What are the requirements?**
- Python 3.10+ (most systems already have this)
- Claude Desktop (for using the MCP server)
- A Twos export file
- *Optional:* Claude Code CLI if you want AI-powered data grooming and entity classification (improves accuracy)

## **Is my data private?**
100%. Everything stays on your machine - nothing gets uploaded to external servers except when Claude Desktop queries its API (same as any Claude conversation). The MCP server runs locally via stdin/stdout.

## **What's this "entity classification" thing?**
The Twos parser sometimes misclassifies sentence-starting verbs and place names as people (e.g., "New York" → person named "New", or "Put the keys away" → person named "Put"). Entity classification uses AI to filter these out and normalize name variants. It's optional but makes queries like "things with Alice" way more accurate.

You can run it with AI (requires Claude Code, takes 2-3 minutes) or skip it entirely. The server works either way.

## **How long does setup take?**
- Basic setup (no AI features): ~2 minutes
- Full setup with AI grooming and entity classification: ~7-10 minutes

The AI steps are optional and can be run later if you want to start fast.

## **What kind of queries can I run?**
Here are some examples I've tried:
- "What things did I have about the renovation last summer?"
- "Show me everything I planned with [person name]"
- "Find things tagged #work from January"
- "When did I first mention the basement project?"
- "Give me a timeline of all contractor-related Things"

Basically anything you could manually search for in Twos, but Claude can now do it conversationally and piece together threads across months.

## **Does it work with the Twos API or real-time sync?**
No, it uses Markdown exports. You export from Twos, run the converter, and load it into SQLite. If you want to update it later, just export again and re-run the loader. Not real-time, but works great for retrospectives and analysis.

## **Can I see an example of it working?**
I don't want to post screenshots with my personal data, but happy to share example queries/results if folks are interested. The README has some example queries too.

## **Wow, did you build this yourself?**
No, this project is the result of a number of iterations with Claude Code and Codex, with a little help from Gemini CLI and Perplexity.

I knew how to architect it and the data flow I wanted, the AIs did the heavy lifting on implementation and debugging.