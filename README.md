# Memex Twos MCP

Transform your [Twos](https://www.twosapp.com/) app exports into a queryable knowledge base for Claude.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## What is this?

Memex Twos MCP is a [Model Context Protocol](https://modelcontextprotocol.io/) server that lets you query your personal task history from the Twos productivity app using natural language through Claude.

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

In the Twos app: Settings -> Export -> Markdown format

### 2. Run setup wizard

```bash
git clone https://github.com/yourusername/memex-twos-mcp.git
cd memex-twos-mcp
python scripts/setup_wizard.py
```

Optional: run the shell wrapper that accepts a Twos export path (and an optional
Claude Desktop config path):

```bash
./setup_wizard.sh /path/to/Twos-Export.md
./setup_wizard.sh /path/to/Twos-Export.md /path/to/claude_desktop_config.json
```

You can also pass the export and Claude config paths directly:

```bash
python scripts/setup_wizard.py --export-file /path/to/Twos-Export.md
python scripts/setup_wizard.py --export-file /path/to/Twos-Export.md --claude-config /path/to/claude_desktop_config.json
```

To override Claude config auto-detection globally, set `MEMEX_CLAUDE_CONFIG`.

The wizard can:
- Create a virtual environment
- Install dependencies
- Convert your export to JSON
- Clean and groom your data (remove duplicates, fix quality issues)
- Load cleaned data to SQLite
- Generate Claude Desktop configuration

### 3. Configure Claude

Restart Claude Desktop and try:
"What is in my task database?"

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
