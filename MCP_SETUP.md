# MCP Server Setup Guide

This guide explains how to set up and use the Memex Twos MCP server with Claude.

## Quick Setup (Recommended)

If you want automated setup, use the setup wizard instead of following the manual steps below:

### Shell Wrapper (Easiest)

```bash
./setup_wizard.sh /path/to/Twos-Export.md
```

**What it does:**
1. Checks for virtual environment (creates if missing)
2. Activates the venv
3. Runs the Python setup wizard with your export file
4. Handles the complete pipeline: convert → groom → load → configure

**Options:**
- Specify custom Claude config: `./setup_wizard.sh /path/to/export.md /path/to/claude_desktop_config.json`
- Auto-creates venv if missing (asks for confirmation)
- Uses venv Python for reliable package imports

### Python Wizard (Interactive)

```bash
python scripts/setup_wizard.py
```

**Interactive mode** - prompts you for:
- Export file location
- Claude Desktop config path (auto-detects or asks)
- Whether to run each step (convert, groom, load, configure)

**Non-interactive mode** - pass paths directly:
```bash
python scripts/setup_wizard.py --export-file /path/to/Twos-Export.md
python scripts/setup_wizard.py --export-file /path/to/export.md --claude-config /path/to/claude_desktop_config.json
```

**Environment variable:** Set `MEMEX_CLAUDE_CONFIG` to override auto-detection globally.

---

**When to use manual setup instead:** Use the manual steps below if you want fine-grained control over each step, need to debug individual components, or want to customize the data processing pipeline.

## Prerequisites

- Python 3.10 or higher
- Virtual environment with MCP SDK installed
- SQLite database loaded with Twos data

## Installation

### 1. Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install "mcp[cli]"
```

### 2. Verify Database

Ensure your SQLite database exists at:
```
data/processed/twos.db
```

If not, run the data loader:
```bash
python3 scripts/load_to_sqlite.py data/processed/twos_data.json
```

## Testing the Server

### Local Testing (CLI)

Test the server using the MCP CLI tool:

```bash
# Activate venv if not already
source .venv/bin/activate

# Test server startup
PYTHONPATH=src python -m memex_twos_mcp.server
# Server should start without errors (Ctrl+C to stop)
```

### Test with MCP Inspector

```bash
# Install MCP inspector
npx @modelcontextprotocol/inspector

# Then in the inspector, connect to:
# Command: python
# Args: ["-m", "memex_twos_mcp.server"]
# Env: {"PYTHONPATH": "<path-to-project>/src"}
```

## Configuring Claude Desktop

### For Claude Desktop App

1. **Find your Claude config file**:
   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux: `~/.config/Claude/claude_desktop_config.json`

2. **Add the server configuration**:

> **Note**: Replace `<path-to-project>` with your actual installation path.  
> Windows: `C:\\Users\\YourName\\memex-twos-mcp`  
> Mac/Linux: `/home/yourname/memex-twos-mcp`

```json
{
  "mcpServers": {
    "memex-twos-v2": {
      "command": "python",
      "args": [
        "-m",
        "memex_twos_mcp.server"
      ],
      "cwd": "<path-to-project>",
      "env": {
        "PYTHONPATH": "<path-to-project>/src"
      }
    }
  }
}
```

3. **Restart Claude Desktop**

   > **Windows Users:** Closing the Claude Desktop window sometimes doesn't exit the process. If the MCP server doesn't appear after "restarting," fully terminate Claude via Task Manager (Ctrl+Shift+Esc → find "Claude" → End Task), then restart the app.

4. **Verify Connection**:
   - Open Claude Desktop
   - Look for MCP indicator/status
   - Try using a tool: "Search my things for 'Alice'"

## Configuring Claude Code (CLI)

### For Claude Code CLI

1. **Find your Claude Code config**:
   ```bash
   # Location varies, check Claude Code docs
   # Typically: ~/.claude/config.json or similar
   ```

2. **Add MCP server** following Claude Code's MCP configuration format

3. **Test in CLI**:
   ```bash
   claude "Search my things for college"
   ```

## Available Tools

Once configured, Claude can use these tools:

### 1. query_things_by_date
Query things within a date range.
```
Example: "Show me things from January 2024"
```

### 2. search_things
Full-text search across all content.
```
Example: "Search for things about 'house renovation'"
```

### 3. get_person_things
Get all things mentioning a person.
```
Example: "Show me all things mentioning Alice"
```

### 4. get_tag_things
Get things with a specific tag.
```
Example: "Show me all #siri things"
```

### 5. get_things_stats
Get database statistics.
```
Example: "What's in my task database?"
```

## Available Resources

Claude can also read these resources:

- `twos://database/stats` - Database statistics
- `twos://database/people` - List of all people
- `twos://database/tags` - List of all tags

## Troubleshooting

### Server won't start

**Check Python path**:
```bash
which python
# Should be inside .venv
```

**Check database exists**:
```bash
ls -lh data/processed/twos.db
```

**Check PYTHONPATH**:
```bash
PYTHONPATH=src python -c "from memex_twos_mcp import database; print('OK')"
```

### Claude can't connect

**Check config file syntax**:
- JSON must be valid (no trailing commas)
- Paths must be absolute
- Use forward slashes even on Windows

**Check logs**:
- Claude Desktop logs location varies by platform
- Look for MCP connection errors

### Tool calls fail

**Check database permissions**:
```bash
sqlite3 data/processed/twos.db "SELECT COUNT(*) FROM things;"
```

**Test tool directly**:
```python
from pathlib import Path
from memex_twos_mcp.database import TwosDatabase

db = TwosDatabase(Path("data/processed/twos.db"))
print(db.get_stats())
```

## Development

### Running Tests

```bash
# Activate venv
source .venv/bin/activate

# Run tests (when created)
pytest tests/
```

### Making Changes

1. Edit code in `src/memex_twos_mcp/`
2. Restart MCP server (restart Claude Desktop)
3. Test changes

### Adding New Tools

1. Add method to `database.py`
2. Add tool definition in `server.py` `list_tools()`
3. Add handler in `server.py` `call_tool()`
4. Restart server

## Security Notes

- The MCP server runs locally and accesses your personal database
- No data is sent to external servers
- Database contains personal information - keep it private
- MCP connection is local (stdin/stdout)

## Next Steps

- Try asking Claude questions about your things
- Experiment with date range queries
- Search for patterns across your data
- Let Claude help you discover insights

## Support

For issues or questions:
- Check GitHub issues
- Review MCP SDK documentation
- Check Claude Code/Desktop documentation
