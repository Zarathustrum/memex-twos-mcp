# Prompt: Prepare memex-twos-mcp for GitHub Public Release

## OBJECTIVE

Transform the current private memex-twos-mcp project into a public GitHub repository that other Twos app users can use, while removing all personal information and improving usability for less-technical users.

## CONTEXT

You are working on `<path-to-project>/memex-twos-mcp`, a fully functional MCP server that converts Twos app exports into a queryable SQLite database with Claude integration. It works perfectly but contains personal information and has technical barriers for novice users.

**Current Project Structure:**
```
memex-twos-mcp/
├── src/memex_twos_mcp/     # MCP server code
├── scripts/                 # Data conversion scripts
├── schema/                  # SQLite schema
├── docs/                    # Documentation + grooming reports
├── data/                    # Personal data (gitignored)
├── MCP_SETUP.md            # Setup instructions (has hardcoded paths)
├── CLAUDE.md               # Personal workflow (to remove)
├── PROMPT_FOR_NEXT_LLM.md  # Personal context (to remove)
└── README.md               # Has personal data stats
```

**Read This First**: `GITHUB_DEPLOYMENT_PLAN.md` contains the complete analysis and plan.

## YOUR MISSION

Execute the deployment plan in phases. Work methodically and commit after each phase.

---

## PHASE 1: AUDIT AND SANITIZE (Remove Personal Info)

### Task 1.1: Find All Personal Data

Search the entire codebase for personal information:

```bash
# Search for specific names mentioned in data
grep -r "Alice" --exclude-dir=.git --exclude-dir=.venv --exclude-dir=data
grep -r "Bob" --exclude-dir=.git --exclude-dir=.venv --exclude-dir=data
grep -r "Carol" --exclude-dir=.git --exclude-dir=.venv --exclude-dir=data
grep -r "Diana" --exclude-dir=.git --exclude-dir=.venv --exclude-dir=data
grep -r "ProjectX" --exclude-dir=.git --exclude-dir=.venv --exclude-dir=data

# Search for hardcoded paths
grep -r "<path-to-home>" --exclude-dir=.git --exclude-dir=.venv
grep -r "192.168" --exclude-dir=.git --exclude-dir=.venv

# Search for personal context
grep -r "~10,000" --exclude-dir=.git --exclude-dir=.venv
```

**Create a report**: `SANITIZATION_REPORT.md` listing every file and line with personal data.

### Task 1.2: Remove Personal Files

```bash
# Files to delete before GitHub
rm CLAUDE.md
rm PROMPT_FOR_NEXT_LLM.md
rm docs/grooming-reports/2025-12-19-initial-grooming.md
rm -rf data/raw/*  # Already gitignored but be sure
rm -rf data/processed/*  # Already gitignored
```

### Task 1.3: Create Generic Grooming Example

**Create**: `docs/grooming-reports/EXAMPLE_GROOMING.md`

Replace real data with generic examples:
- Alice → Alice
- Bob → Bob
- Carol → Carol
- Diana → Diana
- ProjectX → ProjectX
- ~10,000 → ~10,000
- Specific dates → Generic ranges

### Task 1.4: Sanitize README.md

**Current problematic sections**:
```markdown
### Database Stats
- **Tasks**: ~10,000 (Oct 2023 - Dec 2025)
- **People**: 2,961 unique entities
...

Top entities: Alice (243), Bob (235)...
```

**Replace with**:
```markdown
### Example Database Stats
- **Tasks**: ~10,000+ (2+ years of data)
- **People**: ~3,000 unique entities
- **Tags**: Customizable
- **Full-text search**: All content indexed

Your stats will vary based on your Twos usage patterns.
```

### Task 1.5: Sanitize MCP_SETUP.md

Replace all instances of `<path-to-project>/memex-twos-mcp` with `$PROJECT_DIR` or `<path-to-project>`.

Add note:
```markdown
> **Note**: Replace `<path-to-project>` with your actual installation path.
> On Windows: `C:\Users\YourName\memex-twos-mcp`
> On Mac/Linux: `/home/yourname/memex-twos-mcp`
```

**Commit after Phase 1**:
```
Sanitize: remove all personal information for public release

- Removed personal workflow files (CLAUDE.md, PROMPT_FOR_NEXT_LLM.md)
- Created generic grooming example
- Replaced real stats with placeholders
- Generalized all hardcoded paths
- Removed personal data from all documentation

(anthropic claude-code sonnet-4.5)
```

---

## PHASE 2: ADD CONFIGURATION SYSTEM (Make Paths Flexible)

### Task 2.1: Create Configuration Module

**Create**: `src/memex_twos_mcp/config.py`

```python
"""
Configuration management for memex-twos-mcp.

Supports environment variables and config files.
"""

import os
from pathlib import Path
from typing import Optional
import yaml


class MemexConfig:
    """Configuration for Memex Twos MCP server."""

    def __init__(self):
        # Project root
        self.project_root = self._find_project_root()

        # Database path (env var or default)
        self.db_path = Path(os.getenv(
            'MEMEX_DB_PATH',
            self.project_root / 'data' / 'processed' / 'twos.db'
        ))

        # Config file path
        self.config_file = Path(os.getenv(
            'MEMEX_CONFIG',
            Path.home() / '.memex-twos' / 'config.yaml'
        ))

        # Load from config file if exists
        if self.config_file.exists():
            self._load_config_file()

    def _find_project_root(self) -> Path:
        """Find project root directory."""
        # Start from this file's location
        current = Path(__file__).parent

        # Look for markers (pyproject.toml, .git)
        while current != current.parent:
            if (current / 'pyproject.toml').exists():
                return current
            if (current / '.git').exists():
                return current
            current = current.parent

        # Fallback to current directory
        return Path.cwd()

    def _load_config_file(self):
        """Load configuration from YAML file."""
        with open(self.config_file) as f:
            config = yaml.safe_load(f)

        if 'database' in config:
            self.db_path = Path(config['database']['path'])

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if not self.db_path.exists():
            errors.append(f"Database not found: {self.db_path}")

        if not self.db_path.is_file():
            errors.append(f"Database path is not a file: {self.db_path}")

        return errors


# Global config instance
_config = None

def get_config() -> MemexConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = MemexConfig()
    return _config
```

### Task 2.2: Update server.py to Use Config

**Modify**: `src/memex_twos_mcp/server.py`

Find this line:
```python
db_path = Path(__file__).parent.parent.parent / "data" / "processed" / "twos.db"
```

Replace with:
```python
from .config import get_config

config = get_config()
db_path = config.db_path

# Validate before starting
errors = config.validate()
if errors:
    for error in errors:
        print(f"❌ {error}", file=sys.stderr)
    print("\n💡 Run setup wizard: python scripts/setup_wizard.py", file=sys.stderr)
    sys.exit(1)
```

### Task 2.3: Add Config File Template

**Create**: `config.example.yaml`

```yaml
# Memex Twos MCP Configuration
# Copy to ~/.memex-twos/config.yaml and customize

database:
  # Path to SQLite database
  path: /home/yourname/memex-twos-mcp/data/processed/twos.db

mcp:
  # MCP server settings
  log_level: INFO

  # Claude Desktop integration
  claude_desktop_config: auto  # or provide path
```

**Commit after Phase 2**:
```
Config: add flexible configuration system

- Created config.py for path management
- Support environment variables (MEMEX_DB_PATH)
- Support config file (~/.memex-twos/config.yaml)
- Auto-detect project root
- Added validation with helpful error messages
- Updated server.py to use config
- Added config.example.yaml template

(anthropic claude-code sonnet-4.5)
```

---

## PHASE 3: ADD SETUP WIZARD (Improve Usability)

### Task 3.1: Create Setup Wizard

**Create**: `scripts/setup_wizard.py`

```python
#!/usr/bin/env python3
"""
Interactive setup wizard for memex-twos-mcp.
Guides users through installation and configuration.
"""

import sys
import subprocess
from pathlib import Path
import shutil


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def check_python_version():
    """Check Python version is 3.10+"""
    print_header("Step 1: Checking Python Version")

    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required, you have {sys.version}")
        print("Please upgrade Python and try again.")
        sys.exit(1)

    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def setup_virtual_environment():
    """Create and activate virtual environment"""
    print_header("Step 2: Virtual Environment")

    venv_path = Path.cwd() / '.venv'

    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return

    response = input("Create virtual environment? (y/n): ")
    if response.lower() != 'y':
        print("⚠️  Skipping virtual environment")
        return

    print("Creating virtual environment...")
    subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
    print("✅ Virtual environment created")
    print("\n💡 Activate it with:")
    print("   source .venv/bin/activate  (Linux/Mac)")
    print("   .venv\\Scripts\\activate     (Windows)")


def install_dependencies():
    """Install Python dependencies"""
    print_header("Step 3: Installing Dependencies")

    response = input("Install dependencies now? (y/n): ")
    if response.lower() != 'y':
        print("⚠️  Skipping dependency installation")
        print("Run: pip install -e . later")
        return

    print("Installing dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], check=True)
    print("✅ Dependencies installed")


def setup_data_files():
    """Set up data directories and get Twos export"""
    print_header("Step 4: Data Setup")

    # Create directories
    (Path.cwd() / 'data' / 'raw').mkdir(parents=True, exist_ok=True)
    (Path.cwd() / 'data' / 'processed').mkdir(parents=True, exist_ok=True)

    print("Do you have a Twos export file? (markdown format)")
    print("1. Yes, I have an export file")
    print("2. No, use sample data for testing")
    print("3. Skip for now")

    choice = input("Choice (1/2/3): ")

    if choice == '1':
        export_path = input("Path to your Twos export file: ")
        if Path(export_path).exists():
            # Copy to data/raw
            shutil.copy(export_path, 'data/raw/twos_export.md')
            print("✅ Export file copied")
            return 'real'
        else:
            print(f"❌ File not found: {export_path}")
            return None

    elif choice == '2':
        print("Using sample data (not implemented yet)")
        print("TODO: Copy sample data")
        return 'sample'

    else:
        print("⚠️  Skipping data setup")
        return None


def convert_and_load_data(data_type):
    """Convert markdown to JSON and load to SQLite"""
    if not data_type:
        return

    print_header("Step 5: Converting Data")

    print("Converting Twos export to JSON...")
    # Run converter
    subprocess.run([
        sys.executable,
        'src/convert_to_json.py',
        'data/raw/twos_export.md',
        '-o', 'data/processed/twos_data.json',
        '--pretty'
    ], check=True)
    print("✅ Conversion complete")

    print("\nLoading data into SQLite...")
    subprocess.run([
        sys.executable,
        'scripts/load_to_sqlite.py',
        'data/processed/twos_data.json'
    ], check=True)
    print("✅ Database created")


def generate_mcp_config():
    """Generate MCP configuration for Claude Desktop"""
    print_header("Step 6: MCP Configuration")

    print("Generate Claude Desktop configuration?")
    response = input("(y/n): ")

    if response.lower() != 'y':
        print("⚠️  Skipping MCP config")
        print("See MCP_SETUP.md for manual configuration")
        return

    # Run config generator (to be created)
    print("Running config generator...")
    subprocess.run([
        sys.executable,
        'scripts/generate_mcp_config.py'
    ])


def test_server():
    """Test that server can start"""
    print_header("Step 7: Testing Server")

    print("Testing server startup...")
    print("(This will start the server briefly and then stop it)")

    # TODO: Start server, wait 2 seconds, kill it
    print("✅ Server test (TODO)")


def main():
    """Run the setup wizard"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║           Memex Twos MCP Setup Wizard                    ║
    ║                                                           ║
    ║   Transform your Twos exports into a queryable          ║
    ║   knowledge base for Claude                              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    try:
        check_python_version()
        setup_virtual_environment()
        install_dependencies()
        data_type = setup_data_files()
        convert_and_load_data(data_type)
        generate_mcp_config()
        test_server()

        print_header("Setup Complete! 🎉")
        print("Next steps:")
        print("1. Restart Claude Desktop")
        print("2. Try asking: 'What's in my task database?'")
        print("3. See MCP_SETUP.md for more examples")
        print("\nHappy querying! 🚀")

    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("Check MCP_SETUP.md for troubleshooting")
        sys.exit(1)


if __name__ == '__main__':
    main()
```

Make it executable:
```bash
chmod +x scripts/setup_wizard.py
```

### Task 3.2: Create Validation Script

**Create**: `scripts/validate_setup.py`

This should check:
- Python version
- Dependencies installed
- Database exists
- Database has data
- Server can import
- Config file valid
- MCP config exists

### Task 3.3: Create MCP Config Generator

**Create**: `scripts/generate_mcp_config.py`

Detect OS, find Claude Desktop config, generate correct JSON.

**Commit after Phase 3**:
```
Setup: add interactive setup wizard and validators

- Created setup_wizard.py for guided installation
- Added validate_setup.py to check configuration
- Added generate_mcp_config.py for Claude integration
- Improved error messages throughout
- Added platform detection for paths

Makes installation accessible to non-technical users.

(anthropic claude-code sonnet-4.5)
```

---

## PHASE 4: CREATE SAMPLE DATA

### Task 4.1: Create Sample Twos Export

**Create**: `data/sample/sample_export.md`

Generic Twos export with ~200 tasks:
- People: Alice, Bob, Carol, Diana
- Projects: Home Improvement, Work Project, Health Goals
- Dates: Recent 3 months
- No personal information

### Task 4.2: Document Sample Data Usage

**Create**: `data/sample/README.md`

Explain how to use sample data for testing before using real data.

**Commit**:
```
Sample: add generic sample data for testing

- Created sample Twos export with 200 generic tasks
- Uses placeholder names (Alice, Bob, Carol)
- Safe for public demos and testing
- Documented usage in data/sample/README.md

(anthropic claude-code sonnet-4.5)
```

---

## PHASE 5: OPEN SOURCE POLISH

### Task 5.1: Add License

**Create**: `LICENSE`

Use MIT License (most permissive):
```
MIT License

Copyright (c) 2025 memex-twos-mcp contributors

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

### Task 5.2: Add Contributing Guidelines

**Create**: `CONTRIBUTING.md`

```markdown
# Contributing to Memex Twos MCP

Thank you for your interest! Here's how to contribute:

## Reporting Bugs
...

## Suggesting Features
...

## Pull Requests
...

## Code Style
- Follow PEP 8
- Add type hints
- Write tests
...
```

### Task 5.3: Add Code of Conduct

**Create**: `CODE_OF_CONDUCT.md`

Use Contributor Covenant standard template.

### Task 5.4: Add GitHub Templates

**Create**:
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`
- `.github/PULL_REQUEST_TEMPLATE.md`

### Task 5.5: Add Tests

**Create**: `tests/` with basic tests:
- `test_database.py` - Database wrapper tests
- `test_converter.py` - JSON conversion tests
- `test_server.py` - MCP server tests (mock)

### Task 5.6: Add CI/CD

**Create**: `.github/workflows/test.yml`

Run tests on push, multiple Python versions.

**Commit**:
```
Project: add open source governance and CI

- Added MIT License
- Created CONTRIBUTING.md and CODE_OF_CONDUCT.md
- Added GitHub issue/PR templates
- Created initial test suite
- Added GitHub Actions CI workflow

Ready for public collaboration.

(anthropic claude-code sonnet-4.5)
```

---

## PHASE 6: REWRITE README FOR PUBLIC

### Task 6.1: Create New README

**Replace entire `README.md`** with public-friendly version:

```markdown
# Memex Twos MCP

> Transform your [Twos](https://www.twosapp.com/) app exports into a queryable knowledge base for Claude

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## What is this?

Memex Twos MCP is a [Model Context Protocol](https://modelcontextprotocol.io/) server that lets you query your personal task history from the Twos productivity app using natural language through Claude.

Instead of manually searching through thousands of tasks, ask Claude:
- "What tasks did I have about home renovation last summer?"
- "Show me everything I planned with Alice"
- "Find tasks tagged #work from January"

## Features

- 🔍 **Full-text search** across all your tasks (FTS5)
- 📅 **Date-range queries** for temporal analysis
- 👥 **People tracking** - see all tasks mentioning someone
- 🏷️ **Tag filtering** - organize by your custom tags
- 📊 **Pattern analysis** - discover trends in your life
- 🔒 **Privacy-first** - all data stays local
- 🚀 **Easy setup** - interactive wizard included

## Quick Start

### 1. Export your Twos data

In the Twos app: Settings → Export → Markdown format

### 2. Run setup wizard

```bash
git clone https://github.com/yourusername/memex-twos-mcp.git
cd memex-twos-mcp
python scripts/setup_wizard.py
```

The wizard will:
- Create a virtual environment
- Install dependencies
- Convert your export to SQLite
- Generate Claude Desktop configuration

### 3. Configure Claude

Restart Claude Desktop and try:
> "What's in my task database?"

## Requirements

- Python 3.10 or higher
- Twos app with data to export
- Claude Desktop or Claude Code CLI

## Example Queries

Once configured, you can ask Claude:

- "Show me all tasks from last December"
- "What did I plan with Bob this year?"
- "Find tasks about 'doctor appointments'"
- "Show me all tasks tagged #urgent"
- "What are my most common task themes?"

## Documentation

- [Setup Guide](MCP_SETUP.md) - Detailed installation instructions
- [Data Grooming](docs/DATA_GROOMING_PROMPT.md) - How to improve data quality
- [Database Schema](schema/README.md) - Technical details

## How It Works

```
Twos Export (MD) → Parser → JSON → SQLite → MCP Server → Claude
```

1. **Parser**: Extracts tasks, people, tags, timestamps
2. **SQLite**: Stores data with full-text search index
3. **MCP Server**: Exposes tools for Claude to query
4. **Claude**: Natural language interface to your data

## Privacy & Security

- All data processing happens **locally** on your machine
- Nothing is uploaded to external servers
- Your Twos data never leaves your computer
- MCP connection is local (stdin/stdout)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- Inspired by Vannevar Bush's [memex](https://en.wikipedia.org/wiki/Memex) concept
- Built with [Model Context Protocol](https://modelcontextprotocol.io/)
- For [Twos](https://www.twosapp.com/) app users

## Support

- 🐛 [Report bugs](../../issues/new?template=bug_report.md)
- 💡 [Request features](../../issues/new?template=feature_request.md)
- 📖 [Read docs](MCP_SETUP.md)
```

**Commit**:
```
Docs: rewrite README for public audience

- Removed all personal data and specific stats
- Added clear value proposition
- Included badges and links
- Wrote for non-technical users
- Added example queries
- Emphasized privacy and local-first

(anthropic claude-code sonnet-4.5)
```

---

## FINAL CHECKS

### Before Creating GitHub Repo

**Run these checks**:

```bash
# 1. Search for personal data one more time
grep -r "Alice\|Bob\|Carol\|Diana\|ProjectX" . \
  --exclude-dir=.git --exclude-dir=.venv --exclude-dir=data

# 2. Search for hardcoded paths
grep -r "<path-to-home>" . --exclude-dir=.git --exclude-dir=.venv

# 3. Check no data files in git
git ls-files | grep -E "\.db$|data/raw|data/processed"

# 4. Validate all markdown files render correctly
# (use markdown preview)

# 5. Test wizard on fresh clone
git clone <path-to-project>/memex-twos-mcp /tmp/test-memex
cd /tmp/test-memex
python scripts/setup_wizard.py
# Should work without errors

# 6. Run tests
pytest tests/

# 7. Lint code
black src/ scripts/
flake8 src/ scripts/
```

### Create Checklist

**Create**: `PRE_RELEASE_CHECKLIST.md`

- [ ] All personal data removed
- [ ] Setup wizard tested
- [ ] Sample data works
- [ ] README is public-friendly
- [ ] License added (MIT)
- [ ] Contributing guide added
- [ ] Code of conduct added
- [ ] Tests pass
- [ ] CI/CD configured
- [ ] Documentation complete
- [ ] No hardcoded paths
- [ ] Works on Windows/Mac/Linux (tested)

---

## DEPLOYMENT STEPS

Once all phases are complete:

### Option A: New Public Repo (Recommended)

```bash
# 1. Create new repo on GitHub: twos-memex

# 2. Create orphan branch (clean history)
git checkout --orphan github-public

# 3. Add all sanitized files
git add -A

# 4. Initial commit
git commit -m "Initial public release

Full-featured MCP server for querying Twos app exports.

Features:
- Interactive setup wizard
- Full-text search (FTS5)
- Claude Desktop integration
- Sample data for testing
- Comprehensive documentation

(anthropic claude-code sonnet-4.5)"

# 5. Push to GitHub
git remote add github https://github.com/yourusername/twos-memex.git
git push -u github github-public:main

# 6. Set main as default branch on GitHub

# 7. Add topics on GitHub: mcp, twos, claude, productivity, pkm

# 8. Create first release v0.1.0
```

### Option B: Push Current Repo (After Sanitization)

Only if you're confident all personal data is removed.

---

## SUCCESS CRITERIA

You know it's ready when:

1. ✅ Grep for personal names returns nothing
2. ✅ Fresh clone → setup wizard → works
3. ✅ Sample data runs successfully
4. ✅ README makes sense to outsiders
5. ✅ All docs reviewed and sanitized
6. ✅ Tests pass
7. ✅ No hardcoded absolute paths
8. ✅ License and contributing docs present

---

## EXECUTION CHECKLIST FOR YOU (THE LLM)

- [ ] Read GITHUB_DEPLOYMENT_PLAN.md thoroughly
- [ ] Execute Phase 1 (Sanitization) - commit
- [ ] Execute Phase 2 (Configuration) - commit
- [ ] Execute Phase 3 (Setup Wizard) - commit
- [ ] Execute Phase 4 (Sample Data) - commit
- [ ] Execute Phase 5 (Open Source) - commit
- [ ] Execute Phase 6 (README Rewrite) - commit
- [ ] Run final checks
- [ ] Create PRE_RELEASE_CHECKLIST.md
- [ ] Report completion with any warnings

---

## IMPORTANT NOTES

- **Commit after each phase** with proper messages
- **Test incrementally** - don't wait until the end
- **Ask questions** if you find ambiguity
- **Document any issues** you discover
- **Be thorough** with privacy audits
- **Test the wizard** actually works
- **Keep GITHUB_DEPLOYMENT_PLAN.md** as reference

## WHEN YOU'RE DONE

Summarize:
1. What was completed
2. What personal data was found and removed
3. What still needs manual review
4. Any blockers or concerns
5. Recommended next steps

Good luck! 🚀
