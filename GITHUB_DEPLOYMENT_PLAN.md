# GitHub Deployment Plan

## Current State Analysis

### Personal Information to Remove/Sanitize

**Files with Personal Data:**
1. `docs/grooming-reports/2025-12-19-initial-grooming.md` - Contains actual names (Alice, Bob, Carol, Diana)
2. `README.md` - Has specific database stats with real entity names
3. `PROMPT_FOR_NEXT_LLM.md` - Personal context about user's life
4. `CLAUDE.md` - User-specific paths and workflow
5. `MCP_SETUP.md` - Has hardcoded user paths in examples
6. Example outputs in server.py docstrings

**Files to Keep Private:**
- `memex-twos-data/` entire repo (separate, stays private)
- All `/data/` directory contents (gitignored)
- `.venv/` (gitignored)
- Personal grooming reports

### Technical Barriers for Less-Technical Users

**Current Issues:**
1. Manual virtual environment setup
2. Hardcoded absolute paths throughout
3. No installation script
4. MCP configuration is manual JSON editing
5. No validation/error checking for setup
6. Assumes Python 3.10+ knowledge
7. No GUI or simpler interface
8. Database path hardcoded in server.py

### Required Changes for Public Release

#### Phase 1: Sanitization
- Remove personal data from all docs
- Create generic example outputs
- Sanitize grooming report or make it template
- Remove CLAUDE.md and PROMPT_FOR_NEXT_LLM.md (project-specific)
- Replace real stats with placeholders

#### Phase 2: Generalization
- Environment variables for paths (DATABASE_PATH, etc.)
- Config file support (memex-config.yaml or similar)
- Remove hardcoded paths from code
- Make installation path-agnostic
- Add platform detection (Windows/Mac/Linux)

#### Phase 3: Usability
- Add `setup.py` or `install.sh` wizard
- Add `--init` command to create config interactively
- Better error messages with suggestions
- Add `--check` command to validate setup
- Create step-by-step first-run guide
- Add demo/sample data for testing

#### Phase 4: Open Source Polish
- Add proper LICENSE (MIT/Apache/GPL?)
- Add CONTRIBUTING.md guidelines
- Add CODE_OF_CONDUCT.md
- Add issue/PR templates
- Update README for general audience
- Add CHANGELOG.md
- Add badges (build status, license, etc.)
- Create demo GIF/video

---

## Detailed Implementation Plan

### 1. Create Generic Documentation

**New Files:**
- `README.github.md` - Clean, generic version for public
- `GROOMING_EXAMPLE.md` - Sanitized example (Alice/Bob names)
- `DEMO_DATA.md` - How to create sample data for testing

**Update Files:**
- `MCP_SETUP.md` - Replace user paths with `$PROJECT_DIR` placeholders
- `docs/DATA_GROOMING_PROMPT.md` - Remove references to specific people

### 2. Make Code Path-Agnostic

**Changes to `src/memex_twos_mcp/server.py`:**
```python
# Instead of hardcoded path
db_path = Path(__file__).parent.parent.parent / "data" / "processed" / "twos.db"

# Use environment variable or config
db_path = Path(os.getenv('MEMEX_DB_PATH',
    Path.home() / '.memex-twos' / 'twos.db'))
```

**Add `src/memex_twos_mcp/config.py`:**
- Load from `~/.memex-twos/config.yaml`
- Support environment variables
- Provide sensible defaults
- Validate paths exist

### 3. Create Installation Wizard

**New File: `scripts/setup_wizard.py`:**
```python
#!/usr/bin/env python3
"""Interactive setup wizard for memex-twos-mcp"""

def setup_wizard():
    1. Welcome message
    2. Check Python version (>=3.10)
    3. Create virtual environment (offer to)
    4. Install dependencies
    5. Ask for Twos export file path
    6. Ask where to store database
    7. Run conversion
    8. Run data loader
    9. Test server startup
    10. Generate MCP config for Claude Desktop
    11. Success message with next steps
```

**New File: `scripts/validate_setup.py`:**
- Check all dependencies installed
- Verify database exists and is readable
- Test server can start
- Check MCP config syntax
- Provide fix suggestions for each error

### 4. Add Platform-Specific Support

**Detect OS and adjust:**
- Windows: Use `%USERPROFILE%` paths, `.bat` files
- Mac: Use `~/Library/` for config
- Linux: Use `~/.config/` for config

**New File: `scripts/generate_mcp_config.py`:**
- Detect OS
- Find Claude Desktop config location
- Generate correct config with absolute paths
- Offer to merge with existing config

### 5. Create Demo/Sample Data

**New File: `data/sample/sample_export.md`:**
- Generic Twos export with fake data
- Alice, Bob, Carol as people
- Generic tasks (grocery shopping, doctor appointment, etc.)
- Safe for public viewing
- Small (100-200 tasks) for quick testing

**Document how users can test before using real data**

### 6. Better Error Handling

**Add to all scripts:**
```python
try:
    # operation
except FileNotFoundError as e:
    print(f"❌ Database not found: {e}")
    print(f"💡 Did you run: python scripts/load_to_sqlite.py ?")
    print(f"📖 See: MCP_SETUP.md#troubleshooting")
    sys.exit(1)
```

### 7. License and Governance

**Choose License:**
- MIT (most permissive, recommended)
- Apache 2.0 (patent protection)
- GPL v3 (copyleft)

**Add Files:**
- `LICENSE` - Full license text
- `CONTRIBUTING.md` - How to contribute
- `CODE_OF_CONDUCT.md` - Community standards
- `.github/ISSUE_TEMPLATE/` - Bug/feature templates
- `.github/PULL_REQUEST_TEMPLATE.md`

### 8. Improve README for General Audience

**New Structure:**
```markdown
# Memex Twos MCP

> Transform your Twos app exports into a queryable knowledge base for Claude

[Demo GIF here]

## What is this?

Simple explanation for non-technical users

## Quick Start

1. Export your data from Twos
2. Run: `python scripts/setup_wizard.py`
3. Configure Claude Desktop
4. Ask Claude about your life!

## Features

- Natural language queries
- Full-text search
- Timeline analysis
- Pattern detection

## Requirements

- Python 3.10+
- Twos app export (markdown format)
- Claude Desktop or Claude Code

## Installation

[Step by step for beginners]

## Usage Examples

[Generic examples, no personal data]

## Troubleshooting

[Common issues and solutions]

## Privacy

Your data stays local. Nothing is uploaded.

## Contributing

See CONTRIBUTING.md

## License

MIT License
```

### 9. CI/CD and Quality

**Add `.github/workflows/test.yml`:**
- Run tests on push
- Lint with black/flake8
- Type check with mypy
- Test on multiple Python versions

**Add pre-commit hooks:**
- Prevent committing personal data
- Run formatters
- Check for hardcoded paths

---

## Files to Create/Modify

### New Files
```
LICENSE
CONTRIBUTING.md
CODE_OF_CONDUCT.md
CHANGELOG.md
DEMO_DATA.md
.github/
  ├── ISSUE_TEMPLATE/
  │   ├── bug_report.md
  │   └── feature_request.md
  ├── PULL_REQUEST_TEMPLATE.md
  └── workflows/
      └── test.yml
scripts/
  ├── setup_wizard.py
  ├── validate_setup.py
  ├── generate_mcp_config.py
  └── create_sample_data.py
data/sample/
  └── sample_export.md
src/memex_twos_mcp/
  └── config.py
tests/
  ├── __init__.py
  ├── test_database.py
  ├── test_converter.py
  └── test_server.py
```

### Files to Modify
```
README.md - Completely rewrite for public
MCP_SETUP.md - Generalize paths
src/memex_twos_mcp/server.py - Add config support
src/memex_twos_mcp/database.py - Add config support
src/convert_to_json.py - Add config support
scripts/load_to_sqlite.py - Better error messages
pyproject.toml - Add entry points, more metadata
.gitignore - Add more patterns
```

### Files to Remove (before GitHub)
```
CLAUDE.md - User-specific workflow
PROMPT_FOR_NEXT_LLM.md - Personal context
docs/grooming-reports/2025-12-19-initial-grooming.md - Personal data
```

---

## Risk Assessment

### Privacy Risks
- ⚠️ **HIGH**: Personal names in documentation
- ⚠️ **MEDIUM**: Example outputs with real data
- ✅ **LOW**: Actual data files (already gitignored)

**Mitigation**: Audit all text files for names before push

### Usability Risks
- ⚠️ **HIGH**: Users can't install without Python expertise
- ⚠️ **MEDIUM**: MCP config is confusing
- ⚠️ **MEDIUM**: No validation, hard to debug

**Mitigation**: Setup wizard + better docs

### Support Burden
- ⚠️ **HIGH**: Users will have setup questions
- ⚠️ **MEDIUM**: Platform-specific issues (Win/Mac/Linux)

**Mitigation**: Good docs + issue templates + FAQ

---

## Success Criteria

A successful GitHub release has:

1. ✅ Zero personal information in any committed file
2. ✅ Installation wizard that works for beginners
3. ✅ Works on Windows, Mac, and Linux
4. ✅ Clear documentation with examples
5. ✅ Sample data for testing without real data
6. ✅ Proper license and contribution guidelines
7. ✅ Automated tests passing
8. ✅ At least 3 test users successfully install

---

## Recommended Approach

**Option A: Soft Launch**
1. Create `github-public` branch
2. Implement changes there
3. Test with 2-3 friends
4. Iterate based on feedback
5. Merge to main when stable
6. Create GitHub repo
7. Push

**Option B: New Repo**
1. Create fresh repo: `twos-memex`
2. Cherry-pick commits (excluding personal stuff)
3. Rewrite history to be clean
4. Build public version from scratch there
5. Keep `memex-twos-mcp` private for personal use

**Recommendation**: Option B (cleaner, safer)
