# Cognition Checkpoint Agent (CCA) — Implementation Plan

**Version:** 1.0
**Status:** Draft
**Target:** Claude Code CLI Agent

---

## Executive Summary

This plan describes how to implement the Cognition Checkpoint Agent (CCA) as a standalone Claude Code CLI tool. The CCA is a non-interactive agent that captures developer cognition at context-loss moments, producing structured "checkpoint" artifacts that can be consumed by humans or stateless LLMs.

The implementation prioritizes:
1. **Speed of capture** — v0.1 checkpoints in <30 seconds
2. **Zero friction** — pure function behavior, no dialogue
3. **Durable artifacts** — plain-text Markdown, portable and grep-able
4. **Explicit uncertainty** — fact vs belief separation built into the format

---

## 1. Architecture Overview

### 1.1 System Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                     Developer Environment                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌─────────────────┐     ┌──────────────┐ │
│   │   Trigger   │────▶│  CCA CLI Agent  │────▶│  Checkpoint  │ │
│   │ (git hook,  │     │                 │     │   Artifact   │ │
│   │  alias, CI) │     │ • Parse seed    │     │   (.ckpt.md) │ │
│   └─────────────┘     │ • Infer context │     └──────────────┘ │
│                       │ • Generate tier │              │        │
│                       │ • Write file    │              ▼        │
│                       └─────────────────┘     ┌──────────────┐ │
│                                               │   Storage    │ │
│                                               │ (local, git, │ │
│                                               │  issue API)  │ │
│                                               └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Components

| Component | Responsibility | Location |
|-----------|---------------|----------|
| **Checkpoint Model** | Data structures for v0.1/v1.0 artifacts | `cca/models.py` |
| **Context Collector** | Gathers git state, file context, recent edits | `cca/context.py` |
| **Artifact Generator** | Templates and renders Markdown checkpoints | `cca/generator.py` |
| **Storage Backend** | Writes to filesystem, optionally syncs to issue tracker | `cca/storage.py` |
| **CLI Entry Point** | Argument parsing, orchestration | `cca/cli.py` |

### 1.3 Design Constraints

1. **No network calls by default** — Works offline, issue sync is opt-in
2. **No database required** — Plain files, zero setup
3. **No interactive prompts** — Pure function: seed → artifact
4. **No code execution** — Never runs, modifies, or interprets code
5. **Deterministic output** — Same seed + context → same artifact structure

---

## 2. Artifact Format Specification

### 2.1 v0.1 — Memory Anchor

**Purpose:** Stop immediate context loss
**Creation time:** ~30 seconds
**File pattern:** `.checkpoints/v0.1/<timestamp>-<slug>.ckpt.md`

```markdown
---
cca_version: 0.1
created: 2025-01-25T14:32:00Z
trigger: context-switch
branch: feature/auth-refactor
files_touched:
  - src/auth/login.py
  - tests/test_login.py
status: captured
---

# Memory Anchor: Auth token refresh failing silently

## What I Was Doing
Debugging why auth tokens expire but users see no error.

## Current Theory
The refresh endpoint returns 200 even on invalid tokens. Frontend trusts status code.

## Where I Left Off
- [ ] Added logging to refresh handler
- [ ] Haven't deployed to staging yet

## Unknowns
- Is the 200 response intentional or a bug?
- Does this affect mobile clients?

## Next Step When I Return
Check refresh endpoint implementation in `src/api/auth.py:142`

---
<!-- Auto-captured. Delete freely. Promote to v1.0 if this persists >48h. -->
```

**Required fields:** `What I Was Doing`, `Where I Left Off`
**Optional fields:** `Current Theory`, `Unknowns`, `Next Step`
**Forbidden:** Analysis, recommendations, code suggestions

### 2.2 v1.0 — Context Capsule

**Purpose:** Enable cold start by stateless agent
**Creation time:** 5–10 minutes
**File pattern:** `.checkpoints/v1.0/<id>-<slug>.ckpt.md`

```markdown
---
cca_version: 1.0
id: CKP-2025-0042
created: 2025-01-25T14:32:00Z
promoted_from: v0.1/20250125-143200-auth-token.ckpt.md
trigger: extended-absence
severity: high
branch: feature/auth-refactor
files:
  primary:
    - src/api/auth.py
    - src/auth/login.py
  secondary:
    - tests/test_login.py
    - config/auth_settings.yaml
status: active
---

# Context Capsule: Silent Auth Token Refresh Failure

## Problem Statement
Users with expired tokens are silently logged out without error feedback.
First reported: 2025-01-20.
Frequency: ~5% of sessions after 4+ hours.

## Observable Symptoms
1. Token refresh endpoint returns HTTP 200 with `{"success": false}` body
2. Frontend checks only status code, assumes refresh succeeded
3. Subsequent API calls fail with 401, triggering hard logout

## Current Working Theory
**Confidence: Medium**

The refresh endpoint was designed for internal service-to-service calls where
the caller already handles the `success` field. When the frontend was added,
this contract was not communicated.

## Evidence Collected

### Logs
```
2025-01-24 09:14:22 INFO  auth.refresh token=abc123 result=invalid_signature
2025-01-24 09:14:22 INFO  auth.refresh response_code=200 body={"success":false}
```

### Code Inspection
`src/api/auth.py:142-158` — Refresh handler returns 200 regardless of validation result.

## Negative Knowledge (What Failed)

### Attempt 1: Frontend retry logic
**Tried:** Added exponential backoff on 401 responses
**Result:** Failed — by the time 401 fires, session is already corrupt
**Why:** The problem is upstream of the 401

### Attempt 2: Token expiry extension
**Tried:** Extended token lifetime from 4h to 24h
**Result:** Masked symptom, didn't fix root cause
**Why:** Tokens can become invalid for reasons other than expiry (key rotation, revocation)

### AI Suggestion (Claude): Check token signature validation
**Tried:** Reviewed signature validation logic
**Result:** Signature validation is correct
**Why:** The issue is response handling, not validation

## Hypotheses Not Yet Tested
1. [ ] Return 401 from refresh endpoint on validation failure
2. [ ] Add `success` field check to frontend token refresh
3. [ ] Emit frontend analytics event on refresh failure

## Acceptance Criteria
- [ ] User sees explicit error when token refresh fails
- [ ] Error message includes retry action (re-login button)
- [ ] No silent logouts in staging for 24h test period
- [ ] Existing mobile clients handle new response gracefully (backward compat)

## Risks & Constraints
- **Breaking change risk:** Mobile apps may expect 200 response
- **Data loss risk:** None — auth state only, no user data involved
- **Time constraint:** Should fix before Jan 31 release

## Resumption Context for Stateless Agent

When resuming work on this checkpoint, the agent should:

1. **Read first:** `src/api/auth.py:142-158` (refresh handler)
2. **Understand:** The 200-with-failure-body pattern is intentional but problematic
3. **Do NOT retry:** Token lifetime extension (Attempt 2) — already proven ineffective
4. **Test hypothesis:** Start with #2 (frontend success field check) — lowest risk

---
<!-- Promoted from v0.1 on 2025-01-25. Review for graduation or archive after resolution. -->
```

**Required sections:** Problem Statement, Negative Knowledge, Acceptance Criteria
**Structural invariant:** Every claim must be traceable to evidence or marked as theory

---

## 3. CLI Interface Design

### 3.1 Command Structure

```bash
# Primary command: capture checkpoint
cca capture [OPTIONS] [SEED_TEXT]

# Subcommands
cca list                    # List checkpoints by status
cca show <id>               # Display checkpoint content
cca promote <v0.1-path>     # Upgrade v0.1 → v1.0
cca archive <id>            # Mark checkpoint resolved
cca delete <id>             # Remove checkpoint (no guilt)
```

### 3.2 Capture Command Options

```bash
cca capture "Auth tokens failing silently"

Options:
  --tier, -t [0.1|1.0]     Force specific tier (default: auto-detect)
  --trigger, -g TEXT        Capture trigger (context-switch, interruption, bug, etc.)
  --severity, -s [low|medium|high|critical]
  --files, -f PATH          Associate specific files (repeatable)
  --branch TEXT             Override branch detection
  --stdout                  Write to stdout instead of file
  --dry-run                 Show what would be captured without writing
```

### 3.3 Environment Variables

```bash
CCA_CHECKPOINT_DIR=.checkpoints    # Override checkpoint storage location
CCA_DEFAULT_TIER=0.1               # Default tier when auto-detect is ambiguous
CCA_GIT_CONTEXT=true               # Include git state in capture
CCA_ISSUE_SYNC=false               # Disable issue tracker sync by default
```

### 3.4 Example Usage Flows

**Quick capture during context switch:**
```bash
# Alias: alias ckpt='cca capture'
ckpt "debugging auth refresh, check api/auth.py:142"
# → Creates .checkpoints/v0.1/20250125-143200-debugging-auth-refresh.ckpt.md
```

**Promote after returning to stale work:**
```bash
cca list --stale
# Shows v0.1 checkpoints older than 48h

cca promote .checkpoints/v0.1/20250123-091500-auth-refresh.ckpt.md
# Opens template with v0.1 content pre-filled, user completes v1.0 fields
```

**Archive after resolution:**
```bash
cca archive CKP-2025-0042 --resolution "Fixed in commit abc123"
# Moves to .checkpoints/archived/ with resolution metadata
```

---

## 4. Context Collection Strategy

### 4.1 Automatic Context (No User Input Required)

| Source | Data Collected | Use |
|--------|---------------|-----|
| Git | Current branch, recent commits (5), uncommitted files | Branch context |
| Git | Files modified in last 30 min | `files_touched` field |
| Timestamp | ISO 8601 UTC | `created` field |
| Working directory | Project root detection | File path resolution |

### 4.2 Inferred Context (Heuristic)

| Signal | Inference |
|--------|-----------|
| Seed contains "bug", "error", "failing" | `trigger: bug-report` |
| Seed contains "security", "auth", "credential" | `severity: high` |
| >3 files touched in last hour | `trigger: active-development` |
| No commits in last 4h despite file changes | `trigger: stuck` |

### 4.3 Explicit Context (User Provides)

- Seed text (required for meaningful checkpoint)
- Severity override
- Associated files beyond auto-detected
- Negative knowledge entries (for v1.0)

---

## 5. Storage Strategy

### 5.1 Local Filesystem (Primary)

```
project-root/
├── .checkpoints/
│   ├── v0.1/
│   │   ├── 20250125-143200-auth-refresh.ckpt.md
│   │   └── 20250125-160000-api-rate-limit.ckpt.md
│   ├── v1.0/
│   │   └── CKP-2025-0042-auth-token-failure.ckpt.md
│   ├── archived/
│   │   └── CKP-2025-0038-resolved.ckpt.md
│   └── .cca-index.json    # Optional: fast lookup index
├── .gitignore             # Contains: .checkpoints/v0.1/*
└── ...
```

**Gitignore strategy:**
- v0.1 checkpoints: Ignored (ephemeral, personal)
- v1.0 checkpoints: Committed (team-shareable, durable)
- Archived: Committed (audit trail)

### 5.2 Issue Tracker Sync (Optional)

```yaml
# .cca/config.yaml
sync:
  enabled: false
  provider: gitea  # gitea | github | gitlab | linear
  endpoint: https://git.example.com/api/v1
  project: my-project

  # Only sync v1.0 checkpoints with severity >= medium
  filter:
    min_tier: 1.0
    min_severity: medium

  # Map checkpoint fields to issue fields
  mapping:
    title: "Problem Statement (first line)"
    body: "Full checkpoint content"
    labels: ["cca-checkpoint", "{{severity}}"]
```

**Sync behavior:**
- One-way push: CCA → Issue tracker
- Creates issue on v1.0 promotion
- Updates issue on checkpoint edit
- Does NOT pull changes from issue tracker back

---

## 6. Implementation Phases

### Phase 1: Core Capture (MVP)

**Goal:** Working v0.1 capture with minimal friction
**Deliverables:**

1. `cca capture` command with seed text input
2. v0.1 artifact generation with auto-context
3. Local filesystem storage
4. Basic git context collection

**Acceptance criteria:**
- `cca capture "some text"` completes in <5 seconds
- Produces valid Markdown checkpoint
- Works without network, without config, without dependencies beyond Python stdlib

**Implementation tasks:**

```
[ ] Define Checkpoint dataclass (v0.1 fields)
[ ] Implement git context collector (branch, recent commits, modified files)
[ ] Create v0.1 Markdown template
[ ] Build CLI entry point with argparse
[ ] Write to .checkpoints/v0.1/ directory
[ ] Add timestamp and slug generation
[ ] Create installation script / package
```

### Phase 2: Rich Context & v1.0

**Goal:** Full v1.0 checkpoints with negative knowledge tracking
**Deliverables:**

1. v1.0 artifact format and templates
2. `cca promote` command
3. Negative knowledge structured capture
4. Acceptance criteria checklist format

**Acceptance criteria:**
- Promotion flow takes v0.1 → v1.0 with guided expansion
- Negative knowledge section has structured format
- Acceptance criteria are parseable checkboxes

**Implementation tasks:**

```
[ ] Define Checkpoint dataclass (v1.0 fields)
[ ] Create v1.0 Markdown template with all sections
[ ] Implement promote command (copy v0.1, add v1.0 sections)
[ ] Add severity classification heuristics
[ ] Build negative knowledge entry format
[ ] Implement checkpoint ID generation (CKP-YYYY-NNNN)
```

### Phase 3: Lifecycle Management

**Goal:** Complete checkpoint lifecycle (list, archive, delete)
**Deliverables:**

1. `cca list` with filtering (tier, status, age)
2. `cca archive` with resolution capture
3. `cca delete` without guilt
4. Stale checkpoint detection (>48h v0.1)

**Implementation tasks:**

```
[ ] Build checkpoint index (.cca-index.json)
[ ] Implement list command with filters
[ ] Implement archive command with resolution metadata
[ ] Implement delete command
[ ] Add stale detection (configurable threshold)
[ ] Create status field in frontmatter (active, stale, archived)
```

### Phase 4: Integration & Automation

**Goal:** Frictionless integration with developer workflow
**Deliverables:**

1. Git hook templates (pre-commit checkpoint reminder)
2. Shell aliases for common patterns
3. Issue tracker sync (Gitea priority)
4. Claude Code integration hooks

**Implementation tasks:**

```
[ ] Create git hook templates
[ ] Document shell alias patterns
[ ] Implement issue sync provider interface
[ ] Build Gitea sync provider
[ ] Create Claude Code hook configuration
[ ] Add CI/CD checkpoint validation
```

### Phase 5: LLM Consumption Optimization

**Goal:** Checkpoints optimized for stateless LLM cold-start
**Deliverables:**

1. "Resumption Context" section with explicit agent instructions
2. Negative knowledge formatted for LLM prompts
3. Checkpoint → prompt template conversion
4. Validation that LLM can parse and act on checkpoints

**Implementation tasks:**

```
[ ] Design resumption context format
[ ] Create LLM prompt template for checkpoint consumption
[ ] Build checkpoint → prompt converter
[ ] Test with Claude Code: can it resume from checkpoint alone?
[ ] Add file path annotations for agent navigation
```

---

## 7. Technical Specifications

### 7.1 Dependencies

**Required (stdlib only for MVP):**
- Python 3.10+
- `argparse` (CLI)
- `datetime` (timestamps)
- `pathlib` (file operations)
- `subprocess` (git commands)
- `json` (index files)
- `re` (slug generation)

**Optional (for enhanced features):**
- `pyyaml` — Config file parsing
- `rich` — Enhanced CLI output
- `httpx` — Issue tracker sync
- `gitpython` — Richer git context (avoid for MVP, subprocess is simpler)

### 7.2 File Format Details

**Frontmatter:** YAML between `---` delimiters
**Body:** GitHub-flavored Markdown
**Encoding:** UTF-8, LF line endings
**Max size:** Soft limit 10KB (warn if exceeded)

### 7.3 ID Generation

**v0.1:** `<timestamp>-<slug>`
- Timestamp: `YYYYMMDD-HHMMSS`
- Slug: First 40 chars of seed, slugified (lowercase, hyphens, no special chars)

**v1.0:** `CKP-<YYYY>-<NNNN>`
- Year: Four-digit year
- Sequence: Zero-padded counter, per year, from `.cca-index.json`

### 7.4 Error Handling

| Scenario | Behavior |
|----------|----------|
| No seed text provided | Use branch name + recent commit message |
| Git not available | Skip git context, warn user |
| Checkpoint dir not writable | Write to stdout, error exit |
| Malformed existing checkpoint | Skip during list, warn user |
| Duplicate ID detected | Append suffix (-2, -3, etc.) |

---

## 8. Integration Patterns

### 8.1 Git Hooks

**Pre-commit reminder (optional):**
```bash
#!/bin/sh
# .git/hooks/pre-commit

# Check for stale v0.1 checkpoints
stale=$(cca list --tier 0.1 --stale --quiet --count)
if [ "$stale" -gt 0 ]; then
    echo "⚠️  You have $stale stale checkpoints. Consider promoting or deleting."
    cca list --tier 0.1 --stale
fi
```

**Post-checkout capture prompt:**
```bash
#!/bin/sh
# .git/hooks/post-checkout

# Prompt for checkpoint on branch switch
prev_branch=$1
new_branch=$2

if [ "$prev_branch" != "$new_branch" ]; then
    echo "Switched from $prev_branch to $new_branch"
    echo "Create context checkpoint? [y/N]"
    read -r response
    if [ "$response" = "y" ]; then
        cca capture --trigger context-switch "Switched from $prev_branch"
    fi
fi
```

### 8.2 Shell Aliases

```bash
# ~/.bashrc or ~/.zshrc

# Quick capture
alias ckpt='cca capture'
alias ckptb='cca capture --trigger bug'
alias ckptu='cca capture --trigger stuck'

# Review
alias ckls='cca list'
alias ckstale='cca list --tier 0.1 --stale'

# Cleanup
alias ckclean='cca list --tier 0.1 --stale --format=path | xargs rm -i'
```

### 8.3 Claude Code Hooks

```yaml
# .claude/hooks.yaml

on_session_end:
  - name: checkpoint-prompt
    when: uncommitted_changes
    action: |
      echo "You have uncommitted changes. Create checkpoint?"
      # Optionally auto-capture with last conversation summary

on_error_loop:
  - name: negative-knowledge-capture
    when: same_error_3x
    action: |
      cca capture --trigger ai-failure "AI suggested fix failed: {{last_suggestion}}"
```

### 8.4 MCP Integration (Future)

Checkpoints could be exposed as MCP resources:

```python
# MCP resource definition
@server.resource("cca://checkpoints/active")
async def get_active_checkpoints():
    """Return all active v1.0 checkpoints for context injection."""
    return load_checkpoints(status="active", tier="1.0")

@server.resource("cca://checkpoints/{id}")
async def get_checkpoint(id: str):
    """Return specific checkpoint content."""
    return load_checkpoint(id)
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# tests/test_capture.py

def test_v01_capture_minimal():
    """v0.1 capture with just seed text."""
    result = capture("debugging auth")
    assert result.tier == "0.1"
    assert "debugging auth" in result.content
    assert result.file_path.exists()

def test_v01_capture_with_git_context():
    """v0.1 capture includes git branch and files."""
    # In a git repo with changes
    result = capture("testing git context")
    assert result.metadata["branch"] is not None
    assert len(result.metadata["files_touched"]) > 0

def test_capture_no_network():
    """Capture works completely offline."""
    with network_disabled():
        result = capture("offline test")
        assert result.file_path.exists()
```

### 9.2 Integration Tests

```python
# tests/test_lifecycle.py

def test_full_lifecycle():
    """v0.1 → v1.0 → archive flow."""
    # Create v0.1
    v01 = capture("test issue")

    # Promote to v1.0
    v10 = promote(v01.file_path,
                  negative_knowledge=["tried X, failed because Y"],
                  acceptance_criteria=["thing works"])

    # Archive
    archived = archive(v10.id, resolution="fixed in abc123")

    assert archived.status == "archived"
    assert "fixed in abc123" in archived.resolution
```

### 9.3 LLM Consumption Tests

```python
# tests/test_llm_consumption.py

def test_checkpoint_parseable_by_llm():
    """Checkpoint can be parsed and acted upon by Claude."""
    checkpoint = create_sample_v10_checkpoint()

    prompt = f"""
    You are resuming work on a problem. Here is the checkpoint:

    {checkpoint.content}

    What is the first thing you should do?
    """

    response = call_claude(prompt)

    # Should reference the "Next Step" or "Resumption Context"
    assert "auth.py" in response or "refresh handler" in response

    # Should NOT suggest already-failed approaches
    assert "token lifetime" not in response.lower()  # Negative knowledge
```

---

## 10. Success Metrics

### 10.1 Usage Metrics (If Telemetry Enabled)

| Metric | Target | Failure Signal |
|--------|--------|----------------|
| v0.1 captures per day | 3-10 | <1 = too much friction |
| v0.1 deletion rate | 60-80% | <40% = guilt ledger forming |
| v0.1 → v1.0 promotion rate | 10-20% | >50% = tier boundary wrong |
| Time to v0.1 capture | <30 sec | >60 sec = too slow |
| Time to v1.0 promotion | 5-10 min | >20 min = too complex |

### 10.2 Qualitative Signals

**Healthy:**
- User creates checkpoints without prompting
- v0.1 checkpoints are deleted freely
- Cold starts feel faster after using checkpoints
- AI agents avoid repeating documented dead ends

**Unhealthy:**
- User avoids capture because it "takes too long"
- Checkpoint list becomes a guilt backlog
- Checkpoints are overly verbose or speculative
- User edits checkpoints for "presentation" rather than accuracy

---

## 11. Graduation Protocol (v0.1 → v1.0)

### 11.1 Automatic Graduation Signals

The system should suggest (not force) promotion when:

| Signal | Weight |
|--------|--------|
| v0.1 checkpoint age > 48h | High |
| Same files touched again after checkpoint | Medium |
| Checkpoint mentioned in commit message | Medium |
| AI agent asked about same topic | High |
| Severity keyword in original seed | Medium |

### 11.2 Promotion Invariants

A v1.0 checkpoint **must** have:

1. **Problem Statement** — One sentence describing the issue
2. **At least one evidence entry** — Logs, code refs, or observations
3. **At least one acceptance criterion** — Verifiable checkbox
4. **Negative knowledge section** — Even if empty, explicitly present

A v1.0 checkpoint **should** have:

5. Working theory with confidence level
6. Hypotheses not yet tested
7. Resumption context for stateless agents

### 11.3 Promotion Flow

```
1. User runs: cca promote <v0.1-path>

2. System generates v1.0 template pre-filled with:
   - v0.1 content in "What I Was Doing" → "Problem Statement"
   - v0.1 "Current Theory" → "Working Theory"
   - v0.1 "Unknowns" → "Hypotheses Not Yet Tested"
   - Empty "Negative Knowledge" section with example format
   - Empty "Acceptance Criteria" with example format

3. User completes required sections (editor opens)

4. System validates invariants before saving

5. v0.1 file is deleted (or moved to archive)

6. v1.0 file is created with new ID
```

---

## 12. Open Questions

### Resolved During Implementation

- [ ] Should v0.1 checkpoints support multiple "threads" in same file?
- [ ] How to handle checkpoint conflicts when merging branches?
- [ ] Should there be a v0.5 tier for "quick but structured"?

### Deferred to Future Versions

- Multi-user checkpoint sharing and deduplication
- Checkpoint search / semantic retrieval
- Automatic checkpoint generation from git diff + commit message
- Integration with existing knowledge bases (Notion, Obsidian, etc.)

---

## 13. File Structure (Final)

```
cca/
├── __init__.py
├── cli.py              # Entry point, argument parsing
├── models.py           # Checkpoint dataclasses
├── context.py          # Git and environment context collection
├── generator.py        # Markdown template rendering
├── storage.py          # Filesystem operations
├── lifecycle.py        # List, promote, archive, delete
├── sync/
│   ├── __init__.py
│   ├── base.py         # Sync provider interface
│   └── gitea.py        # Gitea sync implementation
└── templates/
    ├── v0.1.md.j2      # v0.1 template
    └── v1.0.md.j2      # v1.0 template

tests/
├── test_capture.py
├── test_lifecycle.py
├── test_context.py
└── test_llm_consumption.py

docs/
├── cca-spec.md         # Original specification
├── cca-implementation-plan.md  # This document
└── cca-user-guide.md   # Usage documentation
```

---

## Appendix A: Sample Prompts for Claude Code Implementation

### Prompt 1: Core Capture (Phase 1)

```
Implement the core CCA capture functionality:

1. Create a Python CLI tool that accepts seed text and creates a v0.1 checkpoint
2. Auto-collect git context (branch, recent commits, modified files)
3. Generate Markdown with YAML frontmatter
4. Write to .checkpoints/v0.1/ directory
5. No external dependencies beyond Python stdlib

The checkpoint should include:
- Timestamp (ISO 8601)
- Trigger type
- Branch name
- Files touched
- The seed text in "What I Was Doing" section
- Placeholder sections for Theory, Unknowns, Next Step

Test by running: python -m cca.cli capture "test checkpoint"
```

### Prompt 2: v1.0 Promotion (Phase 2)

```
Add v1.0 checkpoint support and promotion:

1. Define the v1.0 format with all sections from the spec
2. Implement `cca promote <v0.1-path>` command
3. Pre-fill v1.0 template from v0.1 content
4. Validate required sections before saving
5. Generate sequential ID (CKP-YYYY-NNNN)
6. Delete original v0.1 after successful promotion

The v1.0 must include:
- Problem Statement
- Observable Symptoms
- Working Theory (with confidence)
- Evidence Collected
- Negative Knowledge section (structured)
- Acceptance Criteria (checkboxes)
- Resumption Context for LLMs
```

### Prompt 3: Lifecycle Commands (Phase 3)

```
Implement checkpoint lifecycle management:

1. `cca list` - Show all checkpoints with filters:
   --tier [0.1|1.0]
   --status [active|stale|archived]
   --since DATE
   --format [table|json|paths]

2. `cca archive <id>` - Archive with resolution:
   --resolution TEXT (required)
   Move to .checkpoints/archived/
   Update status in frontmatter

3. `cca delete <id>` - Remove checkpoint:
   No confirmation needed (design principle: no guilt)

4. Stale detection:
   v0.1 older than 48h = stale
   Flag in list output
```

---

## Appendix B: Anti-Patterns to Avoid

| Anti-Pattern | Why It's Bad | What to Do Instead |
|--------------|--------------|-------------------|
| Verbose templates | Discourages capture | Minimal required fields |
| Required severity | Every decision is friction | Auto-infer, allow override |
| Blocking validation | Interrupts flow | Warn but allow save |
| Auto-promotion | User loses control | Suggest only |
| Syncing v0.1 to issues | Creates public guilt ledger | Only sync v1.0 |
| Pretty formatting | Wastes cognitive load | Plain, grep-able text |
| Checkpoint editing UI | Overkill for note-taking | Plain Markdown, any editor |

---

*End of Implementation Plan*
