# Defect Log - Memex Twos MCP v2

**Testing Phase:** Phases 1-3 (BM25, Two-Phase Retrieval, Caching)
**Test Date:** 2025-12-28
**Data:** Ungroomed export (first run)

---

## Active Defects

### DEF-0001: query_things_by_date returns full records instead of previews

**Severity:** P0-Critical
**Status:** Root Cause Found
**Phase:** P2-Retrieval (incomplete implementation)
**Reported:** 2025-12-28 11:20 AM

**Repro Steps:**
1. Load v2 MCP in Claude Desktop
2. Ask: "How many completed things did I have in November?"
3. Claude calls `query_things_by_date(start_date=2024-11-01, end_date=2024-11-30)`
4. Observe response handling

**Expected Behavior:**
- Tool returns preview/candidate data (minimal fields: id, timestamp, is_completed, snippet)
- Response size ~8-10KB for 100 results
- Fits in Claude Desktop context window
- Claude can work with data directly in conversation

**Actual Behavior:**
- Tool returns full records with all 15 fields via `SELECT *`
- Response "too large for context"
- Claude Desktop auto-saves to JSON file: `/mnt/user-data/tool_results/memex-twos-v2_query_things_by_date_*.json`
- Claude forced to read and parse file instead of using MCP API
- User experience degraded

**Root Cause:**
- **File:** `src/memex_twos_mcp/server.py:311`
- **Issue:** Calls `database.query_tasks_by_date()` which uses `SELECT *` (line 94 in database.py)
- **Missing:** Phase 2 two-phase retrieval pattern NOT implemented for date queries
- **Exists for:** FTS search has `search_candidates()` with minimal fields ✅
- **Missing for:** Date queries have no preview/candidate equivalent ❌

**Code Evidence:**
```python
# server.py:311 - MCP tool handler
results = database.query_tasks_by_date(...)  # Old v1 method

# database.py:94 - Database method
query = "SELECT * FROM things WHERE 1=1"  # All 15 fields!
```

**Impact:**
- Any date-range query returning 50+ results breaks Claude Desktop context
- Common queries unusable: "What happened in [month]?", "Show me last week's tasks"
- Phase 2 token reduction benefits (75%) not realized for date queries
- User forced into file-parsing workaround (defeats MCP purpose)

**Fix Needed:**
Create preview version of date queries following Phase 2 pattern:
- Option A: New method `query_tasks_by_date_candidates()` with minimal SELECT
- Option B: Add `preview=True` parameter to existing method
- Update MCP tool handler to use preview method by default

**Assigned To:** TBD (coding session)
**Priority:** High - blocks primary use case
**Blocked Tests:**
- I.1 - End-to-end search flow (date queries)
- B.1/B.2 - Performance benchmarks (date queries)
- P2.4 - Token efficiency measurement

---

## Defect Statistics

- **Total:** 1
- **P0-Critical:** 1
- **P1-High:** 0
- **P2-Medium:** 0
- **P3-Low:** 0
- **Status Breakdown:**
  - Open: 0
  - Investigating: 0
  - Root Cause Found: 1
  - Fixed: 0
  - Verified: 0

---

## Testing Notes

- First run without data grooming (expect data quality issues, not bugs)
- Testing against real dataset (size TBD)
- Platform: macOS, Claude Desktop with MCP
- Branch: `claude/memex-twos-v2-upgrade-d3SNe`
