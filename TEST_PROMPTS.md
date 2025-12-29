# Test Prompts for Claude Desktop - v2 Features

**Purpose:** Copy-paste these prompts into Claude Desktop to test Phase 1-3 features.
**Format:** Each test includes the prompt, what to look for, and how to identify success/failure.

---

## PHASE 1: BM25 Ranking + Snippets

### Test P1.1: BM25 Relevance Ordering

**Copy this prompt to Claude Desktop:**
```
Search my things for "doctor appointment" and show me the first 3 results.
For each result, tell me: the thing ID, the relevance score, and the snippet.
```

**What to look for in Claude's response:**
- ✅ **SUCCESS:** Claude shows results with:
  - Thing IDs (e.g., `task_00123`)
  - Relevance scores (numbers, likely between 0-100)
  - Snippets with highlighted matches (might show as bold or with markers)
  - Results ordered by relevance score (highest first, NOT chronological)

- ❌ **FAILURE:**
  - Results ordered by date instead of relevance
  - No relevance scores shown
  - No snippets, just full content
  - Error message about missing fields

**Ask Claude follow-up:**
```
Are these results ordered by relevance score or by date?
```

---

### Test P1.2: Snippet Highlighting

**Copy this prompt to Claude Desktop:**
```
Search for "meeting" in my things and show me just the snippets from the top 5 results.
Do the snippets contain the word "meeting" highlighted in some way?
```

**What to look for:**
- ✅ **SUCCESS:**
  - Snippets contain the search term
  - Highlighting markers present (like `<b>meeting</b>` or `**meeting**`)
  - Snippets are truncated (not full content)
  - Ellipsis (`...`) at beginning/end of snippets

- ❌ **FAILURE:**
  - Full content returned instead of snippets
  - No highlighting on search terms
  - Snippets don't contain the search term

---

## PHASE 2: Two-Phase Retrieval (Preview Tools)

### Test P2.1: Preview Tool Returns Minimal Data

**Copy this prompt to Claude Desktop:**
```
Use the search_things_preview tool to search for "coffee" and show me
what fields are returned for the first result. List all the field names.
```

**What to look for:**
- ✅ **SUCCESS:** Claude lists ONLY these 7 fields:
  - `id`
  - `relevance_score`
  - `snippet`
  - `timestamp`
  - `tags`
  - `people`
  - `is_completed`

- ❌ **FAILURE:**
  - More than 7 fields returned (like `content_raw`, `section_header`, etc.)
  - Error: tool not found
  - Error: invalid response format

**Follow-up prompt:**
```
How many total fields did that preview result contain?
```
(Should answer: 7)

---

### Test P2.2: Full Record Fetch by IDs

**Copy this prompt to Claude Desktop:**
```
First, use search_things_preview to find things about "birthday".
Then use get_things_by_ids to fetch the full details for the first 2 thing IDs.
Tell me how many fields the preview had vs the full record.
```

**What to look for:**
- ✅ **SUCCESS:**
  - Preview returns ~7 fields
  - Full record returns 15 fields
  - Claude successfully chains the two calls
  - Full record includes fields like: `content`, `content_raw`, `section_header`, `line_number`, etc.

- ❌ **FAILURE:**
  - `get_things_by_ids` tool not found
  - Both preview and full return same number of fields
  - Error fetching by IDs

---

### Test P2.3: Token Efficiency Comparison

**Copy this prompt to Claude Desktop:**
```
Search for "project" using search_things_preview (limit 10).
Then search for "project" using search_things (limit 10).
Compare the size of the responses - which one returned more data?
```

**What to look for:**
- ✅ **SUCCESS:**
  - Claude reports `search_things_preview` returned LESS data
  - Roughly 50-75% smaller response
  - Both tools work and return results

- ❌ **FAILURE:**
  - Both return same amount of data
  - One or both tools error out
  - `search_things_preview` not found

---

## PHASE 3: Connection Pooling + Caching

### Test P3.1: Cache Stats Available

**Copy this prompt to Claude Desktop:**
```
Use the get_cache_stats tool and show me what it returns.
```

**What to look for:**
- ✅ **SUCCESS:** Returns cache statistics including:
  - Cache size (number of entries)
  - Hit rate or hit/miss counts
  - TTL information
  - Some numeric metrics

- ❌ **FAILURE:**
  - Tool not found
  - Error calling tool
  - Empty response

---

### Test P3.2: Cache Behavior (Hit/Miss)

**Copy this prompt to Claude Desktop:**
```
1. Get the current cache stats
2. Search for "vacation" using search_things_preview
3. Get cache stats again
4. Search for "vacation" again (same query)
5. Get cache stats a third time

Tell me: did the cache size or hit count change between steps 3 and 5?
```

**What to look for:**
- ✅ **SUCCESS:**
  - Step 3: Cache miss (new entry added, size increases)
  - Step 5: Cache hit (hit count increases, size stays same)
  - Stats show measurable difference

- ❌ **FAILURE:**
  - No cache stats changes
  - Same query counts as cache miss twice
  - Cache stats tool errors

---

### Test P3.3: Query Normalization (Case Insensitivity)

**Copy this prompt to Claude Desktop:**
```
1. Clear your context (or start fresh)
2. Search for "COFFEE" (uppercase)
3. Search for "coffee" (lowercase)
4. Get cache stats

Did the second search hit the cache, or was it treated as a different query?
```

**What to look for:**
- ✅ **SUCCESS:**
  - Second search is a cache hit
  - Cache normalizes "COFFEE" and "coffee" to same key
  - Only 1 cache entry for both queries

- ❌ **FAILURE:**
  - Second search is cache miss
  - Two separate cache entries
  - Case sensitivity not normalized

---

## INTEGRATION: Cross-Feature Tests

### Test I.1: Full Search Flow with Caching

**Copy this prompt to Claude Desktop:**
```
1. Get cache stats (baseline)
2. Use search_things_preview to find "dinner" (limit 5)
3. Get cache stats (should show new entry)
4. Use get_things_by_ids to fetch full details for the first 2 results
5. Search for "dinner" again (same query)
6. Get cache stats (should show cache hit)

Walk me through what happened at each step.
```

**What to look for:**
- ✅ **SUCCESS:**
  - All tools work in sequence
  - Cache correctly stores preview results
  - Second search hits cache
  - Full fetch works with IDs from preview

- ❌ **FAILURE:**
  - Any tool fails
  - Cache doesn't store or retrieve correctly
  - Can't chain preview → full fetch

---

## TEST EXECUTION NOTES

**Before testing:**
- Make sure v2 MCP server is enabled in Claude Desktop
- Disable main branch MCP if both installed
- Start each test category fresh (or ask Claude to start clean context)

**During testing:**
- Copy prompts exactly as written
- Note any error messages verbatim
- If Claude says "tool not found", that's a critical defect
- If results look wrong, ask Claude to show raw JSON response

**After testing:**
- Report which tests passed/failed
- Paste any error messages or unexpected outputs
- I'll diagnose and update DEFECTS.md

---

## QUICK REFERENCE: Available v2 Tools

According to implementation, these tools should exist:
- ✅ `search_things` (legacy, full records)
- ✅ `search_things_preview` (Phase 2, minimal fields)
- ✅ `get_thing_by_id` (single full record)
- ✅ `get_things_by_ids` (batch full records)
- ✅ `get_cache_stats` (Phase 3)
- ⚠️ `query_things_by_date` (exists but broken - DEF-0001)
- ✅ `get_things_stats` (database stats)
- ✅ `get_person_things` (filter by person)
- ✅ `get_tag_things` (filter by tag)

If Claude says any of these don't exist, that's a defect.
