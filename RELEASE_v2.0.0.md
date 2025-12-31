# v2.0.0: Semantic Search, List Understanding, and Time-Based Summaries

New release of the Twos MCP server. This version adds semantic search, list-aware queries, and monthly summaries for exploring personal data.

## Example use case

Here's what semantic search enables:

```
> when did i replace the control board on our "best/nutone" brand kitchen hood

Found it! February 6, 2025. You had two related tasks:
- "Get instructions for Broan control panel" (completed)
- "Install Broan control panel range hood" (completed)

Both dated 2/6/25. Note that your data shows "Broan" as the brand name,
though you asked about "Best/Nutone" - these are related brands under
the same parent company (Broan-NuTone).
```

I spent time searching my email and Twos for this but couldn't find the right keyword. Semantic search returned it in about three seconds.

**Before v2.0:** You needed to remember the exact brand name ("Broan") and search for "broan control" or similar exact keywords.

**With v2.0:** Describe what you're looking for conceptually ("kitchen hood control board"), and semantic search finds related content even if you get the brand name slightly wrong.

## What's new

### Semantic Search
Combines keyword matching with semantic understanding. Queries like "health stuff" find "doctor appointments", "dentist visits", "medication refills" without needing those exact words.

### List Semantics
Fixed the mismatch between "things timestamped on a date" vs "all things on that date's list." Asking "what's on my list for Dec 30?" now returns everything under that day's section header, not just items with timestamps.

New tools: `get_list_by_date()`, `get_list_by_name()`, `search_within_list()`

### TimePacks (Time-Based Rollups)
Precomputed day/week/month summaries for "what happened last week?" queries. Returns compact packs with highlights, tag/person counts, and top keywords instead of processing hundreds of items.

New tools: `get_timepack()`, `list_timepacks()`

### MonthlySummaries (AI-Powered Context)
LLM-generated semantic summaries for monthly exploration. Identifies themes, suggests follow-up questions, and anchors insights to specific thing IDs.

New tools: `get_month_summary()`, `list_month_summaries()`

### ThreadPacks (Activity Indices)
Activity indices for tags and people. Answers "What's active with Alice?" or "Show me recent #work threads" with pre-indexed data. Tracks activity windows, highlights, and keyword trends.

New tools: `list_threads()`, `get_thread()`, `get_thread_highlights()`

### Incremental Ingestion
Content-hash based change detection for faster database updates.

**Before:** Adding 100 new things to 10K database = full rebuild (~3s)
**After:** Adding 100 new things = incremental update (~0.3s)

### Better People Extraction
Integrated spaCy NER for ~90% precision on people names. Reduces false positives like "Set" (verb) or "March" (month) being classified as people.

### Data Grooming
AI-assisted tools for duplicate removal, broken reference fixes, and entity normalization. Runs automatically during setup with detailed change reports.

## Performance notes

- Semantic search: <200ms median (10K things)
- TimePack queries: <50ms
- Storage overhead: ~1.5KB per thing for embeddings (~15MB for 10K things)
- Monthly summaries: ~1-2 minutes to build 12 months (LLM API calls)

## Known limitations

**Testing:** This codebase hasn't been thoroughly tested beyond my personal use cases. It's worked well for my workflows, but expect some rough edges. I'm happy to field bug reports and will do my best to help troubleshoot.

**Date handling:** Claude doesn't have the strongest sense of how dates work. You may occasionally get off-by-one errors, confused date ranges, or timezone-related quirks. Good prompt engineering can mitigate many of these issues - being explicit about date formats and ranges helps.

**Quirks are manageable:** Some AI interpretation quirks can be addressed with solid prompt engineering. For example, specifying "exact date YYYY-MM-DD" or "show me the raw data" tends to produce more reliable results than open-ended queries.

## Breaking changes

None. All new features are opt-in or automatically enabled with graceful fallbacks if dependencies are missing.

## Getting started

```bash
# Fresh install
git clone https://github.com/Zarathustrum/memex-twos-mcp.git
cd memex-twos-mcp
./setup_wizard.sh /path/to/Twos-Export.md

# Upgrade existing database
python scripts/migrate_add_embeddings.py data/processed/twos.db
python scripts/build_all.py --db data/processed/twos.db
```

See [README.md](README.md) for full setup instructions and optional features.

## What it's good for

Same use cases as before (retrospectives, thread reconstruction, pattern analysis), with faster and more flexible queries:

- "What health-related things happened last year?" (semantic search)
- "What's on my list for today?" (list semantics)
- "What happened last month?" (timepacks + monthly summaries)
- "What's active with Alice?" (threadpacks)
- Daily database updates (incremental ingestion)

## Repo

https://github.com/Zarathustrum/memex-twos-mcp

Thanks for the continued interest. If you've been using v1.0 and have feedback, I'd love to hear it.
