#!/usr/bin/env python3
"""
Memex Twos MCP Server

Model Context Protocol server for querying and analyzing personal task data
from Twos app exports stored in SQLite.
"""

import asyncio
import json
import sys
from typing import Any, cast

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool
from pydantic import AnyUrl

from .config import get_config
from .database import TwosDatabase


# Initialize the MCP server with a stable name used by clients.
app = Server("memex-twos-mcp-v2")

# Database instance (initialized in main and used by request handlers).
db: TwosDatabase | None = None


def require_db() -> TwosDatabase:
    """
    Return the initialized database or raise a runtime error.

    Returns:
        The initialized TwosDatabase wrapper.

    Raises:
        RuntimeError: If the server was started without a database.
    """
    if db is None:
        raise RuntimeError("Database not initialized")
    return db


@app.list_resources()
async def list_resources() -> list[Resource]:
    """
    List available resources.

    Resources are read-only data endpoints that MCP clients can request.

    Returns:
        A list of MCP Resource definitions.
    """
    return [
        Resource(
            uri=cast(AnyUrl, "twos://database/stats"),
            name="Database Statistics",
            mimeType="application/json",
            description="Statistics about the Twos thing database",
        ),
        Resource(
            uri=cast(AnyUrl, "twos://database/people"),
            name="People List",
            mimeType="application/json",
            description="List of all people mentioned in things",
        ),
        Resource(
            uri=cast(AnyUrl, "twos://database/tags"),
            name="Tags List",
            mimeType="application/json",
            description="List of all tags used in things",
        ),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """
    Read a specific resource and return a JSON string payload.

    Args:
        uri: The MCP resource URI requested by the client.

    Returns:
        JSON text for the requested resource.
    """
    database = require_db()
    if uri == "twos://database/stats":
        stats = database.get_stats()
        return json.dumps(stats, indent=2)

    elif uri == "twos://database/people":
        people = database.get_people_list()
        return json.dumps(people, indent=2)

    elif uri == "twos://database/tags":
        tags = database.get_tags_list()
        return json.dumps(tags, indent=2)

    else:
        raise ValueError(f"Unknown resource: {uri}")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    List available tools.

    Tools are callable actions a client can invoke with JSON inputs.

    Returns:
        A list of MCP Tool definitions.
    """
    return [
        Tool(
            name="query_things_by_date",
            description=(
                "Query things within a date range. Returns items sorted by date "
                "(most recent first)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional)",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 100)",
                        "default": 100,
                    },
                },
            },
        ),
        Tool(
            name="search_things",
            description=(
                "Full-text search across all thing content with BM25 relevance ranking. "
                "Returns FULL records ordered by relevance (most relevant first) with highlighted snippets. "
                "Each result includes all fields plus relevance_score and snippet. "
                "For large result sets, consider using search_things_preview instead. "
                "Supports FTS5 search operators: AND, OR, NOT, phrase queries with quotes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (supports FTS5 syntax: AND, OR, NOT, \"phrase queries\")",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 50)",
                        "default": 50,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_things_preview",
            description=(
                "Search things and return minimal candidate previews (two-phase retrieval). "
                "Returns only essential fields: id, relevance_score, snippet, timestamp, tags, people, is_completed. "
                "~75% smaller response size than search_things. Use this for initial search, "
                "then call get_things_by_ids or get_thing_by_id to fetch full content for specific items. "
                "Supports FTS5 search operators: AND, OR, NOT, phrase queries with quotes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (supports FTS5 syntax: AND, OR, NOT, \"phrase queries\")",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of candidate previews (default: 50)",
                        "default": 50,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_thing_by_id",
            description=(
                "Get a single thing by ID with full details including all fields, "
                "tags, people, and links. Use after search_things_preview to fetch "
                "complete content for specific items."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "thing_id": {
                        "type": "string",
                        "description": "The thing ID to fetch (e.g., 'task_00001')",
                    },
                },
                "required": ["thing_id"],
            },
        ),
        Tool(
            name="get_things_by_ids",
            description=(
                "Batch fetch multiple things by IDs with full details. "
                "Use after search_things_preview to fetch complete content for "
                "selected candidates. More efficient than multiple get_thing_by_id calls."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "thing_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of thing IDs to fetch (e.g., ['task_00001', 'task_00002'])",
                    },
                },
                "required": ["thing_ids"],
            },
        ),
        Tool(
            name="get_person_things",
            description=(
                "Get all things mentioning a specific person. Returns items sorted by date."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "person_name": {
                        "type": "string",
                        "description": "Name of person to search for (partial matches work)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 100)",
                        "default": 100,
                    },
                },
                "required": ["person_name"],
            },
        ),
        Tool(
            name="get_tag_things",
            description="Get all things with a specific tag. Tags are normalized to lowercase.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tag_name": {
                        "type": "string",
                        "description": "Tag name (e.g., 'siri', 'memory', 'journal')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 100)",
                        "default": 100,
                    },
                },
                "required": ["tag_name"],
            },
        ),
        Tool(
            name="get_things_stats",
            description=(
                "Get comprehensive statistics about the thing database including counts, "
                "date ranges, and entity information."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_things_count",
            description=(
                "Get a minimal health check payload with database path, "
                "total thing count, and load metadata."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_cache_stats",
            description=(
                "Get query cache performance statistics including cache size, "
                "hit rate, and TTL. Useful for monitoring cache effectiveness."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """
    Handle tool calls and return text content results.

    Args:
        name: Tool name requested by the client.
        arguments: Parsed JSON arguments for the tool.

    Returns:
        A list of TextContent blocks containing JSON-serialized results.
    """
    database = require_db()

    if name == "query_things_by_date":
        # Date-based query; database returns rows as dictionaries.
        results = database.query_tasks_by_date(
            start_date=arguments.get("start_date"),
            end_date=arguments.get("end_date"),
            limit=arguments.get("limit", 100),
        )
        return [
            TextContent(type="text", text=json.dumps(results, indent=2, default=str))
        ]

    elif name == "search_things":
        # Full-text search with BM25 ranking and snippet extraction (full records)
        try:
            results = database.search_content(
                query=arguments["query"], limit=arguments.get("limit", 50)
            )
            return [
                TextContent(type="text", text=json.dumps(results, indent=2, default=str))
            ]
        except ValueError as e:
            # Invalid FTS5 query syntax - return helpful error
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": "Invalid search query",
                            "message": str(e),
                            "query": arguments["query"],
                        },
                        indent=2,
                    ),
                )
            ]

    elif name == "search_things_preview":
        # Two-phase retrieval: return minimal candidate previews
        try:
            candidates = database.search_candidates(
                query=arguments["query"], limit=arguments.get("limit", 50)
            )
            return [
                TextContent(
                    type="text", text=json.dumps(candidates, indent=2, default=str)
                )
            ]
        except ValueError as e:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": "Invalid search query",
                            "message": str(e),
                            "query": arguments["query"],
                        },
                        indent=2,
                    ),
                )
            ]

    elif name == "get_thing_by_id":
        # Fetch single thing by ID
        thing = database.get_thing_by_id(arguments["thing_id"])
        if thing is None:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": "Thing not found",
                            "thing_id": arguments["thing_id"],
                        },
                        indent=2,
                    ),
                )
            ]
        return [TextContent(type="text", text=json.dumps(thing, indent=2, default=str))]

    elif name == "get_things_by_ids":
        # Batch fetch things by IDs
        things = database.get_things_by_ids(arguments["thing_ids"])
        return [
            TextContent(type="text", text=json.dumps(things, indent=2, default=str))
        ]

    elif name == "get_person_things":
        # Person lookup uses a SQL LIKE query for partial matches.
        results = database.get_tasks_by_person(
            person_name=arguments["person_name"], limit=arguments.get("limit", 100)
        )
        return [
            TextContent(type="text", text=json.dumps(results, indent=2, default=str))
        ]

    elif name == "get_tag_things":
        # Tag lookup expects normalized lowercase tags.
        results = database.get_tasks_by_tag(
            tag_name=arguments["tag_name"], limit=arguments.get("limit", 100)
        )
        return [
            TextContent(type="text", text=json.dumps(results, indent=2, default=str))
        ]

    elif name == "get_things_stats":
        stats = database.get_stats()
        return [TextContent(type="text", text=json.dumps(stats, indent=2, default=str))]

    elif name == "get_things_count":
        info = database.get_count_info()
        return [TextContent(type="text", text=json.dumps(info, indent=2, default=str))]

    elif name == "get_cache_stats":
        stats = database.get_cache_stats()
        return [TextContent(type="text", text=json.dumps(stats, indent=2, default=str))]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """
    Main entry point for the MCP server.

    Side effects:
        - Reads configuration from disk and environment variables.
        - Opens a persistent SQLite database connection (connection pooling).
        - Starts a JSON-RPC server over stdin/stdout.

    Returns:
        None. This function runs until the process exits.
    """
    global db

    # Load configuration from environment or config file.
    config = get_config()
    errors = config.validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            "\nTIP: Run setup wizard: python scripts/setup_wizard.py",
            file=sys.stderr,
        )
        sys.exit(1)

    db_path = config.db_path

    # Initialize database wrapper with connection pooling and caching
    db = TwosDatabase(db_path)

    print(
        "MCP stdio server ready (with connection pooling and caching). "
        "This process expects JSON-RPC over stdin; use an MCP client to connect.",
        file=sys.stderr,
    )

    try:
        # Run server over stdio; no network sockets are opened here.
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream, write_stream, app.create_initialization_options()
            )
    finally:
        # Clean shutdown: close database connection
        if db:
            db.close()
            print("Database connection closed.", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
