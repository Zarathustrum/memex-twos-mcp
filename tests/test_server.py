"""Tests for MCP server definitions."""

from __future__ import annotations

import asyncio

from memex_twos_mcp import server


def test_list_tools() -> None:
    """
    Server exposes the expected tools.

    This calls the async list_tools handler without starting the server.

    Returns:
        None.
    """
    tools = asyncio.run(server.list_tools())
    tool_names = {tool.name for tool in tools}

    assert "query_things_by_date" in tool_names
    assert "search_things" in tool_names
    assert "get_person_things" in tool_names
    assert "get_tag_things" in tool_names
    assert "get_things_stats" in tool_names
