import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastmcp import Client

from engine.src.sdk_models import IndexOperationSnapshot
from quickcontext_mcp.server import _now_iso, mcp


def _run(coro):
    return asyncio.run(coro)


class MCPServerTests(unittest.TestCase):
    def test_list_tools_exposes_expected_surface(self) -> None:
        async def _test() -> None:
            async with Client(mcp) as client:
                tools = await client.list_tools()
            tool_map = {tool.name: tool for tool in tools}
            self.assertIn("project_info", tool_map)
            self.assertIn("index", tool_map)
            self.assertIn("search", tool_map)
            self.assertIn("grep", tool_map)
            self.assertIn("symbol_lookup", tool_map)
            self.assertTrue(tool_map["search"].annotations.readOnlyHint)
            self.assertFalse(tool_map["index"].annotations.readOnlyHint)

        _run(_test())

    def test_search_tool_returns_structured_payload(self) -> None:
        fake_response = {
            "query": "how does indexing work",
            "path": "C:/repo",
            "project_name": "repo",
            "mode_requested": "auto",
            "mode_used": "symbol",
            "indexed": True,
            "results": [
                {
                    "kind": "semantic",
                    "file_path": "C:/repo/main.py",
                    "language": "python",
                    "symbol_name": "index_directory",
                    "symbol_kind": "function",
                    "line_start": 10,
                    "line_end": 30,
                    "score": 0.99,
                    "signature": "def index_directory(path: str) -> None",
                    "description": "Indexes a directory.",
                    "snippet": "def index_directory(path): ...",
                    "path_context": "repo / main.py",
                    "matched_terms": [],
                }
            ],
            "related_files": [],
            "related_callers": [],
            "warnings": [],
            "error": None,
        }

        async def _test() -> None:
            with patch("quickcontext_mcp.server._search_impl", return_value=fake_response):
                async with Client(mcp) as client:
                    result = await client.call_tool(
                        "search",
                        {"query": "how does indexing work", "path": "C:/repo"},
                    )
            payload = result.data
            self.assertEqual(payload.project_name, "repo")
            self.assertEqual(payload.mode_used, "symbol")
            self.assertEqual(payload.results[0].symbol_name, "index_directory")

        _run(_test())

    def test_project_info_survives_missing_qdrant(self) -> None:
        fake_response = {
            "path": "C:/repo",
            "project_name": "repo",
            "indexed": False,
            "qdrant_enabled": True,
            "qdrant_available": False,
            "parser_connected": True,
            "cache_entries": 0,
            "cache_disk_bytes": 0,
            "cache_path": "C:/repo/.quickcontext/file_cache.json",
            "collection": {
                "project_name": "repo",
                "indexed": False,
                "real_collection": None,
                "points_count": None,
                "indexed_vectors_count": None,
                "status": None,
            },
            "active_index_runs": [],
            "folders": [],
            "search_modes": ["auto", "text", "semantic", "bundle"],
        }

        async def _test() -> None:
            with patch("quickcontext_mcp.server._project_info_impl", return_value=fake_response):
                async with Client(mcp) as client:
                    result = await client.call_tool("project_info", {"path": "C:/repo"})
            payload = result.data
            self.assertEqual(payload.project_name, "repo")
            self.assertFalse(payload.indexed)

        _run(_test())

    def test_index_status_reads_registry_records(self) -> None:
        record = IndexOperationSnapshot(
            operation_id="run123",
            kind="index",
            path="C:/repo",
            project_name="demo",
            status="running",
            current_stage="upsert",
            updated_at=_now_iso(),
            created_at=_now_iso(),
            started_at=_now_iso(),
            message="Indexing directory",
        )

        async def _test() -> None:
            with patch("quickcontext_mcp.server.QuickContext.get_operation_status", return_value=record):
                async with Client(mcp) as client:
                    result = await client.call_tool("index_status", {"run_id": "run123"})
            payload = result.data
            self.assertEqual(len(payload.runs), 1)
            self.assertEqual(payload.runs[0].status, "running")
            self.assertEqual(payload.runs[0].run_id, "run123")
            self.assertEqual(payload.runs[0].current_stage, "upsert")

        _run(_test())

    def test_index_tool_attaches_to_existing_run(self) -> None:
        existing = IndexOperationSnapshot(
            operation_id="run999",
            kind="index",
            path="C:/repo",
            project_name="repo",
            status="running",
            current_stage="extract",
            updated_at=_now_iso(),
            created_at=_now_iso(),
            started_at=_now_iso(),
            message="Already running",
        )

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            (project_root / "package.json").write_text("{}", encoding="utf-8")

            async def _test() -> None:
                with patch("quickcontext_mcp.server.QuickContext.start_index_directory", return_value=existing):
                    async with Client(mcp) as client:
                        result = await client.call_tool("index", {"path": str(project_root)})
                payload = result.data
                self.assertEqual(payload.status, "running")
                self.assertTrue(payload.attached_to_existing)
                self.assertEqual(payload.duplicate_of, "run999")

            _run(_test())
