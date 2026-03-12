import importlib
import os
import tempfile
import time
import unittest
from unittest import mock
from dataclasses import dataclass
from pathlib import Path
import builtins

from engine.__main__ import _find_command
from engine.sdk import QuickContext
from engine.src.chunker import ChunkBuilder, CodeChunk
from engine.src.config import EngineConfig, QdrantConfig
from engine.src.dedup import deduplicate_chunks
from engine.src.describer import build_fallback_description
from engine.src.filecache import FileSignatureCache
from engine.src.cli import _optimize_search_config
from engine.src.parsing import ImportEdge
from engine.src.qdrant import QdrantConnection
from engine.src.searcher import CodeSearcher, LIGHT_RESULT_PAYLOAD_FIELDS, SearchResult


@dataclass(frozen=True)
class _Symbol:
    name: str
    kind: str
    language: str
    file_path: str
    line_start: int
    line_end: int
    byte_start: int
    byte_end: int
    source: str
    signature: str | None = None
    docstring: str | None = None
    params: str | None = None
    return_type: str | None = None
    parent: str | None = None
    visibility: str | None = None
    role: str | None = None


@dataclass(frozen=True)
class _Extraction:
    file_path: str
    language: str
    symbols: list
    file_hash: str | None = "file-hash"


class _FakeProvider:
    def __init__(self, model: str, dimension: int):
        self.model = model
        self.dimension = dimension

    def embed_single(self, text: str):
        import numpy as np
        return np.array([0.1, 0.2], dtype=np.float32)


class _FakePoint:
    def __init__(self, score: float, payload: dict, point_id: str = "point-1"):
        self.score = score
        self.payload = payload
        self.id = point_id


class _FakeQueryResponse:
    def __init__(self, points: list[_FakePoint]):
        self.points = points


class _FakeClient:
    def __init__(self, points: list[_FakePoint], records: list | None = None):
        self._points = points
        self._records = records or []
        self.last_query_with_payload = None
        self.last_retrieve_with_payload = None
        self.retrieve_calls = 0

    def query_points(self, **kwargs):
        self.last_query_with_payload = kwargs.get("with_payload")
        return _FakeQueryResponse(self._points)

    def query_batch_points(self, collection_name, requests):
        return [_FakeQueryResponse(self._points) for _ in requests]

    def retrieve(self, **kwargs):
        self.last_retrieve_with_payload = kwargs.get("with_payload")
        self.retrieve_calls += 1
        return self._records


class _BlockedImport:
    def __init__(self, blocked_names: set[str]):
        self._blocked_names = blocked_names
        self._orig_import = builtins.__import__

    def __enter__(self):
        def _guard(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split(".")[0]
            if root in self._blocked_names:
                raise AssertionError(f"blocked import attempted: {name}")
            return self._orig_import(name, globals, locals, fromlist, level)

        builtins.__import__ = _guard
        return self

    def __exit__(self, exc_type, exc, tb):
        builtins.__import__ = self._orig_import


class RegressionTests(unittest.TestCase):
    def test_find_command_skips_global_config_option(self) -> None:
        command = _find_command(["--config", "quickcontext.json", "symbol-lookup", "CollectionManager"])
        self.assertEqual(command, "symbol-lookup")

    def test_chunk_ids_differ_for_overloaded_like_symbols(self) -> None:
        builder = ChunkBuilder()
        file_path = str(Path("sample.py").resolve())

        s1 = _Symbol(
            name="connect",
            kind="method",
            language="python",
            file_path=file_path,
            line_start=1,
            line_end=2,
            byte_start=10,
            byte_end=40,
            source="def connect(client):\n    return client",
            signature="connect(client)",
            parent="ApiClient",
        )
        s2 = _Symbol(
            name="connect",
            kind="method",
            language="python",
            file_path=file_path,
            line_start=4,
            line_end=5,
            byte_start=50,
            byte_end=90,
            source="def connect(timeout):\n    return timeout",
            signature="connect(timeout)",
            parent="ApiClient",
        )

        chunks = builder.build_chunks([_Extraction(file_path=file_path, language="python", symbols=[s1, s2])])
        self.assertEqual(len(chunks), 2)
        self.assertNotEqual(chunks[0].chunk_id, chunks[1].chunk_id)

    def test_dedup_is_context_aware(self) -> None:
        c1 = CodeChunk(
            chunk_id="1",
            source="return value",
            language="python",
            file_path="a.py",
            symbol_name="foo",
            symbol_kind="function",
            line_start=1,
            line_end=1,
            byte_start=0,
            byte_end=12,
            signature="foo()",
            docstring=None,
            parent=None,
            visibility=None,
            role=None,
            file_hash="a",
        )
        c2 = CodeChunk(
            chunk_id="2",
            source="return value",
            language="python",
            file_path="b.py",
            symbol_name="bar",
            symbol_kind="function",
            line_start=1,
            line_end=1,
            byte_start=0,
            byte_end=12,
            signature="bar()",
            docstring=None,
            parent=None,
            visibility=None,
            role=None,
            file_hash="b",
        )
        c3 = CodeChunk(
            chunk_id="3",
            source="return value",
            language="python",
            file_path="c.py",
            symbol_name="foo",
            symbol_kind="function",
            line_start=1,
            line_end=1,
            byte_start=0,
            byte_end=12,
            signature="foo()",
            docstring=None,
            parent=None,
            visibility=None,
            role=None,
            file_hash="c",
        )

        result = deduplicate_chunks([c1, c2, c3])
        self.assertEqual(result.unique_count, 2)

    def test_file_signature_cache_uses_external_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_path = root / "sample.py"
            file_path.write_text("print('x')\n", encoding="utf-8")
            stat = file_path.stat()

            cache = FileSignatureCache(root)
            cache.put(file_path, int(stat.st_mtime), stat.st_size, "hash-1")

            cached = cache.is_unchanged_from_metadata(
                file_path,
                file_size=stat.st_size,
                file_mtime=int(stat.st_mtime),
            )
            self.assertEqual(cached, "hash-1")

            changed = cache.is_unchanged_from_metadata(
                file_path,
                file_size=stat.st_size + 1,
                file_mtime=int(stat.st_mtime),
            )
            self.assertIsNone(changed)

    def test_searcher_can_share_query_embedding_when_providers_match(self) -> None:
        provider = _FakeProvider("shared-model", 2)
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=provider,
            desc_provider=provider,
        )
        self.assertTrue(searcher._can_share_query_embedding())

    def test_search_hybrid_uses_cached_vectors_without_thread_pool(self) -> None:
        payload = {
            "file_path": str(Path("engine/src/searcher.py").resolve()),
            "symbol_name": "_embed_query_with_provider_cached",
            "symbol_kind": "function",
            "line_start": 1,
            "line_end": 5,
            "description": "Cache query embedding vectors.",
            "keywords": ["query", "embedding", "cache"],
            "path_context": "engine / src / searcher.py",
        }
        client = _FakeClient([_FakePoint(0.8, payload, point_id="point-1")])
        searcher = CodeSearcher(
            client=client,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        searcher._query_vector_cache[("code", "query embedding cache")] = [0.1, 0.2]
        searcher._query_vector_cache[("description", "query embedding cache")] = [0.3, 0.4]

        with mock.patch("engine.src.searcher.ThreadPoolExecutor", side_effect=AssertionError("thread pool should not be created")):
            results = searcher.search_hybrid(
                query="query embedding cache",
                limit=1,
                use_keywords=False,
                include_source=False,
            )

        self.assertEqual(len(results), 1)

    def test_diversify_results_prefers_distinct_files_for_broad_queries(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        results = [
            SearchResult(0.9, "a.py", "one", "function", 1, 1, "", ""),
            SearchResult(0.8, "a.py", "two", "function", 2, 2, "", ""),
            SearchResult(0.7, "b.py", "three", "function", 1, 1, "", ""),
        ]
        diversified = searcher._diversify_results(results, ["broad"] * 10, 3)
        self.assertEqual([item.file_path for item in diversified], ["a.py", "b.py", "a.py"])

    def test_hybrid_request_limit_uses_lower_candidate_floor(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )

        self.assertEqual(searcher._hybrid_request_limit(5), 16)
        self.assertEqual(searcher._hybrid_request_limit(10), 30)
        self.assertEqual(
            searcher._hybrid_request_limit(
                5,
                ["role", "metadata", "assigned", "assign", "set", "classify", "classified", "rust", "carried", "carry", "propagate", "store", "persist", "payload", "vector"],
            ),
            18,
        )

    def test_path_prefix_filter_builds_server_side_constraint(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        query_filter, normalized_prefix, fetch_limit = searcher._build_query_filter(
            limit=5,
            language="python",
            file_path=None,
            path_prefix="engine/src",
            symbol_kind=None,
            ranking_keywords=["engine", "src"],
            rerank=False,
        )
        self.assertEqual(normalized_prefix, "engine/src")
        self.assertGreaterEqual(fetch_limit, 50)
        self.assertIsNotNone(query_filter)
        must_conditions = query_filter.get("must", [])
        self.assertTrue(any(isinstance(cond, dict) and cond.get("key") == "path_prefixes" for cond in must_conditions))

    def test_batch_query_filter_does_not_reapply_keyword_overfetch(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        _, _, fetch_limit = searcher._build_query_filter(
            limit=30,
            language=None,
            file_path=None,
            path_prefix=None,
            symbol_kind=None,
            ranking_keywords=["query", "cache", "search"],
            rerank=False,
            apply_keyword_overfetch=False,
        )
        self.assertEqual(fetch_limit, 30)

    def test_symbol_tokens_split_snake_case_identifiers(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        tokens = searcher._symbol_tokens("_embed_query_with_provider_cached", None)
        self.assertIn("embed", tokens)
        self.assertIn("query", tokens)
        self.assertIn("provider", tokens)
        self.assertIn("cached", tokens)

    def test_query_keyword_expansion_adds_delete_synonym_for_removed(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        keywords = searcher._extract_keywords_cached("How are stale chunks removed?")
        self.assertIn("delete", keywords)
        self.assertIn("obsolete", keywords)

    def test_query_keyword_expansion_adds_dedup_synonym_for_duplicates(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        keywords = searcher._extract_keywords_cached("How are duplicate chunks collapsed before embedding?")
        self.assertIn("dedup", keywords)

    def test_query_keyword_expansion_bridges_assignment_and_payload_flow_terms(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        keywords = searcher._extract_keywords_cached("How is role metadata assigned and carried into indexed vector payloads?")
        self.assertIn("classify", keywords)
        self.assertIn("store", keywords)

    def test_symbol_tokens_add_dedup_for_deduplicate(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        tokens = searcher._symbol_tokens("deduplicate_chunks", None)
        self.assertIn("dedup", tokens)

    def test_action_query_penalizes_result_container_symbols(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        penalty = searcher._container_symbol_penalty(
            "TextSearchResult",
            "class",
            ["searcher", "hydrate", "source"],
        )
        neutral = searcher._container_symbol_penalty(
            "PersistedTextIndex",
            "struct",
            ["text", "index", "stored"],
        )
        self.assertLess(penalty, 1.0)
        self.assertEqual(neutral, 1.0)

    def test_strong_basename_match_gets_bonus(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        bonus = searcher._basename_hit_bonus(
            str(Path("service/src/text_index.rs").resolve()),
            ["text", "index", "refresh"],
        )
        self.assertGreater(bonus, 0.0)

    def test_provider_path_bonus_prefers_provider_modules_for_loading_queries(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        bonus = searcher._provider_path_bonus(
            str(Path("engine/src/providers/fastembed_provider.py").resolve()),
            ["provider", "dependencies", "loaded", "cli", "start", "faster"],
        )
        neutral = searcher._provider_path_bonus(
            str(Path("engine/src/cli.py").resolve()),
            ["provider", "dependencies", "loaded", "cli", "start", "faster"],
        )
        self.assertGreater(bonus, 0.0)
        self.assertEqual(neutral, 0.0)

    def test_startup_boundary_bonus_prefers_parser_startup_files(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        bonus = searcher._startup_boundary_bonus(
            str(Path("engine/src/parsing.py").resolve()),
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        parser_cli_bonus = searcher._startup_boundary_bonus(
            str(Path("engine/src/parser_cli.py").resolve()),
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        neutral = searcher._startup_boundary_bonus(
            str(Path("engine/src/searcher.py").resolve()),
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        self.assertGreater(bonus, 0.0)
        self.assertGreater(parser_cli_bonus, 0.0)
        self.assertEqual(neutral, 0.0)

    def test_implementation_path_bonus_prefers_searcher_and_indexer(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        searcher_bonus = searcher._implementation_path_bonus(
            str(Path("engine/src/searcher.py").resolve()),
            ["path", "scoped", "semantic", "search", "payloads", "filters"],
        )
        indexer_bonus = searcher._implementation_path_bonus(
            str(Path("engine/src/indexer.py").resolve()),
            ["path", "scoped", "semantic", "search", "payloads", "filters"],
        )
        neutral = searcher._implementation_path_bonus(
            str(Path("engine/sdk.py").resolve()),
            ["path", "scoped", "semantic", "search", "payloads", "filters"],
        )
        self.assertGreater(searcher_bonus, 0.0)
        self.assertGreater(indexer_bonus, 0.0)
        self.assertEqual(neutral, 0.0)

    def test_startup_symbol_bonus_prefers_rust_parser_service_for_startup_query(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        bonus = searcher._startup_symbol_bonus(
            str(Path("engine/src/parsing.py").resolve()),
            "RustParserService",
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        neutral = searcher._startup_symbol_bonus(
            str(Path("engine/src/collection.py").resolve()),
            "CollectionManager",
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        self.assertGreater(bonus, 0.0)
        self.assertEqual(neutral, 0.0)

    def test_role_bonus_prefers_logic_and_penalizes_configuration_for_action_queries(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        logic_bonus = searcher._role_bonus("logic", ["search", "path", "filters"])
        config_penalty = searcher._role_bonus("configuration", ["search", "path", "filters"])
        neutral = searcher._role_bonus("definition", ["class", "struct"])
        self.assertGreater(logic_bonus, 0.0)
        self.assertLess(config_penalty, 0.0)
        self.assertEqual(neutral, 0.0)

    def test_wrapper_symbol_penalty_downranks_sdk_wrappers_for_internal_queries(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        penalty = searcher._wrapper_symbol_penalty(
            str(Path("engine/sdk.py").resolve()),
            "semantic_search",
            ["path", "scoped", "semantic", "search", "payloads", "filters"],
        )
        neutral = searcher._wrapper_symbol_penalty(
            str(Path("engine/src/searcher.py").resolve()),
            "search_hybrid",
            ["path", "scoped", "semantic", "search", "payloads", "filters"],
        )
        self.assertLess(penalty, 1.0)
        self.assertEqual(neutral, 1.0)

    def test_wrapper_symbol_penalty_downranks_cli_init_for_parser_startup_query(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        penalty = searcher._wrapper_symbol_penalty(
            str(Path("engine/src/cli.py").resolve()),
            "init",
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        self.assertLess(penalty, 1.0)

        optimize_penalty = searcher._wrapper_symbol_penalty(
            str(Path("engine/src/cli.py").resolve()),
            "_optimize_search_config",
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        self.assertLess(optimize_penalty, 1.0)

    def test_wrapper_symbol_penalty_downranks_sdk_provider_property_for_parser_startup_query(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        penalty = searcher._wrapper_symbol_penalty(
            str(Path("engine/sdk.py").resolve()),
            "code_provider",
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        self.assertLess(penalty, 1.0)

        get_searcher_penalty = searcher._wrapper_symbol_penalty(
            str(Path("engine/sdk.py").resolve()),
            "_get_searcher",
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        self.assertLess(get_searcher_penalty, 1.0)

    def test_constructor_penalty_downranks_init_for_non_constructor_queries(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        penalty = searcher._constructor_penalty(
            "__init__",
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        neutral = searcher._constructor_penalty(
            "__init__",
            ["constructor", "qdrant", "client"],
        )
        self.assertLess(penalty, 1.0)
        self.assertEqual(neutral, 1.0)

    def test_startup_import_symbol_penalty_downranks_import_rows(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        penalty = searcher._startup_import_symbol_penalty(
            "import",
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        neutral = searcher._startup_import_symbol_penalty(
            "function",
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        self.assertLess(penalty, 1.0)
        self.assertEqual(neutral, 1.0)

    def test_ranking_helper_penalty_downranks_bonus_helpers_for_non_ranking_queries(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        penalty = searcher._ranking_helper_penalty(
            "_startup_boundary_bonus",
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        neutral = searcher._ranking_helper_penalty(
            "_startup_boundary_bonus",
            ["ranking", "score", "boost"],
        )
        self.assertLess(penalty, 1.0)
        self.assertEqual(neutral, 1.0)

    def test_ranking_subsystem_penalty_downranks_search_modules_for_non_search_queries(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        penalty = searcher._ranking_subsystem_penalty(
            str(Path("service/src/structural_boost.rs").resolve()),
            ["role", "metadata", "rust", "indexed", "vector", "payloads"],
        )
        neutral = searcher._ranking_subsystem_penalty(
            str(Path("engine/src/searcher.py").resolve()),
            ["search", "path", "filters"],
        )
        self.assertLess(penalty, 1.0)
        self.assertEqual(neutral, 1.0)

    def test_filecache_bonus_prefers_cache_module_for_unchanged_detection(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        bonus = searcher._filecache_bonus(
            str(Path("engine/src/filecache.py").resolve()),
            "is_unchanged_from_metadata",
            ["detect", "unchanged", "files", "before", "indexing"],
        )
        neutral = searcher._filecache_bonus(
            str(Path("engine/sdk.py").resolve()),
            "refresh_files",
            ["detect", "unchanged", "files", "before", "indexing"],
        )
        self.assertGreater(bonus, 0.0)
        self.assertEqual(neutral, 0.0)

    def test_subsystem_conflict_penalty_downranks_provider_modules_for_parser_startup_query(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        penalty = searcher._subsystem_conflict_penalty(
            str(Path("engine/src/providers/litellm_provider.py").resolve()),
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        neutral = searcher._subsystem_conflict_penalty(
            str(Path("engine/src/parsing.py").resolve()),
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        self.assertLess(penalty, 1.0)
        self.assertEqual(neutral, 1.0)

        qdrant_penalty = searcher._subsystem_conflict_penalty(
            str(Path("engine/src/qdrant.py").resolve()),
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        self.assertLess(qdrant_penalty, 1.0)

        config_penalty = searcher._subsystem_conflict_penalty(
            str(Path("engine/src/config.py").resolve()),
            ["parser", "commands", "qdrant", "imports", "startup"],
        )
        self.assertLess(config_penalty, 1.0)

    def test_action_query_boosts_matching_symbol_action(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        bonus = searcher._action_symbol_bonus(
            ["stale", "delete", "chunks"],
            ["delete", "file"],
        )
        neutral = searcher._action_symbol_bonus(
            ["stale", "delete", "chunks"],
            ["filter", "chunks"],
        )
        self.assertGreater(bonus, 0.0)
        self.assertEqual(neutral, 0.0)

    def test_action_query_boosts_classify_for_role_assignment_queries(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        bonus = searcher._action_symbol_bonus(
            ["role", "assigned", "classify"],
            ["classify", "role"],
        )
        self.assertGreater(bonus, 0.0)

    def test_non_test_query_penalizes_test_paths(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        penalty = searcher._path_signal_multiplier(
            str(Path("engine/tests/test_regressions.py").resolve()),
            ["parser", "python", "commands"],
        )
        no_penalty = searcher._path_signal_multiplier(
            str(Path("engine/tests/test_regressions.py").resolve()),
            ["test", "regression", "imports"],
        )
        self.assertLess(penalty, 1.0)
        self.assertEqual(no_penalty, 1.0)

    def test_non_benchmark_query_penalizes_scripts_paths(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        penalty = searcher._path_signal_multiplier(
            str(Path("scripts/retrieval_benchmark.py").resolve()),
            ["parser", "python", "commands"],
        )
        no_penalty = searcher._path_signal_multiplier(
            str(Path("scripts/retrieval_benchmark.py").resolve()),
            ["benchmark", "retrieval", "cases"],
        )
        self.assertLess(penalty, 1.0)
        self.assertEqual(no_penalty, 1.0)

    def test_default_search_promotes_metadata_match_for_architecture_query(self) -> None:
        general_payload = {
            "file_path": str(Path("engine/__init__.py").resolve()),
            "symbol_name": "semantic_search",
            "symbol_kind": "function",
            "line_start": 1,
            "line_end": 20,
            "source": "def semantic_search(...): ...",
            "description": "Run semantic search over indexed code vectors.",
            "keywords": ["semantic", "search", "project"],
            "path_context": "engine / __init__.py",
        }
        relevant_payload = {
            "file_path": str(Path("engine/src/searcher.py").resolve()),
            "symbol_name": "_embed_query_with_provider_cached",
            "symbol_kind": "function",
            "line_start": 300,
            "line_end": 360,
            "source": "def _embed_query_with_provider_cached(...): ...",
            "description": "Cache query embedding vectors in memory and on disk for repeated searches.",
            "keywords": ["query", "embedding", "cache", "search"],
            "path_context": "engine / src / searcher.py",
            "signature": "_embed_query_with_provider_cached(query, provider, cache_key)",
        }

        client = _FakeClient(
            [
                _FakePoint(0.71, general_payload),
                _FakePoint(0.62, relevant_payload),
            ]
        )
        searcher = CodeSearcher(
            client=client,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )

        results = searcher.search_code(
            query="How are query embeddings cached or reused in the search layer for repeated searches?",
            limit=2,
            use_keywords=False,
        )
        self.assertEqual(results[0].file_path.replace("\\", "/"), relevant_payload["file_path"].replace("\\", "/"))

    def test_fallback_description_uses_docstring_and_path_context(self) -> None:
        chunk = CodeChunk(
            chunk_id="chunk-1",
            source="def _embed_query_with_provider_cached(query, provider, cache_key): ...",
            language="python",
            file_path=str(Path("engine/src/searcher.py").resolve()),
            symbol_name="_embed_query_with_provider_cached",
            symbol_kind="function",
            line_start=1,
            line_end=5,
            byte_start=0,
            byte_end=64,
            signature="_embed_query_with_provider_cached(query, provider, cache_key)",
            docstring="Cache query embedding vectors in memory and on disk for repeated searches.",
            parent="CodeSearcher",
            visibility=None,
            role=None,
            file_hash="hash",
        )

        description = build_fallback_description(chunk)
        self.assertIn("Cache query embedding vectors", description.description)
        self.assertIn("engine/src/searcher.py", description.description.replace("\\", "/"))
        self.assertIn("query", description.keywords)
        self.assertIn("searcher", description.keywords)

    def test_merge_related_edges_deduplicates_related_files(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=None,
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        try:
            related_by_path: dict[str, dict] = {}
            qc._merge_related_edges(
                seed_file="a.py",
                relation="imports",
                edges=[
                    ImportEdge(
                        source_file="a.py",
                        target_file="b.py",
                        import_statement="import b",
                        module_path="b",
                        language="python",
                        line=1,
                    ),
                    ImportEdge(
                        source_file="a.py",
                        target_file="b.py",
                        import_statement="from b import c",
                        module_path="b",
                        language="python",
                        line=2,
                    ),
                ],
                related_by_path=related_by_path,
                excluded_paths={"a.py"},
            )
        finally:
            qc.close()

        self.assertIn("b.py", related_by_path)
        self.assertEqual(len(related_by_path["b.py"]["relations"]), 2)

    def test_related_files_for_results_collects_import_graph_neighbors(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=None,
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        try:
            result_item = type("ResultItem", (), {"file_path": "a.py"})()
            graph_result = type(
                "GraphResult",
                (),
                {
                    "edges": [
                        ImportEdge(
                            source_file="a.py",
                            target_file="b.py",
                            import_statement="import b",
                            module_path="b",
                            language="python",
                            line=1,
                        )
                    ]
                },
            )()
            with mock.patch.object(qc, "import_graph", return_value=graph_result), mock.patch.object(
                qc,
                "find_importers",
                return_value=type("GraphResult", (), {"edges": []})(),
            ):
                related = qc._related_files_for_results(
                    results=[result_item],
                    related_seed_files=1,
                    related_file_limit=4,
                )
        finally:
            qc.close()

        self.assertEqual(len(related), 1)
        self.assertEqual(related[0]["file_path"], "b.py")
        self.assertEqual(related[0]["relations"][0]["relation"], "imports")

    def test_related_callers_for_results_collects_callers_for_top_callable(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=None,
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        try:
            result_item = type(
                "ResultItem",
                (),
                {"symbol_name": "build_chunks", "symbol_kind": "function"},
            )()
            caller_result = type(
                "CallerResult",
                (),
                {
                    "callers": [
                        type(
                            "CallerRow",
                            (),
                            {
                                "caller_name": "index_directory",
                                "caller_kind": "function",
                                "caller_file_path": "engine/sdk.py",
                                "caller_line": 1200,
                                "caller_language": "python",
                            },
                        )()
                    ]
                },
            )()
            with mock.patch.object(qc, "find_callers", return_value=caller_result):
                related = qc._related_callers_for_results([result_item])
        finally:
            qc.close()

        self.assertEqual(len(related), 1)
        self.assertEqual(related[0]["symbol"], "build_chunks")
        self.assertEqual(related[0]["caller_name"], "index_directory")

    def test_split_semantic_bundle_results_moves_extra_distinct_files_to_related(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=None,
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        try:
            results = [
                SearchResult(0.9, "a.py", "one", "function", 1, 1, "", ""),
                SearchResult(0.8, "a.py", "two", "function", 2, 2, "", ""),
                SearchResult(0.7, "b.py", "three", "function", 1, 1, "", ""),
                SearchResult(0.6, "c.py", "four", "function", 1, 1, "", ""),
            ]
            anchors, related = qc._split_semantic_bundle_results(results, anchor_limit=2, related_file_limit=3)
        finally:
            qc.close()

        self.assertEqual([item.file_path for item in anchors], ["a.py", "b.py"])
        self.assertEqual([item["file_path"] for item in related], ["c.py"])
        self.assertEqual(related[0]["relations"][0]["relation"], "semantic_neighbor")

    def test_tooling_related_semantic_neighbors_filters_to_scripts(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=None,
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        try:
            results = [
                SearchResult(0.9, "engine/src/qdrant_search.py", "RestQdrantSearchClient", "class", 1, 1, "", ""),
                SearchResult(0.8, "scripts/search_phase_benchmark.py", "main", "function", 1, 1, "", ""),
                SearchResult(0.7, "scripts/retrieval_benchmark.py", "main", "function", 1, 1, "", ""),
            ]
            with mock.patch.object(qc, "semantic_search", return_value=results):
                related = qc._tooling_related_semantic_neighbors(
                    query="How does the Qdrant phase benchmark separate client-side latency from Qdrant server time?",
                    project_name="quickcontext",
                    excluded_paths={"engine/src/qdrant_search.py"},
                    related_file_limit=2,
                )
        finally:
            qc.close()

        self.assertEqual([item["file_path"] for item in related], ["scripts/search_phase_benchmark.py", "scripts/retrieval_benchmark.py"])
        self.assertEqual(related[0]["relations"][0]["relation"], "semantic_tooling_neighbor")

    def test_looks_like_tooling_query_detects_benchmark_terms(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=None,
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        try:
            self.assertTrue(qc._looks_like_tooling_query("How does the phase benchmark separate client-side latency from server time?"))
            self.assertFalse(qc._looks_like_tooling_query("How are duplicate chunks collapsed before embedding?"))
        finally:
            qc.close()

    def test_should_use_bundle_for_broad_cross_file_query(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=None,
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        try:
            self.assertTrue(
                qc._should_use_bundle_for_query(
                    "How are file path prefixes generated during indexing and then used during retrieval filtering?"
                )
            )
            self.assertFalse(qc._should_use_bundle_for_query("Where are query embeddings cached?"))
        finally:
            qc.close()

    def test_semantic_search_auto_routes_between_search_and_bundle(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=None,
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        try:
            with mock.patch.object(qc, "semantic_search", return_value=["plain"]) as plain, mock.patch.object(
                qc,
                "semantic_search_bundle",
                return_value={"query": "x", "project_name": "p", "results": ["bundle"], "related_files": [], "related_callers": []},
            ) as bundle:
                simple = qc.semantic_search_auto("Where are query embeddings cached?")
                broad = qc.semantic_search_auto(
                    "How are file path prefixes generated during indexing and then used during retrieval filtering?"
                )
        finally:
            qc.close()

        plain.assert_called_once()
        bundle.assert_called_once()
        self.assertEqual(simple["mode"], "search")
        self.assertEqual(simple["results"], ["plain"])
        self.assertEqual(broad["mode"], "bundle")
        self.assertEqual(broad["results"], ["bundle"])

    def test_search_hydrates_final_results_after_lightweight_query(self) -> None:
        payload = {
            "file_path": str(Path("engine/src/searcher.py").resolve()),
            "symbol_name": "_embed_query_cached",
            "symbol_kind": "function",
            "line_start": 10,
            "line_end": 20,
            "description": "Cache query embeddings for repeated searches.",
            "keywords": ["query", "embedding", "cache"],
            "path_context": "engine / src / searcher.py",
        }
        full_payload = dict(payload)
        full_payload["source"] = "def _embed_query_cached(query, using): ..."

        record = type("Record", (), {"id": "point-1", "payload": full_payload})()
        client = _FakeClient(
            [_FakePoint(0.8, payload, point_id="point-1")],
            records=[record],
        )
        searcher = CodeSearcher(
            client=client,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )

        results = searcher.search_code(
            query="How are query embeddings cached or reused in the search layer for repeated searches?",
            limit=1,
            use_keywords=False,
        )

        self.assertEqual(client.last_query_with_payload, LIGHT_RESULT_PAYLOAD_FIELDS)
        self.assertIsNotNone(client.last_retrieve_with_payload)
        self.assertEqual(results[0].source, full_payload["source"])

    def test_search_skips_hydration_when_source_not_requested(self) -> None:
        payload = {
            "file_path": str(Path("engine/src/searcher.py").resolve()),
            "symbol_name": "_embed_query_cached",
            "symbol_kind": "function",
            "line_start": 10,
            "line_end": 20,
            "description": "Cache query embeddings for repeated searches.",
            "keywords": ["query", "embedding", "cache"],
            "path_context": "engine / src / searcher.py",
        }
        client = _FakeClient([_FakePoint(0.8, payload, point_id="point-1")])
        searcher = CodeSearcher(
            client=client,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )

        results = searcher.search_code(
            query="How are query embeddings cached or reused in the search layer for repeated searches?",
            limit=1,
            use_keywords=False,
            include_source=False,
        )

        self.assertEqual(client.retrieve_calls, 0)
        self.assertEqual(results[0].source, "")

    def test_qdrant_connect_skips_eager_verification_by_default(self) -> None:
        fake_client = mock.Mock()

        with mock.patch("engine.src.qdrant.QdrantClient", return_value=fake_client) as ctor:
            conn = QdrantConnection(QdrantConfig())
            conn.connect()

        ctor.assert_called_once()
        fake_client.get_collections.assert_not_called()

    def test_qdrant_connect_verifies_when_requested(self) -> None:
        fake_client = mock.Mock()

        with mock.patch("engine.src.qdrant.QdrantClient", return_value=fake_client):
            conn = QdrantConnection(QdrantConfig())
            conn.connect(verify=True)

        fake_client.get_collections.assert_called_once()

    def test_optimize_search_config_prefers_rest_for_local_qdrant(self) -> None:
        optimized = _optimize_search_config(EngineConfig(qdrant=QdrantConfig(prefer_grpc=True)))
        self.assertFalse(optimized.qdrant.prefer_grpc)


class LazyImportBoundaryTests(unittest.TestCase):
    def _reload_module(self, module_name: str):
        module = importlib.import_module(module_name)
        return importlib.reload(module)

    def test_engine_package_import_does_not_require_qdrant_client(self) -> None:
        with _BlockedImport({"qdrant_client"}):
            module = self._reload_module("engine")
            self.assertTrue(hasattr(module, "__getattr__"))

    def test_parsing_import_does_not_require_qdrant_client(self) -> None:
        with _BlockedImport({"qdrant_client"}):
            module = self._reload_module("engine.src.parsing")
            self.assertTrue(hasattr(module, "RustParserService"))

    def test_fastembed_provider_import_is_lazy(self) -> None:
        with _BlockedImport({"fastembed"}):
            module = self._reload_module("engine.src.providers.fastembed_provider")
            self.assertTrue(hasattr(module, "FastEmbedProvider"))

    def test_litellm_provider_import_is_lazy(self) -> None:
        with _BlockedImport({"litellm"}):
            module = self._reload_module("engine.src.providers.litellm_provider")
            self.assertTrue(hasattr(module, "LiteLLMProvider"))


if __name__ == "__main__":
    unittest.main()
