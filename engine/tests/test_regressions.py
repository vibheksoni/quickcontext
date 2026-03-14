import importlib
import os
import tempfile
import time
import unittest
from unittest import mock
from dataclasses import dataclass, replace
from pathlib import Path
import builtins

from engine.__main__ import _find_command
from engine.sdk import QuickContext
from engine.src.chunker import ChunkBuilder, CodeChunk
from engine.src.collection import CollectionManager
from engine.src.config import CollectionVectorConfig, EngineConfig, QdrantConfig
from engine.src.dedup import deduplicate_chunks
from engine.src.describer import build_fallback_description
from engine.src.filecache import FileSignatureCache
from engine.src.cli import _optimize_search_config
from engine.src.parsing import CompactExtractionResult, ExtractStats, ImportEdge, RustParserService, SymbolLookupItem, SymbolLookupResult
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


class _FakeCollectionSummary:
    def __init__(self, name: str):
        self.name = name


class _FakeAliasSummary:
    def __init__(self, alias_name: str, collection_name: str):
        self.alias_name = alias_name
        self.collection_name = collection_name


class _FakeCollectionList:
    def __init__(self, collections: list[str]):
        self.collections = [_FakeCollectionSummary(name) for name in collections]


class _FakeAliasList:
    def __init__(self, aliases: dict[str, str]):
        self.aliases = [
            _FakeAliasSummary(alias_name=name, collection_name=target)
            for name, target in aliases.items()
        ]


class _FakeCollectionClient:
    def __init__(self, collections: list[str], aliases: dict[str, str] | None = None):
        self.collections = set(collections)
        self.aliases = dict(aliases or {})
        self.payload_index_calls: list[tuple[str, str]] = []

    def get_collections(self):
        return _FakeCollectionList(sorted(self.collections))

    def get_collection(self, name: str):
        resolved = self.aliases.get(name, name)
        if resolved not in self.collections:
            raise RuntimeError(f"missing collection: {name}")
        return object()

    def get_collection_aliases(self, collection_name: str):
        aliases = {
            alias_name: target
            for alias_name, target in self.aliases.items()
            if alias_name == collection_name
        }
        return _FakeAliasList(aliases)

    def get_aliases(self):
        return _FakeAliasList(self.aliases)

    def create_collection(self, collection_name: str, vectors_config):
        self.collections.add(collection_name)

    def create_payload_index(self, collection_name: str, field_name: str, field_schema):
        self.payload_index_calls.append((collection_name, field_name))

    def update_collection_aliases(self, change_aliases_operations):
        for op in change_aliases_operations:
            create_alias = getattr(op, "create_alias", None)
            if create_alias is not None:
                self.aliases[create_alias.alias_name] = create_alias.collection_name
                continue
            delete_alias = getattr(op, "delete_alias", None)
            if delete_alias is not None:
                self.aliases.pop(delete_alias.alias_name, None)

    def delete_collection(self, name: str):
        self.collections.discard(name)


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

    def test_collection_shadow_swap_and_cleanup_keeps_new_active_version(self) -> None:
        client = _FakeCollectionClient(
            collections=["demo_v1", "demo_v2"],
            aliases={"demo": "demo_v2"},
        )
        config = EngineConfig(
            qdrant=QdrantConfig(),
            code_embedding=None,
            desc_embedding=None,
            llm=None,
            vectors=[CollectionVectorConfig(name="code", dimension=8, distance="cosine")],
        )
        manager = CollectionManager(client=client, config=config, collection_name="demo")

        shadow_name = manager.create_shadow()
        old_collection = manager.swap_alias(shadow_name)
        deleted = manager.cleanup_old_versions()

        self.assertEqual(shadow_name, "demo_v3")
        self.assertEqual(old_collection, "demo_v2")
        self.assertEqual(manager.resolve_alias(), "demo_v3")
        self.assertEqual(set(deleted), {"demo_v1", "demo_v2"})
        self.assertIn("demo_v3", client.collections)
        self.assertNotIn("demo_v2", client.collections)

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

    def test_graph_path_bonus_prefers_service_graph_files_for_import_queries(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        service_bonus = searcher._graph_path_bonus(
            str(Path("service/src/import_graph.rs").resolve()),
            "find_importers",
            ["importers", "dependencies", "module", "neighbors"],
        )
        wrapper_bonus = searcher._graph_path_bonus(
            str(Path("engine/sdk.py").resolve()),
            "find_importers",
            ["importers", "dependencies", "module", "neighbors"],
        )
        self.assertGreater(service_bonus, wrapper_bonus)

    def test_graph_wrapper_penalty_downranks_sdk_and_cli_for_graph_queries(self) -> None:
        searcher = CodeSearcher(
            client=None,
            collection_name="x",
            code_provider=_FakeProvider("code", 2),
            desc_provider=_FakeProvider("desc", 2),
        )
        sdk_penalty = searcher._graph_wrapper_penalty(
            str(Path("engine/sdk.py").resolve()),
            ["caller", "trace", "graph", "lookup"],
        )
        cli_penalty = searcher._graph_wrapper_penalty(
            str(Path("engine/src/cli.py").resolve()),
            ["caller", "trace", "graph", "lookup"],
        )
        neutral = searcher._graph_wrapper_penalty(
            str(Path("service/src/symbol_index.rs").resolve()),
            ["caller", "trace", "graph", "lookup"],
        )
        self.assertLess(sdk_penalty, 1.0)
        self.assertLess(cli_penalty, 1.0)
        self.assertEqual(neutral, 1.0)

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
            neighbor_result = type(
                "NeighborResult",
                (),
                {
                    "imports": [
                        ImportEdge(
                            source_file="a.py",
                            target_file="b.py",
                            import_statement="import b",
                            module_path="b",
                            language="python",
                            line=1,
                        )
                    ],
                    "importers": [],
                },
            )()
            with mock.patch.object(
                qc,
                "import_neighbors",
                return_value=neighbor_result,
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
            related = qc._tooling_related_semantic_neighbors(
                results=results,
                tooling_query=True,
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
            self.assertTrue(
                qc._should_use_bundle_for_query(
                    "How does the Qdrant phase benchmark separate client-side latency from Qdrant server time?"
                )
            )
            self.assertTrue(
                qc._should_use_bundle_for_query(
                    "How are unchanged points copied into the new shadow collection during a full reindex?"
                )
            )
            self.assertFalse(qc._should_use_bundle_for_query("Where are query embeddings cached?"))
        finally:
            qc.close()

    def test_should_use_graph_related_for_dependency_queries(self) -> None:
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
            self.assertTrue(qc._should_use_graph_related_for_query("Which files import this module and what dependencies does it have?"))
            self.assertFalse(qc._should_use_graph_related_for_query("How are file path prefixes generated during indexing and then used during retrieval filtering?"))
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

    def test_extract_symbol_query_candidate_prefers_identifier_targets(self) -> None:
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
            self.assertEqual(
                qc._extract_symbol_query_candidate("Where is CodeSearcher.search_hybrid defined?"),
                "CodeSearcher.search_hybrid",
            )
            self.assertEqual(
                qc._extract_symbol_query_candidate("QuickContext._get_searcher"),
                "QuickContext._get_searcher",
            )
            self.assertIsNone(
                qc._extract_symbol_query_candidate(
                    "How does the Python layer decide how to connect to the Rust service on Windows versus Linux?"
                )
            )
        finally:
            qc.close()

    def test_retrieve_context_auto_routes_symbol_queries_to_symbol_lookup(self) -> None:
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
            result = SearchResult(
                1.0,
                "engine/src/searcher.py",
                "search_hybrid",
                "function",
                1,
                10,
                "def search_hybrid(...): ...",
                "search_hybrid(query, limit=10)",
                parent="CodeSearcher",
                language="python",
            )
            with mock.patch.object(qc, "_symbol_lookup_search_results", return_value=[result]) as symbol_search, mock.patch.object(
                qc,
                "semantic_search_auto",
                side_effect=AssertionError("semantic_search_auto should not run for symbol query"),
            ):
                payload = qc.retrieve_context_auto("Where is CodeSearcher.search_hybrid defined?")
        finally:
            qc.close()

        symbol_search.assert_called_once()
        self.assertEqual(payload["mode"], "symbol")
        self.assertEqual(payload["results"][0].symbol_name, "search_hybrid")

    def test_retrieve_context_auto_falls_back_to_semantic_auto_when_symbol_results_are_empty(self) -> None:
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
            with mock.patch.object(qc, "_symbol_lookup_search_results", return_value=[]), mock.patch.object(
                qc,
                "text_search",
                return_value=type("TextResult", (), {"matches": []})(),
            ), mock.patch.object(
                qc,
                "semantic_search_auto",
                return_value={"query": "x", "project_name": "p", "mode": "search", "results": ["semantic"], "related_files": [], "related_callers": []},
            ) as semantic_auto:
                payload = qc.retrieve_context_auto("Where is CodeSearcher.search_hybrid defined?")
        finally:
            qc.close()

        semantic_auto.assert_called_once()
        self.assertEqual(payload["mode"], "search")
        self.assertEqual(payload["results"], ["semantic"])
        self.assertIsNone(payload["symbol_query"])

    def test_retrieve_context_auto_adds_lexical_related_files_for_search_mode(self) -> None:
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
            semantic_result = SearchResult(
                1.0,
                str(Path("engine/src/parsing.py").resolve()),
                "connect",
                "function",
                1,
                10,
                "def connect(...): ...",
                "connect(path)",
                parent=None,
                language="python",
            )
            text_match = type(
                "TextMatch",
                (),
                {
                    "file_path": str(Path("engine/src/pipe.py").resolve()),
                    "language": "python",
                    "snippet_line_start": 12,
                },
            )()
            with mock.patch.object(
                qc,
                "semantic_search_auto",
                return_value={
                    "query": "x",
                    "project_name": "p",
                    "mode": "search",
                    "results": [semantic_result],
                    "related_files": [],
                    "related_callers": [],
                },
            ), mock.patch.object(
                qc,
                "_should_use_text_primary_for_query",
                return_value=False,
            ), mock.patch.object(
                qc,
                "text_search",
                return_value=type("TextResult", (), {"matches": [text_match]})(),
            ):
                payload = qc.retrieve_context_auto(
                    "How does the Python layer decide how to connect to the Rust service on Windows versus Linux?"
                )
        finally:
            qc.close()

        self.assertEqual(payload["mode"], "search")
        self.assertEqual(len(payload["related_files"]), 1)
        self.assertEqual(payload["related_files"][0]["file_path"], str(Path("engine/src/pipe.py").resolve()))
        self.assertEqual(payload["related_files"][0]["relations"][0]["relation"], "lexical_neighbor")

    def test_retrieve_context_auto_adds_graph_lexical_related_files_for_graph_query(self) -> None:
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
            semantic_result = SearchResult(
                1.0,
                str(Path("engine/src/parsing.py").resolve()),
                "import_graph",
                "function",
                1,
                10,
                "def import_graph(...): ...",
                "import_graph(file, path)",
                parent=None,
                language="python",
            )
            base_text_match = type(
                "TextMatch",
                (),
                {
                    "file_path": str(Path("engine/src/pipe.py").resolve()),
                    "language": "python",
                    "snippet_line_start": 12,
                    "matched_terms": ["importers", "dependencies"],
                    "score": 8.0,
                },
            )()
            graph_text_match = type(
                "TextMatch",
                (),
                {
                    "file_path": str(Path("service/src/import_graph.rs").resolve()),
                    "language": "rust",
                    "snippet_line_start": 44,
                    "matched_terms": ["importers", "graph", "dependencies"],
                    "score": 9.0,
                },
            )()
            with mock.patch.object(
                qc,
                "semantic_search_auto",
                return_value={
                    "query": "x",
                    "project_name": "p",
                    "mode": "search",
                    "results": [semantic_result],
                    "related_files": [],
                    "related_callers": [],
                },
            ), mock.patch.object(
                qc,
                "_should_use_text_primary_for_query",
                return_value=False,
            ), mock.patch.object(
                qc,
                "text_search",
                side_effect=[
                    type("TextResult", (), {"matches": [base_text_match]})(),
                    type("TextResult", (), {"matches": [graph_text_match]})(),
                ],
            ):
                payload = qc.retrieve_context_auto(
                    "Which files import this module and what dependencies does it have?"
                )
        finally:
            qc.close()

        all_paths = {str(Path(item.file_path).resolve()) for item in payload["results"]}
        all_paths.update(item["file_path"] for item in payload["related_files"])
        self.assertIn(str(Path("service/src/import_graph.rs").resolve()), all_paths)

    def test_promote_graph_related_results_dedupes_wrappers_and_backfills_impl_files(self) -> None:
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
                SearchResult(1.0, str(Path("engine/src/parsing.py").resolve()), "find_importers", "function", 1, 10, "", "", language="python"),
                SearchResult(0.9, str(Path("engine/src/parsing.py").resolve()), "import_graph", "function", 11, 20, "", "", language="python"),
                SearchResult(0.8, str(Path("engine/sdk.py").resolve()), "find_importers", "function", 21, 30, "", "", language="python"),
            ]
            related_files = [
                {
                    "file_path": str(Path("service/src/import_graph.rs").resolve()),
                    "distance": 1,
                    "relations": [
                        {
                            "relation": "graph_lexical_neighbor",
                            "seed_file": "",
                            "module_path": "",
                            "language": "rust",
                            "line": 44,
                        }
                    ],
                },
                {
                    "file_path": str(Path("engine/src/pipe.py").resolve()),
                    "distance": 1,
                    "relations": [
                        {
                            "relation": "lexical_neighbor",
                            "seed_file": "",
                            "module_path": "",
                            "language": "python",
                            "line": 10,
                        }
                    ],
                },
            ]
            promoted, remaining = qc._promote_graph_related_results(
                "How does the code find importers for a file and merge import neighbors?",
                results,
                related_files,
                limit=3,
            )
        finally:
            qc.close()

        promoted_paths = [Path(item.file_path).resolve() for item in promoted]
        self.assertEqual(
            promoted_paths,
            [
                Path("engine/src/parsing.py").resolve(),
                Path("engine/sdk.py").resolve(),
                Path("service/src/import_graph.rs").resolve(),
            ],
        )
        self.assertEqual(
            [Path(item["file_path"]).resolve() for item in remaining],
            [Path("engine/src/pipe.py").resolve()],
        )

    def test_retrieve_context_auto_routes_non_symbol_query_to_text_primary(self) -> None:
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
            text_matches = [
                type(
                    "TextMatch",
                    (),
                    {
                        "file_path": str(Path("engine/src/pipe.py").resolve()),
                        "language": "python",
                        "snippet_line_start": 12,
                        "snippet_line_end": 18,
                        "snippet": "def connect(...): ...",
                        "matched_terms": ["python", "connect", "windows", "linux"],
                        "score": 12.0,
                    },
                )(),
                type(
                    "TextMatch",
                    (),
                    {
                        "file_path": str(Path("engine/src/parsing.py").resolve()),
                        "language": "python",
                        "snippet_line_start": 24,
                        "snippet_line_end": 31,
                        "snippet": "class RustParserService: ...",
                        "matched_terms": ["python", "rust", "service"],
                        "score": 10.0,
                    },
                )(),
            ]
            with mock.patch.object(
                qc,
                "text_search",
                return_value=type("TextResult", (), {"matches": text_matches})(),
            ), mock.patch.object(
                qc,
                "semantic_search_auto",
                side_effect=AssertionError("semantic_search_auto should not run for text-primary route"),
            ):
                payload = qc.retrieve_context_auto(
                    "How does the Python layer decide how to connect to the Rust service on Windows versus Linux?",
                    limit=1,
                )
        finally:
            qc.close()

        self.assertEqual(payload["mode"], "text")
        self.assertEqual(payload["results"][0].file_path, str(Path("engine/src/pipe.py").resolve()))
        self.assertEqual(payload["related_files"][0]["file_path"], str(Path("engine/src/parsing.py").resolve()))

    def test_retrieve_context_auto_routes_non_graph_bundle_query_to_text_primary(self) -> None:
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
            text_matches = [
                type(
                    "TextMatch",
                    (),
                    {
                        "file_path": str(Path("engine/sdk.py").resolve()),
                        "language": "python",
                        "snippet_line_start": 2923,
                        "snippet_line_end": 2947,
                        "snippet": "shadow_name = collection.create_shadow()",
                        "matched_terms": ["full", "indexing", "unchanged", "shadow", "collection"],
                        "score": 18.0,
                    },
                )(),
                type(
                    "TextMatch",
                    (),
                    {
                        "file_path": str(Path("engine/src/collection.py").resolve()),
                        "language": "python",
                        "snippet_line_start": 323,
                        "snippet_line_end": 333,
                        "snippet": "def create_shadow(self) -> str:",
                        "matched_terms": ["shadow", "collection"],
                        "score": 17.0,
                    },
                )(),
            ]
            with mock.patch.object(
                qc,
                "text_search",
                return_value=type("TextResult", (), {"matches": text_matches})(),
            ), mock.patch.object(
                qc,
                "semantic_search_bundle",
                side_effect=AssertionError("semantic_search_bundle should not run when strong lexical path is enough"),
            ):
                payload = qc.retrieve_context_auto(
                    "How does full indexing avoid replacing unchanged data when it builds a shadow collection?",
                    limit=2,
                )
        finally:
            qc.close()

        self.assertEqual(payload["mode"], "text")
        self.assertEqual(
            [Path(item.file_path).resolve() for item in payload["results"]],
            [Path("engine/sdk.py").resolve(), Path("engine/src/collection.py").resolve()],
        )

    def test_retrieve_context_auto_uses_symbol_bundle_for_broad_symbol_queries(self) -> None:
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
            result = SearchResult(
                1.0,
                "engine/src/searcher.py",
                "search_hybrid",
                "function",
                1,
                10,
                "def search_hybrid(...): ...",
                "search_hybrid(query, limit=10)",
                parent="CodeSearcher",
                language="python",
            )
            with mock.patch.object(qc, "_symbol_lookup_search_results", return_value=[result]), mock.patch.object(
                qc,
                "_should_use_bundle_for_query",
                return_value=True,
            ), mock.patch.object(
                qc,
                "_should_use_graph_related_for_query",
                return_value=True,
            ), mock.patch.object(
                qc,
                "_related_files_for_results",
                return_value=[{"file_path": "engine/src/qdrant_search.py", "distance": 1, "relations": []}],
            ) as related_files, mock.patch.object(
                qc,
                "_related_callers_for_results",
                return_value=[{"symbol": "search_hybrid", "caller_name": "semantic_search"}],
            ) as related_callers:
                payload = qc.retrieve_context_auto(
                    "How do imports and callers around CodeSearcher.search_hybrid flow across the codebase?"
                )
        finally:
            qc.close()

        related_files.assert_called_once()
        related_callers.assert_called_once()
        self.assertEqual(payload["mode"], "symbol_bundle")
        self.assertEqual(payload["results"][0].symbol_name, "search_hybrid")
        self.assertEqual(payload["related_callers"][0]["caller_name"], "semantic_search")

    def test_expand_symbol_context_results_adds_referenced_helper_symbols(self) -> None:
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
            anchor = SearchResult(
                1.0,
                str(Path("engine/src/searcher.py").resolve()),
                "search_hybrid",
                "function",
                281,
                380,
                "code_vector, desc_vector = self._hybrid_query_vectors(query)\n"
                "code_results, desc_results = self._batch_search(requests=[])\n"
                "fused = self._rrf_fuse([code_results, desc_results], [0.5, 0.5])",
                "Hybrid search using RRF fusion across code and description spaces.",
                parent="CodeSearcher",
                language="python",
            )
            fallback = SearchResult(
                0.9,
                str(Path("engine/src/searcher.py").resolve()),
                "CodeSearcher",
                "class",
                115,
                1579,
                "",
                "Semantic code search using dual embeddings.",
                parent=None,
                language="python",
            )
            extraction = _Extraction(
                file_path=str(Path("engine/src/searcher.py").resolve()),
                language="python",
                symbols=[
                    _Symbol(
                        name="_hybrid_query_vectors",
                        kind="function",
                        language="python",
                        file_path=str(Path("engine/src/searcher.py").resolve()),
                        line_start=700,
                        line_end=720,
                        byte_start=0,
                        byte_end=10,
                        source="def _hybrid_query_vectors(self, query): ...",
                        signature="_hybrid_query_vectors(self, query)",
                        parent="CodeSearcher",
                    ),
                    _Symbol(
                        name="_batch_search",
                        kind="function",
                        language="python",
                        file_path=str(Path("engine/src/searcher.py").resolve()),
                        line_start=839,
                        line_end=922,
                        byte_start=0,
                        byte_end=10,
                        source="def _batch_search(self, requests): ...",
                        signature="_batch_search(self, requests)",
                        parent="CodeSearcher",
                    ),
                    _Symbol(
                        name="_rrf_fuse",
                        kind="function",
                        language="python",
                        file_path=str(Path("engine/src/searcher.py").resolve()),
                        line_start=520,
                        line_end=571,
                        byte_start=0,
                        byte_end=10,
                        source="def _rrf_fuse(self, results, weights): ...",
                        signature="_rrf_fuse(self, results, weights)",
                        parent="CodeSearcher",
                    ),
                ],
            )
            with mock.patch.object(qc, "extract_symbols", return_value=[extraction]):
                expanded = qc._expand_symbol_context_results(
                    "How does CodeSearcher.search_hybrid merge code and description vectors?",
                    [anchor, fallback],
                    limit=4,
                )
        finally:
            qc.close()

        names = [item.symbol_name for item in expanded]
        self.assertEqual(names[0], "search_hybrid")
        self.assertEqual(set(names[1:4]), {"_hybrid_query_vectors", "_batch_search", "_rrf_fuse"})

    def test_expand_symbol_context_results_skips_definition_only_queries(self) -> None:
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
            anchor = SearchResult(
                1.0,
                str(Path("engine/src/searcher.py").resolve()),
                "search_hybrid",
                "function",
                281,
                380,
                "return self._batch_search(requests=[])\n",
                "Hybrid search using RRF fusion across code and description spaces.",
                parent="CodeSearcher",
                language="python",
            )
            fallback = SearchResult(
                0.9,
                str(Path("engine/src/searcher.py").resolve()),
                "CodeSearcher",
                "class",
                115,
                1579,
                "",
                "Semantic code search using dual embeddings.",
                parent=None,
                language="python",
            )
            with mock.patch.object(
                qc,
                "extract_symbols",
                side_effect=AssertionError("extract_symbols should not run for definition-only query"),
            ):
                expanded = qc._expand_symbol_context_results(
                    "Where is CodeSearcher.search_hybrid defined?",
                    [anchor, fallback],
                    limit=3,
                )
        finally:
            qc.close()

        self.assertEqual([item.symbol_name for item in expanded], ["search_hybrid", "CodeSearcher"])

    def test_expand_symbol_context_results_penalizes_postprocess_helpers_for_fuse_query(self) -> None:
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
            anchor = SearchResult(
                1.0,
                str(Path("engine/src/searcher.py").resolve()),
                "search_structured",
                "function",
                391,
                509,
                "batch_results = self._batch_search(requests)\n"
                "fused = self._rrf_fuse(result_lists, filtered_weights)\n"
                "if rerank_active:\n"
                "    return self._blend_with_rerank(query, fused, limit)\n"
                "diversified = self._diversify_results(fused, broad_keywords, limit)\n"
                "return self._finalize_results(diversified, include_source=True)",
                "Structured search across typed sub-queries with RRF fusion.",
                parent="CodeSearcher",
                language="python",
            )
            extraction = _Extraction(
                file_path=str(Path("engine/src/searcher.py").resolve()),
                language="python",
                symbols=[
                    _Symbol(
                        name="_rrf_fuse",
                        kind="function",
                        language="python",
                        file_path=str(Path("engine/src/searcher.py").resolve()),
                        line_start=520,
                        line_end=571,
                        byte_start=0,
                        byte_end=10,
                        source="def _rrf_fuse(self, results, weights): ...",
                        signature="_rrf_fuse(self, results, weights)",
                        parent="CodeSearcher",
                    ),
                    _Symbol(
                        name="_batch_search",
                        kind="function",
                        language="python",
                        file_path=str(Path("engine/src/searcher.py").resolve()),
                        line_start=839,
                        line_end=922,
                        byte_start=0,
                        byte_end=10,
                        source="def _batch_search(self, requests): ...",
                        signature="_batch_search(self, requests)",
                        parent="CodeSearcher",
                    ),
                    _Symbol(
                        name="_blend_with_rerank",
                        kind="function",
                        language="python",
                        file_path=str(Path("engine/src/searcher.py").resolve()),
                        line_start=573,
                        line_end=639,
                        byte_start=0,
                        byte_end=10,
                        source="def _blend_with_rerank(self, query, fused, limit): ...",
                        signature="_blend_with_rerank(self, query, fused, limit)",
                        parent="CodeSearcher",
                    ),
                    _Symbol(
                        name="_finalize_results",
                        kind="function",
                        language="python",
                        file_path=str(Path("engine/src/searcher.py").resolve()),
                        line_start=1173,
                        line_end=1179,
                        byte_start=0,
                        byte_end=10,
                        source="def _finalize_results(self, results, include_source): ...",
                        signature="_finalize_results(self, results, include_source)",
                        parent="CodeSearcher",
                    ),
                ],
            )
            with mock.patch.object(qc, "_load_file_symbols", return_value=extraction.symbols):
                expanded = qc._expand_symbol_context_results(
                    "How does CodeSearcher.search_structured fuse typed sub-queries?",
                    [anchor],
                    limit=4,
                )
        finally:
            qc.close()

        names = [item.symbol_name for item in expanded]
        self.assertEqual(names[:3], ["search_structured", "_rrf_fuse", "_batch_search"])
        self.assertNotIn("_finalize_results", names[:3])

    def test_semantic_search_bundle_can_skip_graph_related_expansion(self) -> None:
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
            with mock.patch.object(qc, "semantic_search", return_value=results), mock.patch.object(
                qc,
                "import_graph",
                side_effect=AssertionError("graph expansion should be skipped"),
            ), mock.patch.object(
                qc,
                "find_importers",
                side_effect=AssertionError("graph expansion should be skipped"),
            ):
                payload = qc.semantic_search_bundle(
                    "How does the Qdrant phase benchmark separate client-side latency from Qdrant server time?",
                    project_name="quickcontext",
                    limit=2,
                    related_file_limit=2,
                    include_graph_related=False,
                )
        finally:
            qc.close()

        self.assertEqual(len(payload["related_files"]), 1)
        self.assertIn(payload["related_files"][0]["file_path"], {"scripts/search_phase_benchmark.py", "scripts/retrieval_benchmark.py"})

    def test_semantic_search_bundle_hydrates_final_anchors_only(self) -> None:
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
                SearchResult(0.9, "engine/src/parsing.py", "import_graph", "function", 1, 10, "", ""),
                SearchResult(0.8, "engine/sdk.py", "import_graph", "function", 20, 30, "", ""),
            ]
            hydrated = [replace(results[0], source="def import_graph(...): ...")]
            with mock.patch.object(qc, "semantic_search", return_value=results) as semantic_search, mock.patch.object(
                qc,
                "_hydrate_search_result_sources",
                return_value=hydrated,
            ) as hydrate:
                payload = qc.semantic_search_bundle(
                    "Which files import this module and what dependencies does it have?",
                    project_name="quickcontext",
                    limit=1,
                    related_file_limit=1,
                    include_graph_related=False,
                )
        finally:
            qc.close()

        self.assertFalse(semantic_search.call_args.kwargs["include_source"])
        hydrate.assert_called_once()
        self.assertEqual(payload["results"], hydrated)

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

    def test_extract_compact_accepts_file_style_list_payload(self) -> None:
        service = RustParserService()
        try:
            service._client = mock.Mock()
            service._client.ensure_server = mock.Mock()
            service._client.extract.return_value = [
                {
                    "file_path": str(Path("engine/src/searcher.py").resolve()),
                    "language": "python",
                    "symbols": [
                        {
                            "name": "search_hybrid",
                            "kind": "function",
                            "language": "python",
                            "file_path": str(Path("engine/src/searcher.py").resolve()),
                            "line_start": 281,
                            "line_end": 380,
                            "signature": "search_hybrid(self, query, limit=10)",
                            "parent": "CodeSearcher",
                        }
                    ],
                    "symbol_count": 1,
                }
            ]

            results, stats = service.extract_compact("engine/src/searcher.py")
        finally:
            service.close()

        self.assertEqual(len(results), 1)
        self.assertEqual(type(results[0]).__name__, "CompactExtractionResult")
        self.assertEqual(results[0].symbols[0].name, "search_hybrid")
        self.assertEqual(type(stats).__name__, "ExtractStats")
        self.assertEqual(stats.total_files, 1)
        self.assertEqual(stats.total_symbols, 1)

    def test_warm_project_returns_transport_payload(self) -> None:
        service = RustParserService()
        try:
            service._client = mock.Mock()
            service._client.ensure_server = mock.Mock()
            service._client.warm_project.return_value = {
                "path": str(Path(".").resolve()),
                "symbol_count": 10,
                "text_doc_count": 5,
                "respect_gitignore": True,
            }
            payload = service.warm_project(".")
        finally:
            service.close()

        self.assertEqual(payload["symbol_count"], 10)
        self.assertEqual(payload["text_doc_count"], 5)

    def test_background_warm_starts_only_once(self) -> None:
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
            with mock.patch.object(qc, "_background_warm_once", return_value=None) as warm:
                started = qc.start_background_warm(".", idle_delay_seconds=0.01)
                started_again = qc.start_background_warm(".", idle_delay_seconds=0.01)
                for _ in range(20):
                    if warm.call_count:
                        break
                    time.sleep(0.01)
        finally:
            qc.close()

        self.assertTrue(started)
        self.assertFalse(started_again)
        warm.assert_called_once()

    def test_background_warm_waits_for_activity_to_finish(self) -> None:
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
            with mock.patch.object(qc, "_background_warm_once", return_value=None) as warm:
                with qc._activity_scope():
                    qc.start_background_warm(".", idle_delay_seconds=0.01)
                    time.sleep(0.03)
                    self.assertFalse(warm.called)
                time.sleep(0.05)
        finally:
            qc.close()

        warm.assert_called_once()

    def test_symbol_lookup_uses_requested_path_for_background_warm(self) -> None:
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
            qc._parser_service = mock.Mock()
            qc._parser_service.symbol_lookup.return_value = mock.sentinel.lookup
            with mock.patch.object(qc, "start_background_warm", return_value=True) as start:
                result = qc.symbol_lookup("CollectionManager", path="service")
        finally:
            qc.close()

        self.assertIs(result, mock.sentinel.lookup)
        start.assert_called_once_with(
            "service",
            idle_delay_seconds=qc._background_warm_auto_delay_seconds,
        )

    def test_context_manager_does_not_start_background_warm_implicitly(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=None,
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        with mock.patch.object(qc, "start_background_warm", return_value=True) as start:
            with qc:
                pass
        start.assert_not_called()

    def test_context_manager_does_not_connect_eagerly(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=QdrantConfig(),
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        with mock.patch.object(qc, "connect", return_value=qc) as connect:
            with qc:
                pass
        connect.assert_not_called()

    def test_background_warm_uses_dedicated_parser_service(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=None,
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        service = mock.Mock()
        with mock.patch("engine.sdk.RustParserService", return_value=service) as service_cls:
            qc._background_warm_once(".")

        service_cls.assert_called_once()
        service.warm_project.assert_called_once_with(path=".")
        service.close.assert_called_once()

    def test_collect_symbol_helper_results_keeps_helper_source(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=None,
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        anchor = SearchResult(
            score=1.0,
            file_path=str(Path("engine/src/searcher.py").resolve()),
            symbol_name="search_hybrid",
            symbol_kind="method",
            line_start=10,
            line_end=20,
            source="return self.search_code(query) + self.search_description(query)",
            description="hybrid search",
            parent="CodeSearcher",
            language="python",
        )
        helpers = [
            _Symbol(
                name="search_code",
                kind="method",
                language="python",
                file_path=anchor.file_path,
                line_start=30,
                line_end=35,
                byte_start=0,
                byte_end=10,
                source="def search_code(self, query):\n    return []",
                signature="search_code(self, query)",
                parent="CodeSearcher",
            ),
            _Symbol(
                name="search_description",
                kind="method",
                language="python",
                file_path=anchor.file_path,
                line_start=40,
                line_end=45,
                byte_start=0,
                byte_end=10,
                source="def search_description(self, query):\n    return []",
                signature="search_description(self, query)",
                parent="CodeSearcher",
            ),
        ]

        with mock.patch.object(qc, "_load_file_symbol_candidates", return_value=helpers):
            results = qc._collect_symbol_helper_results(
                query="How does CodeSearcher.search_hybrid merge code and description vectors?",
                anchor=anchor,
                helper_limit=2,
            )

        self.assertEqual(len(results), 2)
        self.assertIn("def search_code", results[0].source)
        self.assertIn("def search_description", results[1].source)

    def test_collect_symbol_helper_results_uses_reference_window_keywords(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=None,
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        anchor = SearchResult(
            score=1.0,
            file_path=str(Path("engine/src/searcher.py").resolve()),
            symbol_name="search_hybrid",
            symbol_kind="method",
            line_start=10,
            line_end=20,
            source=(
                "code_vectors = _batch_search(code_requests)\n"
                "description_vectors = _batch_search(desc_requests)\n"
                "return _blend_with_rerank(fused_vectors)\n"
            ),
            description="hybrid search",
            parent="CodeSearcher",
            language="python",
        )
        helpers = [
            _Symbol(
                name="_batch_search",
                kind="method",
                language="python",
                file_path=anchor.file_path,
                line_start=80,
                line_end=90,
                byte_start=0,
                byte_end=10,
                source="def _batch_search(self, requests):\n    return []",
                signature="_batch_search(self, requests)",
                parent="CodeSearcher",
            ),
            _Symbol(
                name="_blend_with_rerank",
                kind="method",
                language="python",
                file_path=anchor.file_path,
                line_start=40,
                line_end=50,
                byte_start=0,
                byte_end=10,
                source="def _blend_with_rerank(self, fused_vectors):\n    return fused_vectors",
                signature="_blend_with_rerank(self, fused_vectors)",
                parent="CodeSearcher",
            ),
        ]

        with mock.patch.object(qc, "_load_file_symbol_candidates", return_value=helpers):
            results = qc._collect_symbol_helper_results(
                query="How does CodeSearcher.search_hybrid merge code and description vectors?",
                anchor=anchor,
                helper_limit=2,
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].symbol_name, "_batch_search")

    def test_collect_symbol_helper_results_can_use_index_file_symbols(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=None,
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        anchor = SearchResult(
            score=1.0,
            file_path=str(Path("engine/src/searcher.py").resolve()),
            symbol_name="search_hybrid",
            symbol_kind="method",
            line_start=10,
            line_end=20,
            source="return _batch_search(code_requests) + _rrf_fuse(result_lists)",
            description="hybrid search",
            parent="CodeSearcher",
            language="python",
        )
        qc._parser_service = mock.Mock()
        qc._parser_service.file_symbols.return_value = SymbolLookupResult(
            project_root=str(Path(".").resolve()),
            results=[
                SymbolLookupItem(
                    name="_batch_search",
                    kind="method",
                    language="python",
                    file_path=anchor.file_path,
                    line_start=30,
                    line_end=40,
                    parent="CodeSearcher",
                    signature="_batch_search(self, requests)",
                ),
                SymbolLookupItem(
                    name="_rrf_fuse",
                    kind="method",
                    language="python",
                    file_path=anchor.file_path,
                    line_start=50,
                    line_end=60,
                    parent="CodeSearcher",
                    signature="_rrf_fuse(self, lists)",
                ),
            ],
            indexed_files=1,
            indexed_symbols=2,
            from_cache=True,
        )

        with mock.patch.object(qc, "_load_file_symbols", side_effect=AssertionError("fallback extraction should not run")), \
             mock.patch.object(qc, "_read_symbol_context", side_effect=lambda item: (f"source for {getattr(item, 'name', getattr(item, 'symbol_name', ''))}", item.signature, None)):
            results = qc._collect_symbol_helper_results(
                query="How does CodeSearcher.search_hybrid merge code and description vectors?",
                anchor=anchor,
                helper_limit=2,
            )

        self.assertEqual({item.symbol_name for item in results}, {"_batch_search", "_rrf_fuse"})
        self.assertEqual({item.source for item in results}, {"source for _batch_search", "source for _rrf_fuse"})
        qc._parser_service.file_symbols.assert_called_once()

    def test_load_file_symbol_candidates_falls_back_to_extract(self) -> None:
        qc = QuickContext(
            EngineConfig(
                qdrant=None,
                code_embedding=None,
                desc_embedding=None,
                llm=None,
                vectors=[],
            )
        )
        qc._parser_service = mock.Mock()
        qc._parser_service.file_symbols.side_effect = RuntimeError("index unavailable")
        helpers = [_Symbol(
            name="_batch_search",
            kind="method",
            language="python",
            file_path=str(Path("engine/src/searcher.py").resolve()),
            line_start=30,
            line_end=40,
            byte_start=0,
            byte_end=10,
            source="def _batch_search(self, requests):\n    return []",
            signature="_batch_search(self, requests)",
            parent="CodeSearcher",
        )]

        with mock.patch.object(qc, "_load_file_symbols", return_value=helpers) as fallback:
            loaded = qc._load_file_symbol_candidates(str(Path("engine/src/searcher.py").resolve()))

        self.assertEqual(loaded, helpers)
        fallback.assert_called_once()


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
