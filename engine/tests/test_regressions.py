import importlib
import os
import tempfile
import time
import unittest
from unittest import mock
from dataclasses import dataclass
from pathlib import Path
import builtins

from engine.src.chunker import ChunkBuilder, CodeChunk
from engine.src.config import EngineConfig, QdrantConfig
from engine.src.dedup import deduplicate_chunks
from engine.src.describer import build_fallback_description
from engine.src.filecache import FileSignatureCache
from engine.src.cli import _optimize_search_config
from engine.src.qdrant import QdrantConnection
from engine.src.searcher import CodeSearcher, LIGHT_RESULT_PAYLOAD_FIELDS


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

    def query_points(self, **kwargs):
        self.last_query_with_payload = kwargs.get("with_payload")
        return _FakeQueryResponse(self._points)

    def query_batch_points(self, collection_name, requests):
        return [_FakeQueryResponse(self._points) for _ in requests]

    def retrieve(self, **kwargs):
        self.last_retrieve_with_payload = kwargs.get("with_payload")
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
        must_conditions = query_filter.must or []
        self.assertTrue(any(getattr(cond, "key", None) == "path_prefixes" for cond in must_conditions))

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
            file_hash="hash",
        )

        description = build_fallback_description(chunk)
        self.assertIn("Cache query embedding vectors", description.description)
        self.assertIn("engine/src/searcher.py", description.description.replace("\\", "/"))
        self.assertIn("query", description.keywords)
        self.assertIn("searcher", description.keywords)

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
