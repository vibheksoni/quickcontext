import importlib
import os
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
import builtins

from engine.src.chunker import ChunkBuilder, CodeChunk
from engine.src.dedup import deduplicate_chunks
from engine.src.filecache import FileSignatureCache
from engine.src.searcher import CodeSearcher


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
            query_keywords=["engine", "src"],
            rerank=False,
        )
        self.assertEqual(normalized_prefix, "engine/src")
        self.assertGreaterEqual(fetch_limit, 50)
        self.assertIsNotNone(query_filter)
        must_conditions = query_filter.must or []
        self.assertTrue(any(getattr(cond, "key", None) == "path_prefixes" for cond in must_conditions))


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
