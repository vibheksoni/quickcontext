import os
import random
import tempfile
import unittest

import numpy as np

from engine import QuickContext
from engine.src import vector_store as vs
from engine.src.collection import CollectionManager, list_projects, orphaned_versions
from engine.src.config import CollectionVectorConfig, EmbeddingConfig, EngineConfig, StoreConfig
from engine.src.vector_store import (
    CollectionNotFoundError,
    PointRecord,
    SearchFilter,
    SearchRequest,
    VectorStore,
)


SPACES = [CollectionVectorConfig("code", 8, "cosine"), CollectionVectorConfig("description", 4, "cosine")]


def _point(idx: int, rng: np.random.Generator, **payload) -> PointRecord:
    base = {"file_path": f"C:\\repo\\src\\{'api' if idx % 2 else 'core'}\\f{idx % 5}.py", "language": "python" if idx % 3 else "rust"}
    base.update(payload)
    return PointRecord(
        id=f"p{idx}",
        payload=base,
        vectors={"code": rng.standard_normal(8), "description": rng.standard_normal(4)},
    )


class _HashProvider:
    model = "hash"
    dimension = 32

    def _vec(self, text: str) -> np.ndarray:
        vector = np.zeros(32, dtype=np.float32)
        for token in text.lower().split():
            vector[sum(map(ord, token)) % 32] += 1
        return vector

    def embed(self, texts):
        return [self._vec(text) for text in texts]

    def embed_single(self, text):
        return self._vec(text)


class VectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.config = StoreConfig(path=os.path.join(self._tmp.name, "store.db"))
        self.store = VectorStore(self.config).open()
        self.store.create_collection("demo_v1", SPACES)
        self.store.set_alias("demo", "demo_v1")
        self.rng = np.random.default_rng(7)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()

    def test_search_matches_brute_force_cosine_ranking(self) -> None:
        points = [_point(idx, self.rng, source=f"def f{idx}(): ...") for idx in range(200)]
        self.store.upsert("demo", points)
        query = self.rng.standard_normal(8)

        hits = self.store.search("demo", SearchRequest(vector=query, using="code", limit=10, with_payload=["file_path"]))

        matrix = np.array([p.vectors["code"] for p in points], dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        expected = np.argsort(-(matrix @ (query / np.linalg.norm(query))))[:10]
        self.assertEqual([hit.id for hit in hits], [f"p{idx}" for idx in expected])
        self.assertEqual(set(hits[0].payload), {"file_path"})

    def test_text_field_is_only_loaded_when_requested(self) -> None:
        self.store.upsert("demo", [_point(1, self.rng, source="body")])
        light = self.store.retrieve("demo", ["p1"], with_payload=["file_path", "language"])
        full = self.store.retrieve("demo", ["p1"])
        self.assertNotIn("source", light[0].payload)
        self.assertEqual(full[0].payload["source"], "body")

    def test_filters_combine_keywords_and_relative_path_scope(self) -> None:
        self.store.upsert("demo", [_point(idx, self.rng) for idx in range(30)])
        flt = SearchFilter.build(path_prefix="src/api", language="python")
        hits = self.store.search("demo", SearchRequest(vector=self.rng.standard_normal(8), using="code", limit=50, filter=flt))
        self.assertTrue(hits)
        for hit in hits:
            self.assertIn("\\api\\", hit.payload["file_path"])
            self.assertEqual(hit.payload["language"], "python")
        self.assertEqual(self.store.count("demo", flt), len(hits))

    def test_dimension_mismatch_raises_clear_error(self) -> None:
        bad = PointRecord(id="x", payload={}, vectors={"code": [1.0, 2.0], "description": [1, 2, 3, 4]})
        with self.assertRaises(ValueError):
            self.store.upsert("demo", [bad])
        with self.assertRaises(ValueError):
            self.store.search("demo", SearchRequest(vector=[1.0, 2.0], using="code", limit=1))

    def test_missing_collection_raises_not_found(self) -> None:
        with self.assertRaises(CollectionNotFoundError) as ctx:
            self.store.search("nope", SearchRequest(vector=[0.0] * 8, using="code", limit=1))
        self.assertIn("collection not found", str(ctx.exception).lower())

    def test_second_instance_sees_incremental_changes(self) -> None:
        self.store.upsert("demo", [_point(idx, self.rng) for idx in range(50)])
        reader = VectorStore(self.config).open()
        try:
            self.assertEqual(reader.warm("demo"), 50)
            removed = self.store.delete_by_files("demo", ["C:\\repo\\src\\core\\f0.py"])
            target = np.ones(8)
            self.store.upsert("demo", [PointRecord("new", {"file_path": "n.py"}, {"code": target, "description": np.ones(4)})])
            hits = reader.search("demo", SearchRequest(vector=target, using="code", limit=100, with_payload=False))
            self.assertEqual(hits[0].id, "new")
            self.assertAlmostEqual(hits[0].score, 1.0, places=5)
            self.assertEqual(len(hits), 50 - removed + 1)
        finally:
            reader.close()

    def test_randomized_writes_stay_consistent_across_instances(self) -> None:
        original_threshold = vs.TOMBSTONE_PRUNE_THRESHOLD
        vs.TOMBSTONE_PRUNE_THRESHOLD = 100
        reader = VectorStore(self.config).open()
        try:
            rng = random.Random(3)
            truth: dict[str, tuple[np.ndarray, str]] = {}
            for step in range(150):
                roll = rng.random()
                if roll < 0.6:
                    batch = []
                    for _ in range(rng.randint(1, 20)):
                        pid = f"p{rng.randint(0, 200)}"
                        vec = self.rng.standard_normal(8).astype(np.float32)
                        lang = rng.choice(["py", "rs"])
                        batch.append(PointRecord(pid, {"language": lang}, {"code": vec, "description": np.ones(4)}))
                        truth[pid] = (vec / np.linalg.norm(vec), lang)
                    self.store.upsert("demo", batch)
                elif roll < 0.85 and truth:
                    ids = rng.sample(sorted(truth), min(len(truth), rng.randint(1, 15)))
                    self.store.delete_points("demo", ids)
                    for pid in ids:
                        truth.pop(pid)
                else:
                    self.store.delete_by_filter("demo", SearchFilter.build(language="rs"))
                    for pid in [key for key, value in truth.items() if value[1] == "rs"]:
                        truth.pop(pid)
                if step % 3 == 0:
                    query = self.rng.standard_normal(8).astype(np.float32)
                    got = [hit.id for hit in reader.search("demo", SearchRequest(query, "code", 5, with_payload=False))]
                    ranked = sorted(truth.items(), key=lambda item: -float(item[1][0] @ (query / np.linalg.norm(query))))
                    self.assertEqual(got, [pid for pid, _ in ranked[:5]], f"step {step}")
        finally:
            vs.TOMBSTONE_PRUNE_THRESHOLD = original_threshold
            reader.close()

    def test_copy_and_alias_swap(self) -> None:
        self.store.upsert("demo", [_point(idx, self.rng) for idx in range(10)])
        self.store.create_collection("demo_v2", SPACES)
        self.assertEqual(self.store.copy_points("demo", "demo_v2"), 10)
        self.assertEqual(self.store.set_alias("demo", "demo_v2"), "demo_v1")
        self.assertEqual(orphaned_versions(self.store), ["demo_v1"])
        self.assertEqual(self.store.get_collection("demo").points_count, 10)
        self.assertEqual(list_projects(self.store)[0]["real_collection"], "demo_v2")

    def test_collection_manager_ensure_adopts_newest_version(self) -> None:
        self.store.delete_alias("demo")
        self.store.create_collection("demo_v4", SPACES)
        config = EngineConfig(store=self.config, vectors=SPACES)
        manager = CollectionManager(store=self.store, config=config, collection_name="demo")
        self.assertTrue(manager.ensure())
        self.assertEqual(manager.resolve_alias(), "demo_v4")
        self.assertEqual(manager.create_shadow(), "demo_v5")

    def test_file_states_report_hash_and_counts(self) -> None:
        self.store.upsert("demo", [_point(idx, self.rng, file_hash="h") for idx in range(10)])
        states = self.store.file_states("demo", ["C:\\repo\\src\\api\\f1.py", "missing.py"])
        self.assertEqual(states, {"C:\\repo\\src\\api\\f1.py": ("h", 1)})


class DocumentStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        config = EngineConfig(
            store=StoreConfig(path=":memory:"),
            code_embedding=None,
            desc_embedding=EmbeddingConfig(dimension=32),
            llm=None,
        )
        self.qc = QuickContext(config)
        self.qc._desc_provider = _HashProvider()

    def tearDown(self) -> None:
        self.qc.close()

    def test_collect_recall_forget_roundtrip(self) -> None:
        ids = self.qc.collect("notes", [
            "Webhook deliveries retry with exponential backoff",
            {"text": "Auth tokens are cached in Redis for 15 minutes", "tags": ["auth"], "source": "docs/auth.md", "priority": 2},
            {"text": "Login page uses server components", "kind": "finding"},
        ], tags="sprint")

        hits = self.qc.recall("notes", "how long are auth tokens cached")
        self.assertEqual(hits[0].id, ids[1])
        self.assertEqual(hits[0].source, "docs/auth.md")
        self.assertEqual(hits[0].metadata, {"priority": 2})
        self.assertEqual(set(hits[0].tags), {"sprint", "auth"})

        self.assertEqual([hit.id for hit in self.qc.recall("notes", "page", kind="finding", mode="keyword")], [ids[2]])
        self.assertEqual([hit.id for hit in self.qc.recall("notes", "tokens", where={"priority": 2})], [ids[1]])
        self.assertEqual(self.qc.collect("notes", "Webhook deliveries retry with exponential backoff"), [ids[0]])
        self.assertEqual(self.qc.documents.count("notes"), 3)

        self.assertEqual(self.qc.forget("notes", tags="auth"), 1)
        self.assertEqual(self.qc.forget("notes", ids[0]), 1)
        self.assertEqual([doc.id for doc in self.qc.documents.list("notes")], [ids[2]])

    def test_keyword_search_matches_source_and_tags(self) -> None:
        self.qc.collect("notes", {"text": "Retry budget is three attempts", "source": "runbooks/payments.md", "tags": ["billing"]})
        self.assertEqual(len(self.qc.recall("notes", "payments runbook", mode="keyword")), 1)
        self.assertEqual(len(self.qc.recall("notes", "billing", mode="keyword")), 1)

    def test_dedupe_threshold_merges_near_duplicates(self) -> None:
        first = self.qc.collect("notes", "cache tokens in redis", tags="a")
        second = self.qc.collect("notes", "Cache tokens in Redis", tags="b", dedupe_threshold=0.95)
        self.assertEqual(first, second)
        self.assertEqual(self.qc.documents.count("notes"), 1)
        self.assertEqual(set(self.qc.documents.get("notes", first)[0].tags), {"a", "b"})

    def test_recency_half_life_prefers_newer_documents(self) -> None:
        old_id, new_id = self.qc.collect("notes", ["deploy uses blue green", "deploy uses blue green rollout"])
        stale = self.qc.documents.get("notes", old_id)[0]
        payload = {"text": stale.text, "tags": stale.tags, "added_at": stale.added_at - 30 * 86400}
        self.qc.store.upsert("notes", [vs.PointRecord(old_id, payload, {"text": _HashProvider()._vec(stale.text)})])
        hits = self.qc.recall("notes", "deploy uses blue green", half_life_days=7)
        self.assertEqual(hits[0].id, new_id)

    def test_mmr_diversity_skips_redundant_hits(self) -> None:
        self.qc.collect("notes", [
            "redis cache ttl",
            "redis cache ttl again",
            "postgres connection pool size",
        ])
        plain = [doc.text for doc in self.qc.recall("notes", "redis cache ttl pool", limit=2, mode="semantic")]
        diverse = [doc.text for doc in self.qc.recall("notes", "redis cache ttl pool", limit=2, mode="semantic", diversity=0.7)]
        self.assertTrue(all("redis" in text for text in plain))
        self.assertIn("postgres connection pool size", diverse)

    def test_keyword_only_collection_needs_no_embeddings(self) -> None:
        self.qc.documents.create("scratch", semantic=False)
        self.qc.collect("scratch", ["alpha beta", "gamma delta"])
        self.assertEqual(self.qc.recall("scratch", "gamma")[0].text, "gamma delta")
        self.assertEqual(self.qc.documents.collections()[0]["vectors"], {})

    def test_document_api_refuses_code_collections(self) -> None:
        self.qc.store.create_collection("project_v1", SPACES)
        with self.assertRaises(ValueError):
            self.qc.documents.drop("project_v1")
        with self.assertRaises(ValueError):
            self.qc.collect("project_v1", "text")


if __name__ == "__main__":
    unittest.main()
