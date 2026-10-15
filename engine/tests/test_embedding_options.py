import os
import unittest
from unittest import mock

import numpy as np

from engine.src.config import EmbeddingConfig, EngineConfig
from engine.src.embedder import estimate_tokens, plan_remote_batches
from engine.src.providers import local_models
from engine.src.providers.base import EmbeddingProvider
from engine.src.searcher import embed_query


REMOTE = {
    "provider": "litellm",
    "model": "openrouter/qwen/qwen3-embedding-8b",
    "dimension": 4096,
    "api_key": "secret",
    "query_prefix": "Instruct: find code\nQuery:",
}
LOCAL = {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5", "dimension": 384}


def _config(**extra) -> dict:
    data = {"code_embedding": dict(REMOTE), "desc_embedding": dict(REMOTE), "llm": None}
    data.update(extra)
    return data


class _RecordingProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.seen: list[str] = []

    def embed(self, texts):
        return [self.embed_single(text) for text in texts]

    def embed_single(self, text):
        self.seen.append(text)
        return np.zeros(self.dimension, dtype=np.float32)


class PlanRemoteBatchesTests(unittest.TestCase):
    def test_every_index_planned_once_longest_first(self):
        texts = ["x" * size for size in (10, 4000, 50, 900, 3, 2000, 70)]
        batches = plan_remote_batches(texts, batch_size=3, token_budget=None)
        flat = [idx for batch in batches for idx in batch]
        self.assertEqual(sorted(flat), list(range(len(texts))))
        self.assertEqual(flat[0], 1)
        self.assertTrue(all(len(batch) <= 3 for batch in batches))

    def test_token_budget_splits_heavy_requests(self):
        texts = ["x" * 4000] * 4 + ["y" * 40] * 8
        batches = plan_remote_batches(texts, batch_size=32, token_budget=2048)
        for batch in batches:
            tokens = sum(estimate_tokens(texts[idx]) for idx in batch)
            self.assertTrue(len(batch) == 1 or tokens <= 2048)
        self.assertGreater(len(batches), 1)

    def test_oversized_text_gets_own_request(self):
        texts = ["x" * 40000, "short"]
        batches = plan_remote_batches(texts, batch_size=8, token_budget=1000)
        self.assertEqual(batches, [[0], [1]])

    def test_empty_input(self):
        self.assertEqual(plan_remote_batches([], batch_size=8, token_budget=1000), [])


class ProfileTests(unittest.TestCase):
    def test_no_profile_keeps_base_and_lists_names(self):
        config = EngineConfig.from_dict(_config(profiles={"local": {}}))
        self.assertIsNone(config.profile)
        self.assertEqual(tuple(config.available_profiles), ("local",))
        self.assertEqual(config.code_embedding.provider, "litellm")

    def test_model_swap_replaces_section_and_rederives_vectors(self):
        data = _config(
            vectors=[{"name": "code", "dimension": 4096}, {"name": "description", "dimension": 4096}],
            profiles={"local": {"code_embedding": dict(LOCAL), "desc_embedding": dict(LOCAL)}},
        )
        config = EngineConfig.from_dict(data, profile="local")
        self.assertEqual(config.profile, "local")
        self.assertEqual(config.code_embedding.provider, "fastembed")
        self.assertIsNone(config.code_embedding.api_key)
        self.assertIsNone(config.code_embedding.query_prefix)
        self.assertEqual({v.name: v.dimension for v in config.vectors}, {"code": 384, "description": 384})
        self.assertTrue(config.store.path.endswith("store-local.db"))

    def test_tuning_override_merges_into_base_section(self):
        data = _config(profiles={"fast": {"code_embedding": {"concurrency": 64}}})
        config = EngineConfig.from_dict(data, profile="fast")
        self.assertEqual(config.code_embedding.concurrency, 64)
        self.assertEqual(config.code_embedding.api_key, "secret")
        self.assertEqual(config.code_embedding.query_prefix, REMOTE["query_prefix"])

    def test_profile_selected_by_key_and_env(self):
        data = _config(profile="local", profiles={"local": {"code_embedding": dict(LOCAL)}, "remote": {}})
        self.assertEqual(EngineConfig.from_dict(data).profile, "local")
        with mock.patch.dict(os.environ, {"QC_PROFILE": "remote"}):
            self.assertEqual(EngineConfig.from_dict(data).profile, "remote")
        self.assertEqual(EngineConfig.from_dict(data, profile="remote").profile, "remote")

    def test_explicit_store_path_is_kept(self):
        data = _config(store={"path": "/tmp/custom.db"}, profiles={"local": {"code_embedding": dict(LOCAL)}})
        self.assertEqual(EngineConfig.from_dict(data, profile="local").store.path, "/tmp/custom.db")

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError):
            EngineConfig.from_dict(_config(profiles={"local": {}}), profile="missing")


class QueryPrefixTests(unittest.TestCase):
    def test_prefix_applies_to_queries_only(self):
        provider = _RecordingProvider(EmbeddingConfig(**{**LOCAL, "dimension": 4, "query_prefix": "Q: "}))
        embed_query(provider, "where is auth")
        provider.embed(["def auth(): pass"])
        self.assertEqual(provider.seen, ["Q: where is auth", "def auth(): pass"])

    def test_no_prefix_and_providers_without_embed_query(self):
        provider = _RecordingProvider(EmbeddingConfig(**{**LOCAL, "dimension": 4}))
        self.assertEqual(provider.query_prefix, "")
        embed_query(provider, "plain")
        self.assertEqual(provider.seen, ["plain"])

        class Bare:
            def embed_single(self, text):
                return text

        self.assertEqual(embed_query(Bare(), "plain"), "plain")


class LocalModelTests(unittest.TestCase):
    def test_device_candidates(self):
        self.assertEqual(local_models._device_candidates("cpu"), ["cpu"])
        self.assertEqual(local_models._device_candidates("directml"), ["directml", "cpu"])
        with mock.patch.object(local_models, "available_providers", return_value=["DmlExecutionProvider", "CPUExecutionProvider"]):
            self.assertEqual(local_models._device_candidates("auto"), ["directml", "cpu"])
        with mock.patch.object(local_models, "available_providers", return_value=["CPUExecutionProvider"]):
            self.assertEqual(local_models._device_candidates("gpu"), ["cpu"])
        with self.assertRaises(ValueError):
            local_models._device_candidates("tpu")

    def test_gpu_failure_falls_back_to_cpu_and_models_are_shared(self):
        created = []

        def fake_create(model_name, device):
            if device == "cuda":
                raise RuntimeError("no CUDA")
            created.append((model_name, device))
            return object()

        with mock.patch.object(local_models, "_create", side_effect=fake_create), \
                mock.patch.dict(local_models._MODELS, clear=True):
            first, device = local_models.get_local_model("m", "cuda")
            second, _ = local_models.get_local_model("m", "cpu")
        self.assertEqual(device, "cpu")
        self.assertIs(first, second)
        self.assertEqual(created, [("m", "cpu")])

    def test_effective_batch_size(self):
        self.assertEqual(local_models.effective_batch_size(16, "cpu"), 16)
        self.assertEqual(local_models.effective_batch_size(16, "directml"), local_models.GPU_MIN_BATCH_SIZE)
        self.assertEqual(local_models.effective_batch_size(256, "cuda"), 256)


if __name__ == "__main__":
    unittest.main()
