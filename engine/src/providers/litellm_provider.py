import os
from typing import Sequence

import numpy as np

from engine.src.config import EmbeddingConfig
from engine.src.providers.base import EmbeddingProvider


def _get_litellm_embedding():
    """
    Lazy-import litellm embedding entrypoint and suppress debug output.

    Returns: callable — litellm.embedding function.
    """
    import litellm
    from litellm import embedding as litellm_embedding

    os.environ.setdefault("LITELLM_LOG", "ERROR")
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    return litellm_embedding


class LiteLLMProvider(EmbeddingProvider):
    """
    Cloud embedding provider via litellm unified gateway.
    Supports OpenAI, Cohere, Voyage, Azure, Bedrock, HuggingFace, vLLM, Infinity.

    API keys resolved in order:
        1. EmbeddingConfig.api_key (passed directly to litellm)
        2. Environment variables (OPENAI_API_KEY, COHERE_API_KEY, etc.)
    """

    def __init__(self, config: EmbeddingConfig):
        """
        config: EmbeddingConfig — Must have provider="litellm".
            model format: "openai/text-embedding-3-small", "cohere/embed-english-v3.0", etc.
            api_key: Optional override for provider API key.
            api_base: Optional custom endpoint for self-hosted models.
        """
        super().__init__(config)

    def _call_litellm(self, texts: list[str]) -> list[np.ndarray]:
        """
        Call litellm.embedding() with configured parameters.

        texts: list[str] — Input texts to embed.
        Returns: list[np.ndarray] — Embedding vectors.
        """
        kwargs = {
            "model": self._config.model,
            "input": texts,
        }

        if self._config.api_key:
            kwargs["api_key"] = self._config.api_key

        if self._config.api_base:
            kwargs["api_base"] = self._config.api_base

        if self._config.request_timeout_seconds is not None:
            kwargs["timeout"] = float(self._config.request_timeout_seconds)

        if self._config.openrouter_provider:
            kwargs["extra_body"] = {"provider": {"order": [self._config.openrouter_provider]}}

        response = _get_litellm_embedding()(**kwargs)

        vectors = []
        for item in response.data:
            vectors.append(np.array(item["embedding"], dtype=np.float32))

        return vectors

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        """
        Generate embeddings for a batch of texts via litellm.
        Chunks into batch_size groups to avoid API limits.

        texts: Sequence[str] — Input text strings.
        Returns: list[np.ndarray] — One embedding vector per input text.
        """
        text_list = list(texts)
        all_vectors: list[np.ndarray] = []

        for i in range(0, len(text_list), self._config.batch_size):
            batch = text_list[i:i + self._config.batch_size]
            all_vectors.extend(self._call_litellm(batch))

        return all_vectors

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text via litellm.

        text: str — Input text string.
        Returns: np.ndarray — Embedding vector.
        """
        results = self._call_litellm([text])
        return results[0]
