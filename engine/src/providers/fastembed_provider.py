from typing import Sequence
import numpy as np

from engine.src.config import EmbeddingConfig
from engine.src.providers.base import EmbeddingProvider


class FastEmbedProvider(EmbeddingProvider):
    """
    Local ONNX-based embedding provider via fastembed.
    No API keys required — fully offline.

    _model: TextEmbedding — Loaded fastembed model instance.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        config: EmbeddingConfig — Must have provider="fastembed".
        """
        super().__init__(config)
        self._model = None
        self._device = "cpu"

    def _ensure_model(self):
        """
        Lazy-load the fastembed model on first use.

        Returns: TextEmbedding — Loaded model instance.
        """
        if self._model is None:
            from engine.src.providers.local_models import get_local_model

            self._model, self._device = get_local_model(
                self._config.model,
                device=self._config.device,
                max_input_tokens=self._config.max_input_tokens,
            )
        return self._model

    @property
    def device(self) -> str:
        """
        Returns: str — Resolved execution device ("cpu", "cuda", or "directml").
        """
        self._ensure_model()
        return self._device

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        """
        Generate embeddings for a batch of texts using fastembed.

        texts: Sequence[str] — Input text strings.
        Returns: list[np.ndarray] — One embedding vector per input text.
        """
        from engine.src.embedder import embed_length_sorted
        from engine.src.providers.local_models import effective_batch_size

        model = self._ensure_model()
        return embed_length_sorted(model, list(texts), effective_batch_size(self._config.batch_size, self._device))

    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text using fastembed.

        text: str — Input text string.
        Returns: np.ndarray — Embedding vector.
        """
        model = self._ensure_model()
        results = list(model.embed(documents=[text], batch_size=1))
        return results[0]
