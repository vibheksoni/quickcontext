from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

from engine.src.config import EmbeddingConfig


class EmbeddingProvider(ABC):
    """
    Abstract base for all embedding providers.

    _config: EmbeddingConfig — Provider configuration.
    """

    def __init__(self, config: EmbeddingConfig):
        """
        config: EmbeddingConfig — Embedding provider settings.
        """
        self._config = config

    @property
    def model(self) -> str:
        """
        Returns: str — Model identifier.
        """
        return self._config.model

    @property
    def dimension(self) -> int:
        """
        Returns: int — Output embedding dimensionality.
        """
        return self._config.dimension

    @property
    def query_prefix(self) -> str:
        """
        Returns: str — Instruction prepended to search queries ("" when unset).
        """
        return self._config.query_prefix or ""

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a search query, applying the configured query instruction.

        text: str — Query text.
        Returns: np.ndarray — Query vector.
        """
        return self.embed_single(self.query_prefix + text)

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        """
        Generate embeddings for a batch of texts.

        texts: Sequence[str] — Input text strings.
        Returns: list[np.ndarray] — One embedding vector per input text.
        """

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        text: str — Input text string.
        Returns: np.ndarray — Embedding vector.
        """
