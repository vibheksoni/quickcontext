from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
import json


CONFIG_FILENAMES = ["quickcontext.json", ".quickcontext.json"]


@dataclass(frozen=True, slots=True)
class QdrantConfig:
    """
    Qdrant vector database connection configuration.

    host: str — Qdrant server hostname.
    port: int — Qdrant HTTP API port.
    grpc_port: int — Qdrant gRPC API port.
    prefer_grpc: bool — Use gRPC transport when available.
    collection: Optional[str] — Default collection name (deprecated, use project-based naming).
    timeout: float — Request timeout in seconds.
    api_key: Optional[str] — Qdrant Cloud API key (None for local).
    upsert_batch_size: int — Number of points per upsert request.
    upsert_concurrency: int — Number of concurrent upsert workers.
    """

    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = True
    collection: Optional[str] = None
    timeout: float = 30.0
    api_key: Optional[str] = None
    upsert_batch_size: int = 100
    upsert_concurrency: int = 1

    @property
    def url(self) -> str:
        """
        url: str — Full HTTP URL for the Qdrant instance.
        """
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """
    Embedding provider configuration.

    provider: str — Provider backend ("fastembed" or "litellm").
    model: str — Model identifier.
        fastembed: "jinaai/jina-embeddings-v2-base-code", "BAAI/bge-small-en-v1.5", etc.
        litellm: "openai/text-embedding-3-small", "cohere/embed-english-v3.0",
                 "voyage/voyage-3.5", "bedrock/amazon.titan-embed-text-v1", etc.
    dimension: int — Output embedding dimension.
    batch_size: int — Batch size for embedding generation.
    min_batch_size: int — Minimum adaptive batch size when shrinking on errors.
    max_batch_size: int — Maximum adaptive batch size when growing on stable responses.
    adaptive_batching: bool — Enable adaptive batch-size controller for litellm embeddings.
    adaptive_target_latency_ms: int — Target latency threshold for adaptive resizing decisions.
    concurrency: int — Max concurrent embedding request workers.
    max_retries: int — Max retries per failed embedding request.
    retry_base_delay_ms: int — Exponential backoff base delay in milliseconds.
    retry_max_delay_ms: int — Exponential backoff max delay in milliseconds.
    request_timeout_seconds: Optional[float] — Optional per-request timeout.
    api_key: Optional[str] — API key for cloud providers (litellm only).
        Falls back to env vars: OPENAI_API_KEY, COHERE_API_KEY, VOYAGE_API_KEY, etc.
    api_base: Optional[str] — Custom API endpoint URL (litellm only).
        For self-hosted models (vLLM, Infinity, HuggingFace TEI).
    openrouter_provider: Optional[str] — Force specific OpenRouter upstream (e.g. "Nebius", "Together").
    """

    provider: str = "fastembed"
    model: str = "jinaai/jina-embeddings-v2-base-code"
    dimension: int = 768
    batch_size: int = 32
    min_batch_size: int = 8
    max_batch_size: int = 128
    adaptive_batching: bool = True
    adaptive_target_latency_ms: int = 1500
    concurrency: int = 4
    max_retries: int = 3
    retry_base_delay_ms: int = 250
    retry_max_delay_ms: int = 4000
    request_timeout_seconds: Optional[float] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    openrouter_provider: Optional[str] = None


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """
    LLM provider configuration for code description generation.

    provider: str — Provider backend ("litellm").
    model: str — Model identifier (litellm format, e.g. "openrouter/openai/gpt-oss-20b").
    max_tokens: int — Max output tokens per completion.
    temperature: float — Sampling temperature (0.0 = deterministic).
    api_key: Optional[str] — API key for the provider.
    api_base: Optional[str] — Custom API endpoint URL.
    openrouter_provider: Optional[str] — Force specific OpenRouter upstream (e.g. "Groq", "DeepInfra").
    """

    provider: str = "litellm"
    model: str = "openrouter/openai/gpt-oss-20b"
    max_tokens: int = 256
    temperature: float = 0.0
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    openrouter_provider: Optional[str] = None
    artifact_metadata_enabled: bool = False
    artifact_metadata_batch_size: int = 4
    artifact_metadata_max_tokens: int = 512
    artifact_metadata_chunks_per_file: int = 2


@dataclass(frozen=True, slots=True)
class CollectionVectorConfig:
    """
    Named vector space configuration for a Qdrant collection.

    name: str — Vector space name.
    dimension: int — Vector dimensionality.
    distance: str — Distance metric ("cosine", "euclid", "dot").
    """

    name: str = "code"
    dimension: int = 768
    distance: str = "cosine"


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """
    Top-level engine configuration. All subsystems are optional.
    Set a field to None to disable that subsystem entirely.

    qdrant: Optional[QdrantConfig] — Qdrant connection settings. None = no vector DB.
    code_embedding: Optional[EmbeddingConfig] — Embedding config for code vectors. None = no code embeddings.
    desc_embedding: Optional[EmbeddingConfig] — Embedding config for description vectors. None = no desc embeddings.
    llm: Optional[LLMConfig] — LLM config for code description generation. None = no LLM descriptions.
    vectors: list[CollectionVectorConfig] — Named vector spaces for the collection.
    """

    qdrant: Optional[QdrantConfig] = field(default_factory=QdrantConfig)
    code_embedding: Optional[EmbeddingConfig] = field(default_factory=EmbeddingConfig)
    desc_embedding: Optional[EmbeddingConfig] = field(
        default_factory=lambda: EmbeddingConfig(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
    )
    llm: Optional[LLMConfig] = field(default_factory=LLMConfig)
    vectors: list[CollectionVectorConfig] = field(
        default_factory=lambda: [
            CollectionVectorConfig(name="code", dimension=768, distance="cosine"),
            CollectionVectorConfig(name="description", dimension=384, distance="cosine"),
        ]
    )

    @staticmethod
    def from_dict(data: dict) -> "EngineConfig":
        """
        Build EngineConfig from a flat or nested dictionary.

        data: dict — Configuration dictionary. Supports nested keys:
            qdrant.host, qdrant.port, code_embedding.provider, etc.
        Returns: EngineConfig — Fully constructed config.
        """
        _sentinel = object()

        qdrant_raw = data.get("qdrant", _sentinel)
        code_raw = data.get("code_embedding", _sentinel)
        desc_raw = data.get("desc_embedding", _sentinel)
        llm_raw = data.get("llm", _sentinel)
        vectors_data = data.get("vectors", None)

        if qdrant_raw is None:
            qdrant = None
        elif qdrant_raw is _sentinel or not qdrant_raw:
            qdrant = QdrantConfig()
        else:
            qdrant = QdrantConfig(**{
                k: v for k, v in qdrant_raw.items()
                if k in QdrantConfig.__dataclass_fields__
            })

        if code_raw is None:
            code_embedding = None
        elif code_raw is _sentinel or not code_raw:
            code_embedding = EmbeddingConfig()
        else:
            code_embedding = EmbeddingConfig(**{
                k: v for k, v in code_raw.items()
                if k in EmbeddingConfig.__dataclass_fields__
            })

        if desc_raw is None:
            desc_embedding = None
        elif desc_raw is _sentinel or not desc_raw:
            desc_embedding = EmbeddingConfig(
                model="BAAI/bge-small-en-v1.5",
                dimension=384,
            )
        else:
            desc_embedding = EmbeddingConfig(**{
                k: v for k, v in desc_raw.items()
                if k in EmbeddingConfig.__dataclass_fields__
            })

        if llm_raw is None:
            llm = None
        elif llm_raw is _sentinel or not llm_raw:
            llm = LLMConfig()
        else:
            llm = LLMConfig(**{
                k: v for k, v in llm_raw.items()
                if k in LLMConfig.__dataclass_fields__
            })

        vectors = [
            CollectionVectorConfig(**v) for v in vectors_data
        ] if vectors_data else [
            CollectionVectorConfig(name="code", dimension=768, distance="cosine"),
            CollectionVectorConfig(name="description", dimension=384, distance="cosine"),
        ]

        return EngineConfig(
            qdrant=qdrant,
            code_embedding=code_embedding,
            desc_embedding=desc_embedding,
            llm=llm,
            vectors=vectors,
        )

    @staticmethod
    def from_json(path: str) -> "EngineConfig":
        """
        Load EngineConfig from a JSON file.

        path: str — Path to JSON config file.
        Returns: EngineConfig — Parsed config.
        """
        with open(path, "r", encoding="utf-8") as f:
            return EngineConfig.from_dict(json.load(f))

    @staticmethod
    def from_env() -> "EngineConfig":
        """
        Build EngineConfig from environment variables.

        Reads: QC_QDRANT_HOST, QC_QDRANT_PORT, QC_QDRANT_COLLECTION,
               QC_CODE_PROVIDER, QC_CODE_MODEL, QC_CODE_DIMENSION, QC_CODE_API_KEY,
               QC_DESC_PROVIDER, QC_DESC_MODEL, QC_DESC_DIMENSION, QC_DESC_API_KEY,
               QC_LLM_PROVIDER, QC_LLM_MODEL, QC_LLM_MAX_TOKENS, QC_LLM_API_KEY,
               QC_LLM_ARTIFACT_METADATA_ENABLED, QC_LLM_ARTIFACT_METADATA_BATCH_SIZE,
               QC_LLM_ARTIFACT_METADATA_MAX_TOKENS, QC_LLM_ARTIFACT_METADATA_CHUNKS_PER_FILE.
        Returns: EngineConfig — Config built from env vars with defaults.
        """
        qdrant = QdrantConfig(
            host=os.environ.get("QC_QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QC_QDRANT_PORT", "6333")),
            grpc_port=int(os.environ.get("QC_QDRANT_GRPC_PORT", "6334")),
            collection=os.environ.get("QC_QDRANT_COLLECTION", "codebase"),
            timeout=float(os.environ.get("QC_QDRANT_TIMEOUT", "30.0")),
            api_key=os.environ.get("QC_QDRANT_API_KEY"),
            upsert_batch_size=int(os.environ.get("QC_QDRANT_UPSERT_BATCH_SIZE", "100")),
            upsert_concurrency=int(os.environ.get("QC_QDRANT_UPSERT_CONCURRENCY", "1")),
        )

        code_embedding = EmbeddingConfig(
            provider=os.environ.get("QC_CODE_PROVIDER", "fastembed"),
            model=os.environ.get("QC_CODE_MODEL", "jinaai/jina-embeddings-v2-base-code"),
            dimension=int(os.environ.get("QC_CODE_DIMENSION", "768")),
            batch_size=int(os.environ.get("QC_CODE_BATCH_SIZE", "32")),
            min_batch_size=int(os.environ.get("QC_CODE_MIN_BATCH_SIZE", "8")),
            max_batch_size=int(os.environ.get("QC_CODE_MAX_BATCH_SIZE", "128")),
            adaptive_batching=os.environ.get("QC_CODE_ADAPTIVE_BATCHING", "true").lower() in {"1", "true", "yes", "on"},
            adaptive_target_latency_ms=int(os.environ.get("QC_CODE_ADAPTIVE_TARGET_LATENCY_MS", "1500")),
            concurrency=int(os.environ.get("QC_CODE_CONCURRENCY", "4")),
            max_retries=int(os.environ.get("QC_CODE_MAX_RETRIES", "3")),
            retry_base_delay_ms=int(os.environ.get("QC_CODE_RETRY_BASE_DELAY_MS", "250")),
            retry_max_delay_ms=int(os.environ.get("QC_CODE_RETRY_MAX_DELAY_MS", "4000")),
            request_timeout_seconds=(
                float(os.environ["QC_CODE_REQUEST_TIMEOUT_SECONDS"])
                if "QC_CODE_REQUEST_TIMEOUT_SECONDS" in os.environ
                else None
            ),
            api_key=os.environ.get("QC_CODE_API_KEY"),
            api_base=os.environ.get("QC_CODE_API_BASE"),
        )

        desc_embedding = EmbeddingConfig(
            provider=os.environ.get("QC_DESC_PROVIDER", "fastembed"),
            model=os.environ.get("QC_DESC_MODEL", "BAAI/bge-small-en-v1.5"),
            dimension=int(os.environ.get("QC_DESC_DIMENSION", "384")),
            batch_size=int(os.environ.get("QC_DESC_BATCH_SIZE", "32")),
            min_batch_size=int(os.environ.get("QC_DESC_MIN_BATCH_SIZE", "8")),
            max_batch_size=int(os.environ.get("QC_DESC_MAX_BATCH_SIZE", "128")),
            adaptive_batching=os.environ.get("QC_DESC_ADAPTIVE_BATCHING", "true").lower() in {"1", "true", "yes", "on"},
            adaptive_target_latency_ms=int(os.environ.get("QC_DESC_ADAPTIVE_TARGET_LATENCY_MS", "1500")),
            concurrency=int(os.environ.get("QC_DESC_CONCURRENCY", "4")),
            max_retries=int(os.environ.get("QC_DESC_MAX_RETRIES", "3")),
            retry_base_delay_ms=int(os.environ.get("QC_DESC_RETRY_BASE_DELAY_MS", "250")),
            retry_max_delay_ms=int(os.environ.get("QC_DESC_RETRY_MAX_DELAY_MS", "4000")),
            request_timeout_seconds=(
                float(os.environ["QC_DESC_REQUEST_TIMEOUT_SECONDS"])
                if "QC_DESC_REQUEST_TIMEOUT_SECONDS" in os.environ
                else None
            ),
            api_key=os.environ.get("QC_DESC_API_KEY"),
            api_base=os.environ.get("QC_DESC_API_BASE"),
        )

        llm = LLMConfig(
            provider=os.environ.get("QC_LLM_PROVIDER", "litellm"),
            model=os.environ.get("QC_LLM_MODEL", "openrouter/openai/gpt-oss-20b"),
            max_tokens=int(os.environ.get("QC_LLM_MAX_TOKENS", "256")),
            temperature=float(os.environ.get("QC_LLM_TEMPERATURE", "0.0")),
            api_key=os.environ.get("QC_LLM_API_KEY"),
            api_base=os.environ.get("QC_LLM_API_BASE"),
            artifact_metadata_enabled=os.environ.get("QC_LLM_ARTIFACT_METADATA_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
            artifact_metadata_batch_size=int(os.environ.get("QC_LLM_ARTIFACT_METADATA_BATCH_SIZE", "4")),
            artifact_metadata_max_tokens=int(os.environ.get("QC_LLM_ARTIFACT_METADATA_MAX_TOKENS", "512")),
            artifact_metadata_chunks_per_file=int(os.environ.get("QC_LLM_ARTIFACT_METADATA_CHUNKS_PER_FILE", "2")),
        )

        return EngineConfig(
            qdrant=qdrant,
            code_embedding=code_embedding,
            desc_embedding=desc_embedding,
            llm=llm,
        )

    @staticmethod
    def _find_config_file(start: Path | None = None) -> Path | None:
        """
        Walk up from start directory looking for a config file.

        start: Path | None — Starting directory (defaults to CWD).
        Returns: Path | None — First matching config file path, or None.
        """
        current = start or Path.cwd()
        current = current.resolve()

        while True:
            for name in CONFIG_FILENAMES:
                candidate = current / name
                if candidate.is_file():
                    return candidate

            parent = current.parent
            if parent == current:
                break
            current = parent

        return None

    @staticmethod
    def auto() -> "EngineConfig":
        """
        Auto-discover configuration. Priority:
        1. quickcontext.json / .quickcontext.json (walk up from CWD)
        2. Environment variables (QC_* prefix)
        3. Built-in defaults (fastembed, localhost Qdrant)

        Returns: EngineConfig — Resolved configuration.
        """
        config_path = EngineConfig._find_config_file()
        if config_path:
            return EngineConfig.from_json(str(config_path))
        return EngineConfig.from_env()
