from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional
import os
import json


CONFIG_FILENAMES = ["quickcontext.json", ".quickcontext.json"]
WINDOWS_PIPE_NAME = r"\\.\pipe\quickcontext"


DEFAULT_KEYWORD_FIELDS = ("language", "file_path", "symbol_kind", "kind", "source", "tags")


def default_store_path() -> Path:
    """
    Resolve the default on-disk location of the embedded vector store.

    Order: QC_STORE_PATH, then %LOCALAPPDATA% on Windows, then $XDG_DATA_HOME,
    then ~/.local/share.
    """
    override = os.environ.get("QC_STORE_PATH")
    if override:
        return Path(override).expanduser()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    elif os.environ.get("XDG_DATA_HOME"):
        base = Path(os.environ["XDG_DATA_HOME"])
    else:
        base = Path.home() / ".local" / "share"
    return base / "quickcontext" / "store.db"


@dataclass(frozen=True, slots=True)
class StoreConfig:
    """
    Embedded vector store configuration (SQLite + in-memory numpy search, no server).

    path: Optional[str] — SQLite database file. None uses default_store_path(); ":memory:" keeps it in RAM.
    collection: Optional[str] — Default project name for legacy single-collection commands.
    upsert_batch_size: int — Number of points written per transaction.
    busy_timeout_seconds: float — How long writers wait on a lock held by another process.
    max_cached_collections: int — Collections whose vector matrices stay resident in memory.
    """

    path: Optional[str] = None
    collection: Optional[str] = None
    upsert_batch_size: int = 512
    busy_timeout_seconds: float = 30.0
    max_cached_collections: int = 4

    @property
    def resolved_path(self) -> str:
        """
        resolved_path: str — Absolute database path, or ":memory:".
        """
        if self.path == ":memory:":
            return self.path
        if self.path:
            return str(Path(self.path).expanduser().resolve())
        return str(default_store_path().resolve())

    @property
    def url(self) -> str:
        """
        url: str — Human-readable location used in status output.
        """
        return f"sqlite:///{self.resolved_path}"


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    """
    Rust service runtime configuration.

    path: Optional[str] — Explicit path to the quickcontext-service binary.
    """

    path: Optional[str] = None


@dataclass(frozen=True, slots=True)
class TransportConfig:
    """
    Local IPC transport configuration.

    windows_pipe_name: str — Named pipe used on Windows.
    unix_socket_path: Optional[str] — Explicit Unix socket path override.
    """

    windows_pipe_name: str = WINDOWS_PIPE_NAME
    unix_socket_path: Optional[str] = None


@dataclass(frozen=True, slots=True)
class MCPConfig:
    """
    FastMCP server runtime configuration.

    transport: str — MCP transport, usually "stdio" or "http".
    host: str — HTTP bind host when transport is HTTP.
    port: int — HTTP bind port when transport is HTTP.
    http_path: str — HTTP route path when transport is HTTP.
    stateless_http: bool — Whether HTTP mode should be stateless.
    """

    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8000
    http_path: str = "/mcp/"
    stateless_http: bool = False


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
    adaptive_batching: bool — Pack litellm requests by estimated token budget (longest texts
        first) instead of a fixed item count.
    adaptive_target_latency_ms: int — Deprecated; accepted for old configs and ignored.
    concurrency: int — Max concurrent embedding requests in flight (litellm only; 429s are retried).
    max_retries: int — Max retries per failed embedding request.
    retry_base_delay_ms: int — Exponential backoff base delay in milliseconds.
    retry_max_delay_ms: int — Exponential backoff max delay in milliseconds.
    request_timeout_seconds: Optional[float] — Optional per-request timeout.
    api_key: Optional[str] — API key for cloud providers (litellm only).
        Falls back to env vars: OPENAI_API_KEY, COHERE_API_KEY, VOYAGE_API_KEY, etc.
    api_base: Optional[str] — Custom API endpoint URL (litellm only).
        For self-hosted models (vLLM, Infinity, HuggingFace TEI).
    openrouter_provider: Optional[str] — Force specific OpenRouter upstream (e.g. "Nebius", "Together").
    contextual_header: bool — Prepend a short file/symbol header to code chunks before embedding
        (deterministic "contextual retrieval"; only used for code_embedding).
    device: str — Local (fastembed) execution device: "auto" (CUDA, then DirectML, then CPU),
        "cpu", "gpu", "cuda", or "directml". GPU needs onnxruntime-gpu or onnxruntime-directml.
    max_input_tokens: Optional[int] — Cap on tokens embedded per input for local models
        (None keeps the model's own limit).
    query_prefix: Optional[str] — Text prepended to search queries (not documents) before
        embedding, for instruction-tuned models such as Qwen3-Embedding or E5.
    """

    provider: str = "fastembed"
    model: str = "jinaai/jina-embeddings-v2-base-code"
    dimension: int = 768
    batch_size: int = 32
    min_batch_size: int = 8
    max_batch_size: int = 128
    adaptive_batching: bool = True
    adaptive_target_latency_ms: int = 1500
    concurrency: int = 16
    max_retries: int = 3
    retry_base_delay_ms: int = 250
    retry_max_delay_ms: int = 4000
    request_timeout_seconds: Optional[float] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    openrouter_provider: Optional[str] = None
    contextual_header: bool = True
    device: str = "auto"
    max_input_tokens: Optional[int] = None
    query_prefix: Optional[str] = None


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
    Named vector space configuration for a store collection.

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

    store: Optional[StoreConfig] — Embedded vector store settings. None = no vector store.
    code_embedding: Optional[EmbeddingConfig] — Embedding config for code vectors. None = no code embeddings.
    desc_embedding: Optional[EmbeddingConfig] — Embedding config for description vectors. None = no desc embeddings.
    llm: Optional[LLMConfig] — LLM config for code description generation. None = no LLM descriptions.
    service: ServiceConfig — Rust service runtime configuration.
    transport: TransportConfig — Local IPC transport configuration.
    mcp: MCPConfig — MCP runtime configuration.
    vectors: list[CollectionVectorConfig] — Named vector spaces for code collections.
        Derived from code_embedding/desc_embedding dimensions when not given explicitly.
    profile: Optional[str] — Active embedding profile name, if any.
    available_profiles: tuple[str, ...] — Profile names defined in the config file.
    """

    store: Optional[StoreConfig] = field(default_factory=StoreConfig)
    code_embedding: Optional[EmbeddingConfig] = field(default_factory=EmbeddingConfig)
    desc_embedding: Optional[EmbeddingConfig] = field(
        default_factory=lambda: EmbeddingConfig(
            model="BAAI/bge-small-en-v1.5",
            dimension=384,
        )
    )
    llm: Optional[LLMConfig] = field(default_factory=LLMConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    vectors: list[CollectionVectorConfig] = field(
        default_factory=lambda: [
            CollectionVectorConfig(name="code", dimension=768, distance="cosine"),
            CollectionVectorConfig(name="description", dimension=384, distance="cosine"),
        ]
    )
    profile: Optional[str] = None
    available_profiles: tuple[str, ...] = ()

    @staticmethod
    def from_dict(data: dict, profile: Optional[str] = None) -> "EngineConfig":
        """
        Build EngineConfig from a flat or nested dictionary.

        data: dict — Configuration dictionary. Supports nested keys:
            store.path, code_embedding.provider, etc. A legacy "qdrant" section is
            accepted for its collection and upsert_batch_size keys.
        profile: Optional[str] — Named entry of data["profiles"] to apply; defaults to
            $QC_PROFILE, then data["profile"].
        Returns: EngineConfig — Fully constructed config.
        """
        data, active_profile = _apply_profile(data, profile)
        _sentinel = object()

        store_raw = data.get("store", _sentinel)
        if store_raw is _sentinel and "qdrant" in data:
            legacy = data.get("qdrant")
            store_raw = None if legacy is None else {
                k: v for k, v in (legacy or {}).items()
                if k in ("collection", "upsert_batch_size")
            }
        code_raw = data.get("code_embedding", _sentinel)
        desc_raw = data.get("desc_embedding", _sentinel)
        llm_raw = data.get("llm", _sentinel)
        service_raw = data.get("service", _sentinel)
        transport_raw = data.get("transport", _sentinel)
        mcp_raw = data.get("mcp", _sentinel)
        vectors_data = data.get("vectors", None)

        if store_raw is None:
            store = None
        elif store_raw is _sentinel or not store_raw:
            store = StoreConfig()
        else:
            store = StoreConfig(**{
                k: v for k, v in store_raw.items()
                if k in StoreConfig.__dataclass_fields__
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

        if service_raw is _sentinel or not service_raw:
            service = ServiceConfig()
        else:
            service = ServiceConfig(**{
                k: v for k, v in service_raw.items()
                if k in ServiceConfig.__dataclass_fields__
            })

        if transport_raw is _sentinel or not transport_raw:
            transport = TransportConfig()
        else:
            transport = TransportConfig(**{
                k: v for k, v in transport_raw.items()
                if k in TransportConfig.__dataclass_fields__
            })

        if mcp_raw is _sentinel or not mcp_raw:
            mcp = MCPConfig()
        else:
            mcp = MCPConfig(**{
                k: v for k, v in mcp_raw.items()
                if k in MCPConfig.__dataclass_fields__
            })

        vectors = [
            CollectionVectorConfig(**v) for v in vectors_data
        ] if vectors_data else _derive_vectors(code_embedding, desc_embedding)

        return EngineConfig(
            store=store,
            code_embedding=code_embedding,
            desc_embedding=desc_embedding,
            llm=llm,
            service=service,
            transport=transport,
            mcp=mcp,
            vectors=vectors,
            profile=active_profile,
            available_profiles=tuple(sorted(data.get("_profile_names", ()))),
        )

    @staticmethod
    def from_json(path: str, profile: Optional[str] = None) -> "EngineConfig":
        """
        Load EngineConfig from a JSON file.

        path: str — Path to JSON config file.
        profile: Optional[str] — Embedding profile to activate (see from_dict).
        Returns: EngineConfig — Parsed config.
        """
        with open(path, "r", encoding="utf-8") as f:
            config = EngineConfig.from_dict(json.load(f), profile=profile)
        store = config.store
        if (
            store is not None
            and store.path
            and store.path != ":memory:"
            and not Path(store.path).expanduser().is_absolute()
        ):
            anchored = str((Path(path).resolve().parent / store.path).resolve())
            config = replace(config, store=replace(store, path=anchored))
        return config

    @staticmethod
    def from_env() -> "EngineConfig":
        """
        Build EngineConfig from environment variables.

        Reads QC_STORE_PATH, QC_STORE_COLLECTION, QC_STORE_UPSERT_BATCH_SIZE,
               QC_CODE_PROVIDER, QC_CODE_MODEL, QC_CODE_DIMENSION, QC_CODE_API_KEY,
               QC_DESC_PROVIDER, QC_DESC_MODEL, QC_DESC_DIMENSION, QC_DESC_API_KEY,
               QC_LLM_PROVIDER, QC_LLM_MODEL, QC_LLM_MAX_TOKENS, QC_LLM_API_KEY,
               QC_LLM_ARTIFACT_METADATA_ENABLED, QC_LLM_ARTIFACT_METADATA_BATCH_SIZE,
               QC_LLM_ARTIFACT_METADATA_MAX_TOKENS, QC_LLM_ARTIFACT_METADATA_CHUNKS_PER_FILE,
               QC_SERVICE_PATH, QC_SOCKET_PATH, QC_MCP_*.
        Returns: EngineConfig — Config built from env vars with defaults.
        """
        store = StoreConfig(
            path=os.environ.get("QC_STORE_PATH"),
            collection=os.environ.get("QC_STORE_COLLECTION", "codebase"),
            upsert_batch_size=int(os.environ.get("QC_STORE_UPSERT_BATCH_SIZE", "512")),
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
            concurrency=int(os.environ.get("QC_CODE_CONCURRENCY", "16")),
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
            concurrency=int(os.environ.get("QC_DESC_CONCURRENCY", "16")),
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

        service = ServiceConfig(
            path=os.environ.get("QC_SERVICE_PATH"),
        )

        transport = TransportConfig(
            windows_pipe_name=os.environ.get("QC_WINDOWS_PIPE_NAME", WINDOWS_PIPE_NAME),
            unix_socket_path=os.environ.get("QC_SOCKET_PATH"),
        )

        mcp = MCPConfig(
            transport=os.environ.get("QC_MCP_TRANSPORT", "stdio"),
            host=os.environ.get("QC_MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("QC_MCP_PORT", "8000")),
            http_path=os.environ.get("QC_MCP_HTTP_PATH", "/mcp/"),
            stateless_http=os.environ.get("QC_MCP_STATELESS_HTTP", "false").lower() in {"1", "true", "yes", "on"},
        )

        return EngineConfig(
            store=store,
            code_embedding=code_embedding,
            desc_embedding=desc_embedding,
            llm=llm,
            service=service,
            transport=transport,
            mcp=mcp,
            vectors=_derive_vectors(code_embedding, desc_embedding),
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
    def auto(profile: Optional[str] = None) -> "EngineConfig":
        """
        Auto-discover configuration. Priority:
        1. quickcontext.json / .quickcontext.json (walk up from CWD)
        2. Legacy environment variables (QC_* prefix)
        3. Built-in defaults

        profile: Optional[str] — Embedding profile to activate in a JSON config.
        Returns: EngineConfig — Resolved configuration.
        """
        config_path = EngineConfig._find_config_file()
        if config_path:
            return EngineConfig.from_json(str(config_path), profile=profile)
        return EngineConfig.from_env()


def _derive_vectors(
    code_embedding: Optional[EmbeddingConfig],
    desc_embedding: Optional[EmbeddingConfig],
) -> list[CollectionVectorConfig]:
    """
    Build code-collection vector spaces from the embedding dimensions so the two
    can never silently disagree.
    """
    return [
        CollectionVectorConfig(
            name="code",
            dimension=code_embedding.dimension if code_embedding is not None else 768,
            distance="cosine",
        ),
        CollectionVectorConfig(
            name="description",
            dimension=desc_embedding.dimension if desc_embedding is not None else 384,
            distance="cosine",
        ),
    ]


_PROFILE_EMBEDDING_KEYS = ("code_embedding", "desc_embedding")


def _apply_profile(data: dict, profile: Optional[str]) -> tuple[dict, Optional[str]]:
    """
    Merge a named profile from data["profiles"] over the base config.

    Sections in the profile are merged key by key over the base sections (a null
    section disables it). An embedding section that sets "model" or "provider"
    replaces the base section instead, so API keys, query prefixes and other
    model-specific settings never leak into a different model. When a profile changes the embedding models but does
    not set "vectors", vector spaces are re-derived from the new dimensions.
    Without an explicit store path, each profile gets its own database file so
    switching profiles never replaces another profile's index.
    """
    profiles = data.get("profiles") or {}
    names = tuple(profiles)
    active = profile or os.environ.get("QC_PROFILE") or data.get("profile")
    base = {key: value for key, value in data.items() if key not in ("profiles", "profile")}
    base["_profile_names"] = names
    if not active:
        return base, None
    if active not in profiles:
        available = ", ".join(names) or "none defined"
        raise ValueError(f"Unknown config profile {active!r} (available: {available})")

    overrides = profiles[active] or {}
    merged = dict(base)
    for key, value in overrides.items():
        swaps_model = key in _PROFILE_EMBEDDING_KEYS and isinstance(value, dict) and (
            "model" in value or "provider" in value
        )
        if isinstance(value, dict) and isinstance(merged.get(key), dict) and not swaps_model:
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    if "vectors" not in overrides and any(key in overrides for key in _PROFILE_EMBEDDING_KEYS):
        merged.pop("vectors", None)

    store = merged.get("store", {})
    if store is not None:
        store = dict(store or {})
        if not store.get("path"):
            store["path"] = str(default_store_path().with_name(f"store-{active}.db"))
        merged["store"] = store
    return merged, active
