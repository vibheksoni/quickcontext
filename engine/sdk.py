from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
import re
import time
from threading import Event, Lock, Thread
from typing import Any, Optional, TYPE_CHECKING

from engine.src.chunk_filter import ChunkFilterConfig, filter_chunks
from engine.src.chunker import ChunkBuilder, CodeChunk
from engine.src.compressor import CompressionStats, compress_grep_line, compress_source
from engine.src.config import EngineConfig
from engine.src.artifact_index import (
    ARTIFACT_FALLBACK_ERROR,
    ArtifactIndexProfile,
    analyze_artifact_candidate,
    should_downgrade_artifact_profile,
)
from engine.src.artifact_semantics import extract_artifact_semantic_signals
from engine.src.dedup import (
    DeduplicationResult,
    content_hash,
    deduplicate_chunks,
    expand_descriptions,
    expand_embeddings,
)
from engine.src.filecache import FileSignatureCache
from engine.src.lsp_setup import build_lsp_check_plan, build_lsp_setup_plan
from engine.src.operation_status import GLOBAL_OPERATION_REGISTRY, OperationProgressReporter, snapshot_to_dict

if TYPE_CHECKING:
    from engine.src.collection import CollectionManager
    from engine.src.describer import ChunkDescription, DescriptionGenerator
    from engine.src.embedder import DualEmbedder, EmbeddedChunk
    from engine.src.indexer import IndexStats, IndexedFileState, QdrantIndexer
    from engine.src.index_resume import ResumeFile, ResumeState
    from engine.src.providers import EmbeddingProvider
    from engine.src.qdrant import QdrantConnection
    from engine.src.searcher import CodeSearcher, SearchResult
    from engine.src.watcher import FileWatcher
    from engine.src.reranker import ColBERTReranker

from engine.src.parsing import (
    CallGraphTraceResult,
    CallerLookupResult,
    ExtractedSymbol,
    ExtractionResult,
    ExtractSymbolResult,
    FileEditResult,
    FileEditRevertResult,
    FileReadLine,
    FileReadResult,
    GrepResult,
    ImportEdge,
    ImportGraphResult,
    ImportNeighborsResult,
    PatternMatchCapture,
    PatternMatchItem,
    PatternMatchResult,
    ProtocolContract,
    ProtocolEvidence,
    ProtocolInputField,
    ProtocolOutputField,
    ProtocolSearchResult,
    RewriteEdit,
    RewriteFileResult,
    RewriteResult,
    RustParserService,
    SkeletonResult,
    SymbolLookupResult,
    TextSearchMatch,
    TextSearchResult,
    TraceEdge,
    TraceNode,
)
from engine.src.differ import ChunkDiffer
from engine.src.keywords import STOP_WORDS, extract_keywords
from engine.src.packer import (
    PackedOutput,
    PackedResult,
    count_tokens,
    pack_grep_results,
    pack_search_results,
    pack_skeleton,
    truncate_source,
)
from engine.src.project import detect_project_name
from engine.src.sdk_models import (
    IndexOperationSnapshot,
    LspCheckPlanInfo,
    LspInstallStepInfo,
    LspServerCheckInfo,
    LspServerSetupInfo,
    LspSetupPlanInfo,
    ProjectCollectionInfo,
    ProjectFolderInfo,
    ProjectInfo,
)
from engine.src.search_modes import BIAS_NAMES, SearchBias, apply_bias, get_bias

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_UNDERSCORE_SPLIT_PATTERN = re.compile(r"_+")
_CAMEL_BOUNDARY_PATTERN = re.compile(r"([a-z0-9])([A-Z])")
_GENERATED_BUNDLE_NAME_PATTERN = re.compile(r"(?:^|[._-])[0-9a-f]{6,}(?:[._-]|$)")
_LSP_SYMBOL_KIND_MAP = {
    1: "file",
    2: "module",
    3: "namespace",
    4: "package",
    5: "class",
    6: "method",
    7: "property",
    8: "field",
    9: "constructor",
    10: "enum",
    11: "interface",
    12: "function",
    13: "variable",
    14: "constant",
    15: "string",
    16: "number",
    17: "boolean",
    18: "array",
    19: "object",
    20: "key",
    21: "null",
    22: "enum_member",
    23: "struct",
    24: "event",
    25: "operator",
    26: "type_parameter",
}
_LSP_SKIPPED_KINDS = {
    "string",
    "number",
    "boolean",
    "array",
    "object",
    "key",
    "null",
    "event",
    "operator",
    "type_parameter",
}
_LSP_HIGH_SIGNAL_KINDS = {
    "module",
    "namespace",
    "package",
    "class",
    "method",
    "property",
    "field",
    "constructor",
    "enum",
    "interface",
    "function",
    "enum_member",
    "struct",
}
_LSP_LANGUAGE_ALIASES = {
    "bash": {"bash"},
    "c": {"c"},
    "c_sharp": {"csharp"},
    "cpp": {"c"},
    "cxx": {"c"},
    "css": {"css"},
    "dockerfile": {"dockerfile"},
    "elixir": {"elixir"},
    "erlang": {"erlang"},
    "go": {"go"},
    "haskell": {"haskell"},
    "html": {"html"},
    "java": {"java"},
    "javascript": {"typescript"},
    "json": {"json"},
    "jsonc": {"json"},
    "jsx": {"typescript"},
    "kotlin": {"kotlin"},
    "lua": {"lua"},
    "markdown": {"markdown"},
    "nim": {"nim"},
    "ocaml": {"ocaml"},
    "perl": {"perl"},
    "php": {"php"},
    "python": {"python"},
    "r": {"r"},
    "ruby": {"ruby", "ruby_solargraph"},
    "rust": {"rust"},
    "scala": {"scala"},
    "svelte": {"svelte"},
    "swift": {"swift"},
    "terraform": {"terraform"},
    "toml": {"toml"},
    "tsx": {"typescript"},
    "typescript": {"typescript"},
    "verilog": {"verilog"},
    "vhdl": {"vhdl"},
    "vue": {"vue"},
    "yaml": {"yaml"},
    "zig": {"zig"},
}


def _human_readable_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ("B", "KB", "MB", "GB")
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    return f"{value:.2f} {units[unit_idx]}"


class QuickContext:
    """
    Main SDK entry point. All subsystems are lazy and optional.
    Qdrant, embeddings, and LLM are only initialized when actually used.
    Rust parser service works independently of all other subsystems.

    _config: EngineConfig — Full engine configuration.
    _conn: QdrantConnection | None — Qdrant connection manager (lazy, None if qdrant disabled).
    _collections: dict[str, CollectionManager] — Per-project collection managers.
    _code_provider: EmbeddingProvider | None — Code embedding provider (lazy).
    _desc_provider: EmbeddingProvider | None — Description embedding provider (lazy).
    _parser_service: RustParserService | None — Rust AST extraction service wrapper (lazy).
    _chunker: ChunkBuilder | None — Chunk builder for symbol-to-chunk conversion (lazy).
    _describer: DescriptionGenerator | None — LLM-based description generator (lazy).
    _embedder: DualEmbedder | None — Dual embedding generator (lazy).
    _indexers: dict[str, QdrantIndexer] — Per-project Qdrant indexers.
    _searchers: dict[str, CodeSearcher] — Per-project semantic searchers.
    _reranker: ColBERTReranker | None — ColBERT reranker for post-retrieval reranking (lazy).
    """

    def __init__(self, config: EngineConfig | None = None):
        """
        config: EngineConfig | None — Engine config. Defaults to EngineConfig() if None.
        """
        self._config = config or EngineConfig()
        self._conn: "QdrantConnection | None" = None
        self._collections: dict = {}
        self._code_provider: "EmbeddingProvider | None" = None
        self._desc_provider: "EmbeddingProvider | None" = None
        self._parser_service: RustParserService | None = None
        self._chunker: ChunkBuilder | None = None
        self._describer: "DescriptionGenerator | None" = None
        self._embedder: "DualEmbedder | None" = None
        self._indexers: dict = {}
        self._searchers: dict = {}
        self._file_caches: dict[str, FileSignatureCache] = {}
        self._symbol_file_compact_cache: dict[str, tuple[int, int, list]] = {}
        self._symbol_file_extract_cache: dict[str, tuple[int, int, list]] = {}
        self._lsp_ready_language_cache: dict[str, set[str]] = {}
        self._reranker: "ColBERTReranker | None" = None
        self._activity_lock = Lock()
        self._active_operations = 0
        self._last_activity_at = time.monotonic()
        self._background_warm_thread: Thread | None = None
        self._background_warm_stop = Event()
        self._background_warm_path: str | None = None
        self._background_warm_started = False
        self._background_warm_auto_delay_seconds = 0.5

    @property
    def config(self) -> EngineConfig:
        """
        Returns: EngineConfig — Current engine configuration.
        """
        return self._config

    @property
    def connection(self) -> "QdrantConnection":
        """
        Returns: QdrantConnection — Qdrant connection manager (lazy-created).
        Raises: RuntimeError — If Qdrant is disabled in config.
        """
        if self._config.qdrant is None:
            raise RuntimeError(
                "Qdrant is disabled (config.qdrant is None). "
                "Enable it in your config to use vector search features."
            )
        if self._conn is None:
            from engine.src.qdrant import QdrantConnection
            self._conn = QdrantConnection(self._config.qdrant)
        return self._conn

    def connect(self, verify: bool = False) -> "QuickContext":
        """
        Connect to Qdrant server. No-op if Qdrant is disabled.

        Returns: QuickContext — Self for chaining.
        Raises: ConnectionError — If Qdrant is unreachable.
        """
        if self._config.qdrant is None:
            return self
        self.connection.connect(verify=verify)
        return self

    @contextmanager
    def _activity_scope(self):
        with self._activity_lock:
            self._active_operations += 1
            self._last_activity_at = time.monotonic()
        try:
            yield
        finally:
            with self._activity_lock:
                self._active_operations = max(0, self._active_operations - 1)
                self._last_activity_at = time.monotonic()

    def start_background_warm(
        self,
        path: str | Path = ".",
        idle_delay_seconds: float = 0.05,
    ) -> bool:
        """
        Start a best-effort background warm after the session goes idle.
        """
        resolved_path = str(Path(path).resolve())
        with self._activity_lock:
            if self._background_warm_started and self._background_warm_path == resolved_path:
                return False
            if self._background_warm_thread is not None and self._background_warm_thread.is_alive():
                return False
            self._background_warm_started = True
            self._background_warm_path = resolved_path
            self._background_warm_stop.clear()

        def _runner() -> None:
            while not self._background_warm_stop.is_set():
                with self._activity_lock:
                    active = self._active_operations
                    idle_for = time.monotonic() - self._last_activity_at
                if active == 0 and idle_for >= idle_delay_seconds:
                    break
                self._background_warm_stop.wait(0.05)

            if self._background_warm_stop.is_set():
                return

            try:
                self._background_warm_once(resolved_path)
            except Exception:
                return

        thread = Thread(target=_runner, name="quickcontext-warm", daemon=True)
        self._background_warm_thread = thread
        thread.start()
        return True

    def _background_warm_once(self, path: str | Path) -> None:
        service = RustParserService()
        try:
            service.warm_project(path=path)
        finally:
            service.close()

    def _ensure_background_warm_started(self, path: str | Path = ".") -> None:
        self.start_background_warm(
            path,
            idle_delay_seconds=self._background_warm_auto_delay_seconds,
        )

    def _resolve_retrieval_root(self, path: str | Path | None = None) -> Path:
        """
        Resolve the repo root used by parser-first and text-first retrieval flows.
        """
        return Path(path).resolve() if path is not None else Path.cwd().resolve()

    def _get_collection(self, project_name: str) -> "CollectionManager":
        """
        Get or create CollectionManager for a project.

        project_name: str — Project name (used as collection name).
        Returns: CollectionManager — Collection manager for the project.
        Raises: RuntimeError — If Qdrant is disabled.
        """
        if project_name not in self._collections:
            from engine.src.collection import CollectionManager
            conn = self.connection
            client = conn.client

            self._collections[project_name] = CollectionManager(
                client=client,
                config=self._config,
                collection_name=project_name,
            )
            self._collections[project_name].ensure()

        return self._collections[project_name]

    def _get_indexer(self, project_name: str) -> "QdrantIndexer":
        """
        Get or create QdrantIndexer for a project.

        project_name: str — Project name.
        Returns: QdrantIndexer — Indexer for the project.
        """
        if project_name not in self._indexers:
            from engine.src.indexer import QdrantIndexer
            collection = self._get_collection(project_name)
            self._indexers[project_name] = QdrantIndexer(
                client=collection.client,
                collection_name=project_name,
                batch_size=self._config.qdrant.upsert_batch_size,
                upsert_concurrency=self._config.qdrant.upsert_concurrency,
            )

        return self._indexers[project_name]

    def _get_reranker(self) -> "ColBERTReranker":
        """
        Get or create ColBERT reranker (lazy singleton).

        Returns: ColBERTReranker — Initialized reranker instance.
        """
        if self._reranker is None:
            from engine.src.reranker import ColBERTReranker
            self._reranker = ColBERTReranker()
        return self._reranker

    def _get_searcher(self, project_name: str, rerank: bool = False) -> "CodeSearcher":
        """
        Get or create CodeSearcher for a project.

        project_name: str — Project name.
        rerank: bool — Whether to attach ColBERT reranker.
        Returns: CodeSearcher — Searcher for the project.
        """
        reranker = self._get_reranker() if rerank else None
        cache_key = f"{project_name}:rerank={rerank}"

        if cache_key not in self._searchers:
            if self._config.qdrant is None:
                raise RuntimeError("Qdrant is disabled (config.qdrant is None).")
            from engine.src.qdrant_search import RestQdrantSearchClient
            from engine.src.searcher import CodeSearcher
            self._searchers[cache_key] = CodeSearcher(
                client=RestQdrantSearchClient(self._config.qdrant),
                collection_name=project_name,
                code_provider=self.code_provider,
                desc_provider=self.desc_provider,
                reranker=reranker,
            )

        return self._searchers[cache_key]

    def _get_file_cache(self, directory: str | Path) -> FileSignatureCache:
        """
        Get or create FileSignatureCache for a project directory.

        directory: str | Path — Project root directory.
        Returns: FileSignatureCache — File signature cache for the directory.
        """
        key = str(Path(directory).resolve())
        if key not in self._file_caches:
            self._file_caches[key] = FileSignatureCache(key)
        return self._file_caches[key]

    @property
    def collection(self) -> "CollectionManager":
        """
        Returns: CollectionManager — Collection manager (deprecated, use project-specific access).
        Raises: ValueError — If no project specified.
        Raises: RuntimeError — If Qdrant is disabled.
        """
        if self._config.qdrant is None:
            raise RuntimeError("Qdrant is disabled (config.qdrant is None).")
        if self._config.qdrant.collection:
            return self._get_collection(self._config.qdrant.collection)

        raise ValueError(
            "No default collection configured. Use project-based indexing instead."
        )

    @property
    def code_provider(self) -> "EmbeddingProvider":
        """
        Returns: EmbeddingProvider — Lazy-initialized code embedding provider.
        Raises: RuntimeError — If code embedding is disabled.
        """
        if self._config.code_embedding is None:
            raise RuntimeError(
                "Code embedding is disabled (config.code_embedding is None). "
                "Enable it in your config to use embedding features."
            )
        if self._code_provider is None:
            from engine.src.providers import create_provider
            self._code_provider = create_provider(self._config.code_embedding)
        return self._code_provider

    @property
    def desc_provider(self) -> "EmbeddingProvider":
        """
        Returns: EmbeddingProvider — Lazy-initialized description embedding provider.
        Raises: RuntimeError — If description embedding is disabled.
        """
        if self._config.desc_embedding is None:
            raise RuntimeError(
                "Description embedding is disabled (config.desc_embedding is None). "
                "Enable it in your config to use embedding features."
            )
        if self._desc_provider is None:
            from engine.src.providers import create_provider
            self._desc_provider = create_provider(self._config.desc_embedding)
        return self._desc_provider

    @property
    def parser_service(self) -> RustParserService:
        """
        Returns: RustParserService — Lazy-initialized Rust pipe parser wrapper.
        """
        if self._parser_service is None:
            self._parser_service = RustParserService()
        return self._parser_service

    @property
    def chunker(self) -> ChunkBuilder:
        """
        Returns: ChunkBuilder — Lazy-initialized chunk builder.
        """
        if self._chunker is None:
            self._chunker = ChunkBuilder()
        return self._chunker

    @property
    def describer(self) -> "DescriptionGenerator":
        """
        Returns: DescriptionGenerator — Lazy-initialized description generator.
        Raises: RuntimeError — If LLM is disabled.
        """
        if self._config.llm is None:
            raise RuntimeError(
                "LLM is disabled (config.llm is None). "
                "Enable it in your config to use description generation."
            )
        if self._describer is None:
            from engine.src.describer import DescriptionGenerator
            llm_config = self._config.llm
            self._describer = DescriptionGenerator(
                model=llm_config.model,
                api_key=llm_config.api_key,
                api_base=llm_config.api_base,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                openrouter_provider=llm_config.openrouter_provider,
            )
        return self._describer

    @property
    def embedder(self) -> "DualEmbedder":
        """
        Returns: DualEmbedder — Lazy-initialized dual embedder.
        Raises: RuntimeError — If code or description embedding is disabled.
        """
        if self._config.code_embedding is None or self._config.desc_embedding is None:
            raise RuntimeError(
                "Embeddings are disabled (code_embedding or desc_embedding is None). "
                "Enable both in your config to use embedding features."
            )
        if self._embedder is None:
            from engine.src.embedder import DualEmbedder
            code_cfg = self._config.code_embedding
            desc_cfg = self._config.desc_embedding
            self._embedder = DualEmbedder(
                code_provider=code_cfg.provider,
                code_model=code_cfg.model,
                code_dimension=code_cfg.dimension,
                desc_provider=desc_cfg.provider,
                desc_model=desc_cfg.model,
                desc_dimension=desc_cfg.dimension,
                code_api_key=code_cfg.api_key,
                desc_api_key=desc_cfg.api_key,
                code_api_base=code_cfg.api_base,
                desc_api_base=desc_cfg.api_base,
                code_batch_size=code_cfg.batch_size,
                desc_batch_size=desc_cfg.batch_size,
                code_min_batch_size=code_cfg.min_batch_size,
                desc_min_batch_size=desc_cfg.min_batch_size,
                code_max_batch_size=code_cfg.max_batch_size,
                desc_max_batch_size=desc_cfg.max_batch_size,
                code_adaptive_batching=code_cfg.adaptive_batching,
                desc_adaptive_batching=desc_cfg.adaptive_batching,
                code_adaptive_target_latency_ms=code_cfg.adaptive_target_latency_ms,
                desc_adaptive_target_latency_ms=desc_cfg.adaptive_target_latency_ms,
                code_concurrency=code_cfg.concurrency,
                desc_concurrency=desc_cfg.concurrency,
                code_max_retries=code_cfg.max_retries,
                desc_max_retries=desc_cfg.max_retries,
                code_retry_base_delay_ms=code_cfg.retry_base_delay_ms,
                desc_retry_base_delay_ms=desc_cfg.retry_base_delay_ms,
                code_retry_max_delay_ms=code_cfg.retry_max_delay_ms,
                desc_retry_max_delay_ms=desc_cfg.retry_max_delay_ms,
                code_request_timeout_seconds=code_cfg.request_timeout_seconds,
                desc_request_timeout_seconds=desc_cfg.request_timeout_seconds,
                code_openrouter_provider=code_cfg.openrouter_provider,
                desc_openrouter_provider=desc_cfg.openrouter_provider,
            )
        return self._embedder


    def extract_symbols(
        self,
        path: str | Path,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> list[ExtractionResult]:
        """
        Extract symbols from a file or directory via the Rust named-pipe service.

        path: str | Path — File or directory to parse.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: list[ExtractionResult] — Typed extraction results.
        """
        return self.parser_service.extract(
            path=path,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def extract_symbol(
        self,
        file: str | Path,
        symbol: str,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> ExtractSymbolResult:
        """
        Extract a specific symbol by name from a single file.

        file: str | Path — Absolute path to the source file.
        symbol: str — Symbol name or "Parent.name" for disambiguation.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: ExtractSymbolResult — Matched symbols with metadata.
        """
        return self.parser_service.extract_symbol(
            file=file,
            symbol=symbol,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )



    def file_read(
        self,
        file: str | Path,
        start_line: int | None = None,
        end_line: int | None = None,
        max_bytes: int | None = None,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> FileReadResult:
        """
        Read file content with optional line slicing via Rust service.
        """
        return self.parser_service.file_read(
            file=file,
            start_line=start_line,
            end_line=end_line,
            max_bytes=max_bytes,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def file_edit(
        self,
        file: str | Path,
        mode: str,
        edits: list[dict] | None = None,
        text: str | None = None,
        dry_run: bool = False,
        expected_hash: str | None = None,
        record_undo: bool = True,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> FileEditResult:
        """
        Apply a file edit operation via Rust service.
        """
        return self.parser_service.file_edit(
            file=file,
            mode=mode,
            edits=edits,
            text=text,
            dry_run=dry_run,
            expected_hash=expected_hash,
            record_undo=record_undo,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def file_edit_revert(
        self,
        edit_id: str,
        dry_run: bool = False,
        expected_hash: str | None = None,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> FileEditRevertResult:
        """
        Revert a prior file_edit operation by undo edit ID.
        """
        return self.parser_service.file_edit_revert(
            edit_id=edit_id,
            dry_run=dry_run,
            expected_hash=expected_hash,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def grep_text(
        self,
        query: str,
        path: str | Path | None = None,
        respect_gitignore: bool = True,
        limit: int = 200,
        before_context: int = 0,
        after_context: int = 0,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> GrepResult:
        """
        Perform fast literal grep through the Rust service.

        query: str — Literal text pattern to search.
        path: str | Path | None — Optional file/directory scope. Defaults to current directory.
        respect_gitignore: bool — Apply .gitignore/.ignore/.quick-ignore when True.
        limit: int — Maximum number of matches to return.
        before_context: int — Number of lines before each match.
        after_context: int — Number of lines after each match.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: GrepResult — Grep results with match metadata and timing.
        """
        return self.parser_service.grep(
            query=query,
            path=path,
            respect_gitignore=respect_gitignore,
            limit=limit,
            before_context=before_context,
            after_context=after_context,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def symbol_lookup(
        self,
        query: str,
        path: str | Path | None = None,
        respect_gitignore: bool = True,
        limit: int = 50,
        intent_mode: bool = False,
        intent_level: int = 2,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> SymbolLookupResult:
        """
        Look up symbols by keyword via the Rust in-memory index.

        query: str — Symbol name or keyword query.
        path: str | Path | None — Project root or search scope. Uses cwd when None.
        respect_gitignore: bool — Apply ignore rules when True.
        limit: int — Maximum symbols returned.
        intent_mode: bool — Enable non-embedding intent expansion.
        intent_level: int — Intent expansion aggressiveness from 1..=3.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: SymbolLookupResult — Typed symbol lookup payload.
        """
        self._ensure_background_warm_started(path or Path.cwd())
        with self._activity_scope():
            return self.parser_service.symbol_lookup(
                query=query,
                path=path,
                respect_gitignore=respect_gitignore,
                limit=limit,
                intent_mode=intent_mode,
                intent_level=intent_level,
                ensure_server=ensure_server,
                timeout_ms=timeout_ms,
            )

    def find_callers(
        self,
        symbol: str,
        path: str | Path | None = None,
        respect_gitignore: bool = True,
        limit: int = 100,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> CallerLookupResult:
        """
        Find all call sites for a symbol via the Rust call index.

        symbol: str — Target symbol name.
        path: str | Path | None — Project root or search scope. Uses cwd when None.
        respect_gitignore: bool — Apply ignore rules when True.
        limit: int — Maximum caller rows returned.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: CallerLookupResult — Typed caller lookup payload.
        """
        self._ensure_background_warm_started(path or Path.cwd())
        with self._activity_scope():
            return self.parser_service.find_callers(
                symbol=symbol,
                path=path,
                respect_gitignore=respect_gitignore,
                limit=limit,
                ensure_server=ensure_server,
                timeout_ms=timeout_ms,
            )

    def trace_call_graph(
        self,
        symbol: str,
        path: str | Path | None = None,
        respect_gitignore: bool = True,
        direction: str = "both",
        max_depth: int = 5,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> CallGraphTraceResult:
        """
        Trace multi-hop call graph from a symbol via BFS.

        symbol: str — Root symbol name to trace from.
        path: str | Path | None — Project root or search scope. Uses cwd when None.
        respect_gitignore: bool — Apply ignore rules when True.
        direction: str — Trace direction: "upstream", "downstream", or "both".
        max_depth: int — Maximum BFS traversal depth.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: CallGraphTraceResult — Typed call graph trace payload.
        """
        self._ensure_background_warm_started(path or Path.cwd())
        with self._activity_scope():
            return self.parser_service.trace_call_graph(
                symbol=symbol,
                path=path,
                respect_gitignore=respect_gitignore,
                direction=direction,
                max_depth=max_depth,
                ensure_server=ensure_server,
                timeout_ms=timeout_ms,
            )

    def skeleton(
        self,
        path: str | Path,
        max_depth: int | None = None,
        include_signatures: bool = True,
        include_line_numbers: bool = False,
        collapse_threshold: int = 0,
        respect_gitignore: bool = True,
        format: str = "json",
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> SkeletonResult:
        """
        Generate a repo skeleton for a file or directory.

        path: str | Path — File or directory to skeletonize.
        max_depth: int | None — Max directory recursion depth. None uses server default (20).
        include_signatures: bool — Include function/class signatures.
        include_line_numbers: bool — Include line number ranges.
        collapse_threshold: int — Collapse dirs with fewer files than this.
        respect_gitignore: bool — Apply .gitignore/.ignore/.quick-ignore when True.
        format: str — Output format: "json" or "markdown".
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: SkeletonResult — Typed skeleton result payload.
        """
        return self.parser_service.skeleton(
            path=path,
            max_depth=max_depth,
            include_signatures=include_signatures,
            include_line_numbers=include_line_numbers,
            collapse_threshold=collapse_threshold,
            respect_gitignore=respect_gitignore,
            format=format,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def import_graph(
        self,
        file: str | Path,
        path: str | Path | None = None,
        respect_gitignore: bool = True,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> ImportGraphResult:
        """
        Get import dependencies for a file (outgoing edges).

        file: str | Path — Absolute path to the source file.
        path: str | Path | None — Project root directory. Uses cwd when None.
        respect_gitignore: bool — Apply ignore rules when True.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: ImportGraphResult — Import edges and graph metadata.
        """
        return self.parser_service.import_graph(
            file=file,
            path=path,
            respect_gitignore=respect_gitignore,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def find_importers(
        self,
        file: str | Path,
        path: str | Path | None = None,
        respect_gitignore: bool = True,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> ImportGraphResult:
        """
        Find files that import a given file (incoming edges).

        file: str | Path — Absolute path to the target file.
        path: str | Path | None — Project root directory. Uses cwd when None.
        respect_gitignore: bool — Apply ignore rules when True.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: ImportGraphResult — Importer edges and graph metadata.
        """
        return self.parser_service.find_importers(
            file=file,
            path=path,
            respect_gitignore=respect_gitignore,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def import_neighbors(
        self,
        file: str | Path,
        path: str | Path | None = None,
        respect_gitignore: bool = True,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> ImportNeighborsResult:
        """
        Get both imports and importers for a file in one request.
        """
        return self.parser_service.import_neighbors(
            file=file,
            path=path,
            respect_gitignore=respect_gitignore,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def text_search(
        self,
        query: str,
        path: str | Path | None = None,
        respect_gitignore: bool = True,
        limit: int = 20,
        intent_mode: bool = False,
        intent_level: int = 2,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> TextSearchResult:
        """
        Perform BM25 full-text search via the Rust service.

        query: str — Search query with optional operators (AND/OR/+/-/"exact"/file:/ext:/lang:).
        path: str | Path | None — Directory to search in. Uses cwd when None.
        respect_gitignore: bool — Apply .gitignore/.ignore/.quick-ignore when True.
        limit: int — Maximum number of ranked results returned.
        intent_mode: bool — Enable non-embedding intent expansion.
        intent_level: int — Intent expansion aggressiveness from 1..=3.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: TextSearchResult — Typed BM25 search result payload.
        """
        self._ensure_background_warm_started(path or Path.cwd())
        with self._activity_scope():
            return self.parser_service.text_search(
                query=query,
                path=path,
                respect_gitignore=respect_gitignore,
                limit=limit,
                intent_mode=intent_mode,
                intent_level=intent_level,
                ensure_server=ensure_server,
                timeout_ms=timeout_ms,
            )

    def warm_project(
        self,
        path: str | Path = ".",
        respect_gitignore: bool = True,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Warm persisted Rust symbol and text indices for a project path.
        """
        return self.parser_service.warm_project(
            path=path,
            respect_gitignore=respect_gitignore,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def pattern_search(
        self,
        pattern: str,
        language: str,
        path: str | Path | None = None,
        respect_gitignore: bool = True,
        limit: int = 50,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> PatternMatchResult:
        """
        Search for AST pattern matches via the Rust service.

        pattern: str — Code pattern with metavariables ($NAME, $$$, $_).
        language: str — Target language name (python, rust, javascript, etc.).
        path: str | Path | None — Directory or file to search in. Uses cwd when None.
        respect_gitignore: bool — Apply .gitignore/.ignore/.quick-ignore when True.
        limit: int — Maximum number of matches returned.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: PatternMatchResult — Typed pattern match result payload.
        """
        return self.parser_service.pattern_search(
            pattern=pattern,
            language=language,
            path=path,
            respect_gitignore=respect_gitignore,
            limit=limit,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def protocol_search(
        self,
        query: str,
        path: str | Path | None = None,
        respect_gitignore: bool = True,
        limit: int = 20,
        context_radius: int | None = None,
        min_score: float | None = None,
        include_markers: list[str] | None = None,
        exclude_markers: list[str] | None = None,
        max_input_fields: int | None = None,
        max_output_fields: int | None = None,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> ProtocolSearchResult:
        """
        Extract protocol request/response contracts via the Rust service.

        query: str — Intent query for protocol extraction.
        path: str | Path | None — Directory or file to search in. Uses cwd when None.
        respect_gitignore: bool — Apply .gitignore/.ignore/.quick-ignore when True.
        limit: int — Maximum number of protocol contracts returned.
        context_radius: int | None — Context line radius around matched protocol markers.
        min_score: float | None — Minimum score threshold for returned contracts.
        include_markers: list[str] | None — Additional markers treated as protocol signals.
        exclude_markers: list[str] | None — Markers that skip candidate files.
        max_input_fields: int | None — Maximum inferred input fields per contract.
        max_output_fields: int | None — Maximum inferred output fields per contract.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: ProtocolSearchResult — Typed protocol extraction result payload.
        """
        return self.parser_service.protocol_search(
            query=query,
            path=path,
            respect_gitignore=respect_gitignore,
            limit=limit,
            context_radius=context_radius,
            min_score=min_score,
            include_markers=include_markers,
            exclude_markers=exclude_markers,
            max_input_fields=max_input_fields,
            max_output_fields=max_output_fields,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def pattern_rewrite(
        self,
        pattern: str,
        replacement: str,
        language: str,
        path: str | Path | None = None,
        respect_gitignore: bool = True,
        limit: int = 50,
        dry_run: bool = True,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> RewriteResult:
        """
        Rewrite code matching an AST pattern via the Rust service.

        pattern: str — Code pattern with metavariables ($NAME, $$$, $_).
        replacement: str — Replacement template with metavariable substitution.
        language: str — Target language name (python, rust, javascript, etc.).
        path: str | Path | None — Directory or file to search in. Uses cwd when None.
        respect_gitignore: bool — Apply .gitignore/.ignore/.quick-ignore when True.
        limit: int — Maximum number of files to rewrite.
        dry_run: bool — Compute edits without writing files when True.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: RewriteResult — Typed rewrite result payload.
        """
        return self.parser_service.pattern_rewrite(
            pattern=pattern,
            replacement=replacement,
            language=language,
            path=path,
            respect_gitignore=respect_gitignore,
            limit=limit,
            dry_run=dry_run,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def lsp_definition(
        self,
        file: str | Path,
        line: int,
        character: int,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Go to definition of symbol at position via LSP.

        file: str | Path — Absolute path to the source file.
        line: int — Zero-based line number.
        character: int — Zero-based character offset.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP definition location(s).
        """
        return self.parser_service.lsp_definition(
            file=file,
            line=line,
            character=character,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def lsp_references(
        self,
        file: str | Path,
        line: int,
        character: int,
        include_declaration: bool = True,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Find all references to symbol at position via LSP.

        file: str | Path — Absolute path to the source file.
        line: int — Zero-based line number.
        character: int — Zero-based character offset.
        include_declaration: bool — Include the declaration itself.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP reference locations.
        """
        return self.parser_service.lsp_references(
            file=file,
            line=line,
            character=character,
            include_declaration=include_declaration,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def lsp_hover(
        self,
        file: str | Path,
        line: int,
        character: int,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Get hover information for symbol at position via LSP.

        file: str | Path — Absolute path to the source file.
        line: int — Zero-based line number.
        character: int — Zero-based character offset.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP hover contents (type info, docs).
        """
        return self.parser_service.lsp_hover(
            file=file,
            line=line,
            character=character,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def lsp_symbols(
        self,
        file: str | Path,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Get document symbols (outline) for a file via LSP.

        file: str | Path — Absolute path to the source file.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP hierarchical symbol tree.
        """
        return self.parser_service.lsp_symbols(
            file=file,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def lsp_format(
        self,
        file: str | Path,
        tab_size: int = 4,
        insert_spaces: bool = True,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Format a document via LSP.

        file: str | Path — Absolute path to the source file.
        tab_size: int — Spaces per tab.
        insert_spaces: bool — Use spaces instead of tabs.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP text edits to apply.
        """
        return self.parser_service.lsp_format(
            file=file,
            tab_size=tab_size,
            insert_spaces=insert_spaces,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def lsp_diagnostics(
        self,
        file: str | Path,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Request diagnostics for a file via LSP.

        file: str | Path — Absolute path to the source file.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP diagnostic request status.
        """
        return self.parser_service.lsp_diagnostics(
            file=file,
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def lsp_completion(
        self,
        file: str | Path,
        line: int,
        character: int,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Request completions at position via LSP.

        file: str | Path — Absolute path to the source file.
        line: int — Zero-based line number.
        character: int — Zero-based character offset.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP completion items.
        """
        return self.parser_service.lsp_completion(
            file=file, line=line, character=character,
            ensure_server=ensure_server, timeout_ms=timeout_ms,
        )

    def lsp_rename(
        self,
        file: str | Path,
        line: int,
        character: int,
        new_name: str,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Rename symbol at position via LSP.

        file: str | Path — Absolute path to the source file.
        line: int — Zero-based line number.
        character: int — Zero-based character offset.
        new_name: str — New name for the symbol.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP workspace edit (file changes).
        """
        return self.parser_service.lsp_rename(
            file=file, line=line, character=character,
            new_name=new_name,
            ensure_server=ensure_server, timeout_ms=timeout_ms,
        )

    def lsp_prepare_rename(
        self,
        file: str | Path,
        line: int,
        character: int,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Validate rename and get symbol range at position via LSP.

        file: str | Path — Absolute path to the source file.
        line: int — Zero-based line number.
        character: int — Zero-based character offset.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP rename range and placeholder text.
        """
        return self.parser_service.lsp_prepare_rename(
            file=file, line=line, character=character,
            ensure_server=ensure_server, timeout_ms=timeout_ms,
        )

    def lsp_code_actions(
        self,
        file: str | Path,
        start_line: int,
        start_character: int,
        end_line: int,
        end_character: int,
        diagnostics: list[dict] | None = None,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Get available code actions for a range via LSP.

        file: str | Path — Absolute path to the source file.
        start_line: int — Zero-based start line.
        start_character: int — Zero-based start character.
        end_line: int — Zero-based end line.
        end_character: int — Zero-based end character.
        diagnostics: list[dict] | None — Diagnostics to include in context.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP available code actions.
        """
        return self.parser_service.lsp_code_actions(
            file=file,
            start_line=start_line, start_character=start_character,
            end_line=end_line, end_character=end_character,
            diagnostics=diagnostics,
            ensure_server=ensure_server, timeout_ms=timeout_ms,
        )

    def lsp_signature_help(
        self,
        file: str | Path,
        line: int,
        character: int,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Get signature help at position via LSP.

        file: str | Path — Absolute path to the source file.
        line: int — Zero-based line number.
        character: int — Zero-based character offset.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP active signature and parameter info.
        """
        return self.parser_service.lsp_signature_help(
            file=file, line=line, character=character,
            ensure_server=ensure_server, timeout_ms=timeout_ms,
        )

    def lsp_workspace_symbols(
        self,
        query: str,
        file: str | Path | None = None,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Search for symbols across the workspace via LSP.

        query: str — Symbol search query.
        file: str | Path | None — File to identify which language server to use.
        ensure_server: bool — Auto-start Rust service when unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP matching workspace symbols.
        """
        return self.parser_service.lsp_workspace_symbols(
            query=query, file=file,
            ensure_server=ensure_server, timeout_ms=timeout_ms,
        )

    def lsp_sessions(
        self,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        List active LSP server sessions tracked by the Rust service.
        """
        return self.parser_service.lsp_sessions(
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def lsp_shutdown_all(
        self,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Shut down all active LSP server sessions tracked by the Rust service.
        """
        return self.parser_service.lsp_shutdown_all(
            ensure_server=ensure_server,
            timeout_ms=timeout_ms,
        )

    def semantic_search(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 5,
        language: Optional[str] = None,
        path_prefix: Optional[str] = None,
        project_name: Optional[str] = None,
        use_keywords: bool = False,
        keyword_weight: float = 0.3,
        rerank: bool = False,
        rrf_k: int = 60,
        top_rank_bonus_1: float = 0.05,
        top_rank_bonus_2_3: float = 0.02,
        rerank_top3_retrieval_weight: float = 0.75,
        rerank_top10_retrieval_weight: float = 0.60,
        rerank_tail_retrieval_weight: float = 0.40,
        rerank_candidate_multiplier: int = 4,
        include_source: bool = True,
        path: str | Path | None = None,
    ) -> list:
        """
        Run semantic search over indexed code vectors.

        query: str — Search query text.
        mode: str — Search mode (code, desc, hybrid).
        limit: int — Maximum number of results.
        language: Optional[str] — Optional language filter.
        path_prefix: Optional[str] — Optional file path prefix filter.
        project_name: Optional[str] — Project collection override.
        use_keywords: bool — Enable keyword overlap boosting.
        keyword_weight: float — Keyword blend weight.
        rerank: bool — Enable reranker blending.
        rrf_k: int — RRF constant.
        top_rank_bonus_1: float — Bonus added for rank #1 in each list.
        top_rank_bonus_2_3: float — Bonus added for ranks #2-3 in each list.
        rerank_top3_retrieval_weight: float — Retrieval weight for top 1-3 candidates.
        rerank_top10_retrieval_weight: float — Retrieval weight for top 4-10 candidates.
        rerank_tail_retrieval_weight: float — Retrieval weight for lower-ranked candidates.
        rerank_candidate_multiplier: int — Candidate multiplier before reranking.
        Returns: list — SearchResult list.
        """
        retrieval_root = self._resolve_retrieval_root(path)
        self._ensure_background_warm_started(retrieval_root)
        with self._activity_scope():
            project = project_name if project_name else detect_project_name(retrieval_root, manual_override=None)
            searcher = self._get_searcher(project, rerank=rerank)

            if mode == "code":
                results = searcher.search_code(
                    query=query,
                    limit=limit,
                    language=language,
                    path_prefix=path_prefix,
                    use_keywords=use_keywords,
                    keyword_weight=keyword_weight,
                    rerank=rerank,
                    include_source=include_source,
                )
                return self._focus_artifact_results(query, results, retrieval_root) if include_source else results

            if mode == "desc":
                results = searcher.search_description(
                    query=query,
                    limit=limit,
                    language=language,
                    path_prefix=path_prefix,
                    use_keywords=use_keywords,
                    keyword_weight=keyword_weight,
                    rerank=rerank,
                    include_source=include_source,
                )
                return self._focus_artifact_results(query, results, retrieval_root) if include_source else results

            results = searcher.search_hybrid(
                query=query,
                limit=limit,
                language=language,
                path_prefix=path_prefix,
                use_keywords=use_keywords,
                keyword_weight=keyword_weight,
                rerank=rerank,
                rrf_k=rrf_k,
                top_rank_bonus_1=top_rank_bonus_1,
                top_rank_bonus_2_3=top_rank_bonus_2_3,
                rerank_top3_retrieval_weight=rerank_top3_retrieval_weight,
                rerank_top10_retrieval_weight=rerank_top10_retrieval_weight,
                rerank_tail_retrieval_weight=rerank_tail_retrieval_weight,
                rerank_candidate_multiplier=rerank_candidate_multiplier,
                include_source=include_source,
            )
            return self._focus_artifact_results(query, results, retrieval_root) if include_source else results

    def semantic_search_bundle(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 5,
        language: Optional[str] = None,
        path_prefix: Optional[str] = None,
        project_name: Optional[str] = None,
        use_keywords: bool = False,
        keyword_weight: float = 0.3,
        rerank: bool = False,
        related_seed_files: int = 1,
        related_file_limit: int = 8,
        include_graph_related: bool = True,
        include_source: bool = True,
        path: str | Path | None = None,
    ) -> dict:
        """
        Run semantic retrieval and expand related files from the import graph around top hits.

        This is intended for deeper AI workflows that need both the best semantic anchors
        and a small set of adjacent files to inspect next.
        """
        retrieval_root = self._resolve_retrieval_root(path)
        self._ensure_background_warm_started(retrieval_root)
        with self._activity_scope():
            project = project_name if project_name else detect_project_name(retrieval_root, manual_override=None)
            tooling_query = self._looks_like_tooling_query(query)
            semantic_limit = max(limit, 1) * 4
            if tooling_query:
                semantic_limit = max(semantic_limit, 20)
            results = self.semantic_search(
                query=query,
                mode=mode,
                limit=semantic_limit,
                language=language,
                path_prefix=path_prefix,
                project_name=project,
                use_keywords=use_keywords,
                keyword_weight=keyword_weight,
                rerank=rerank,
                include_source=False,
                path=retrieval_root,
            )
            anchors, semantic_neighbors = self._split_semantic_bundle_results(
                results=results,
                anchor_limit=limit,
                related_file_limit=related_file_limit,
            )
            tooling_neighbors = self._tooling_related_semantic_neighbors(
                results=results,
                tooling_query=tooling_query,
                excluded_paths={item.file_path for item in anchors},
                related_file_limit=related_file_limit,
            )
            tooling_paths = {item["file_path"] for item in tooling_neighbors}
            prioritized_related = tooling_neighbors + [
                item
                for item in semantic_neighbors
                if item["file_path"] not in tooling_paths
            ]

            related = []
            if include_graph_related:
                related = self._related_files_for_results(
                    results=anchors,
                    related_seed_files=related_seed_files,
                    related_file_limit=max(0, related_file_limit - len(prioritized_related)),
                    path=retrieval_root,
                )
            related_callers = self._related_callers_for_results(results, path=retrieval_root)
            final_anchors = self._hydrate_search_result_sources(anchors) if include_source else anchors
            if include_source:
                final_anchors = self._focus_artifact_results(query, final_anchors, retrieval_root)

            return {
                "query": query,
                "project_name": project,
                "results": final_anchors,
                "related_files": prioritized_related[:related_file_limit] + related,
                "related_callers": related_callers,
            }

    def semantic_search_auto(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 5,
        language: Optional[str] = None,
        path_prefix: Optional[str] = None,
        project_name: Optional[str] = None,
        use_keywords: bool = False,
        keyword_weight: float = 0.3,
        rerank: bool = False,
        related_seed_files: int = 1,
        related_file_limit: int = 8,
        include_source: bool = True,
        path: str | Path | None = None,
    ) -> dict:
        """
        Automatically choose between plain semantic search and the graph-aware bundle primitive.
        """
        retrieval_root = self._resolve_retrieval_root(path)
        self._ensure_background_warm_started(retrieval_root)
        with self._activity_scope():
            project = project_name if project_name else detect_project_name(retrieval_root, manual_override=None)
            if self._should_use_bundle_for_query(query):
                bundle = self.semantic_search_bundle(
                    query=query,
                    mode=mode,
                    limit=limit,
                    language=language,
                    path_prefix=path_prefix,
                    project_name=project,
                    use_keywords=use_keywords,
                    keyword_weight=keyword_weight,
                    rerank=rerank,
                    related_seed_files=related_seed_files,
                    related_file_limit=related_file_limit,
                    include_graph_related=self._should_use_graph_related_for_query(query),
                    path=retrieval_root,
                )
                bundle["mode"] = "bundle"
                return bundle

            return {
                "query": query,
                "project_name": project,
                "mode": "search",
                "results": self.semantic_search(
                    query=query,
                    mode=mode,
                    limit=limit,
                    language=language,
                    path_prefix=path_prefix,
                    project_name=project,
                    use_keywords=use_keywords,
                    keyword_weight=keyword_weight,
                    rerank=rerank,
                    include_source=include_source,
                    path=retrieval_root,
                ),
                "related_files": [],
                "related_callers": [],
            }

    def retrieve_context_auto(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 5,
        language: Optional[str] = None,
        path_prefix: Optional[str] = None,
        project_name: Optional[str] = None,
        use_keywords: bool = False,
        keyword_weight: float = 0.3,
        rerank: bool = False,
        related_seed_files: int = 1,
        related_file_limit: int = 8,
        path: str | Path | None = None,
    ) -> dict:
        """
        Automatically choose the best retrieval primitive for AI workflows.

        Exact symbol-oriented questions use the Rust symbol index first so the SDK can
        return the real definition source without paying an embedding round trip. Broader
        natural-language questions fall back to semantic_search_auto(...).
        """
        retrieval_root = self._resolve_retrieval_root(path)
        self._ensure_background_warm_started(retrieval_root)
        with self._activity_scope():
            project = project_name if project_name else detect_project_name(retrieval_root, manual_override=None)
            symbol_query = self._extract_symbol_query_candidate(query)
            if symbol_query:
                expand_symbol_context = self._should_expand_symbol_context_results(query)
                results = self._symbol_lookup_search_results(
                    query=symbol_query,
                    limit=1 if expand_symbol_context else limit,
                    language=language,
                    path_prefix=path_prefix,
                    path=retrieval_root,
                )
                if results:
                    if expand_symbol_context:
                        results = self._expand_symbol_context_results(
                            query=query,
                            results=results,
                            limit=limit,
                            path=retrieval_root,
                        )
                    if self._should_use_bundle_for_query(query):
                        return {
                            "query": query,
                            "project_name": project,
                            "mode": "symbol_bundle",
                            "symbol_query": symbol_query,
                            "results": results,
                            "related_files": self._related_files_for_results(
                                results=results,
                                related_seed_files=related_seed_files,
                                related_file_limit=related_file_limit,
                                path=retrieval_root,
                            ) if self._should_use_graph_related_for_query(query) else [],
                            "related_callers": self._related_callers_for_results(results, path=retrieval_root),
                        }

                    return {
                        "query": query,
                        "project_name": project,
                        "mode": "symbol",
                        "symbol_query": symbol_query,
                        "results": results,
                        "related_files": [],
                        "related_callers": [],
                    }

            should_use_bundle = self._should_use_bundle_for_query(query)
            should_use_graph_related = self._should_use_graph_related_for_query(query)

            if not should_use_graph_related:
                try:
                    text_result = self.text_search(
                        query=query,
                        path=retrieval_root,
                        limit=max(limit + related_file_limit, 8),
                        intent_mode=True,
                        intent_level=2,
                    )
                except Exception:
                    text_result = None
                if text_result and self._should_use_text_primary_for_query(query, text_result):
                    return self._text_primary_payload(
                        query=query,
                        project_name=project,
                        text_result=text_result,
                        limit=limit,
                        related_file_limit=related_file_limit,
                        path=retrieval_root,
                    )
            else:
                text_result = None

            if should_use_bundle:
                try:
                    bundle = self.semantic_search_bundle(
                        query=query,
                        mode=mode,
                        limit=limit,
                        language=language,
                        path_prefix=path_prefix,
                        project_name=project,
                        use_keywords=use_keywords,
                        keyword_weight=keyword_weight,
                        rerank=rerank,
                        related_seed_files=related_seed_files,
                        related_file_limit=related_file_limit,
                        include_graph_related=should_use_graph_related,
                        path=retrieval_root,
                    )
                except Exception as exc:
                    fallback = self._text_fallback_payload_for_vector_error(
                        exc=exc,
                        query=query,
                        project_name=project,
                        path=retrieval_root,
                        limit=limit,
                        related_file_limit=related_file_limit,
                        text_result=text_result,
                    )
                    if fallback is not None:
                        return fallback
                    raise
                bundle["mode"] = "bundle"
                bundle["symbol_query"] = None
                return bundle

            try:
                payload = self.semantic_search_auto(
                    query=query,
                    mode=mode,
                    limit=limit,
                    language=language,
                    path_prefix=path_prefix,
                    project_name=project,
                    use_keywords=use_keywords,
                    keyword_weight=keyword_weight,
                    rerank=rerank,
                    related_seed_files=related_seed_files,
                    related_file_limit=related_file_limit,
                    include_source=False,
                    path=retrieval_root,
                )
            except Exception as exc:
                fallback = self._text_fallback_payload_for_vector_error(
                    exc=exc,
                    query=query,
                    project_name=project,
                    path=retrieval_root,
                    limit=limit,
                    related_file_limit=related_file_limit,
                    text_result=text_result,
                )
                if fallback is not None:
                    return fallback
                raise
            payload["symbol_query"] = None
            if payload.get("mode") == "search":
                payload["results"] = self._focus_artifact_results(query, payload["results"], retrieval_root)
                lexical_related = self._lexical_related_files_for_query(
                    query=query,
                    results=payload["results"],
                    related_file_limit=related_file_limit,
                    path=retrieval_root,
                )
                if self._should_add_graph_lexical_companions(query):
                    graph_related = self._graph_lexical_related_files_for_query(
                        query=query,
                        results=payload["results"],
                        existing_related=lexical_related,
                        related_file_limit=related_file_limit,
                        path=retrieval_root,
                    )
                    reserved_graph = graph_related[:1]
                    seen_graph_paths = {entry["file_path"] for entry in reserved_graph}
                    payload["related_files"] = reserved_graph + [
                        item for item in lexical_related
                        if item["file_path"] not in seen_graph_paths
                    ] + [
                        item for item in graph_related[1:]
                        if item["file_path"] not in {entry["file_path"] for entry in lexical_related}
                    ]
                    payload["related_files"] = payload["related_files"][:related_file_limit]
                else:
                    payload["related_files"] = lexical_related
                if self._should_use_graph_related_for_query(query):
                    payload["results"], payload["related_files"] = self._promote_graph_related_results(
                        query=query,
                        results=payload["results"],
                        related_files=payload["related_files"],
                        limit=limit,
                    )
                payload["results"] = self._hydrate_search_result_sources(payload["results"])
            return payload

    def _text_fallback_payload_for_vector_error(
        self,
        exc: Exception,
        query: str,
        project_name: str,
        path: str | Path,
        limit: int,
        related_file_limit: int,
        text_result: TextSearchResult | None = None,
    ) -> dict | None:
        """
        Fall back to Rust text retrieval when vector search is unavailable for a target project.
        """
        if not self._looks_like_missing_vector_index_error(exc):
            return None

        if text_result is None:
            try:
                text_result = self.text_search(
                    query=query,
                    path=path,
                    limit=max(limit + related_file_limit, 8),
                    intent_mode=True,
                    intent_level=2,
                )
            except Exception:
                return None

        if not text_result or not text_result.matches:
            return None

        return self._text_primary_payload(
            query=query,
            project_name=project_name,
            text_result=text_result,
            limit=limit,
            related_file_limit=related_file_limit,
            path=path,
        )

    def _looks_like_missing_vector_index_error(self, exc: Exception) -> bool:
        """
        Detect missing or uninitialized vector-search collections for semantic retrieval.
        """
        message = str(exc).lower()
        return (
            "404 client error" in message
            or ("collection" in message and "not found" in message)
            or "/points/query/batch" in message
            or "/points/query" in message
        )

    def structured_search(
        self,
        query: str,
        limit: int = 5,
        language: Optional[str] = None,
        path_prefix: Optional[str] = None,
        project_name: Optional[str] = None,
        rerank: bool = False,
        use_keywords: bool = True,
        keyword_weight: float = 0.3,
        first_query_weight: float = 2.0,
        rrf_k: int = 60,
        top_rank_bonus_1: float = 0.05,
        top_rank_bonus_2_3: float = 0.02,
        rerank_top3_retrieval_weight: float = 0.75,
        rerank_top10_retrieval_weight: float = 0.60,
        rerank_tail_retrieval_weight: float = 0.40,
        rerank_candidate_multiplier: int = 4,
    ) -> list:
        """
        Run structured typed-query semantic search (lex/vec/hyde lines).

        query: str — Structured query text with type prefixes.
        limit: int — Maximum number of results.
        language: Optional[str] — Optional language filter.
        path_prefix: Optional[str] — Optional file path prefix filter.
        project_name: Optional[str] — Project collection override.
        rerank: bool — Enable reranker blending.
        use_keywords: bool — Enable keyword overlap boosting.
        keyword_weight: float — Keyword blend weight.
        first_query_weight: float — Extra weight for first sub-query list.
        rrf_k: int — RRF constant.
        top_rank_bonus_1: float — Bonus added for rank #1 in each list.
        top_rank_bonus_2_3: float — Bonus added for ranks #2-3 in each list.
        rerank_top3_retrieval_weight: float — Retrieval weight for top 1-3 candidates.
        rerank_top10_retrieval_weight: float — Retrieval weight for top 4-10 candidates.
        rerank_tail_retrieval_weight: float — Retrieval weight for lower-ranked candidates.
        rerank_candidate_multiplier: int — Candidate multiplier before reranking.
        Returns: list — SearchResult list.
        """
        from engine.src.query_dsl import parse_structured_query

        project = project_name if project_name else detect_project_name(Path.cwd(), manual_override=None)
        sub_queries = parse_structured_query(query)
        if not sub_queries:
            return []

        searcher = self._get_searcher(project, rerank=rerank)
        return searcher.search_structured(
            sub_queries=sub_queries,
            limit=limit,
            language=language,
            path_prefix=path_prefix,
            use_keywords=use_keywords,
            keyword_weight=keyword_weight,
            rerank=rerank,
            first_query_weight=first_query_weight,
            rrf_k=rrf_k,
            top_rank_bonus_1=top_rank_bonus_1,
            top_rank_bonus_2_3=top_rank_bonus_2_3,
            rerank_top3_retrieval_weight=rerank_top3_retrieval_weight,
            rerank_top10_retrieval_weight=rerank_top10_retrieval_weight,
            rerank_tail_retrieval_weight=rerank_tail_retrieval_weight,
            rerank_candidate_multiplier=rerank_candidate_multiplier,
        )

    def build_dynamic_search_instructions(self, project_name: Optional[str] = None) -> str:
        """
        Build dynamic search guidance string for MCP/system prompts.

        project_name: Optional[str] — Project collection override.
        Returns: str — Human-readable instructions based on current index status.
        """
        project = project_name if project_name else detect_project_name(Path.cwd(), manual_override=None)
        collection = self._get_collection(project)
        info = collection.info()
        points = int(info.get("points_count", 0))

        lines = [
            f"QuickContext semantic index for project '{project}' has {points} chunks.",
            "Use 'search' for standard semantic retrieval.",
            "Use retrieve_context_auto(...) as the default AI entrypoint when queries may be exact symbols, broad architecture questions, or a mix of both.",
            "Use semantic_search_auto(...) when you want the SDK to choose between fast direct retrieval and a deeper graph-aware bundle.",
            "Use structured mode with lex:/vec:/hyde: lines for intent-controlled retrieval.",
            "Use semantic_search_bundle(...) for broader cross-file or architecture questions that need follow-up files.",
            "Use --rerank for higher precision when candidate count is broad.",
            "Use --path to scope retrieval and reduce noise.",
        ]

        if points == 0:
            lines.append("Index appears empty. Run index before relying on semantic search.")

        return "\n".join(lines)

    def _should_use_text_primary_for_query(self, query: str, text_result: TextSearchResult) -> bool:
        """
        Decide when Rust text search is strong enough to become the primary result set.
        """
        if not text_result.matches:
            return False

        query_keywords = set(extract_keywords(query, max_keywords=20))
        for item in text_result.matches[:5]:
            if self._should_skip_lexical_related_path(query, item.file_path):
                continue

            lexical_hits = len(item.matched_terms)
            if lexical_hits >= 4:
                return True

            if query_keywords:
                matched = {term.lower() for term in item.matched_terms}
                if len(query_keywords.intersection(matched)) >= 3:
                    return True

        return False

    def _text_primary_payload(
        self,
        query: str,
        project_name: str,
        text_result: TextSearchResult,
        limit: int,
        related_file_limit: int,
        path: str | Path,
    ) -> dict:
        """
        Build an AI-facing payload from Rust text search results.
        """
        primary_matches, related_matches = self._split_text_matches_for_query(
            query=query,
            matches=text_result.matches,
            limit=limit,
        )
        primary_results = self._text_matches_to_search_results(primary_matches)
        primary_results = self._focus_generated_file_match_results(query, primary_results, path=path)
        import_related = self._text_import_related_files_for_results(
            query=query,
            results=primary_results,
            related_seed_files=min(2, len(primary_results)),
            related_file_limit=max(related_file_limit * 2, 8),
            path=path,
        )
        module_related = self._module_local_related_files_for_results(
            query=query,
            results=primary_results,
            related_file_limit=max(related_file_limit * 2, 8),
        )
        lexical_related = self._text_matches_to_related_files(
            query=query,
            matches=related_matches,
            related_file_limit=max(related_file_limit * 2, 8),
            excluded_paths={str(item.file_path) for item in primary_matches},
        )
        return {
            "query": query,
            "project_name": project_name,
            "mode": "text",
            "symbol_query": None,
            "results": primary_results,
            "related_files": self._merge_ranked_related_file_lists(
                module_related,
                import_related,
                lexical_related,
                limit=related_file_limit,
            ),
            "related_callers": [],
        }

    def _focus_generated_file_match_results(
        self,
        query: str,
        results: list,
        path: str | Path,
        max_files: int = 2,
    ) -> list:
        """
        Replace low-signal generated-bundle file matches with a tighter grep-backed focus snippet.
        """
        if not results or max_files <= 0:
            return results

        query_terms = [term for term in extract_keywords(query, max_keywords=6) if len(term) >= 4]
        if not query_terms:
            return results

        focused: list = []
        refined = 0
        for item in results:
            file_path = str(getattr(item, "file_path", "") or "")
            symbol_kind = str(getattr(item, "symbol_kind", "") or "").lower()
            if (
                refined >= max_files
                or symbol_kind != "file_match"
                or not self._looks_like_generated_bundle_path(file_path)
            ):
                focused.append(item)
                continue

            refined_item = self._focus_single_artifact_result(query_terms=query_terms, item=item, path=path)
            focused.append(refined_item or item)
            if refined_item is not None:
                refined += 1
        focused.sort(key=lambda item: float(getattr(item, "score", 0.0)), reverse=True)
        return focused

    def _looks_like_generated_bundle_path(self, file_path: str) -> bool:
        normalized = str(file_path or "").replace("\\", "/")
        name = Path(normalized).name.lower()
        return bool(
            name.endswith(".js")
            and (
                _GENERATED_BUNDLE_NAME_PATTERN.search(name)
                or name.startswith("bundle.")
                or name.startswith("vendor.")
                or "~" in name
            )
        )

    def _text_import_related_files_for_results(
        self,
        query: str,
        results: list,
        related_seed_files: int,
        related_file_limit: int,
        path: str | Path,
    ) -> list[dict]:
        """
        Collect import-neighbor companions around text-primary anchors and rank them for context usefulness.
        """
        if (
            not results
            or related_file_limit <= 0
            or related_seed_files <= 0
            or not self._should_expand_text_primary_related(query)
        ):
            return []

        seeds: list[str] = []
        seen_seed_files: set[str] = set()
        for result in results:
            file_path = str(result.file_path)
            if file_path in seen_seed_files:
                continue
            seen_seed_files.add(file_path)
            seeds.append(file_path)
            if len(seeds) >= related_seed_files:
                break

        related_by_path: dict[str, dict] = {}
        excluded_paths = set(seeds)
        project_root = self._resolve_retrieval_root(path)

        for seed_file in seeds:
            try:
                neighbors = self.import_neighbors(file=seed_file, path=project_root)
            except Exception:
                continue
            self._merge_related_edges(
                seed_file=seed_file,
                relation="imports",
                edges=neighbors.imports,
                related_by_path=related_by_path,
                excluded_paths=excluded_paths,
            )
            self._merge_related_edges(
                seed_file=seed_file,
                relation="imported_by",
                edges=neighbors.importers,
                related_by_path=related_by_path,
                excluded_paths=excluded_paths,
            )

        related = list(related_by_path.values())
        related.sort(
            key=lambda item: (
                -self._text_related_file_priority(query, item),
                item["distance"],
                item["file_path"],
            )
        )
        return related[:related_file_limit]

    def _module_local_related_files_for_results(
        self,
        query: str,
        results: list,
        related_file_limit: int,
    ) -> list[dict]:
        """
        Collect nearby module-local files around text-primary anchors using path structure and query overlap.
        """
        if not results or related_file_limit <= 0 or not self._should_expand_text_primary_related(query):
            return []

        query_keywords = set(extract_keywords(query, max_keywords=20))
        related_by_path: dict[str, dict] = {}
        excluded_paths = {
            str(getattr(item, "file_path"))
            for item in results
            if getattr(item, "file_path", None)
        }

        for result in results[: min(2, len(results))]:
            raw_seed_file = str(getattr(result, "file_path", "")).replace("\\\\?\\", "")
            if not raw_seed_file:
                continue
            seed_path = Path(raw_seed_file)
            if not seed_path.exists():
                continue
            for candidate_path, distance in self._iter_module_local_candidate_paths(seed_path):
                normalized_candidate = str(candidate_path).replace("\\\\?\\", "")
                if normalized_candidate in excluded_paths or self._should_skip_lexical_related_path(query, normalized_candidate):
                    continue
                priority = self._module_local_candidate_priority(
                    query_keywords=query_keywords,
                    seed_path=seed_path,
                    candidate_path=candidate_path,
                    distance=distance,
                )
                if priority <= 0:
                    continue

                item = related_by_path.setdefault(
                    normalized_candidate,
                    {
                        "file_path": normalized_candidate,
                        "distance": distance,
                        "relations": [],
                        "_priority": priority,
                    },
                )
                item["distance"] = min(item["distance"], distance)
                item["_priority"] = max(item["_priority"], priority)
                item["relations"].append(
                    {
                        "relation": "module_local_neighbor",
                        "seed_file": str(seed_path),
                        "module_path": str(candidate_path.parent),
                        "language": getattr(result, "language", None),
                        "line": 0,
                    }
                )

        related = sorted(
            related_by_path.values(),
            key=lambda item: (-item["_priority"], item["distance"], item["file_path"]),
        )
        for item in related:
            item.pop("_priority", None)
        return related[:related_file_limit]

    def _iter_module_local_candidate_paths(self, seed_path: Path) -> list[tuple[Path, int]]:
        """
        Yield nearby candidate files from the seed directory, its immediate children, and its parent directory.
        """
        candidates: list[tuple[Path, int]] = []
        seen: set[Path] = set()
        resolved_seed = seed_path.resolve()

        def add_directory(directory: Path, distance: int, include_children: bool = False) -> None:
            if not directory.exists() or not directory.is_dir():
                return
            try:
                entries = list(directory.iterdir())
            except OSError:
                return
            for entry in entries:
                if entry.is_file():
                    resolved = entry.resolve()
                    if resolved == resolved_seed or resolved in seen:
                        continue
                    if not self._is_supported_module_local_candidate(entry, seed_path):
                        continue
                    seen.add(resolved)
                    candidates.append((resolved, distance))
                elif include_children and entry.is_dir():
                    try:
                        child_entries = list(entry.iterdir())
                    except OSError:
                        continue
                    for child in child_entries:
                        if not child.is_file():
                            continue
                        resolved = child.resolve()
                        if resolved == resolved_seed or resolved in seen:
                            continue
                        if not self._is_supported_module_local_candidate(child, seed_path):
                            continue
                        seen.add(resolved)
                        candidates.append((resolved, distance + 1))

        add_directory(seed_path.parent, 1, include_children=True)
        parent_directory = seed_path.parent.parent
        if parent_directory != seed_path.parent:
            add_directory(parent_directory, 2, include_children=False)
        return candidates

    def _is_supported_module_local_candidate(self, candidate_path: Path, seed_path: Path) -> bool:
        """
        Keep module-local companion expansion focused on likely code files.
        """
        code_suffixes = {
            ".c",
            ".cc",
            ".cpp",
            ".cjs",
            ".cs",
            ".go",
            ".h",
            ".hpp",
            ".java",
            ".js",
            ".jsx",
            ".kt",
            ".mjs",
            ".php",
            ".py",
            ".rb",
            ".rs",
            ".swift",
            ".ts",
            ".tsx",
        }
        candidate_suffix = candidate_path.suffix.lower()
        seed_suffix = seed_path.suffix.lower()
        if candidate_suffix and candidate_suffix == seed_suffix:
            return True
        return candidate_suffix in code_suffixes

    def _module_local_candidate_priority(
        self,
        query_keywords: set[str],
        seed_path: Path,
        candidate_path: Path,
        distance: int,
    ) -> int:
        candidate_tokens = self._split_symbol_text_tokens(candidate_path.as_posix())
        seed_tokens = self._split_symbol_text_tokens(seed_path.as_posix())
        overlap = len(query_keywords.intersection(candidate_tokens))
        seed_overlap = len(seed_tokens.intersection(candidate_tokens))

        score = overlap * 4
        score += min(seed_overlap, 3)

        if candidate_path.parent == seed_path.parent:
            score += 3
        elif candidate_path.parent.parent == seed_path.parent:
            score += 2
        elif candidate_path.parent == seed_path.parent.parent:
            score += 1

        generic_tokens = {"index", "main", "logger", "types", "type", "utils", "util", "constants", "shared"}
        if candidate_tokens.intersection(generic_tokens) and not query_keywords.intersection(candidate_tokens):
            score -= 2

        if distance > 1:
            score -= distance - 1

        return score

    def _text_related_file_priority(self, query: str, related_file: dict) -> int:
        """
        Score related files for text-primary context so module-local implementation files win over low-signal utilities.
        """
        file_path = str(related_file["file_path"]).replace("\\", "/")
        query_keywords = set(extract_keywords(query, max_keywords=20))
        candidate_tokens = self._split_symbol_text_tokens(file_path.replace("/", " "))

        score = len(query_keywords.intersection(candidate_tokens)) * 3
        relations = related_file.get("relations", [])
        score += min(len(relations), 2)

        seed_file = ""
        if relations:
            seed_file = str(relations[0].get("seed_file", "")).replace("\\", "/")
        if seed_file:
            seed_path = Path(seed_file)
            candidate_path = Path(file_path)
            if candidate_path.parent == seed_path.parent:
                score += 4
            elif candidate_path.parent.parent == seed_path.parent:
                score += 3
            elif candidate_path.parent == seed_path.parent.parent:
                score += 1

        low_signal_tokens = {"logger", "types", "type", "utils", "util", "constants", "events", "schemas", "shared"}
        if candidate_tokens.intersection(low_signal_tokens) and not query_keywords.intersection(candidate_tokens):
            score -= 2

        return score

    def _merge_ranked_related_file_lists(
        self,
        *related_lists: list[dict],
        limit: int,
    ) -> list[dict]:
        """
        Merge ranked related-file lists while preserving the earlier list priority and deduping paths.
        """
        if limit <= 0:
            return []

        merged_by_path: dict[str, dict] = {}
        ordered: list[dict] = []
        for items in related_lists:
            for item in items:
                file_path = str(item["file_path"])
                existing = merged_by_path.get(file_path)
                if existing is None:
                    merged_item = {
                        "file_path": file_path,
                        "distance": int(item.get("distance", 1)),
                        "relations": list(item.get("relations", [])),
                    }
                    merged_by_path[file_path] = merged_item
                    ordered.append(merged_item)
                else:
                    existing["distance"] = min(existing["distance"], int(item.get("distance", 1)))
                    existing["relations"].extend(item.get("relations", []))
                if len(ordered) >= limit:
                    break
            if len(ordered) >= limit:
                break
        return ordered[:limit]

    def _should_expand_text_primary_related(self, query: str) -> bool:
        """
        Detect broad natural-language questions where text-primary results need richer multi-file companions.
        """
        keywords = set(extract_keywords(query, max_keywords=20))
        if len(keywords) < 5:
            return False

        flow_terms = {
            "across",
            "archive",
            "archived",
            "back",
            "coordinate",
            "flow",
            "history",
            "initialize",
            "manage",
            "merge",
            "migration",
            "pipeline",
            "route",
            "routing",
            "session",
            "start",
            "stored",
            "support",
            "through",
            "trash",
            "version",
            "watching",
            "workspace",
        }
        return bool(keywords.intersection(flow_terms))

    def _split_text_matches_for_query(
        self,
        query: str,
        matches: list[TextSearchMatch],
        limit: int,
    ) -> tuple[list[TextSearchMatch], list[TextSearchMatch]]:
        """
        Keep the strongest text hit first, then fill remaining primary slots with filtered matches.
        """
        if not matches:
            return [], []

        seed_index = 0
        for idx, item in enumerate(matches[:5]):
            if not self._should_skip_lexical_related_path(query, str(item.file_path)):
                seed_index = idx
                break

        primary: list[TextSearchMatch] = [matches[seed_index]]
        related_pool: list[TextSearchMatch] = []
        seen_paths = {str(matches[seed_index].file_path)}

        for idx, item in enumerate(matches):
            if idx == seed_index:
                continue
            file_path = str(item.file_path)
            if file_path in seen_paths:
                continue
            seen_paths.add(file_path)
            if len(primary) < max(limit, 1) and not self._should_skip_lexical_related_path(query, file_path):
                primary.append(item)
                continue
            related_pool.append(item)

        return primary, related_pool

    def _should_expand_symbol_context_results(self, query: str) -> bool:
        """
        Detect exact-symbol questions that need helper-level implementation context.
        """
        keywords = set(extract_keywords(query, max_keywords=20))
        behavior_terms = {
            "how",
            "why",
            "work",
            "works",
            "flow",
            "logic",
            "behavior",
            "merge",
            "fuse",
            "route",
            "routed",
            "decide",
            "choose",
            "select",
            "build",
            "create",
            "construct",
            "compose",
            "convert",
            "update",
            "delete",
            "remove",
            "expand",
            "hydrate",
            "rank",
            "score",
            "pack",
            "copy",
            "connect",
            "upsert",
            "filter",
        }
        definition_terms = {"where", "defined", "definition", "locate", "find"}
        return bool(keywords.intersection(behavior_terms)) and not (
            keywords.intersection(definition_terms) and not keywords.intersection(behavior_terms)
        )

    def _lexical_related_files_for_query(
        self,
        query: str,
        results: list,
        related_file_limit: int,
        path: str | Path,
    ) -> list[dict]:
        """
        Add fast lexical file companions for non-symbol search-mode queries.
        """
        if related_file_limit <= 0:
            return []

        excluded_paths = {
            str(getattr(item, "file_path"))
            for item in results
            if getattr(item, "file_path", None)
        }
        try:
            text_result = self.text_search(
                query=query,
                path=path,
                limit=max(related_file_limit * 4, 12),
                intent_mode=True,
                intent_level=2,
            )
        except Exception:
            return []

        related: list[dict] = []
        for item in text_result.matches:
            file_path = str(item.file_path).replace("\\\\?\\", "")
            if file_path in excluded_paths:
                continue
            if self._should_skip_lexical_related_path(query, file_path):
                continue
            excluded_paths.add(file_path)
            related.append(
                {
                    "file_path": file_path,
                    "distance": 1,
                    "relations": [
                        {
                            "relation": "lexical_neighbor",
                            "seed_file": "",
                            "module_path": "",
                            "language": item.language,
                            "line": item.snippet_line_start,
                        }
                    ],
                }
            )
            if len(related) >= related_file_limit:
                break
        return related

    def _graph_lexical_related_files_for_query(
        self,
        query: str,
        results: list,
        existing_related: list[dict],
        related_file_limit: int,
        path: str | Path,
    ) -> list[dict]:
        """
        Add graph-oriented lexical companions for dependency and caller style queries.
        """
        if related_file_limit <= 0:
            return []

        graph_query = self._graph_companion_query(query)
        if not graph_query:
            return []

        excluded_paths = {
            str(getattr(item, "file_path", ""))
            for item in results
            if getattr(item, "file_path", None)
        }
        excluded_paths.update(item["file_path"] for item in existing_related)

        try:
            text_result = self.text_search(
                query=graph_query,
                path=path,
                limit=max(related_file_limit * 4, 12),
                intent_mode=True,
                intent_level=2,
            )
        except Exception:
            return []

        related: list[dict] = []
        for item in text_result.matches:
            file_path = str(item.file_path).replace("\\\\?\\", "")
            if file_path in excluded_paths:
                continue
            if self._should_skip_lexical_related_path(query, file_path):
                continue
            excluded_paths.add(file_path)
            related.append(
                {
                    "file_path": file_path,
                    "distance": 1,
                    "relations": [
                        {
                            "relation": "graph_lexical_neighbor",
                            "seed_file": "",
                            "module_path": "",
                            "language": item.language,
                            "line": item.snippet_line_start,
                        }
                    ],
                }
            )
            if len(related) >= related_file_limit:
                break
        return related

    def _promote_graph_related_results(
        self,
        query: str,
        results: list,
        related_files: list[dict],
        limit: int,
    ) -> tuple[list, list[dict]]:
        """
        Dedupe primary files and backfill graph-implementation files from related context.
        """
        primary: list = []
        seen_paths: set[str] = set()
        for item in results:
            file_path = str(getattr(item, "file_path", ""))
            if not file_path or file_path in seen_paths:
                continue
            seen_paths.add(file_path)
            primary.append(item)

        if len(primary) >= max(limit, 1):
            return primary[: max(limit, 1)], related_files

        scored_related = sorted(
            related_files,
            key=lambda item: self._graph_related_file_priority(query, item),
            reverse=True,
        )

        promoted_paths: set[str] = set()
        for item in scored_related:
            file_path = str(item["file_path"])
            if file_path in seen_paths:
                continue
            primary.append(self._related_file_to_search_result(item))
            seen_paths.add(file_path)
            promoted_paths.add(file_path)
            if len(primary) >= max(limit, 1):
                break

        remaining_related = [item for item in related_files if item["file_path"] not in promoted_paths]
        return primary[: max(limit, 1)], remaining_related

    def _graph_related_file_priority(self, query: str, related_file: dict) -> int:
        file_path = str(related_file["file_path"]).replace("\\", "/").lower()
        relation = related_file["relations"][0]["relation"]
        keywords = set(extract_keywords(query, max_keywords=20))
        path_tokens = self._split_symbol_text_tokens(file_path.replace("/", " "))

        import_query = bool(keywords.intersection({"import", "imports", "importer", "importers", "dependency", "dependencies", "neighbor", "neighbors", "module"}))
        call_query = bool(keywords.intersection({"call", "calls", "caller", "callers", "trace", "tracing", "traversal", "lookup", "edge", "edges"}))
        traversal_query = bool(keywords.intersection({"trace", "tracing", "traversal"}))

        score = 0
        if relation == "graph_lexical_neighbor":
            score += 2
        if import_query:
            score += len(path_tokens.intersection({"import", "imports", "importer", "importers", "graph", "dependency", "dependencies", "neighbor", "neighbors", "module"})) * 2
            if file_path.endswith("/engine/src/parsing.py"):
                score += 1
        if call_query:
            score += len(path_tokens.intersection({"call", "calls", "caller", "callers", "trace", "tracing", "graph", "lookup", "edge", "edges", "index", "type", "types"})) * 2
            if file_path.endswith("/engine/src/parsing.py"):
                score += 1

        if file_path.endswith("/engine/sdk.py"):
            if import_query:
                score += 2
            if traversal_query:
                score += 5
            score -= 2
        elif file_path.endswith("/engine/src/cli.py") or file_path.endswith("/engine/src/pipe.py"):
            score -= 3

        if traversal_query and file_path.endswith("/service/src/types.rs"):
            score -= 1

        return score

    def _graph_companion_query(self, query: str) -> str:
        """
        Build a condensed lexical query for graph- and dependency-oriented questions.
        """
        keywords = set(extract_keywords(query, max_keywords=20))
        if not keywords:
            return ""

        terms: list[str] = []
        seen: set[str] = set()

        def add(*values: str) -> None:
            for value in values:
                if value in seen:
                    continue
                seen.add(value)
                terms.append(value)

        if keywords.intersection({"import", "imports", "importer", "importers", "module", "dependency", "dependencies", "neighbor", "neighbors"}):
            add("import", "imports", "importers", "graph", "dependencies", "module")
        if keywords.intersection({"call", "calls", "caller", "callers", "trace", "tracing", "traversal", "lookup", "edges"}):
            add("call", "caller", "lookup", "graph", "edges", "trace")

        return " ".join(terms)

    def _should_add_graph_lexical_companions(self, query: str) -> bool:
        keywords = set(extract_keywords(query, max_keywords=20))
        return bool(keywords.intersection({"dependency", "dependencies", "module", "imports", "importers", "neighbors"}))

    def _text_matches_to_search_results(self, matches: list[TextSearchMatch]) -> list:
        """
        Convert Rust text-search matches into SearchResult rows for AI-facing payloads.
        """
        from engine.src.searcher import SearchResult

        results: list[SearchResult] = []
        seen: set[tuple[str, int, int]] = set()
        for item in matches:
            file_path = str(item.file_path).replace("\\\\?\\", "")
            key = (file_path, int(item.snippet_line_start), int(item.snippet_line_end))
            if key in seen:
                continue
            seen.add(key)
            results.append(
                SearchResult(
                    score=float(item.score),
                    file_path=file_path,
                    symbol_name=Path(file_path).name,
                    symbol_kind="file_match",
                    line_start=int(item.snippet_line_start),
                    line_end=int(item.snippet_line_end),
                    source=item.snippet,
                    description=item.snippet,
                    language=item.language,
                    path_context=self._build_symbol_path_context(file_path),
                )
            )
        return results

    def _hydrate_search_result_sources(self, results: list) -> list:
        """
        Hydrate source text only for the final semantic results kept by the router.
        """
        hydrated: list = []
        for item in results:
            if not getattr(item, "file_path", None):
                hydrated.append(item)
                continue
            if getattr(item, "source", ""):
                hydrated.append(item)
                continue
            source, _signature, _docstring = self._read_symbol_context(item)
            hydrated.append(replace(item, source=source))
        return hydrated

    def _focus_artifact_results(
        self,
        query: str,
        results: list,
        path: str | Path,
        max_files: int = 2,
    ) -> list:
        """
        Replace broad artifact blobs with a tighter file-local grep-backed focus snippet.
        """
        if not results or max_files <= 0:
            return results

        query_terms = [term for term in extract_keywords(query, max_keywords=6) if len(term) >= 4]
        if not query_terms:
            return results

        focused: list = []
        refined_files = 0
        for item in results:
            if (
                refined_files >= max_files
                or str(getattr(item, "symbol_kind", "")).lower() != "file_artifact"
                or not getattr(item, "file_path", None)
            ):
                focused.append(item)
                continue

            refined = self._focus_single_artifact_result(
                query_terms=query_terms,
                item=item,
                path=path,
            )
            focused.append(refined or item)
            if refined is not None:
                refined_files += 1
        return focused

    def _focus_single_artifact_result(
        self,
        query_terms: list[str],
        item: object,
        path: str | Path,
    ):
        """
        Build a more focused snippet inside one generated artifact file.
        """
        file_path = str(getattr(item, "file_path", "") or "")
        if not file_path:
            return None

        best_match = None
        best_score = 0
        seen_lines: dict[int, dict] = {}
        generic_action_terms = {"start", "show", "load", "open", "run", "begin", "init"}
        project_terms = {term for term in self._split_symbol_text_tokens(Path(str(path or "")).name) if len(term) >= 4}
        domain_terms = [term for term in query_terms if term not in generic_action_terms]
        domain_terms = [term for term in domain_terms if term not in project_terms]
        for term in query_terms[:4]:
            try:
                grep = self.grep_text(
                    term,
                    path=file_path,
                    limit=12,
                    before_context=2,
                    after_context=2,
                )
            except Exception:
                continue

            for match in grep.matches:
                state = seen_lines.setdefault(
                    int(match.line_number),
                    {"match": match, "terms": set()},
                )
                state["terms"].add(term)

        for state in seen_lines.values():
            match = state["match"]
            block = "\n".join([*match.context_before, match.line, *match.context_after]).lower()
            coverage = len(state["terms"])
            exact_mentions = sum(1 for term in query_terms if term in block)
            score = (coverage * 3) + exact_mentions
            if state["terms"].issubset(generic_action_terms) and domain_terms and not any(term in block for term in domain_terms):
                score *= 0.25
            if score > best_score:
                best_score = score
                best_match = match

        if best_match is None or best_score <= 0:
            return None

        snippet_lines = [*best_match.context_before, best_match.line, *best_match.context_after]
        start_line = max(1, int(best_match.line_number) - len(best_match.context_before))
        end_line = start_line + len(snippet_lines) - 1
        penalty_factor = 1.0
        if seen_lines and best_match is not None:
            state = seen_lines.get(int(best_match.line_number))
            matched_terms = state["terms"] if state is not None else set()
            block = "\n".join(snippet_lines).lower()
            if matched_terms and matched_terms.issubset(project_terms) and domain_terms and not any(term in block for term in domain_terms):
                penalty_factor = 0.55
        return replace(
            item,
            score=(float(getattr(item, "score", 0.0)) * penalty_factor) + (best_score * 0.05),
            symbol_name="<artifact_focus>",
            line_start=start_line,
            line_end=end_line,
            source="\n".join(snippet_lines),
        )

    def _text_matches_to_related_files(
        self,
        query: str,
        matches: list[TextSearchMatch],
        related_file_limit: int,
        excluded_paths: set[str],
    ) -> list[dict]:
        """
        Convert trailing Rust text-search matches into related-file payloads.
        """
        if related_file_limit <= 0:
            return []

        related: list[dict] = []
        seen_paths = set(excluded_paths)
        for item in matches:
            file_path = str(item.file_path).replace("\\\\?\\", "")
            if file_path in seen_paths:
                continue
            if self._should_skip_lexical_related_path(query, file_path):
                continue
            seen_paths.add(file_path)
            related.append(
                {
                    "file_path": file_path,
                    "distance": 1,
                    "relations": [
                        {
                            "relation": "lexical_neighbor",
                            "seed_file": "",
                            "module_path": "",
                            "language": item.language,
                            "line": item.snippet_line_start,
                        }
                    ],
                }
            )
            if len(related) >= related_file_limit:
                break
        return related

    def _should_skip_lexical_related_path(self, query: str, file_path: str) -> bool:
        normalized = file_path.replace("\\", "/").lower()
        keywords = set(extract_keywords(query, max_keywords=20))

        tooling_terms = {
            "benchmark",
            "latency",
            "coverage",
            "phase",
            "tooling",
            "script",
            "scripts",
            "instrumentation",
            "timing",
        }
        if not keywords.intersection(tooling_terms):
            if (
                "/scripts/" in normalized
                or normalized.startswith("scripts/")
                or "benchmark" in normalized
                or normalized.endswith("_cases.json")
                or normalized.endswith(".bench.rs")
            ):
                return True

        doc_terms = {"doc", "docs", "documentation", "readme", "guide", "guides", "manual"}
        if not keywords.intersection(doc_terms):
            if (
                normalized.endswith(".md")
                or normalized.endswith(".rst")
                or "/docs/" in normalized
                or "/doc/" in normalized
                or normalized.endswith("readme")
                or normalized.endswith("readme.md")
                or normalized.endswith("ai_docs.md")
            ):
                return True

        test_terms = {"test", "tests", "spec", "specs", "regression", "unittest", "pytest"}
        if not keywords.intersection(test_terms):
            if (
                "/tests/" in normalized
                or "/test/" in normalized
                or "/__tests__/" in normalized
                or normalized.endswith(".test.ts")
                or normalized.endswith(".test.js")
                or normalized.endswith("_test.py")
                or normalized.endswith("_spec.py")
                or "test_regressions.py" in normalized
            ):
                return True

        return False

    def _related_file_to_search_result(self, related_file: dict) -> object:
        """
        Convert a related-file payload into a lightweight SearchResult row.
        """
        from engine.src.searcher import SearchResult

        file_path = str(related_file["file_path"]).replace("\\\\?\\", "")
        relation = related_file["relations"][0]
        line = int(relation.get("line") or 0)
        source = ""
        if line > 0:
            try:
                snippet = self.file_read(file=file_path, start_line=line, end_line=line + 6)
                source = "\n".join(item.text for item in snippet.lines)
            except Exception:
                source = ""

        return SearchResult(
            score=0.0,
            file_path=file_path,
            symbol_name=Path(file_path).name,
            symbol_kind="file_match",
            line_start=line,
            line_end=max(line, line + 6),
            source=source,
            description=relation.get("relation", "related_file"),
            language=relation.get("language"),
            path_context=self._build_symbol_path_context(file_path),
        )

    def _expand_symbol_context_results(
        self,
        query: str,
        results: list,
        limit: int,
        path: str | Path,
    ) -> list:
        """
        Expand exact symbol anchors with nearby helper symbols from the same implementation file.
        """
        if not results or limit <= 1 or not self._should_expand_symbol_context_results(query):
            return results[: max(limit, 1)]

        anchor = results[0]
        helpers = self._collect_symbol_helper_results(
            query=query,
            anchor=anchor,
            helper_limit=max(limit - 1, 0),
            path=path,
        )
        return self._merge_symbol_context_results(
            anchor=anchor,
            helper_results=helpers,
            fallback_results=results[1:],
            limit=limit,
        )

    def _extract_symbol_query_candidate(self, query: str) -> Optional[str]:
        """
        Extract a likely symbol target from an identifier-heavy query.
        """
        raw_query = query.strip()
        if not raw_query:
            return None

        normalized_query = self._normalize_symbol_candidate(raw_query)
        if normalized_query and self._symbol_candidate_score(normalized_query, query) >= 5:
            return normalized_query

        candidates: list[str] = []
        for item in re.findall(r"`([^`]+)`", query):
            normalized = self._normalize_symbol_candidate(item)
            if normalized:
                candidates.append(normalized)

        token_pattern = r"[A-Za-z_][A-Za-z0-9_]*(?:(?:\.|::|#)[A-Za-z_][A-Za-z0-9_]*)*"
        for item in re.findall(token_pattern, query):
            normalized = self._normalize_symbol_candidate(item)
            if normalized:
                candidates.append(normalized)

        best_candidate: str | None = None
        best_score = 0
        seen: set[str] = set()
        for candidate in candidates:
            lowered = candidate.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            score = self._symbol_candidate_score(candidate, query)
            if score > best_score:
                best_candidate = candidate
                best_score = score

        if best_score >= 5:
            return best_candidate
        return None

    def _normalize_symbol_candidate(self, candidate: str) -> Optional[str]:
        """
        Normalize symbol-like tokens extracted from user queries.
        """
        normalized = candidate.strip().strip("`'\"")
        normalized = re.sub(r"\(\)$", "", normalized)
        normalized = re.sub(r"^[^A-Za-z_]+|[^A-Za-z0-9_:.#]+$", "", normalized)
        if not normalized:
            return None
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:(?:\.|::|#)[A-Za-z_][A-Za-z0-9_]*)*", normalized):
            return None
        return normalized

    def _symbol_candidate_score(self, candidate: str, query: str) -> int:
        """
        Score how likely a candidate token is to be the intended symbol target.
        """
        lower_query = query.lower()
        keywords = set(extract_keywords(query, max_keywords=20))
        score = 0

        if query.strip().strip("`'\"").strip("?!.,") == candidate:
            score += 5
        if f"`{candidate}`".lower() in lower_query:
            score += 2
        if "." in candidate or "::" in candidate or "#" in candidate:
            score += 4
        if "_" in candidate:
            score += 1
        if any(char.isupper() for char in candidate):
            score += 2
        if re.search(r"[a-z][A-Z]", candidate):
            score += 1
        if len(keywords) <= 5:
            score += 1
        if keywords.intersection({"function", "method", "class", "symbol", "definition", "define", "defined"}):
            score += 2
        if candidate.lower() in {
            "where",
            "which",
            "defined",
            "definition",
            "function",
            "method",
            "class",
            "symbol",
        }:
            score -= 6
        if "." not in candidate and "::" not in candidate and "#" not in candidate and candidate.islower() and "_" not in candidate:
            score -= 2

        return score

    def _symbol_lookup_search_results(
        self,
        query: str,
        limit: int,
        language: Optional[str] = None,
        path_prefix: Optional[str] = None,
        path: str | Path | None = None,
    ) -> list:
        """
        Run symbol lookup and hydrate top symbol hits into SearchResult rows with source.
        """
        lookup = self.symbol_lookup(
            query=query,
            path=path,
            limit=max(limit * 2, 8),
        )
        prioritized = self._prioritize_symbol_lookup_results(lookup.results, query)
        filtered = self._filter_symbol_lookup_results(
            results=prioritized,
            language=language,
            path_prefix=path_prefix,
        )
        return self._hydrate_symbol_lookup_results(filtered[: max(limit, 1)])

    def _filter_symbol_lookup_results(
        self,
        results: list[object],
        language: Optional[str],
        path_prefix: Optional[str],
    ) -> list:
        """
        Apply language and path-prefix filters to raw symbol lookup rows.
        """
        normalized_prefix = self._normalize_symbol_path_prefix(path_prefix)
        filtered: list = []
        for item in results:
            item_language = str(getattr(item, "language", "")).lower()
            if language and item_language != language.lower():
                continue

            file_path = str(getattr(item, "file_path", ""))
            if normalized_prefix and not self._symbol_path_matches_prefix(file_path, normalized_prefix):
                continue

            filtered.append(item)
        return filtered

    def _normalize_symbol_path_prefix(self, path_prefix: Optional[str]) -> Optional[str]:
        if not path_prefix:
            return None
        return path_prefix.replace("\\", "/").strip("/").lower()

    def _collect_symbol_helper_results(
        self,
        query: str,
        anchor: object,
        helper_limit: int,
        path: str | Path,
    ) -> list:
        """
        Collect same-file helper symbols referenced by the anchor implementation.
        """
        if helper_limit <= 0:
            return []

        anchor_source = str(getattr(anchor, "source", "") or "")
        anchor_file = str(getattr(anchor, "file_path", "") or "")
        if not anchor_source or not anchor_file:
            return []

        try:
            extracted_symbols = self._load_file_symbol_candidates(anchor_file, path=path)
        except Exception:
            return []

        anchor_name = str(getattr(anchor, "symbol_name", ""))
        anchor_parent = getattr(anchor, "parent", None)
        anchor_line_start = int(getattr(anchor, "line_start", 0))
        query_keywords = set(extract_keywords(query, max_keywords=20))
        query_keywords.update(self._split_symbol_text_tokens(query))
        source_context = self._build_symbol_source_context(anchor_source)
        reference_keywords = source_context.get("reference_keywords", {})

        candidates: list[tuple[tuple[int, int, int, int, str], object]] = []
        for symbol in extracted_symbols:
            if symbol.name == anchor_name and symbol.parent == anchor_parent and symbol.line_start == anchor_line_start:
                continue
            if str(symbol.kind).lower() not in {"function", "method", "class", "struct", "interface", "trait"}:
                continue

            query_overlap_tokens, mismatch_tokens = self._symbol_candidate_token_sets(symbol)
            mentioned = self._symbol_name_in_source(symbol.name, source_context)
            same_parent = bool(anchor_parent and symbol.parent == anchor_parent)
            mismatch_penalty = self._symbol_helper_mismatch_penalty(mismatch_tokens, query_keywords)
            base_overlap = self._symbol_query_overlap(query_overlap_tokens, query_keywords)
            if (
                isinstance(reference_keywords, dict)
                and base_overlap < 2
                and mismatch_penalty <= 1
            ):
                query_overlap_tokens = set(query_overlap_tokens)
                query_overlap_tokens.update(reference_keywords.get(str(symbol.name).lower(), set()))
            query_overlap = self._symbol_query_overlap(query_overlap_tokens, query_keywords)
            reference_overlap = self._symbol_reference_overlap(symbol.name, source_context, query_keywords)
            container_match = bool(anchor_parent and symbol.name == anchor_parent)
            if not mentioned and query_overlap == 0 and reference_overlap == 0 and not container_match:
                continue

            line_distance = abs(int(symbol.line_start) - anchor_line_start)
            score = (
                1 if mentioned else 0,
                1 if same_parent else 0,
                query_overlap,
                -mismatch_penalty,
                reference_overlap,
                1 if container_match else 0,
                -line_distance,
                symbol.name,
            )
            candidates.append((score, symbol))

        candidates.sort(reverse=True)
        helper_results = [
            self._search_result_from_extracted_symbol(symbol, score=max(0.2, 0.94 - (idx * 0.04)))
            for idx, (_, symbol) in enumerate(candidates[:helper_limit], 1)
        ]
        return self._hydrate_search_result_sources(helper_results)

    def _load_file_symbol_candidates(self, file_path: str, path: str | Path) -> list:
        """
        Load same-file symbol metadata from the Rust symbol index, falling back to extraction.
        """
        normalized_path = self._normalize_symbol_file_path(file_path)
        try:
            lookup = self.parser_service.file_symbols(
                file=normalized_path,
                path=path,
                limit=2048,
            )
            if lookup.results:
                return lookup.results
        except Exception:
            pass

        return self._load_file_symbols(normalized_path)

    def _build_symbol_source_context(self, source: str) -> dict[str, object]:
        lines = source.splitlines()
        line_keywords: list[set[str]] = []
        identifier_lines: dict[str, set[int]] = {}

        for idx, line in enumerate(lines):
            keywords = set(extract_keywords(line, max_keywords=20))
            keywords.update(self._split_symbol_text_tokens(line))
            line_keywords.append(keywords)

            identifiers_on_line = {
                raw.lower()
                for raw in _IDENTIFIER_PATTERN.findall(line)
            }
            for identifier in identifiers_on_line:
                identifier_lines.setdefault(identifier, set()).add(idx)

        reference_keywords: dict[str, set[str]] = {}
        for identifier, matched_indexes in identifier_lines.items():
            window_keywords: set[str] = set()
            for idx in matched_indexes:
                for candidate_idx in range(max(0, idx - 1), min(len(lines), idx + 2)):
                    window_keywords.update(line_keywords[candidate_idx])
            reference_keywords[identifier] = window_keywords

        return {
            "identifiers": set(identifier_lines),
            "reference_keywords": reference_keywords,
        }

    def _symbol_name_in_source(self, symbol_name: str, source_context: dict[str, object]) -> bool:
        identifiers = source_context.get("identifiers", set())
        return symbol_name.lower() in identifiers

    def _symbol_candidate_token_sets(self, symbol: object) -> tuple[set[str], set[str]]:
        candidate_text = " ".join(
            part
            for part in (
                str(getattr(symbol, "name", "")),
                str(getattr(symbol, "signature", "") or ""),
                str(getattr(symbol, "docstring", "") or ""),
            )
            if part
        )
        candidate_keywords = set(extract_keywords(candidate_text, max_keywords=20))
        candidate_keywords.update(self._split_symbol_text_tokens(candidate_text))
        mismatch_text = " ".join(
            part
            for part in (
                str(getattr(symbol, "name", "")),
                str(getattr(symbol, "signature", "") or ""),
            )
            if part
        )
        mismatch_tokens = self._split_symbol_text_tokens(mismatch_text)
        return candidate_keywords, mismatch_tokens

    def _symbol_query_overlap(self, candidate_keywords: set[str], query_keywords: set[str]) -> int:
        if not candidate_keywords or not query_keywords:
            return 0
        return len(query_keywords.intersection(candidate_keywords))

    def _symbol_reference_overlap(
        self,
        symbol_name: str,
        source_context: dict[str, object],
        query_keywords: set[str],
    ) -> int:
        if not query_keywords:
            return 0
        reference_keywords = source_context.get("reference_keywords", {})
        if not isinstance(reference_keywords, dict):
            return 0
        candidate_keywords = reference_keywords.get(symbol_name.lower(), set())
        if not candidate_keywords:
            return 0
        return len(query_keywords.intersection(candidate_keywords))

    def _symbol_helper_mismatch_penalty(self, candidate_tokens: set[str], query_keywords: set[str]) -> int:
        specialized_groups = {
            "rerank": ({"rerank", "blend"}, 2),
            "embedding": ({"embed", "embedding", "cached", "cache"}, 2),
            "limit": ({"limit", "request", "requests", "threshold", "budget"}, 1),
            "keyword": ({"keyword", "keywords"}, 1),
            "postprocess": ({"finalize", "final", "diversify", "hydrate", "hydration", "postprocess"}, 2),
            "legacy": ({"legacy"}, 2),
        }
        penalty = 0
        for terms, weight in specialized_groups.values():
            if not candidate_tokens.intersection(terms):
                continue
            if query_keywords.intersection(terms):
                continue
            penalty += weight
        return penalty

    def _split_symbol_text_tokens(self, text: str) -> set[str]:
        tokens: set[str] = set()

        def add_token(value: str) -> None:
            lowered_value = value.lower()
            if len(lowered_value) < 3 or lowered_value in STOP_WORDS:
                return
            tokens.add(lowered_value)
            aliases = {
                "desc": ("description",),
                "cfg": ("config", "configuration"),
                "req": ("request",),
                "resp": ("response",),
                "rrf": ("fuse", "fusion", "merge"),
            }
            for alias in aliases.get(lowered_value, ()):
                tokens.add(alias)

        for raw in _IDENTIFIER_PATTERN.findall(text):
            add_token(raw)

            underscore_parts = [part for part in _UNDERSCORE_SPLIT_PATTERN.split(raw.strip("_")) if part]
            for part in underscore_parts:
                add_token(part)

            camel_text = _CAMEL_BOUNDARY_PATTERN.sub(r"\1 \2", raw.replace("_", " "))
            for part in camel_text.split():
                add_token(part)
        return tokens

    def _merge_symbol_context_results(
        self,
        anchor: object,
        helper_results: list,
        fallback_results: list,
        limit: int,
    ) -> list:
        """
        Keep the primary anchor first, then helper symbols, then fallback symbol hits.
        """
        combined: list = []
        seen: set[tuple[str, str, str | None, int, int]] = set()
        for item in [anchor, *helper_results, *fallback_results]:
            key = (
                str(getattr(item, "file_path", "")),
                str(getattr(item, "symbol_name", "")),
                getattr(item, "parent", None),
                int(getattr(item, "line_start", 0)),
                int(getattr(item, "line_end", 0)),
            )
            if key in seen:
                continue
            seen.add(key)
            combined.append(item)
            if len(combined) >= max(limit, 1):
                break
        return combined

    def _prioritize_symbol_lookup_results(self, results: list, query: str) -> list:
        """
        Re-rank symbol lookup rows so exact member hits beat nearby sibling noise.
        """
        parent_query, name_query = self._split_symbol_candidate(query)
        if not name_query:
            return list(results)

        name_lower = name_query.lower()
        parent_lower = parent_query.lower() if parent_query else None
        container_kinds = {"class", "struct", "module", "interface", "trait", "enum"}

        def score(item: object) -> tuple[int, int, str]:
            value = 0
            item_name = str(getattr(item, "name", "")).lower()
            item_parent = str(getattr(item, "parent", "") or "").lower()
            item_kind = str(getattr(item, "kind", "")).lower()

            if item_name == name_lower:
                value += 6
            if parent_lower and item_parent == parent_lower:
                value += 4 if item_name == name_lower else 1
            if item_name == name_lower and item_kind in {"function", "method", "class", "struct", "interface", "trait"}:
                value += 2
            if parent_lower and item_name == parent_lower and item_kind in container_kinds:
                value += 3

            return (-value, int(getattr(item, "line_start", 0)), str(getattr(item, "file_path", "")))

        return sorted(results, key=score)

    def _split_symbol_candidate(self, candidate: str) -> tuple[Optional[str], str]:
        if "::" in candidate:
            parent, name = candidate.rsplit("::", 1)
            return parent or None, name
        for separator in (".", "#"):
            if separator in candidate:
                parent, name = candidate.rsplit(separator, 1)
                return parent or None, name
        return None, candidate

    def _symbol_path_matches_prefix(self, file_path: str, normalized_prefix: str) -> bool:
        normalized_path = file_path.replace("\\", "/").lower()
        if normalized_path.startswith(normalized_prefix):
            return True
        return f"/{normalized_prefix}" in normalized_path

    def _hydrate_symbol_lookup_results(self, results: list) -> list:
        """
        Hydrate symbol lookup rows into SearchResult objects with source text.
        """
        from engine.src.searcher import SearchResult

        hydrated: list[SearchResult] = []
        seen: set[tuple[str, str, str | None, int, int]] = set()
        for rank, item in enumerate(results, 1):
            key = (
                str(getattr(item, "file_path", "")),
                str(getattr(item, "name", "")),
                getattr(item, "parent", None),
                int(getattr(item, "line_start", 0)),
                int(getattr(item, "line_end", 0)),
            )
            if key in seen:
                continue
            seen.add(key)

            source, signature, docstring = self._read_symbol_context(item)
            hydrated.append(
                SearchResult(
                    score=max(0.0, 1.0 - ((rank - 1) * 0.05)),
                    file_path=self._normalize_symbol_file_path(str(getattr(item, "file_path", ""))),
                    symbol_name=str(getattr(item, "name", "")),
                    symbol_kind=str(getattr(item, "kind", "")),
                    line_start=int(getattr(item, "line_start", 0)),
                    line_end=int(getattr(item, "line_end", 0)),
                    source=source,
                    description=self._build_symbol_description(item, docstring, signature),
                    signature=signature,
                    docstring=docstring,
                    parent=getattr(item, "parent", None),
                    language=getattr(item, "language", None),
                    path_context=self._build_symbol_path_context(
                        self._normalize_symbol_file_path(str(getattr(item, "file_path", "")))
                    ),
                )
            )
        return hydrated

    def _normalize_symbol_file_path(self, file_path: str) -> str:
        if file_path.startswith("\\\\?\\"):
            return file_path[4:]
        if file_path.startswith("//?/"):
            return file_path[4:]
        return file_path

    def _lsp_ready_language_ids(self, path: str | Path) -> set[str]:
        """
        Return the set of language ids with a ready or installed LSP for the target path.
        """
        resolved = str(Path(path).resolve())
        cached = self._lsp_ready_language_cache.get(resolved)
        if cached is not None:
            return cached

        try:
            plan = self.lsp_check(resolved)
        except Exception:
            ready: set[str] = set()
        else:
            ready = {
                str(server.language_id)
                for server in plan.servers
                if str(server.status).lower() in {"ready", "installed"}
            }
        self._lsp_ready_language_cache[resolved] = ready
        return ready

    def _language_lsp_aliases(self, language: str) -> set[str]:
        """
        Map parser language names to one or more LSP language ids.
        """
        normalized = str(language or "").strip().lower().replace("-", "_")
        aliases = _LSP_LANGUAGE_ALIASES.get(normalized)
        if aliases is not None:
            return aliases
        if normalized:
            return {normalized}
        return set()

    def _file_text_and_offsets(self, file_path: str) -> tuple[str, list[int]]:
        """
        Read file text once and compute line-start character offsets.
        """
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        offsets = [0]
        for idx, char in enumerate(text):
            if char == "\n":
                offsets.append(idx + 1)
        return text, offsets

    def _position_to_char_offset(
        self,
        line_offsets: list[int],
        text: str,
        line: int,
        character: int,
    ) -> int:
        """
        Convert an LSP line/character position to a character offset.
        """
        if not line_offsets:
            return 0
        safe_line = max(0, min(int(line), len(line_offsets) - 1))
        line_start = line_offsets[safe_line]
        line_end = line_offsets[safe_line + 1] if safe_line + 1 < len(line_offsets) else len(text)
        return max(line_start, min(line_start + max(0, int(character)), line_end))

    def _flatten_lsp_document_symbols(
        self,
        nodes: list[Any],
        parent: str | None = None,
    ) -> list[tuple[dict[str, Any], str | None]]:
        """
        Flatten hierarchical LSP document symbols into a parent-aware list.
        """
        flattened: list[tuple[dict[str, Any], str | None]] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            flattened.append((node, parent))
            child_parent = str(node.get("name", "") or "").strip() or parent
            children = node.get("children")
            if isinstance(children, list) and children:
                flattened.extend(self._flatten_lsp_document_symbols(children, parent=child_parent))
        return flattened

    def _lsp_symbols_to_extracted_symbols(
        self,
        file_path: str,
        language: str,
        payload: Any,
    ) -> list[ExtractedSymbol]:
        """
        Convert LSP document symbols into extracted-symbol records for chunking.
        """
        if not isinstance(payload, list):
            return []

        try:
            text, line_offsets = self._file_text_and_offsets(file_path)
        except Exception:
            return []

        out: list[ExtractedSymbol] = []
        seen_ranges: set[tuple[str, int, int]] = set()
        for node, parent in self._flatten_lsp_document_symbols(payload):
            name = str(node.get("name", "") or "").strip()
            if not name:
                continue
            kind = _LSP_SYMBOL_KIND_MAP.get(int(node.get("kind", 0) or 0), "symbol")
            if kind in _LSP_SKIPPED_KINDS:
                continue

            node_range = node.get("range")
            if not isinstance(node_range, dict):
                continue
            start = node_range.get("start") or {}
            end = node_range.get("end") or {}
            start_line = int(start.get("line", 0) or 0)
            end_line = int(end.get("line", start_line) or start_line)
            start_char = int(start.get("character", 0) or 0)
            end_char = int(end.get("character", 0) or 0)

            char_start = self._position_to_char_offset(line_offsets, text, start_line, start_char)
            char_end = self._position_to_char_offset(line_offsets, text, end_line, end_char)
            if char_end <= char_start:
                fallback_line = min(end_line + 1, len(line_offsets) - 1)
                char_end = self._position_to_char_offset(line_offsets, text, fallback_line, 0)
            if char_end <= char_start:
                continue

            source = text[char_start:char_end].strip()
            if not source:
                continue

            range_key = (name, char_start, char_end)
            if range_key in seen_ranges:
                continue
            seen_ranges.add(range_key)

            byte_start = len(text[:char_start].encode("utf-8"))
            byte_end = len(text[:char_end].encode("utf-8"))
            detail = node.get("detail")
            signature = str(detail).strip() if detail is not None and str(detail).strip() else None
            out.append(
                ExtractedSymbol(
                    name=name,
                    kind=kind,
                    language=language,
                    file_path=file_path,
                    line_start=start_line,
                    line_end=max(start_line, end_line),
                    byte_start=byte_start,
                    byte_end=byte_end,
                    source=source,
                    signature=signature,
                    docstring=None,
                    params=None,
                    return_type=None,
                    parent=parent,
                    visibility=None,
                    role=None,
                )
            )

        if len(out) > 48:
            preferred = [symbol for symbol in out if symbol.kind in _LSP_HIGH_SIGNAL_KINDS]
            if preferred:
                out = preferred

        return out

    def _enrich_extraction_results_with_lsp_symbols(
        self,
        results: list[ExtractionResult],
        root_path: str | Path,
    ) -> tuple[list[ExtractionResult], int]:
        """
        Fill empty parser extraction results with LSP document symbols when a ready server exists.
        """
        if not results:
            return results, 0

        candidates = [
            result
            for result in results
            if not result.symbols and not result.errors and self._language_lsp_aliases(result.language)
        ]
        if not candidates:
            return results, 0

        ready_language_ids = self._lsp_ready_language_ids(root_path)
        if not ready_language_ids:
            return results, 0

        enriched_count = 0
        enriched_results: list[ExtractionResult] = []
        for result in results:
            aliases = self._language_lsp_aliases(result.language)
            if result.symbols or result.errors or not aliases.intersection(ready_language_ids):
                enriched_results.append(result)
                continue

            try:
                payload = self.lsp_symbols(result.file_path)
                lsp_symbols = self._lsp_symbols_to_extracted_symbols(
                    file_path=result.file_path,
                    language=result.language,
                    payload=payload,
                )
            except Exception:
                lsp_symbols = []

            if lsp_symbols:
                enriched_results.append(replace(result, symbols=lsp_symbols))
                enriched_count += 1
            else:
                enriched_results.append(result)

        return enriched_results, enriched_count

    def _maybe_generate_lightweight_artifact_metadata(
        self,
        unique_chunks: list[CodeChunk],
        descriptions: list,
        progress_callback=None,
    ) -> list:
        """
        Optionally replace fallback metadata for artifact chunks with compact batched LLM output.
        """
        llm_cfg = self._config.llm
        if llm_cfg is None or not llm_cfg.artifact_metadata_enabled:
            return descriptions

        artifact_chunks = [chunk for chunk in unique_chunks if chunk.symbol_kind == "file_artifact"]
        if not artifact_chunks:
            return descriptions

        max_per_file = max(1, int(llm_cfg.artifact_metadata_chunks_per_file))
        selected_chunks: list[CodeChunk] = []
        for file_path in sorted({chunk.file_path for chunk in artifact_chunks}):
            file_chunks = [chunk for chunk in artifact_chunks if chunk.file_path == file_path]

            def _artifact_score(chunk: CodeChunk) -> float:
                if chunk.symbol_name == "<artifact_summary>":
                    return 100.0
                signals = extract_artifact_semantic_signals(chunk.source, file_name=Path(chunk.file_path).name)
                score = 0.0
                score += len(signals.call_targets) * 2.0
                score += len(signals.methods) * 1.4
                score += len(signals.field_names) * 0.9
                score += len(signals.message_types) * 1.1
                score += len(signals.string_literals) * 0.5
                if "Raw excerpt:" in chunk.source:
                    score += 0.25
                return score

            ranked = sorted(file_chunks, key=_artifact_score, reverse=True)
            selected_chunks.extend(ranked[:max_per_file])

        generated = self.describer.generate_lightweight_metadata_batch(
            selected_chunks,
            request_batch_size=max(1, int(llm_cfg.artifact_metadata_batch_size)),
            max_tokens=max(64, int(llm_cfg.artifact_metadata_max_tokens)),
            progress_callback=progress_callback,
        )
        generated_by_id = {item.chunk_id: item for item in generated}
        return [generated_by_id.get(item.chunk_id, item) for item in descriptions]

    def _load_file_compact_symbols(self, file_path: str) -> list:
        """
        Load compact extracted symbols for one file with stat-based cache invalidation.
        """
        normalized_path = self._normalize_symbol_file_path(file_path)
        path = Path(normalized_path)
        try:
            resolved = str(path.resolve())
            stat = path.stat()
            cache_key = resolved
            cached = self._symbol_file_compact_cache.get(cache_key)
            if cached and cached[0] == int(stat.st_mtime_ns) and cached[1] == int(stat.st_size):
                return cached[2]
        except Exception:
            cache_key = normalized_path
            cached = self._symbol_file_compact_cache.get(cache_key)
            if cached:
                return cached[2]
            stat = None

        results, _stats = self.parser_service.extract_compact(normalized_path)
        symbols = results[0].symbols if results else []
        if stat is not None:
            self._symbol_file_compact_cache[cache_key] = (
                int(stat.st_mtime_ns),
                int(stat.st_size),
                symbols,
            )
        else:
            self._symbol_file_compact_cache[cache_key] = (0, 0, symbols)
        return symbols

    def _load_file_symbols(self, file_path: str) -> list:
        """
        Load extracted symbols for one file with stat-based cache invalidation.
        """
        normalized_path = self._normalize_symbol_file_path(file_path)
        path = Path(normalized_path)
        try:
            resolved = str(path.resolve())
            stat = path.stat()
            cache_key = resolved
            cached = self._symbol_file_extract_cache.get(cache_key)
            if cached and cached[0] == int(stat.st_mtime_ns) and cached[1] == int(stat.st_size):
                return cached[2]
        except Exception:
            cache_key = normalized_path
            cached = self._symbol_file_extract_cache.get(cache_key)
            if cached:
                return cached[2]
            stat = None

        extracted_files = self.extract_symbols(normalized_path)
        symbols = extracted_files[0].symbols if extracted_files else []
        if stat is not None:
            self._symbol_file_extract_cache[cache_key] = (
                int(stat.st_mtime_ns),
                int(stat.st_size),
                symbols,
            )
        else:
            self._symbol_file_extract_cache[cache_key] = (0, 0, symbols)
        return symbols

    def _read_symbol_context(self, item: object) -> tuple[str, Optional[str], Optional[str]]:
        """
        Read precise source for a symbol lookup hit.
        """
        name = str(getattr(item, "name", getattr(item, "symbol_name", "")))
        parent = getattr(item, "parent", None)
        file_path = str(getattr(item, "file_path", ""))
        signature = getattr(item, "signature", None)

        try:
            line_start = max(1, int(getattr(item, "line_start", 0)) + 1)
            line_end = max(line_start, int(getattr(item, "line_end", 0)) + 1)
            snippet = self.file_read(
                file=file_path,
                start_line=line_start,
                end_line=line_end,
            )
            return (
                "\n".join(line.text for line in snippet.lines),
                signature,
                None,
            )
        except Exception:
            pass

        qualified_name = f"{parent}.{name}" if parent else name
        try:
            extracted = self.extract_symbol(file=file_path, symbol=qualified_name)
            for symbol in extracted.symbols:
                if symbol.name != name:
                    continue
                if parent is not None and symbol.parent != parent:
                    continue
                return symbol.source, symbol.signature or signature, symbol.docstring
        except Exception:
            return "", signature, None
        return "", signature, None

    def _build_symbol_description(
        self,
        item: object,
        docstring: Optional[str],
        signature: Optional[str],
    ) -> str:
        """
        Build a compact description for hydrated symbol lookup results.
        """
        if docstring:
            return docstring.strip()
        if signature:
            return signature

        kind = str(getattr(item, "kind", "symbol")).replace("_", " ").strip()
        name = str(getattr(item, "name", ""))
        parent = getattr(item, "parent", None)
        if parent:
            return f"{kind} {parent}.{name}".strip()
        return f"{kind} {name}".strip()

    def _search_result_from_extracted_symbol(self, symbol: object, score: float) -> object:
        from engine.src.searcher import SearchResult

        file_path = self._normalize_symbol_file_path(str(getattr(symbol, "file_path", "")))
        return SearchResult(
            score=score,
            file_path=file_path,
            symbol_name=str(getattr(symbol, "name", "")),
            symbol_kind=str(getattr(symbol, "kind", "")),
            line_start=int(getattr(symbol, "line_start", 0)),
            line_end=int(getattr(symbol, "line_end", 0)),
            source=str(getattr(symbol, "source", "") or ""),
            description=self._build_symbol_description(
                symbol,
                getattr(symbol, "docstring", None),
                getattr(symbol, "signature", None),
            ),
            signature=getattr(symbol, "signature", None),
            docstring=getattr(symbol, "docstring", None),
            parent=getattr(symbol, "parent", None),
            language=getattr(symbol, "language", None),
            path_context=self._build_symbol_path_context(file_path),
        )

    def _build_symbol_path_context(self, file_path: str) -> str:
        parts = [segment for segment in Path(file_path).parts if segment not in ("", "/", "\\")]
        if not parts:
            return ""
        return " / ".join(parts[-5:])

    def _related_files_for_results(
        self,
        results: list,
        related_seed_files: int,
        related_file_limit: int,
        path: str | Path,
    ) -> list[dict]:
        """
        Collect a small, deduplicated set of related files around the top semantic hits.
        """
        if not results or related_file_limit <= 0 or related_seed_files <= 0:
            return []

        seeds: list[str] = []
        seen_seed_files: set[str] = set()
        for result in results:
            file_path = str(result.file_path)
            if file_path in seen_seed_files:
                continue
            seen_seed_files.add(file_path)
            seeds.append(file_path)
            if len(seeds) >= related_seed_files:
                break

        related_by_path: dict[str, dict] = {}
        excluded_paths = set(seeds)
        project_root = self._resolve_retrieval_root(path)

        for seed_file in seeds:
            neighbors = self.import_neighbors(file=seed_file, path=project_root)
            self._merge_related_edges(
                seed_file=seed_file,
                relation="imports",
                edges=neighbors.imports,
                related_by_path=related_by_path,
                excluded_paths=excluded_paths,
            )
            self._merge_related_edges(
                seed_file=seed_file,
                relation="imported_by",
                edges=neighbors.importers,
                related_by_path=related_by_path,
                excluded_paths=excluded_paths,
            )

        related = sorted(
            related_by_path.values(),
            key=lambda item: (item["distance"], item["file_path"]),
        )
        return related[:related_file_limit]

    def _merge_related_edges(
        self,
        seed_file: str,
        relation: str,
        edges: list[ImportEdge],
        related_by_path: dict[str, dict],
        excluded_paths: set[str],
    ) -> None:
        """
        Merge import-graph edges into a deduplicated related-file map.
        """
        for edge in edges:
            candidate = edge.target_file if relation == "imports" else edge.source_file
            if not candidate or candidate in excluded_paths:
                continue
            item = related_by_path.setdefault(
                candidate,
                {
                    "file_path": candidate,
                    "distance": 1,
                    "relations": [],
                },
            )
            item["relations"].append(
                {
                    "relation": relation,
                    "seed_file": seed_file,
                    "module_path": edge.module_path,
                    "language": edge.language,
                    "line": edge.line,
                }
            )

    def _split_semantic_bundle_results(
        self,
        results: list,
        anchor_limit: int,
        related_file_limit: int,
    ) -> tuple[list, list[dict]]:
        """
        Keep top semantic anchors while surfacing additional distinct semantic files as related context.
        """
        anchors: list = []
        semantic_neighbors: list[dict] = []
        seen_paths: set[str] = set()
        seed_file: str | None = results[0].file_path if results else None

        for item in results:
            file_path = str(item.file_path)
            if file_path in seen_paths:
                continue
            seen_paths.add(file_path)

            if len(anchors) < anchor_limit:
                anchors.append(item)
                continue

            if len(semantic_neighbors) >= related_file_limit:
                break

            semantic_neighbors.append(
                {
                    "file_path": file_path,
                    "distance": 1,
                    "relations": [
                        {
                            "relation": "semantic_neighbor",
                            "seed_file": seed_file,
                            "module_path": "",
                            "language": getattr(item, "language", None),
                            "line": getattr(item, "line_start", 0),
                        }
                    ],
                }
            )

        return anchors, semantic_neighbors

    def _tooling_related_semantic_neighbors(
        self,
        results: list,
        tooling_query: bool,
        excluded_paths: set[str],
        related_file_limit: int,
    ) -> list[dict]:
        """
        Add script-centric semantic neighbors for benchmark and tooling questions.
        """
        if related_file_limit <= 0 or not tooling_query:
            return []

        related: list[dict] = []
        for item in results:
            file_path = str(item.file_path)
            normalized = file_path.replace("\\", "/").lower()
            if "/scripts/" not in normalized and not normalized.startswith("scripts/"):
                continue
            if file_path in excluded_paths:
                continue
            excluded_paths.add(file_path)
            related.append(
                {
                    "file_path": file_path,
                    "distance": 1,
                    "relations": [
                        {
                            "relation": "semantic_tooling_neighbor",
                            "seed_file": "",
                            "module_path": "",
                            "language": getattr(item, "language", None),
                            "line": getattr(item, "line_start", 0),
                        }
                    ],
                }
            )
            if len(related) >= related_file_limit:
                break
        return related

    def _looks_like_tooling_query(self, query: str) -> bool:
        """
        Detect benchmark, instrumentation, and tooling-oriented questions.
        """
        keywords = set(extract_keywords(query, max_keywords=20))
        return bool(
            keywords.intersection(
                {
                    "benchmark",
                    "latency",
                    "server",
                    "client",
                    "coverage",
                    "wrapper",
                    "wrappers",
                    "timing",
                    "instrumentation",
                    "phase",
                }
            )
        )

    def _should_use_bundle_for_query(self, query: str) -> bool:
        """
        Detect broad cross-file and pipeline questions that benefit from bundle expansion.
        """
        if self._looks_like_tooling_query(query):
            return True

        keywords = set(extract_keywords(query, max_keywords=20))
        if len(keywords) < 6:
            return False

        flow_terms = {
            "across",
            "through",
            "between",
            "without",
            "before",
            "after",
            "into",
            "carried",
            "carry",
            "copied",
            "copy",
            "generated",
            "route",
            "routed",
            "dependencies",
            "dependency",
            "importers",
            "callers",
            "cross",
        }
        architecture_terms = {
            "path",
            "prefix",
            "payload",
            "payloads",
            "vector",
            "vectors",
            "index",
            "indexed",
            "indexing",
            "filter",
            "filters",
            "collection",
            "graph",
            "context",
            "benchmark",
            "latency",
            "server",
            "client",
            "coverage",
            "wrapper",
            "wrappers",
            "shadow",
            "reindex",
        }
        if keywords.intersection({"copy", "copied", "shadow", "reindex"}) and keywords.intersection({"collection", "index", "indexed", "points"}):
            return True
        return bool(keywords.intersection(flow_terms)) and len(keywords.intersection(architecture_terms)) >= 2

    def _should_use_graph_related_for_query(self, query: str) -> bool:
        """
        Detect when graph expansion is likely to add value beyond semantic neighbors.
        """
        keywords = set(extract_keywords(query, max_keywords=20))
        graph_terms = {
            "imports",
            "importers",
            "dependencies",
            "dependency",
            "callers",
            "caller",
            "graph",
            "neighbor",
            "neighbors",
            "cross",
            "cross_file",
        }
        return bool(keywords.intersection(graph_terms))

    def _related_callers_for_results(self, results: list, path: str | Path) -> list[dict]:
        """
        Collect caller rows around the top callable semantic anchor.
        """
        if not results:
            return []

        top = results[0]
        if str(getattr(top, "symbol_kind", "")).lower() not in {"function", "method"}:
            return []

        caller_rows = self.find_callers(symbol=top.symbol_name, path=path, limit=8)
        related: list[dict] = []
        seen: set[tuple[str, str, int]] = set()
        for caller in caller_rows.callers:
            key = (caller.caller_file_path, caller.caller_name, caller.caller_line)
            if key in seen:
                continue
            seen.add(key)
            related.append(
                {
                    "symbol": top.symbol_name,
                    "caller_name": caller.caller_name,
                    "caller_kind": caller.caller_kind,
                    "caller_file_path": caller.caller_file_path,
                    "caller_line": caller.caller_line,
                    "language": caller.caller_language,
                }
            )
        return related

    def _chunks_by_file(self, chunks: list[CodeChunk]) -> dict[str, list[CodeChunk]]:
        grouped: dict[str, list[CodeChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.file_path, []).append(chunk)
        return grouped

    def _resume_pending_files(
        self,
        expected_files: list["ResumeFile"],
        indexed_state: dict[str, "IndexedFileState"],
    ) -> tuple[list[str], list[str]]:
        """
        Split resumable shadow files into completed and pending sets.
        """
        completed: list[str] = []
        pending: list[str] = []

        for file_state in expected_files:
            indexed = indexed_state.get(file_state.file_path)
            if indexed and indexed.file_hash == file_state.file_hash and indexed.point_count == file_state.chunk_count:
                completed.append(file_state.file_path)
            else:
                pending.append(file_state.file_path)

        return completed, pending

    def _merge_index_stats(self, left: "IndexStats", right: "IndexStats") -> "IndexStats":
        """
        Merge two indexing stats objects.
        """
        return replace(
            left,
            total_chunks=left.total_chunks + right.total_chunks,
            upserted_points=left.upserted_points + right.upserted_points,
            failed_points=left.failed_points + right.failed_points,
            total_tokens=left.total_tokens + right.total_tokens,
            duration_seconds=left.duration_seconds + right.duration_seconds,
            llm_cost_usd=left.llm_cost_usd + right.llm_cost_usd,
            embedding_cost_usd=left.embedding_cost_usd + right.embedding_cost_usd,
            skipped_small_chunks=left.skipped_small_chunks + right.skipped_small_chunks,
            skipped_minified_chunks=left.skipped_minified_chunks + right.skipped_minified_chunks,
            skipped_capped_chunks=left.skipped_capped_chunks + right.skipped_capped_chunks,
            files_capped=left.files_capped + right.files_capped,
            embedding_requests=left.embedding_requests + right.embedding_requests,
            embedding_retries=left.embedding_retries + right.embedding_retries,
            embedding_failed_requests=left.embedding_failed_requests + right.embedding_failed_requests,
            embedding_input_count=left.embedding_input_count + right.embedding_input_count,
            embedding_stage_duration_seconds=(
                left.embedding_stage_duration_seconds + right.embedding_stage_duration_seconds
            ),
            embedding_batch_shrink_events=left.embedding_batch_shrink_events + right.embedding_batch_shrink_events,
            embedding_batch_grow_events=left.embedding_batch_grow_events + right.embedding_batch_grow_events,
            embedding_final_batch_size=max(left.embedding_final_batch_size, right.embedding_final_batch_size),
            scan_stage_duration_seconds=left.scan_stage_duration_seconds + right.scan_stage_duration_seconds,
            artifact_profile_stage_duration_seconds=(
                left.artifact_profile_stage_duration_seconds + right.artifact_profile_stage_duration_seconds
            ),
            extract_stage_duration_seconds=left.extract_stage_duration_seconds + right.extract_stage_duration_seconds,
            chunk_build_stage_duration_seconds=(
                left.chunk_build_stage_duration_seconds + right.chunk_build_stage_duration_seconds
            ),
            filter_stage_duration_seconds=left.filter_stage_duration_seconds + right.filter_stage_duration_seconds,
            dedup_stage_duration_seconds=left.dedup_stage_duration_seconds + right.dedup_stage_duration_seconds,
            description_stage_duration_seconds=(
                left.description_stage_duration_seconds + right.description_stage_duration_seconds
            ),
            point_build_stage_duration_seconds=(
                left.point_build_stage_duration_seconds + right.point_build_stage_duration_seconds
            ),
            upsert_stage_duration_seconds=left.upsert_stage_duration_seconds + right.upsert_stage_duration_seconds,
        )

    def _zero_index_stats(
        self,
        started_at: float,
        generate_descriptions: bool,
        skipped_small_chunks: int = 0,
        skipped_minified_chunks: int = 0,
        skipped_capped_chunks: int = 0,
        files_capped: int = 0,
    ) -> "IndexStats":
        from engine.src.indexer import IndexStats

        return IndexStats(
            total_chunks=0,
            upserted_points=0,
            failed_points=0,
            total_tokens=0,
            duration_seconds=time.time() - started_at,
            llm_cost_usd=0.0,
            embedding_cost_usd=0.0,
            skipped_small_chunks=skipped_small_chunks,
            skipped_minified_chunks=skipped_minified_chunks,
            skipped_capped_chunks=skipped_capped_chunks,
            files_capped=files_capped,
            descriptions_enabled=generate_descriptions,
        )

    def _collect_artifact_profiles(
        self,
        entries: list,
        fast: bool,
        skip_minified: bool,
    ) -> dict[str, ArtifactIndexProfile]:
        """
        Detect large generated/minified bundle files that should use coarse artifact chunks in fast mode.
        """
        if not fast or not skip_minified:
            return {}

        profiles: dict[str, ArtifactIndexProfile] = {}
        for entry in entries:
            if int(getattr(entry, "file_size", 0) or 0) < 128 * 1024:
                continue
            language = str(getattr(entry, "language", "")).lower()
            if language not in {"javascript", "typescript", "tsx", "jsx"}:
                continue
            try:
                profile = analyze_artifact_candidate(
                    file_path=entry.file_path,
                    language=entry.language,
                    file_size=entry.file_size,
                    file_mtime=entry.file_mtime,
                )
            except Exception:
                continue
            if should_downgrade_artifact_profile(profile):
                profiles[entry.file_path] = profile
        return profiles

    def _build_resume_files(
        self,
        changed_results: list[ExtractionResult],
        chunks: list[CodeChunk],
    ) -> tuple[list["ResumeFile"], list[str]]:
        from engine.src.index_resume import ResumeFile

        chunk_counts: dict[str, int] = {}
        for chunk in chunks:
            chunk_counts[chunk.file_path] = chunk_counts.get(chunk.file_path, 0) + 1

        resumable: list[ResumeFile] = []
        delete_only: list[str] = []
        for result in changed_results:
            chunk_count = chunk_counts.get(result.file_path, 0)
            if chunk_count > 0:
                resumable.append(
                    ResumeFile(
                        file_path=result.file_path,
                        file_hash=result.file_hash,
                        chunk_count=chunk_count,
                    )
                )
            else:
                delete_only.append(result.file_path)

        resumable.sort(key=lambda item: item.file_path)
        delete_only.sort()
        return resumable, delete_only

    def _finalize_shadow_collection(
        self,
        collection: "CollectionManager",
        shadow_name: str,
        show_progress: bool,
    ) -> None:
        old_collection = collection.swap_alias(shadow_name)
        if show_progress:
            print(f"Swapped alias to {shadow_name}")

        if old_collection is not None:
            try:
                collection.client.delete_collection(old_collection)
                if show_progress:
                    print(f"Deleted old collection: {old_collection}")
            except Exception:
                pass

        collection.cleanup_old_versions()

    def _ensure_indexing_backends_ready(
        self,
        *,
        operation_name: str,
        generate_descriptions: bool,
    ) -> None:
        missing = []
        if self._config.qdrant is None:
            missing.append("qdrant")
        if self._config.code_embedding is None:
            missing.append("code_embedding")
        if self._config.desc_embedding is None:
            missing.append("desc_embedding")
        if generate_descriptions and self._config.llm is None:
            missing.append("llm")
        if missing:
            raise RuntimeError(
                f"Cannot {operation_name}: disabled subsystems: {', '.join(missing)}. "
                "Indexing requires qdrant, code_embedding, desc_embedding, and llm when descriptions are enabled."
            )

        try:
            self.connect(verify=True)
        except ConnectionError as exc:
            assert self._config.qdrant is not None
            raise RuntimeError(
                f"Cannot {operation_name}: Qdrant is unreachable at {self._config.qdrant.url}. "
                "Start Qdrant or update the active config before indexing."
            ) from exc

    def _apply_fast_embedding_defaults(
        self,
        *,
        fast: bool,
        embedding_concurrency: Optional[int],
        embedding_batch_size: Optional[int],
        embedding_max_retries: Optional[int],
        embedding_adaptive_batching: Optional[bool],
    ) -> tuple[Optional[int], Optional[int], Optional[int], Optional[bool]]:
        """
        Apply SDK-level fast-mode defaults for remote embedding throughput.

        The CLI already exposed more aggressive one-shot settings, but SDK callers and
        wrappers were previously stuck with conservative defaults unless they knew the
        tuning knobs. Fast mode should carry its speed intent through the whole SDK.
        """
        if not fast:
            return (
                embedding_concurrency,
                embedding_batch_size,
                embedding_max_retries,
                embedding_adaptive_batching,
            )

        providers = {
            cfg.provider
            for cfg in (self._config.code_embedding, self._config.desc_embedding)
            if cfg is not None
        }
        if "litellm" not in providers:
            return (
                embedding_concurrency,
                embedding_batch_size,
                embedding_max_retries,
                embedding_adaptive_batching,
            )

        if embedding_concurrency is None:
            embedding_concurrency = max(
                16,
                max(
                    (cfg.concurrency for cfg in (self._config.code_embedding, self._config.desc_embedding) if cfg is not None),
                    default=4,
                ),
            )
        if embedding_batch_size is None:
            embedding_batch_size = max(
                64,
                max(
                    (cfg.batch_size for cfg in (self._config.code_embedding, self._config.desc_embedding) if cfg is not None),
                    default=32,
                ),
            )
        if embedding_adaptive_batching is None:
            embedding_adaptive_batching = True

        return (
            embedding_concurrency,
            embedding_batch_size,
            embedding_max_retries,
            embedding_adaptive_batching,
        )

    def index_directory(
        self,
        directory: str | Path,
        force_refresh: bool = False,
        show_progress: bool = True,
        project_name: Optional[str] = None,
        fast: bool = False,
        generate_descriptions: bool = True,
        min_chunk_bytes: int = 120,
        max_chunks_per_file: int = 400,
        max_total_chunks: Optional[int] = None,
        compress_for_embedding: Optional[str] = None,
        incremental_resume: bool = False,
        resume_batch_files: int = 200,
        skip_minified: bool = True,
        embedding_concurrency: Optional[int] = None,
        embedding_max_retries: Optional[int] = None,
        embedding_batch_size: Optional[int] = None,
        embedding_adaptive_batching: Optional[bool] = None,
        _progress_reporter: OperationProgressReporter | None = None,
    ) -> "IndexStats":
        """
        Index a directory: extract symbols, chunk, optionally describe, embed, and upsert to Qdrant.

        directory: str | Path — Directory to index.
        force_refresh: bool — Force re-indexing even if files haven't changed.
        show_progress: bool — Print progress messages during indexing.
        project_name: Optional[str] — Manual project name override. Auto-detected if None.
        fast: bool — Fast mode: skips description generation and uses stricter filtering defaults.
        generate_descriptions: bool — Generate LLM descriptions for chunks when True.
        min_chunk_bytes: int — Minimum chunk size in bytes to keep for indexing.
        max_chunks_per_file: int — Maximum chunks kept per file after quality ranking.
        max_total_chunks: Optional[int] — Maximum total chunks kept after filtering.
        compress_for_embedding: Optional[str] — Compression level applied to source before embedding.
        incremental_resume: bool — Resume indexing on live collection instead of shadow swap.
        resume_batch_files: int — Number of files processed per resume batch.
        skip_minified: bool — Skip likely minified/noise-heavy chunks.
        embedding_concurrency: Optional[int] — Override embedding request concurrency for this run.
        embedding_max_retries: Optional[int] — Override embedding request retry count for this run.
        embedding_batch_size: Optional[int] — Override embedding request batch size for this run.
        embedding_adaptive_batching: Optional[bool] — Override adaptive embedding batch resizing.
        Returns: IndexStats — Indexing statistics (chunks, tokens, duration).
        """
        from engine.src.indexer import IndexStats

        started_at = time.time()
        if _progress_reporter is not None:
            _progress_reporter.start("scan", "Scanning supported files")
        self._ensure_indexing_backends_ready(
            operation_name="index",
            generate_descriptions=generate_descriptions,
        )

        directory = Path(directory).resolve()

        detected_project = detect_project_name(directory, manual_override=project_name)
        collection = self._get_collection(detected_project)
        file_cache = self._get_file_cache(directory)
        active_collection_empty = False
        try:
            active_collection_empty = (collection.info().get("points_count", 0) or 0) == 0
        except Exception:
            active_collection_empty = False

        if show_progress:
            print(f"Indexing project: {detected_project}")
            print(f"Scanning supported files in {directory}...")

        scan_started = time.time()
        scan_entries = self.parser_service.scan_files(directory)
        scan_stage_duration_seconds = time.time() - scan_started
        if _progress_reporter is not None:
            _progress_reporter.update(
                files_discovered=len(scan_entries),
                message=f"Scanned {len(scan_entries)} supported files",
            )

        indexer = self._get_indexer(detected_project)
        candidate_entries = []
        unchanged_files = 0

        for entry in scan_entries:
            cached_hash = None
            if not force_refresh and not active_collection_empty:
                cached_hash = file_cache.is_unchanged_from_metadata(
                    entry.file_path,
                    entry.file_size,
                    entry.file_mtime,
                )
            if cached_hash is not None:
                unchanged_files += 1
                continue
            candidate_entries.append(entry)
        if _progress_reporter is not None:
            _progress_reporter.update(
                files_unchanged=unchanged_files,
                message=f"Identified {len(candidate_entries)} candidate files",
            )

        indexed_hashes = indexer.get_file_hashes([entry.file_path for entry in candidate_entries])
        changed_results: list[ExtractionResult] = []
        if _progress_reporter is not None:
            _progress_reporter.set_stage("artifact_profile", "Profiling artifact candidates")
        artifact_profile_started = time.time()
        artifact_profiles = self._collect_artifact_profiles(
            candidate_entries,
            fast=fast,
            skip_minified=skip_minified,
        )
        artifact_profile_stage_duration_seconds = time.time() - artifact_profile_started
        artifact_downgraded_files = 0
        artifact_downgraded_bytes = 0
        extraction_entries = []

        for entry in candidate_entries:
            profile = artifact_profiles.get(entry.file_path)
            if profile is None:
                extraction_entries.append(entry)
                continue

            file_cache.update_from_extraction(
                entry.file_path,
                profile.file_hash,
                entry.file_size,
                entry.file_mtime,
            )

            indexed_hash = indexed_hashes.get(entry.file_path)
            if (
                not force_refresh
                and indexed_hash is not None
                and indexed_hash == profile.file_hash
            ):
                unchanged_files += 1
                continue

            changed_results.append(
                ExtractionResult(
                    file_path=entry.file_path,
                    language=entry.language,
                    symbols=[],
                    errors=[ARTIFACT_FALLBACK_ERROR],
                    file_hash=profile.file_hash,
                    file_size=entry.file_size,
                    file_mtime=entry.file_mtime,
                )
            )
            artifact_downgraded_files += 1
            artifact_downgraded_bytes += int(entry.file_size or 0)

        if show_progress and extraction_entries:
            print(f"Extracting symbols from {len(extraction_entries)} candidate files...")
        if show_progress and artifact_downgraded_files > 0:
            print(
                "Downgrading "
                f"{artifact_downgraded_files} large minified artifact files "
                f"({_human_readable_bytes(artifact_downgraded_bytes)}) to coarse file chunks"
            )

        candidate_paths = {entry.file_path for entry in extraction_entries}
        if _progress_reporter is not None:
            _progress_reporter.set_stage("extract", "Extracting symbols")
        extract_started = time.time()
        if extraction_entries and (
            len(extraction_entries) == len(scan_entries)
            or len(extraction_entries) > 100
        ):
            extracted_candidates = [
                result
                for result in self.extract_symbols(directory)
                if result.file_path in candidate_paths
            ]
            extracted_paths = {result.file_path for result in extracted_candidates}
            missing_paths = sorted(candidate_paths - extracted_paths)
            for file_path in missing_paths:
                extracted_candidates.extend(self.extract_symbols(file_path))
        else:
            extracted_candidates = []
            for entry in extraction_entries:
                extracted_candidates.extend(self.extract_symbols(entry.file_path))
        extract_stage_duration_seconds = time.time() - extract_started
        extracted_candidates, lsp_enriched_files = self._enrich_extraction_results_with_lsp_symbols(
            extracted_candidates,
            root_path=directory,
        )
        if show_progress and lsp_enriched_files > 0:
            print(f"Enriched {lsp_enriched_files} files with LSP document symbols")

        for result in extracted_candidates:
            file_cache.update_from_extraction(
                result.file_path,
                result.file_hash,
                result.file_size,
                result.file_mtime,
            )

            indexed_hash = indexed_hashes.get(result.file_path)
            if (
                not force_refresh
                and indexed_hash is not None
                and indexed_hash == result.file_hash
            ):
                unchanged_files += 1
                continue
            changed_results.append(result)
        if _progress_reporter is not None:
            _progress_reporter.update(
                files_unchanged=unchanged_files,
                files_planned=len(changed_results),
                files_remaining=len(changed_results),
                files_analyzed=len(changed_results),
                artifact_downgraded_files=artifact_downgraded_files,
                symbols_extracted=sum(len(result.symbols) for result in changed_results),
                message=(
                    f"Prepared {len(changed_results)} changed files for indexing"
                    + (f" ({lsp_enriched_files} LSP-enriched)" if lsp_enriched_files else "")
                ),
            )

        if show_progress and unchanged_files > 0:
            print(f"Skipping {unchanged_files} files (already indexed with same hash)")

        if not changed_results:
            if show_progress:
                print("No file changes detected; keeping active collection unchanged")
            empty_stats = IndexStats(
                total_chunks=0,
                upserted_points=0,
                failed_points=0,
                total_tokens=0,
                duration_seconds=time.time() - started_at,
                llm_cost_usd=0.0,
                embedding_cost_usd=0.0,
                descriptions_enabled=generate_descriptions,
            )
            if _progress_reporter is not None:
                _progress_reporter.complete(asdict(empty_stats), message="No file changes detected")
            return empty_stats

        if incremental_resume:
            if show_progress:
                print("Incremental resume mode enabled: indexing directly into live collection")

            changed_paths = [r.file_path for r in changed_results]
            total_stats = IndexStats(
                total_chunks=0,
                upserted_points=0,
                failed_points=0,
                total_tokens=0,
                duration_seconds=0.0,
                llm_cost_usd=0.0,
                embedding_cost_usd=0.0,
                descriptions_enabled=generate_descriptions,
            )

            batch_size = max(1, int(resume_batch_files))
            for offset in range(0, len(changed_paths), batch_size):
                batch_paths = changed_paths[offset:offset + batch_size]
                batch_stats = self.refresh_files(
                    file_paths=batch_paths,
                    show_progress=show_progress,
                    project_name=detected_project,
                    fast=fast,
                    generate_descriptions=generate_descriptions,
                    min_chunk_bytes=min_chunk_bytes,
                    max_chunks_per_file=max_chunks_per_file,
                    max_total_chunks=max_total_chunks,
                    compress_for_embedding=compress_for_embedding,
                    skip_minified=skip_minified,
                    embedding_concurrency=embedding_concurrency,
                    embedding_max_retries=embedding_max_retries,
                    embedding_batch_size=embedding_batch_size,
                    embedding_adaptive_batching=embedding_adaptive_batching,
                    _progress_reporter=_progress_reporter,
                )
                total_stats = replace(
                    total_stats,
                    total_chunks=total_stats.total_chunks + batch_stats.total_chunks,
                    upserted_points=total_stats.upserted_points + batch_stats.upserted_points,
                    failed_points=total_stats.failed_points + batch_stats.failed_points,
                    total_tokens=total_stats.total_tokens + batch_stats.total_tokens,
                    llm_cost_usd=total_stats.llm_cost_usd + batch_stats.llm_cost_usd,
                    embedding_cost_usd=total_stats.embedding_cost_usd + batch_stats.embedding_cost_usd,
                    skipped_small_chunks=total_stats.skipped_small_chunks + batch_stats.skipped_small_chunks,
                    skipped_minified_chunks=total_stats.skipped_minified_chunks + batch_stats.skipped_minified_chunks,
                    skipped_capped_chunks=total_stats.skipped_capped_chunks + batch_stats.skipped_capped_chunks,
                    files_capped=total_stats.files_capped + batch_stats.files_capped,
                    embedding_requests=total_stats.embedding_requests + batch_stats.embedding_requests,
                    embedding_retries=total_stats.embedding_retries + batch_stats.embedding_retries,
                    embedding_failed_requests=total_stats.embedding_failed_requests + batch_stats.embedding_failed_requests,
                    embedding_input_count=total_stats.embedding_input_count + batch_stats.embedding_input_count,
                    embedding_stage_duration_seconds=(
                        total_stats.embedding_stage_duration_seconds + batch_stats.embedding_stage_duration_seconds
                    ),
                    embedding_batch_shrink_events=(
                        total_stats.embedding_batch_shrink_events + batch_stats.embedding_batch_shrink_events
                    ),
                    embedding_batch_grow_events=(
                        total_stats.embedding_batch_grow_events + batch_stats.embedding_batch_grow_events
                    ),
                    embedding_final_batch_size=max(
                        total_stats.embedding_final_batch_size,
                        batch_stats.embedding_final_batch_size,
                    ),
                )

            total_stats = replace(total_stats, duration_seconds=time.time() - started_at)
            return total_stats

        if show_progress:
            total_files = len(changed_results)
            total_symbols = sum(len(r.symbols) for r in changed_results)
            print(f"Extracted {total_symbols} symbols from {total_files} changed files")

        if show_progress:
            print("Building chunks...")

        if _progress_reporter is not None:
            _progress_reporter.set_stage("chunk_build", "Building chunks")
        chunk_build_started = time.time()
        chunks = self.chunker.build_chunks(changed_results)
        chunk_build_stage_duration_seconds = time.time() - chunk_build_started
        chunk_count_before_filter = len(chunks)

        if fast:
            generate_descriptions = False
            min_chunk_bytes = max(min_chunk_bytes, 200)
            max_chunks_per_file = min(max_chunks_per_file, 180)
            skip_minified = True
            if max_total_chunks is None:
                max_total_chunks = 50000
        (
            embedding_concurrency,
            embedding_batch_size,
            embedding_max_retries,
            embedding_adaptive_batching,
        ) = self._apply_fast_embedding_defaults(
            fast=fast,
            embedding_concurrency=embedding_concurrency,
            embedding_batch_size=embedding_batch_size,
            embedding_max_retries=embedding_max_retries,
            embedding_adaptive_batching=embedding_adaptive_batching,
        )

        filter_started = time.time()
        chunks, filter_stats = filter_chunks(
            chunks,
            ChunkFilterConfig(
                min_chunk_bytes=max(1, int(min_chunk_bytes)),
                max_chunks_per_file=max(1, int(max_chunks_per_file)),
                skip_minified=bool(skip_minified),
            ),
        )

        if max_total_chunks is not None and max_total_chunks > 0 and len(chunks) > max_total_chunks:
            ranked = sorted(chunks, key=lambda chunk: len(chunk.source or ""), reverse=True)
            dropped = len(chunks) - max_total_chunks
            chunks = ranked[:max_total_chunks]
            filter_stats = replace(
                filter_stats,
                kept_chunks=len(chunks),
                skipped_per_file_cap=filter_stats.skipped_per_file_cap + dropped,
            )
            if show_progress:
                print(f"Total chunk cap applied: kept {len(chunks)}/{len(ranked)}")
        filter_stage_duration_seconds = time.time() - filter_started
        if _progress_reporter is not None:
            _progress_reporter.update(
                chunks_built=chunk_count_before_filter,
                chunks_kept=len(chunks),
                message=f"Built {chunk_count_before_filter} chunks and kept {len(chunks)}",
            )

        if show_progress:
            print(
                "Chunk filtering: "
                f"kept {filter_stats.kept_chunks}/{filter_stats.input_chunks}, "
                f"small={filter_stats.skipped_small}, "
                f"minified={filter_stats.skipped_minified}, "
                f"capped={filter_stats.skipped_per_file_cap}"
            )

        if not chunks and incremental_resume:
            file_cache.save()
            if show_progress:
                print("No chunks to index after filtering")
            return self._zero_index_stats(
                started_at=started_at,
                generate_descriptions=generate_descriptions,
                skipped_small_chunks=filter_stats.skipped_small,
                skipped_minified_chunks=filter_stats.skipped_minified,
                skipped_capped_chunks=filter_stats.skipped_per_file_cap,
                files_capped=filter_stats.files_capped,
            )

        from engine.src.dedup import deduplicate_chunks, expand_descriptions, expand_embeddings
        from engine.src.index_resume import (
            ResumeState,
            build_settings_fingerprint,
            clear_state,
            load_state,
            save_state,
        )
        from engine.src.indexer import IndexStats, QdrantIndexer

        resume_files, delete_only_files = self._build_resume_files(changed_results, chunks)
        changed_paths = sorted({result.file_path for result in changed_results})
        resume_settings = build_settings_fingerprint(
            {
                "directory": str(directory),
                "project_name": detected_project,
                "force_refresh": force_refresh,
                "fast": fast,
                "generate_descriptions": generate_descriptions,
                "min_chunk_bytes": min_chunk_bytes,
                "max_chunks_per_file": max_chunks_per_file,
                "max_total_chunks": max_total_chunks,
                "compress_for_embedding": compress_for_embedding,
                "skip_minified": skip_minified,
                "resume_batch_files": max(1, int(resume_batch_files)),
                "files": [
                    {
                        "file_path": item.file_path,
                        "file_hash": item.file_hash,
                        "chunk_count": item.chunk_count,
                    }
                    for item in resume_files
                ],
                "delete_only_files": delete_only_files,
            }
        )

        active_collection = collection.resolve_alias()
        resume_state = load_state(directory, detected_project)
        shadow_name: str | None = None

        def shadow_exists(name: str) -> bool:
            try:
                collection.client.get_collection(name)
                return True
            except Exception:
                return False

        if resume_state is not None:
            expected_manifest = {
                (item.file_path, item.file_hash, item.chunk_count)
                for item in resume_files
            }
            stored_manifest = {
                (item.file_path, item.file_hash, item.chunk_count)
                for item in resume_state.files
            }
            if active_collection == resume_state.shadow_collection:
                clear_state(directory, detected_project)
                resume_state = None
            elif (
                resume_state.settings_fingerprint == resume_settings
                and stored_manifest == expected_manifest
                and resume_state.delete_only_files == delete_only_files
                and resume_state.source_collection == active_collection
                and shadow_exists(resume_state.shadow_collection)
            ):
                shadow_name = resume_state.shadow_collection
                if show_progress:
                    print(f"Resuming shadow collection: {shadow_name}")
            else:
                if shadow_exists(resume_state.shadow_collection) and active_collection != resume_state.shadow_collection:
                    try:
                        collection.client.delete_collection(resume_state.shadow_collection)
                    except Exception:
                        pass
                clear_state(directory, detected_project)
                resume_state = None

        if resume_state is None:
            shadow_name = collection.create_shadow()
            resume_state = ResumeState(
                manifest_version=1,
                project_name=detected_project,
                directory=str(directory),
                shadow_collection=shadow_name,
                source_collection=active_collection,
                settings_fingerprint=resume_settings,
                copied_points=False,
                deleted_changed_paths=False,
                files=resume_files,
                delete_only_files=delete_only_files,
            )
            save_state(directory, resume_state)
            if show_progress:
                print(f"Created shadow collection: {shadow_name}")

        shadow_indexer = QdrantIndexer(
            client=collection.client,
            collection_name=shadow_name,
            batch_size=self._config.qdrant.upsert_batch_size,
            upsert_concurrency=self._config.qdrant.upsert_concurrency,
        )

        if resume_state.source_collection is not None and not force_refresh and not resume_state.copied_points:
            copied_points = collection.copy_points(
                source_collection=resume_state.source_collection,
                target_collection=shadow_name,
                batch_size=self._config.qdrant.upsert_batch_size,
            )
            resume_state = replace(resume_state, copied_points=True)
            save_state(directory, resume_state)
            if show_progress:
                print(f"Copied {copied_points} existing points into {shadow_name}")

        if changed_paths and not resume_state.deleted_changed_paths:
            shadow_indexer.delete_by_files(changed_paths)
            resume_state = replace(resume_state, deleted_changed_paths=True)
            save_state(directory, resume_state)
            if show_progress:
                print(f"Removed stale points for {len(changed_paths)} changed files from {shadow_name}")

        indexed_state = shadow_indexer.get_file_index_state([item.file_path for item in resume_files])
        completed_paths, pending_paths = self._resume_pending_files(resume_files, indexed_state)
        if show_progress and completed_paths:
            print(f"Recovered {len(completed_paths)} already indexed files from {shadow_name}")
        if _progress_reporter is not None:
            _progress_reporter.update(
                files_planned=len(resume_files),
                files_indexed=len(completed_paths),
                files_remaining=len(pending_paths),
                message=f"Recovered {len(completed_paths)} files from shadow state",
            )

        if not pending_paths:
            self._finalize_shadow_collection(collection, shadow_name, show_progress)
            clear_state(directory, detected_project)
            file_cache.save()
            if show_progress:
                if resume_files:
                    print("Recovered shadow collection and applied alias swap")
                else:
                    print("Applied filtered deletions with no replacement chunks")
            zero_stats = self._zero_index_stats(
                started_at=started_at,
                generate_descriptions=generate_descriptions,
                skipped_small_chunks=filter_stats.skipped_small,
                skipped_minified_chunks=filter_stats.skipped_minified,
                skipped_capped_chunks=filter_stats.skipped_per_file_cap,
                files_capped=filter_stats.files_capped,
            )
            if _progress_reporter is not None:
                _progress_reporter.update(files_remaining=0, files_indexed=len(resume_files))
            return zero_stats

        pending_path_set = set(pending_paths)
        pending_chunks = [chunk for chunk in chunks if chunk.file_path in pending_path_set]

        if _progress_reporter is not None:
            _progress_reporter.set_stage("dedup", "Deduplicating chunks")
        dedup_started = time.time()
        dedup_result = deduplicate_chunks(pending_chunks)
        dedup_stage_duration_seconds = time.time() - dedup_started
        if _progress_reporter is not None:
            _progress_reporter.update(
                duplicate_chunks=dedup_result.duplicate_count,
                files_indexed=len(completed_paths),
                files_remaining=len(pending_paths),
                message=f"Deduplicated {dedup_result.total_chunks} chunks to {dedup_result.unique_count} unique chunks",
            )

        if compress_for_embedding:
            for idx, chunk in enumerate(dedup_result.unique_chunks):
                compressed_source, _ = compress_source(chunk.source, level=compress_for_embedding, language=chunk.language)
                if compressed_source and compressed_source != chunk.source:
                    dedup_result.unique_chunks[idx] = replace(chunk, source=compressed_source)

        if show_progress and dedup_result.duplicate_count > 0:
            print(
                f"Deduplicated: {dedup_result.total_chunks} chunks -> "
                f"{dedup_result.unique_count} unique, "
                f"{dedup_result.duplicate_count} duplicates skipped"
            )

        if generate_descriptions:
            if show_progress:
                print(f"Generating descriptions for {dedup_result.unique_count} unique chunks...")

            if _progress_reporter is not None:
                _progress_reporter.set_stage("describe", "Generating descriptions")
            description_started = time.time()
            unique_descriptions = self.describer.generate_batch(
                dedup_result.unique_chunks,
                progress_callback=(
                    (lambda completed, total: _progress_reporter.update(
                        description_chunks_completed=completed,
                        message=f"Generated descriptions for {completed}/{total} chunks",
                    ))
                    if _progress_reporter is not None else None
                ),
            )
            descriptions = expand_descriptions(unique_descriptions, dedup_result)
            llm_cost = sum(d.cost_usd for d in unique_descriptions)
            description_stage_duration_seconds = time.time() - description_started

            if show_progress:
                total_tokens = sum(d.token_count for d in unique_descriptions)
                print(f"Generated descriptions ({total_tokens} tokens, ${llm_cost:.4f})")
        else:
            from engine.src.describer import build_fallback_descriptions

            if _progress_reporter is not None:
                _progress_reporter.set_stage("describe", "Building fallback descriptions")
            description_started = time.time()
            unique_descriptions = build_fallback_descriptions(dedup_result.unique_chunks)
            unique_descriptions = self._maybe_generate_lightweight_artifact_metadata(
                dedup_result.unique_chunks,
                unique_descriptions,
                progress_callback=(
                    (lambda completed, total: _progress_reporter.update(
                        description_chunks_completed=completed,
                        message=f"Generated lightweight metadata for {completed}/{total} artifact chunks",
                    ))
                    if _progress_reporter is not None else None
                ),
            )
            descriptions = expand_descriptions(unique_descriptions, dedup_result)
            llm_cost = sum(item.cost_usd for item in unique_descriptions)
            description_stage_duration_seconds = time.time() - description_started
            if _progress_reporter is not None:
                _progress_reporter.update(
                    description_chunks_completed=dedup_result.unique_count,
                    message=f"Built fallback descriptions for {dedup_result.unique_count} chunks",
                )
            if show_progress:
                print("Skipping description generation (fast/no-descriptions mode)")

        if show_progress:
            print("Generating embeddings...")

        if _progress_reporter is not None:
            _progress_reporter.set_stage("embed", "Generating embeddings")
        unique_embedded, embedding_cost = self.embedder.embed_batch(
            dedup_result.unique_chunks,
            unique_descriptions,
            concurrency_override=embedding_concurrency,
            max_retries_override=embedding_max_retries,
            batch_size_override=embedding_batch_size,
            adaptive_batching_override=embedding_adaptive_batching,
            progress_callback=(
                (lambda completed, total: _progress_reporter.update(
                    embedding_chunks_completed=completed,
                    message=f"Embedded {completed}/{total} chunks",
                ))
                if _progress_reporter is not None else None
            ),
        )
        embed_stats = self.embedder.last_run_stats
        embedded_chunks = expand_embeddings(unique_embedded, pending_chunks, descriptions, dedup_result)
        embedded_by_file: dict[str, list] = {}
        for chunk in embedded_chunks:
            embedded_by_file.setdefault(chunk.chunk.file_path, []).append(chunk)

        total_upsert_stats = IndexStats(
            total_chunks=0,
            upserted_points=0,
            failed_points=0,
            total_tokens=0,
            duration_seconds=0.0,
            llm_cost_usd=0.0,
            embedding_cost_usd=0.0,
            descriptions_enabled=generate_descriptions,
        )

        batch_size = max(1, int(resume_batch_files))
        processed_pending_files = 0
        for offset in range(0, len(pending_paths), batch_size):
            batch_paths = pending_paths[offset:offset + batch_size]
            batch_embedded = []
            for file_path in batch_paths:
                batch_embedded.extend(embedded_by_file.get(file_path, []))
            if not batch_embedded:
                continue

            if show_progress:
                print(f"Upserting {len(batch_embedded)} chunks to {shadow_name} for {len(batch_paths)} files...")
            if _progress_reporter is not None:
                _progress_reporter.set_stage("upsert", "Upserting chunks")
            points_before = total_upsert_stats.upserted_points
            failed_before = total_upsert_stats.failed_points
            batch_stats = shadow_indexer.upsert_chunks(
                batch_embedded,
                progress_callback=(
                    (lambda completed, total, failed: _progress_reporter.update(
                        points_upserted=points_before + completed,
                        points_failed=failed_before + failed,
                        message=f"Upserting {points_before + completed}/{points_before + total} points",
                    ))
                    if _progress_reporter is not None else None
                ),
            )
            total_upsert_stats = self._merge_index_stats(total_upsert_stats, batch_stats)
            processed_pending_files += len(batch_paths)
            if _progress_reporter is not None:
                _progress_reporter.update(
                    files_indexed=len(completed_paths) + processed_pending_files,
                    files_remaining=max(0, len(pending_paths) - processed_pending_files),
                    points_upserted=total_upsert_stats.upserted_points,
                    points_failed=total_upsert_stats.failed_points,
                    message=(
                        f"Upserted {total_upsert_stats.upserted_points} points across "
                        f"{len(completed_paths) + processed_pending_files}/{len(resume_files)} files"
                    ),
                )

        self._finalize_shadow_collection(collection, shadow_name, show_progress)
        clear_state(directory, detected_project)

        stats = replace(
            total_upsert_stats,
            duration_seconds=time.time() - started_at,
            llm_cost_usd=llm_cost,
            embedding_cost_usd=embedding_cost,
            skipped_small_chunks=filter_stats.skipped_small,
            skipped_minified_chunks=filter_stats.skipped_minified,
            skipped_capped_chunks=filter_stats.skipped_per_file_cap,
            files_capped=filter_stats.files_capped,
            descriptions_enabled=generate_descriptions,
            embedding_requests=embed_stats.request_count,
            embedding_retries=embed_stats.retry_count,
            embedding_failed_requests=embed_stats.failed_request_count,
            embedding_input_count=embed_stats.input_count,
            embedding_stage_duration_seconds=embed_stats.duration_seconds,
            embedding_final_batch_size=embed_stats.final_batch_size,
            embedding_batch_shrink_events=embed_stats.batch_shrink_events,
            embedding_batch_grow_events=embed_stats.batch_grow_events,
            scan_stage_duration_seconds=scan_stage_duration_seconds,
            artifact_profile_stage_duration_seconds=artifact_profile_stage_duration_seconds,
            extract_stage_duration_seconds=extract_stage_duration_seconds,
            chunk_build_stage_duration_seconds=chunk_build_stage_duration_seconds,
            filter_stage_duration_seconds=filter_stage_duration_seconds,
            dedup_stage_duration_seconds=dedup_stage_duration_seconds,
            description_stage_duration_seconds=description_stage_duration_seconds,
            point_build_stage_duration_seconds=total_upsert_stats.point_build_stage_duration_seconds,
            upsert_stage_duration_seconds=total_upsert_stats.upsert_stage_duration_seconds,
        )

        file_cache.save()

        if show_progress:
            total_cost = llm_cost + embedding_cost
            print(f"Indexed {stats.upserted_points} chunks in {stats.duration_seconds:.2f}s")
            print(f"Total cost: ${total_cost:.4f} (LLM: ${llm_cost:.4f}, Embeddings: ${embedding_cost:.4f})")
            print(
                "Embedding telemetry: "
                f"requests={stats.embedding_requests}, "
                f"retries={stats.embedding_retries}, "
                f"failed_requests={stats.embedding_failed_requests}, "
                f"inputs={stats.embedding_input_count}, "
                f"duration={stats.embedding_stage_duration_seconds:.2f}s, "
                f"final_batch={stats.embedding_final_batch_size}, "
                f"shrink={stats.embedding_batch_shrink_events}, "
                f"grow={stats.embedding_batch_grow_events}"
            )
            print(
                "Index phase timings: "
                f"scan={stats.scan_stage_duration_seconds:.2f}s, "
                f"artifact_profile={stats.artifact_profile_stage_duration_seconds:.2f}s, "
                f"extract={stats.extract_stage_duration_seconds:.2f}s, "
                f"chunk={stats.chunk_build_stage_duration_seconds:.2f}s, "
                f"filter={stats.filter_stage_duration_seconds:.2f}s, "
                f"dedup={stats.dedup_stage_duration_seconds:.2f}s, "
                f"describe={stats.description_stage_duration_seconds:.2f}s, "
                f"embed={stats.embedding_stage_duration_seconds:.2f}s, "
                f"point_build={stats.point_build_stage_duration_seconds:.2f}s, "
                f"upsert={stats.upsert_stage_duration_seconds:.2f}s"
            )
            if stats.failed_points > 0:
                print(f"Failed: {stats.failed_points} chunks")

        return stats

    def refresh_files(
        self,
        file_paths: list[str | Path],
        show_progress: bool = True,
        project_name: Optional[str] = None,
        fast: bool = False,
        generate_descriptions: bool = True,
        min_chunk_bytes: int = 120,
        max_chunks_per_file: int = 400,
        max_total_chunks: Optional[int] = None,
        compress_for_embedding: Optional[str] = None,
        skip_minified: bool = True,
        embedding_concurrency: Optional[int] = None,
        embedding_max_retries: Optional[int] = None,
        embedding_batch_size: Optional[int] = None,
        embedding_adaptive_batching: Optional[bool] = None,
        _progress_reporter: OperationProgressReporter | None = None,
    ) -> "IndexStats":
        """
        Refresh specific files: re-extract, re-chunk, re-embed, and re-index.
        Uses mtime cache to skip unchanged files and chunk-level diffing for changed ones.

        file_paths: list[str | Path] — Files to refresh.
        show_progress: bool — Print progress messages.
        project_name: Optional[str] — Manual project name override. Auto-detected from first file if None.
        fast: bool — Fast mode: skips descriptions and tightens chunk filtering defaults.
        generate_descriptions: bool — Generate LLM descriptions when True.
        min_chunk_bytes: int — Minimum chunk size in bytes to keep.
        max_chunks_per_file: int — Maximum chunks kept per file after quality ranking.
        max_total_chunks: Optional[int] — Maximum total chunks kept after filtering.
        compress_for_embedding: Optional[str] — Compression level applied to source before embedding.
        skip_minified: bool — Skip likely minified/noise-heavy chunks.
        embedding_concurrency: Optional[int] — Override embedding request concurrency for this run.
        embedding_max_retries: Optional[int] — Override embedding request retry count for this run.
        embedding_batch_size: Optional[int] — Override embedding request batch size for this run.
        embedding_adaptive_batching: Optional[bool] — Override adaptive embedding batch resizing.
        Returns: IndexStats — Indexing statistics.
        """
        from engine.src.indexer import IndexStats

        started_at = time.time()
        if _progress_reporter is not None:
            _progress_reporter.start("prepare", "Preparing refresh inputs")

        resolved_paths = [str(Path(p).resolve()) for p in file_paths]

        if not resolved_paths:
            raise ValueError("No file paths provided")

        lsp_enrich_root = Path(resolved_paths[0]).parent
        detected_project = detect_project_name(lsp_enrich_root, manual_override=project_name)
        indexer = self._get_indexer(detected_project)
        file_cache = self._get_file_cache(lsp_enrich_root)
        existing_paths: list[str] = []
        deleted_paths: list[str] = []
        for fp in resolved_paths:
            if Path(fp).exists():
                existing_paths.append(fp)
            else:
                deleted_paths.append(fp)
        if _progress_reporter is not None:
            _progress_reporter.update(
                files_discovered=len(resolved_paths),
                files_deleted=len(deleted_paths),
                message=f"Prepared {len(resolved_paths)} refresh targets",
            )

        if show_progress:
            print(f"Refreshing files in project: {detected_project}")
            print(f"Checking mtime cache for {len(resolved_paths)} files...")

        paths_to_extract: list[str] = []
        mtime_skipped = 0
        for fp in existing_paths:
            cached_hash = file_cache.is_unchanged(fp)
            if cached_hash is not None:
                mtime_skipped += 1
            else:
                paths_to_extract.append(fp)
        if _progress_reporter is not None:
            _progress_reporter.update(
                files_unchanged=mtime_skipped,
                files_planned=len(paths_to_extract),
                files_remaining=len(paths_to_extract),
                message=f"Refreshing {len(paths_to_extract)} changed files",
            )

        if show_progress and mtime_skipped > 0:
            print(f"Skipping {mtime_skipped} files (mtime+size unchanged)")

        if deleted_paths:
            if show_progress:
                print(f"Deleting chunks for {len(deleted_paths)} missing files...")
            indexer.delete_by_files(deleted_paths)
            for fp in deleted_paths:
                file_cache.remove(fp)

        if not paths_to_extract and not deleted_paths:
            if show_progress:
                print("All files unchanged (mtime cache hit)")
            empty_stats = IndexStats(
                total_chunks=0,
                upserted_points=0,
                failed_points=0,
                total_tokens=0,
                duration_seconds=time.time() - started_at,
                llm_cost_usd=0.0,
                embedding_cost_usd=0.0,
                descriptions_enabled=generate_descriptions,
            )
            if _progress_reporter is not None:
                _progress_reporter.complete(asdict(empty_stats), message="All files unchanged")
            return empty_stats

        if not paths_to_extract:
            file_cache.save()
            if show_progress:
                print("No files to refresh after filtering")
            empty_stats = IndexStats(
                total_chunks=0,
                upserted_points=0,
                failed_points=0,
                total_tokens=0,
                duration_seconds=time.time() - started_at,
                llm_cost_usd=0.0,
                embedding_cost_usd=0.0,
                descriptions_enabled=generate_descriptions,
            )
            if _progress_reporter is not None:
                _progress_reporter.complete(asdict(empty_stats), message="Deleted missing files only")
            return empty_stats

        self._ensure_indexing_backends_ready(
            operation_name="refresh",
            generate_descriptions=generate_descriptions,
        )

        if show_progress:
            print(f"Analyzing changes in {len(paths_to_extract)} files...")
        if _progress_reporter is not None:
            _progress_reporter.set_stage("extract", "Analyzing changed files")

        if fast:
            generate_descriptions = False
            min_chunk_bytes = max(min_chunk_bytes, 200)
            max_chunks_per_file = min(max_chunks_per_file, 180)
            skip_minified = True
            if max_total_chunks is None:
                max_total_chunks = 50000
        (
            embedding_concurrency,
            embedding_batch_size,
            embedding_max_retries,
            embedding_adaptive_batching,
        ) = self._apply_fast_embedding_defaults(
            fast=fast,
            embedding_concurrency=embedding_concurrency,
            embedding_batch_size=embedding_batch_size,
            embedding_max_retries=embedding_max_retries,
            embedding_adaptive_batching=embedding_adaptive_batching,
        )

        extract_stage_duration_seconds = 0.0
        chunk_build_stage_duration_seconds = 0.0
        filter_stage_duration_seconds = 0.0
        dedup_stage_duration_seconds = 0.0
        description_stage_duration_seconds = 0.0
        artifact_profiles_by_path: dict[str, ArtifactIndexProfile] = {}
        if _progress_reporter is not None:
            _progress_reporter.set_stage("artifact_profile", "Profiling artifact candidates")
        artifact_profile_started = time.time()
        if fast and skip_minified:
            for file_path in paths_to_extract:
                path_obj = Path(file_path)
                file_size = int(path_obj.stat().st_size) if path_obj.exists() else 0
                if file_size < 128 * 1024:
                    continue
                suffix_language = {
                    ".js": "javascript",
                    ".jsx": "jsx",
                    ".ts": "typescript",
                    ".tsx": "tsx",
                }.get(path_obj.suffix.lower())
                if suffix_language is None:
                    continue
                try:
                    profile = analyze_artifact_candidate(
                        file_path=file_path,
                        language=suffix_language,
                        file_size=file_size,
                        file_mtime=int(path_obj.stat().st_mtime),
                    )
                except Exception:
                    continue
                if should_downgrade_artifact_profile(profile):
                    artifact_profiles_by_path[file_path] = profile
        artifact_profile_stage_duration_seconds = time.time() - artifact_profile_started

        if show_progress and artifact_profiles_by_path:
            total_bytes = sum(profile.file_size for profile in artifact_profiles_by_path.values())
            print(
                "Downgrading "
                f"{len(artifact_profiles_by_path)} refreshed artifact files "
                f"({_human_readable_bytes(total_bytes)}) to coarse file chunks"
            )

        all_chunks_to_reindex: list[CodeChunk] = []
        all_identities_to_delete: list = []
        total_unchanged = 0
        total_skipped_small = 0
        total_skipped_minified = 0
        total_skipped_capped = 0
        total_files_capped = 0
        old_chunks_by_file = indexer.get_chunks_by_files(paths_to_extract)
        analyzed_files = 0

        for file_path in paths_to_extract:
            old_chunks = old_chunks_by_file.get(file_path, [])

            artifact_profile = artifact_profiles_by_path.get(file_path)
            extract_started = time.time()
            if artifact_profile is not None:
                results = [
                    ExtractionResult(
                        file_path=file_path,
                        language=artifact_profile.language,
                        symbols=[],
                        errors=[ARTIFACT_FALLBACK_ERROR],
                        file_hash=artifact_profile.file_hash,
                        file_size=artifact_profile.file_size,
                        file_mtime=artifact_profile.file_mtime,
                    )
                ]
            else:
                results = self.extract_symbols(file_path)
                results, _ = self._enrich_extraction_results_with_lsp_symbols(
                    results,
                    root_path=lsp_enrich_root,
                )
            extract_stage_duration_seconds += time.time() - extract_started
            chunk_started = time.time()
            new_chunks = self.chunker.build_chunks(results)
            chunk_build_stage_duration_seconds += time.time() - chunk_started

            for result in results:
                file_cache.update_from_extraction(
                    result.file_path, result.file_hash, result.file_size, result.file_mtime,
                )

            filter_started = time.time()
            filtered_chunks, filter_stats = filter_chunks(
                new_chunks,
                ChunkFilterConfig(
                    min_chunk_bytes=max(1, int(min_chunk_bytes)),
                    max_chunks_per_file=max(1, int(max_chunks_per_file)),
                    skip_minified=bool(skip_minified),
                ),
            )
            filter_stage_duration_seconds += time.time() - filter_started
            diff = ChunkDiffer.diff(old_chunks, filtered_chunks)

            all_chunks_to_reindex.extend(diff.needs_reindex)
            all_identities_to_delete.extend(diff.needs_deletion)
            total_unchanged += len(diff.unchanged)
            total_skipped_small += filter_stats.skipped_small
            total_skipped_minified += filter_stats.skipped_minified
            total_skipped_capped += filter_stats.skipped_per_file_cap
            total_files_capped += filter_stats.files_capped
            analyzed_files += 1
            if _progress_reporter is not None:
                _progress_reporter.update(
                    files_analyzed=analyzed_files,
                    chunks_built=len(all_chunks_to_reindex) + total_unchanged,
                    chunks_kept=len(all_chunks_to_reindex),
                    artifact_downgraded_files=len(artifact_profiles_by_path),
                    message=f"Analyzed {analyzed_files}/{len(paths_to_extract)} files",
                )

            if show_progress:
                print(
                    f"  {file_path}: +{len(diff.added)} ~{len(diff.modified)} "
                    f"-{len(diff.deleted)} ={len(diff.unchanged)} "
                    f"kept={len(filtered_chunks)}/{len(new_chunks)}"
                )

        if max_total_chunks is not None and max_total_chunks > 0 and len(all_chunks_to_reindex) > max_total_chunks:
            ranked = sorted(all_chunks_to_reindex, key=lambda chunk: len(chunk.source or ""), reverse=True)
            dropped = len(all_chunks_to_reindex) - max_total_chunks
            all_chunks_to_reindex = ranked[:max_total_chunks]
            total_skipped_capped += dropped
            if show_progress:
                print(f"Total chunk cap applied: kept {len(all_chunks_to_reindex)}/{len(ranked)}")

        if show_progress:
            print(
                f"Total: {len(all_chunks_to_reindex)} chunks to re-index, "
                f"{len(all_identities_to_delete)} to delete, {total_unchanged} unchanged"
            )
            print(
                "Chunk filtering: "
                f"small={total_skipped_small}, "
                f"minified={total_skipped_minified}, "
                f"capped={total_skipped_capped}, "
                f"files_capped={total_files_capped}"
            )

        if all_identities_to_delete:
            if show_progress:
                print(f"Deleting {len(all_identities_to_delete)} obsolete chunks...")
            indexer.delete_by_identities(all_identities_to_delete)

        if not all_chunks_to_reindex:
            file_cache.save()
            if show_progress:
                print("No chunks to re-index (all unchanged)")
            empty_stats = IndexStats(
                total_chunks=0,
                upserted_points=0,
                failed_points=0,
                total_tokens=0,
                duration_seconds=time.time() - started_at,
                llm_cost_usd=0.0,
                embedding_cost_usd=0.0,
                skipped_small_chunks=total_skipped_small,
                skipped_minified_chunks=total_skipped_minified,
                skipped_capped_chunks=total_skipped_capped,
                files_capped=total_files_capped,
                descriptions_enabled=generate_descriptions,
            )
            if _progress_reporter is not None:
                _progress_reporter.complete(asdict(empty_stats), message="No chunks to re-index")
            return empty_stats

        from engine.src.dedup import deduplicate_chunks, expand_descriptions, expand_embeddings
        if _progress_reporter is not None:
            _progress_reporter.set_stage("dedup", "Deduplicating chunks")
        dedup_started = time.time()
        dedup_result = deduplicate_chunks(all_chunks_to_reindex)
        dedup_stage_duration_seconds = time.time() - dedup_started
        if _progress_reporter is not None:
            _progress_reporter.update(
                duplicate_chunks=dedup_result.duplicate_count,
                message=f"Deduplicated {dedup_result.total_chunks} chunks to {dedup_result.unique_count} unique chunks",
            )

        if compress_for_embedding:
            for idx, chunk in enumerate(dedup_result.unique_chunks):
                compressed_source, _ = compress_source(chunk.source, level=compress_for_embedding, language=chunk.language)
                if compressed_source and compressed_source != chunk.source:
                    dedup_result.unique_chunks[idx] = replace(chunk, source=compressed_source)

        if show_progress and dedup_result.duplicate_count > 0:
            print(
                f"Deduplicated: {dedup_result.total_chunks} chunks -> "
                f"{dedup_result.unique_count} unique, "
                f"{dedup_result.duplicate_count} duplicates skipped"
            )

        if generate_descriptions:
            if show_progress:
                print(f"Generating descriptions for {dedup_result.unique_count} unique chunks...")

            if _progress_reporter is not None:
                _progress_reporter.set_stage("describe", "Generating descriptions")
            description_started = time.time()
            unique_descriptions = self.describer.generate_batch(
                dedup_result.unique_chunks,
                progress_callback=(
                    (lambda completed, total: _progress_reporter.update(
                        description_chunks_completed=completed,
                        message=f"Generated descriptions for {completed}/{total} chunks",
                    ))
                    if _progress_reporter is not None else None
                ),
            )
            descriptions = expand_descriptions(unique_descriptions, dedup_result)
            llm_cost = sum(d.cost_usd for d in unique_descriptions)
            description_stage_duration_seconds = time.time() - description_started

            if show_progress:
                total_tokens = sum(d.token_count for d in unique_descriptions)
                print(f"Generated descriptions ({total_tokens} tokens, ${llm_cost:.4f})")
        else:
            from engine.src.describer import build_fallback_descriptions

            if _progress_reporter is not None:
                _progress_reporter.set_stage("describe", "Building fallback descriptions")
            description_started = time.time()
            unique_descriptions = build_fallback_descriptions(dedup_result.unique_chunks)
            unique_descriptions = self._maybe_generate_lightweight_artifact_metadata(
                dedup_result.unique_chunks,
                unique_descriptions,
                progress_callback=(
                    (lambda completed, total: _progress_reporter.update(
                        description_chunks_completed=completed,
                        message=f"Generated lightweight metadata for {completed}/{total} artifact chunks",
                    ))
                    if _progress_reporter is not None else None
                ),
            )
            descriptions = expand_descriptions(unique_descriptions, dedup_result)
            llm_cost = sum(item.cost_usd for item in unique_descriptions)
            description_stage_duration_seconds = time.time() - description_started
            if _progress_reporter is not None:
                _progress_reporter.update(
                    description_chunks_completed=dedup_result.unique_count,
                    message=f"Built fallback descriptions for {dedup_result.unique_count} chunks",
                )
            if show_progress:
                print("Skipping description generation (fast/no-descriptions mode)")

        if show_progress:
            print("Generating embeddings...")

        if _progress_reporter is not None:
            _progress_reporter.set_stage("embed", "Generating embeddings")
        unique_embedded, embedding_cost = self.embedder.embed_batch(
            dedup_result.unique_chunks,
            unique_descriptions,
            concurrency_override=embedding_concurrency,
            max_retries_override=embedding_max_retries,
            batch_size_override=embedding_batch_size,
            adaptive_batching_override=embedding_adaptive_batching,
            progress_callback=(
                (lambda completed, total: _progress_reporter.update(
                    embedding_chunks_completed=completed,
                    message=f"Embedded {completed}/{total} chunks",
                ))
                if _progress_reporter is not None else None
            ),
        )
        embed_stats = self.embedder.last_run_stats
        embedded_chunks = expand_embeddings(
            unique_embedded, all_chunks_to_reindex, descriptions, dedup_result,
        )

        if show_progress:
            print(f"Upserting {len(embedded_chunks)} chunks to Qdrant...")

        if _progress_reporter is not None:
            _progress_reporter.set_stage("upsert", "Upserting chunks")
        indexer_stats = indexer.upsert_chunks(
            embedded_chunks,
            progress_callback=(
                (lambda completed, total, failed: _progress_reporter.update(
                    points_upserted=completed,
                    points_failed=failed,
                    message=f"Upserting {completed}/{total} points",
                ))
                if _progress_reporter is not None else None
            ),
        )
        if _progress_reporter is not None:
            _progress_reporter.update(
                files_indexed=len(paths_to_extract),
                files_remaining=0,
                points_upserted=indexer_stats.upserted_points,
                points_failed=indexer_stats.failed_points,
                message=f"Upserted {indexer_stats.upserted_points} points",
            )

        stats = IndexStats(
            total_chunks=indexer_stats.total_chunks,
            upserted_points=indexer_stats.upserted_points,
            failed_points=indexer_stats.failed_points,
            total_tokens=indexer_stats.total_tokens,
            duration_seconds=time.time() - started_at,
            llm_cost_usd=llm_cost,
            embedding_cost_usd=embedding_cost,
            skipped_small_chunks=total_skipped_small,
            skipped_minified_chunks=total_skipped_minified,
            skipped_capped_chunks=total_skipped_capped,
            files_capped=total_files_capped,
            descriptions_enabled=generate_descriptions,
            embedding_requests=embed_stats.request_count,
            embedding_retries=embed_stats.retry_count,
            embedding_failed_requests=embed_stats.failed_request_count,
            embedding_input_count=embed_stats.input_count,
            embedding_stage_duration_seconds=embed_stats.duration_seconds,
            embedding_final_batch_size=embed_stats.final_batch_size,
            embedding_batch_shrink_events=embed_stats.batch_shrink_events,
            embedding_batch_grow_events=embed_stats.batch_grow_events,
            artifact_profile_stage_duration_seconds=artifact_profile_stage_duration_seconds,
            extract_stage_duration_seconds=extract_stage_duration_seconds,
            chunk_build_stage_duration_seconds=chunk_build_stage_duration_seconds,
            filter_stage_duration_seconds=filter_stage_duration_seconds,
            dedup_stage_duration_seconds=dedup_stage_duration_seconds,
            description_stage_duration_seconds=description_stage_duration_seconds,
            point_build_stage_duration_seconds=indexer_stats.point_build_stage_duration_seconds,
            upsert_stage_duration_seconds=indexer_stats.upsert_stage_duration_seconds,
        )

        file_cache.save()

        if show_progress:
            total_cost = llm_cost + embedding_cost
            print(f"Indexed {stats.upserted_points} chunks in {stats.duration_seconds:.2f}s")
            print(f"Total cost: ${total_cost:.4f} (LLM: ${llm_cost:.4f}, Embeddings: ${embedding_cost:.4f})")
            print(
                "Embedding telemetry: "
                f"requests={stats.embedding_requests}, "
                f"retries={stats.embedding_retries}, "
                f"failed_requests={stats.embedding_failed_requests}, "
                f"inputs={stats.embedding_input_count}, "
                f"duration={stats.embedding_stage_duration_seconds:.2f}s, "
                f"final_batch={stats.embedding_final_batch_size}, "
                f"shrink={stats.embedding_batch_shrink_events}, "
                f"grow={stats.embedding_batch_grow_events}"
            )
            print(
                "Refresh phase timings: "
                f"artifact_profile={stats.artifact_profile_stage_duration_seconds:.2f}s, "
                f"extract={stats.extract_stage_duration_seconds:.2f}s, "
                f"chunk={stats.chunk_build_stage_duration_seconds:.2f}s, "
                f"filter={stats.filter_stage_duration_seconds:.2f}s, "
                f"dedup={stats.dedup_stage_duration_seconds:.2f}s, "
                f"describe={stats.description_stage_duration_seconds:.2f}s, "
                f"embed={stats.embedding_stage_duration_seconds:.2f}s, "
                f"point_build={stats.point_build_stage_duration_seconds:.2f}s, "
                f"upsert={stats.upsert_stage_duration_seconds:.2f}s"
            )
            if stats.failed_points > 0:
                print(f"Failed: {stats.failed_points} chunks")

        return stats

    def qdrant_available(self, verify: bool = True) -> bool:
        """
        Return True when the configured Qdrant backend is reachable.

        verify: bool — When True, actively probe Qdrant instead of only checking an
        existing cached client state.
        """
        if self._config.qdrant is None:
            return False

        if not verify:
            return self._conn.is_alive() if self._conn is not None else False

        try:
            self.connect(verify=True)
            return True
        except ConnectionError:
            return False

    def _project_collection_from_info(
        self,
        project_name: str,
        info: dict | None,
        indexed: bool,
    ) -> ProjectCollectionInfo:
        if not info:
            return ProjectCollectionInfo(project_name=project_name, indexed=indexed)

        return ProjectCollectionInfo(
            project_name=project_name,
            indexed=indexed,
            real_collection=info.get("real_collection"),
            points_count=info.get("points_count"),
            indexed_vectors_count=info.get("indexed_vectors_count"),
            segments_count=info.get("segments_count"),
            status=info.get("status"),
            vectors=dict(info.get("vectors", {})),
        )

    def project_collection_info(self, project_name: str) -> ProjectCollectionInfo:
        """
        Return collection metadata for one logical project.
        """
        if self._config.qdrant is None or not self.qdrant_available(verify=True):
            return ProjectCollectionInfo(project_name=project_name, indexed=False)

        try:
            info = self._get_collection(project_name).info()
        except Exception:
            return ProjectCollectionInfo(project_name=project_name, indexed=False)

        points_count = int(info.get("points_count", 0) or 0)
        return self._project_collection_from_info(
            project_name=project_name,
            info=info,
            indexed=points_count > 0,
        )

    def list_projects(self) -> list[ProjectCollectionInfo]:
        """
        List indexed projects visible through the configured Qdrant backend.
        """
        if self._config.qdrant is None or not self.qdrant_available(verify=True):
            return []

        from engine.src.collection import CollectionManager

        manager = CollectionManager(
            client=self.connection.client,
            config=self._config,
            collection_name="_sdk_projects",
        )
        projects = []
        for item in manager.list_all_projects():
            points_count = int(item.get("points_count", 0) or 0)
            projects.append(
                self._project_collection_from_info(
                    project_name=str(item.get("name", "")),
                    info=item,
                    indexed=points_count > 0,
                )
            )
        return projects

    def _list_project_folders(
        self,
        project_root: str | Path,
        max_depth: int = 5,
        limit: int = 200,
    ) -> list[ProjectFolderInfo]:
        root = Path(project_root).resolve()
        if not root.exists() or not root.is_dir():
            return []

        folders = [ProjectFolderInfo(relative_path=".", absolute_path=str(root))]
        for candidate in sorted(root.rglob("*")):
            if not candidate.is_dir():
                continue
            try:
                relative = candidate.relative_to(root)
            except ValueError:
                continue
            if len(relative.parts) > max_depth:
                continue
            if any(part.startswith(".") for part in relative.parts):
                continue
            folders.append(
                ProjectFolderInfo(
                    relative_path=str(relative).replace("\\", "/"),
                    absolute_path=str(candidate.resolve()),
                )
            )
            if len(folders) >= limit:
                break

        return folders

    def project_info(
        self,
        path: str | Path = ".",
        project_name: Optional[str] = None,
        include_folders: bool = False,
        folder_max_depth: int = 5,
        folder_limit: int = 200,
    ) -> ProjectInfo:
        """
        Return stable project discovery metadata for downstream wrappers.
        """
        scope = Path(path).resolve()
        anchor = scope.parent if scope.is_file() else scope
        resolved_project = detect_project_name(anchor, manual_override=project_name)
        qdrant_enabled = self._config.qdrant is not None
        qdrant_available = self.qdrant_available(verify=True) if qdrant_enabled else False

        cache = self.cache_status(anchor)
        collection = self.project_collection_info(resolved_project)
        folders = (
            self._list_project_folders(anchor, max_depth=folder_max_depth, limit=folder_limit)
            if include_folders else []
        )

        return ProjectInfo(
            path=str(anchor),
            project_name=resolved_project,
            indexed=collection.indexed,
            qdrant_enabled=qdrant_enabled,
            qdrant_available=qdrant_available,
            parser_connected=self.parser_service.connected,
            cache_entries=int(cache.get("entries", 0)),
            cache_disk_bytes=int(cache.get("disk_bytes", 0)),
            cache_path=str(cache.get("cache_path", "")),
            collection=collection,
            folders=folders,
        )

    def lsp_setup_plan(self, path: str | Path) -> LspSetupPlanInfo:
        """
        Build a project-specific LSP setup plan for downstream wrappers and the CLI.
        """
        plan = build_lsp_setup_plan(path)
        return LspSetupPlanInfo(
            target_path=plan.target_path,
            platform=plan.platform,
            servers=[
                LspServerSetupInfo(
                    name=server.name,
                    language_id=server.language_id,
                    binary=server.binary,
                    installed=server.installed,
                    detection_reasons=list(server.detection_reasons),
                    auto_install_supported=server.auto_install_supported,
                    install_steps=[
                        LspInstallStepInfo(
                            manager=step.manager,
                            command=step.command,
                            note=step.note,
                        )
                        for step in server.install_steps
                    ],
                    notes=list(server.notes),
                )
                for server in plan.servers
            ],
        )

    def lsp_check(self, path: str | Path) -> LspCheckPlanInfo:
        """
        Build a project-specific LSP readiness report.
        """
        plan = build_lsp_check_plan(path)
        return LspCheckPlanInfo(
            target_path=plan.target_path,
            platform=plan.platform,
            servers=[
                LspServerCheckInfo(
                    name=server.name,
                    language_id=server.language_id,
                    binary=server.binary,
                    status=server.status,
                    installed=server.installed,
                    auto_install_supported=server.auto_install_supported,
                    detection_reasons=list(server.detection_reasons),
                    install_steps=[
                        LspInstallStepInfo(
                            manager=step.manager,
                            command=step.command,
                            note=step.note,
                        )
                        for step in server.install_steps
                    ],
                    notes=list(server.notes),
                    probe_command=server.probe_command,
                    probe_message=server.probe_message,
                )
                for server in plan.servers
            ],
        )

    def start_index_directory(
        self,
        directory: str | Path,
        force_refresh: bool = False,
        project_name: Optional[str] = None,
        fast: bool = False,
        generate_descriptions: bool = True,
        min_chunk_bytes: int = 120,
        max_chunks_per_file: int = 400,
        max_total_chunks: Optional[int] = None,
        compress_for_embedding: Optional[str] = None,
        incremental_resume: bool = False,
        resume_batch_files: int = 200,
        skip_minified: bool = True,
        embedding_concurrency: Optional[int] = None,
        embedding_max_retries: Optional[int] = None,
        embedding_batch_size: Optional[int] = None,
        embedding_adaptive_batching: Optional[bool] = None,
    ) -> IndexOperationSnapshot:
        """
        Start a background index operation owned by the SDK and return a pollable snapshot.
        """
        target_directory = Path(directory).resolve()
        resolved_project = detect_project_name(target_directory, manual_override=project_name)
        snapshot, attached = GLOBAL_OPERATION_REGISTRY.start_or_attach(
            kind="index",
            path=str(target_directory),
            project_name=resolved_project,
        )
        if attached:
            return replace(snapshot, message="Already running")

        def _runner() -> None:
            worker = QuickContext(self._config)
            reporter = OperationProgressReporter(GLOBAL_OPERATION_REGISTRY, snapshot.operation_id)
            try:
                stats = worker.index_directory(
                    directory=target_directory,
                    force_refresh=force_refresh,
                    show_progress=False,
                    project_name=resolved_project,
                    fast=fast,
                    generate_descriptions=generate_descriptions,
                    min_chunk_bytes=min_chunk_bytes,
                    max_chunks_per_file=max_chunks_per_file,
                    max_total_chunks=max_total_chunks,
                    compress_for_embedding=compress_for_embedding,
                    incremental_resume=incremental_resume,
                    resume_batch_files=resume_batch_files,
                    skip_minified=skip_minified,
                    embedding_concurrency=embedding_concurrency,
                    embedding_max_retries=embedding_max_retries,
                    embedding_batch_size=embedding_batch_size,
                    embedding_adaptive_batching=embedding_adaptive_batching,
                    _progress_reporter=reporter,
                )
                current = self.get_operation_status(snapshot.operation_id)
                if current is not None and current.status not in {"completed", "failed"}:
                    reporter.complete(asdict(stats), message="Index complete")
            except Exception as exc:
                reporter.fail(str(exc), stage="failed", message="Index failed")
            finally:
                worker.close()

        Thread(target=_runner, name=f"qc-index-{snapshot.operation_id}", daemon=True).start()
        return snapshot

    def start_refresh_files(
        self,
        file_paths: list[str | Path],
        project_name: Optional[str] = None,
        fast: bool = False,
        generate_descriptions: bool = True,
        min_chunk_bytes: int = 120,
        max_chunks_per_file: int = 400,
        max_total_chunks: Optional[int] = None,
        compress_for_embedding: Optional[str] = None,
        skip_minified: bool = True,
        embedding_concurrency: Optional[int] = None,
        embedding_max_retries: Optional[int] = None,
        embedding_batch_size: Optional[int] = None,
        embedding_adaptive_batching: Optional[bool] = None,
    ) -> IndexOperationSnapshot:
        """
        Start a background refresh operation owned by the SDK and return a pollable snapshot.
        """
        resolved_paths = [str(Path(item).resolve()) for item in file_paths]
        if not resolved_paths:
            raise ValueError("No file paths provided")
        anchor = Path(resolved_paths[0]).parent
        resolved_project = detect_project_name(anchor, manual_override=project_name)
        operation_path = str(anchor)
        snapshot, attached = GLOBAL_OPERATION_REGISTRY.start_or_attach(
            kind="refresh",
            path=operation_path,
            project_name=resolved_project,
        )
        if attached:
            return replace(snapshot, message="Already running")

        def _runner() -> None:
            worker = QuickContext(self._config)
            reporter = OperationProgressReporter(GLOBAL_OPERATION_REGISTRY, snapshot.operation_id)
            try:
                stats = worker.refresh_files(
                    file_paths=resolved_paths,
                    show_progress=False,
                    project_name=resolved_project,
                    fast=fast,
                    generate_descriptions=generate_descriptions,
                    min_chunk_bytes=min_chunk_bytes,
                    max_chunks_per_file=max_chunks_per_file,
                    max_total_chunks=max_total_chunks,
                    compress_for_embedding=compress_for_embedding,
                    skip_minified=skip_minified,
                    embedding_concurrency=embedding_concurrency,
                    embedding_max_retries=embedding_max_retries,
                    embedding_batch_size=embedding_batch_size,
                    embedding_adaptive_batching=embedding_adaptive_batching,
                    _progress_reporter=reporter,
                )
                current = self.get_operation_status(snapshot.operation_id)
                if current is not None and current.status not in {"completed", "failed"}:
                    reporter.complete(asdict(stats), message="Refresh complete")
            except Exception as exc:
                reporter.fail(str(exc), stage="failed", message="Refresh failed")
            finally:
                worker.close()

        Thread(target=_runner, name=f"qc-refresh-{snapshot.operation_id}", daemon=True).start()
        return snapshot

    def get_operation_status(self, operation_id: str) -> IndexOperationSnapshot | None:
        """
        Return one operation snapshot by ID.
        """
        return GLOBAL_OPERATION_REGISTRY.get(operation_id)

    def list_operation_statuses(
        self,
        active_only: bool = False,
        limit: int = 20,
    ) -> list[IndexOperationSnapshot]:
        """
        Return recent SDK-owned operation snapshots.
        """
        return GLOBAL_OPERATION_REGISTRY.list(active_only=active_only, limit=limit)

    def get_operation_status_for_target(
        self,
        *,
        kind: str,
        path: str | Path,
        project_name: Optional[str] = None,
    ) -> list[IndexOperationSnapshot]:
        """
        Return the latest operation snapshot for a specific logical target.
        """
        target_path = Path(path).resolve()
        resolved_project = detect_project_name(target_path, manual_override=project_name)
        return GLOBAL_OPERATION_REGISTRY.get_for_target(
            kind=kind,
            path=str(target_path),
            project_name=resolved_project,
        )

    def status(self, verify_qdrant: bool = True) -> dict:
        """
        Get engine status: connection health, collection info, provider details.

        Returns: dict — Status report.
        """
        result = {}

        if self._config.qdrant is not None:
            alive = self.qdrant_available(verify=verify_qdrant)
            result["qdrant"] = {
                "url": self._config.qdrant.url,
                "configured": True,
                "alive": alive,
            }
        else:
            result["qdrant"] = {"status": "disabled"}

        if self._config.code_embedding is not None:
            result["code_embedding"] = {
                "provider": self._config.code_embedding.provider,
                "model": self._config.code_embedding.model,
                "dimension": self._config.code_embedding.dimension,
            }
        else:
            result["code_embedding"] = {"status": "disabled"}

        if self._config.desc_embedding is not None:
            result["desc_embedding"] = {
                "provider": self._config.desc_embedding.provider,
                "model": self._config.desc_embedding.model,
                "dimension": self._config.desc_embedding.dimension,
            }
        else:
            result["desc_embedding"] = {"status": "disabled"}

        if self._config.llm is not None:
            result["llm"] = {
                "provider": self._config.llm.provider,
                "model": self._config.llm.model,
            }
        else:
            result["llm"] = {"status": "disabled"}

        result["parser"] = {
            "pipe_name": self.parser_service._client._pipe_name,
            "connected": self.parser_service.connected,
        }

        alive = result.get("qdrant", {}).get("alive", False)
        if alive and self._collections:
            result["collections"] = {}
            for project_name, collection in self._collections.items():
                try:
                    result["collections"][project_name] = collection.info()
                except Exception:
                    result["collections"][project_name] = {"error": "collection not found"}

        result["operations"] = [snapshot_to_dict(item) for item in self.list_operation_statuses(active_only=True, limit=20)]

        return result

    def cache_status(self, directory: str | Path) -> dict:
        """
        Get mtime file cache statistics for a project directory.

        directory: str | Path — Project root directory.
        Returns: dict — Cache stats (entries, disk_bytes, cache_path, project_root).
        """
        cache = self._get_file_cache(directory)
        return cache.stats()

    def cache_clear(self, directory: str | Path) -> int:
        """
        Clear the mtime file cache for a project directory.

        directory: str | Path — Project root directory.
        Returns: int — Number of entries cleared.
        """
        cache = self._get_file_cache(directory)
        count = cache.entry_count
        cache.clear()
        cache.save()
        return count

    def watch(
        self,
        directory: str | Path,
        project_name: Optional[str] = None,
        debounce_seconds: float = 2.0,
        show_progress: bool = True,
    ) -> None:
        """
        Watch a directory for file changes and trigger incremental re-indexing.
        Blocks until KeyboardInterrupt (Ctrl+C).

        directory: str | Path — Directory to watch.
        project_name: Optional[str] — Manual project name override.
        debounce_seconds: float — Seconds to wait after last change before refresh.
        show_progress: bool — Print progress messages during refresh.
        """
        directory = Path(directory).resolve()

        def _refresh(paths: list[str], proj: Optional[str]) -> None:
            if show_progress:
                print(f"\n[watch] {len(paths)} file(s) changed, refreshing...")
            stats = self.refresh_files(
                file_paths=paths,
                show_progress=show_progress,
                project_name=proj,
            )
            if show_progress:
                total = stats.upserted_points
                cost = stats.llm_cost_usd + stats.embedding_cost_usd
                if total > 0:
                    print(f"[watch] Refreshed {total} chunks (${cost:.4f})")
                else:
                    print("[watch] No chunks changed")

        def _on_error(exc: Exception) -> None:
            if show_progress:
                print(f"[watch] Refresh error: {exc}")

        from engine.src.watcher import FileWatcher
        watcher = FileWatcher(
            directory=directory,
            project_name=project_name,
            debounce_seconds=debounce_seconds,
            on_error=_on_error,
        )

        if show_progress:
            detected = detect_project_name(directory, manual_override=project_name)
            print(f"Watching project: {detected}")
            print(f"Directory: {directory}")
            print(f"Debounce: {debounce_seconds}s")
            print("Press Ctrl+C to stop\n")

        watcher.run_forever(_refresh)

        if show_progress:
            print("\nWatcher stopped")

    def close(self) -> None:
        """
        Close all connections and release resources.
        """
        self._background_warm_stop.set()
        if self._background_warm_thread is not None and self._background_warm_thread.is_alive():
            self._background_warm_thread.join(timeout=0.2)
        for cache in self._file_caches.values():
            cache.save()
        self._file_caches.clear()
        if self._parser_service is not None:
            self._parser_service.close()
        if self._conn is not None:
            self._conn.close()
        self._collections.clear()
        self._indexers.clear()
        for searcher in self._searchers.values():
            close = getattr(searcher, "close", None)
            if callable(close):
                close()
        self._searchers.clear()

    def __enter__(self) -> "QuickContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        qdrant_info = "disabled" if self._config.qdrant is None else self._config.qdrant.url
        return f"QuickContext(qdrant={qdrant_info!r})"
