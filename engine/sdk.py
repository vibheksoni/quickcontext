from dataclasses import replace
from pathlib import Path
import re
import time
from typing import Optional, TYPE_CHECKING

from engine.src.chunk_filter import ChunkFilterConfig, filter_chunks
from engine.src.chunker import ChunkBuilder, CodeChunk
from engine.src.compressor import CompressionStats, compress_grep_line, compress_source
from engine.src.config import EngineConfig
from engine.src.dedup import (
    DeduplicationResult,
    content_hash,
    deduplicate_chunks,
    expand_descriptions,
    expand_embeddings,
)
from engine.src.filecache import FileSignatureCache

if TYPE_CHECKING:
    from engine.src.collection import CollectionManager
    from engine.src.describer import ChunkDescription, DescriptionGenerator
    from engine.src.embedder import DualEmbedder, EmbeddedChunk
    from engine.src.indexer import IndexStats, QdrantIndexer
    from engine.src.providers import EmbeddingProvider
    from engine.src.qdrant import QdrantConnection
    from engine.src.searcher import CodeSearcher, SearchResult
    from engine.src.watcher import FileWatcher
    from engine.src.reranker import ColBERTReranker

from engine.src.parsing import (
    CallGraphTraceResult,
    CallerLookupResult,
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
from engine.src.search_modes import BIAS_NAMES, SearchBias, apply_bias, get_bias


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
        self._reranker: "ColBERTReranker | None" = None

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
        project = project_name if project_name else detect_project_name(Path.cwd(), manual_override=None)
        searcher = self._get_searcher(project, rerank=rerank)

        if mode == "code":
            return searcher.search_code(
                query=query,
                limit=limit,
                language=language,
                path_prefix=path_prefix,
                use_keywords=use_keywords,
                keyword_weight=keyword_weight,
                rerank=rerank,
            )

        if mode == "desc":
            return searcher.search_description(
                query=query,
                limit=limit,
                language=language,
                path_prefix=path_prefix,
                use_keywords=use_keywords,
                keyword_weight=keyword_weight,
                rerank=rerank,
            )

        return searcher.search_hybrid(
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
        )

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
    ) -> dict:
        """
        Run semantic retrieval and expand related files from the import graph around top hits.

        This is intended for deeper AI workflows that need both the best semantic anchors
        and a small set of adjacent files to inspect next.
        """
        project = project_name if project_name else detect_project_name(Path.cwd(), manual_override=None)
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
            )
        related_callers = self._related_callers_for_results(results)

        return {
            "query": query,
            "project_name": project,
            "results": anchors,
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
    ) -> dict:
        """
        Automatically choose between plain semantic search and the graph-aware bundle primitive.
        """
        project = project_name if project_name else detect_project_name(Path.cwd(), manual_override=None)
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
    ) -> dict:
        """
        Automatically choose the best retrieval primitive for AI workflows.

        Exact symbol-oriented questions use the Rust symbol index first so the SDK can
        return the real definition source without paying an embedding round trip. Broader
        natural-language questions fall back to semantic_search_auto(...).
        """
        project = project_name if project_name else detect_project_name(Path.cwd(), manual_override=None)
        symbol_query = self._extract_symbol_query_candidate(query)
        if symbol_query:
            expand_symbol_context = self._should_expand_symbol_context_results(query)
            results = self._symbol_lookup_search_results(
                query=symbol_query,
                limit=1 if expand_symbol_context else limit,
                language=language,
                path_prefix=path_prefix,
            )
            if results:
                if expand_symbol_context:
                    results = self._expand_symbol_context_results(
                        query=query,
                        results=results,
                        limit=limit,
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
                        ) if self._should_use_graph_related_for_query(query) else [],
                        "related_callers": self._related_callers_for_results(results),
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
        )
        payload["symbol_query"] = None
        if payload.get("mode") == "search":
            payload["related_files"] = self._lexical_related_files_for_query(
                query=query,
                results=payload["results"],
                related_file_limit=related_file_limit,
            )
        return payload

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
                path=Path.cwd(),
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

    def _expand_symbol_context_results(
        self,
        query: str,
        results: list,
        limit: int,
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
    ) -> list:
        """
        Run symbol lookup and hydrate top symbol hits into SearchResult rows with source.
        """
        lookup = self.symbol_lookup(
            query=query,
            path=Path.cwd(),
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
            extracted_symbols = self._load_file_compact_symbols(anchor_file)
        except Exception:
            return []

        anchor_name = str(getattr(anchor, "symbol_name", ""))
        anchor_parent = getattr(anchor, "parent", None)
        anchor_line_start = int(getattr(anchor, "line_start", 0))
        query_keywords = set(extract_keywords(query, max_keywords=20))
        query_keywords.update(self._split_symbol_text_tokens(query))
        source_lower = anchor_source.lower()

        candidates: list[tuple[tuple[int, int, int, int, str], object]] = []
        for symbol in extracted_symbols:
            if symbol.name == anchor_name and symbol.parent == anchor_parent and symbol.line_start == anchor_line_start:
                continue
            if str(symbol.kind).lower() not in {"function", "method", "class", "struct", "interface", "trait"}:
                continue

            mentioned = self._symbol_name_in_source(symbol.name, source_lower)
            same_parent = bool(anchor_parent and symbol.parent == anchor_parent)
            query_overlap = self._symbol_query_overlap(symbol, query_keywords)
            reference_overlap = self._symbol_reference_overlap(anchor_source, symbol.name, query_keywords)
            mismatch_penalty = self._symbol_helper_mismatch_penalty(symbol, query_keywords)
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
        return helper_results

    def _symbol_name_in_source(self, symbol_name: str, source_lower: str) -> bool:
        pattern = rf"(?<![A-Za-z0-9_]){re.escape(symbol_name.lower())}(?![A-Za-z0-9_])"
        return re.search(pattern, source_lower) is not None

    def _symbol_query_overlap(self, symbol: object, query_keywords: set[str]) -> int:
        if not query_keywords:
            return 0
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
        return len(query_keywords.intersection(candidate_keywords))

    def _symbol_reference_overlap(self, source: str, symbol_name: str, query_keywords: set[str]) -> int:
        if not source or not query_keywords:
            return 0
        lines = source.splitlines()
        matched_indexes = [
            idx
            for idx, line in enumerate(lines)
            if self._symbol_name_in_source(symbol_name, line.lower())
        ]
        if not matched_indexes:
            return 0
        window_indexes: set[int] = set()
        for idx in matched_indexes:
            for candidate_idx in range(max(0, idx - 1), min(len(lines), idx + 2)):
                window_indexes.add(candidate_idx)
        line_text = " ".join(lines[idx] for idx in sorted(window_indexes))
        line_keywords = set(extract_keywords(line_text, max_keywords=20))
        line_keywords.update(self._split_symbol_text_tokens(line_text))
        return len(query_keywords.intersection(line_keywords))

    def _symbol_helper_mismatch_penalty(self, symbol: object, query_keywords: set[str]) -> int:
        candidate_text = " ".join(
            part
            for part in (
                str(getattr(symbol, "name", "")),
                str(getattr(symbol, "signature", "") or ""),
            )
            if part
        )
        candidate_tokens = self._split_symbol_text_tokens(candidate_text)
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

        for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
            add_token(raw)

            underscore_parts = [part for part in re.split(r"_+", raw.strip("_")) if part]
            for part in underscore_parts:
                add_token(part)

            camel_text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw.replace("_", " "))
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
        project_root = Path.cwd().resolve()

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

    def _related_callers_for_results(self, results: list) -> list[dict]:
        """
        Collect caller rows around the top callable semantic anchor.
        """
        if not results:
            return []

        top = results[0]
        if str(getattr(top, "symbol_kind", "")).lower() not in {"function", "method"}:
            return []

        caller_rows = self.find_callers(symbol=top.symbol_name, path=Path.cwd(), limit=8)
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
                f"Cannot index: disabled subsystems: {', '.join(missing)}. "
                "Indexing requires qdrant, code_embedding, desc_embedding, and llm when descriptions are enabled."
            )

        directory = Path(directory).resolve()

        detected_project = detect_project_name(directory, manual_override=project_name)
        collection = self._get_collection(detected_project)
        file_cache = self._get_file_cache(directory)

        if show_progress:
            print(f"Indexing project: {detected_project}")
            print(f"Scanning supported files in {directory}...")

        scan_entries = self.parser_service.scan_files(directory)

        indexer = self._get_indexer(detected_project)
        candidate_entries = []
        unchanged_files = 0

        for entry in scan_entries:
            cached_hash = None
            if not force_refresh:
                cached_hash = file_cache.is_unchanged_from_metadata(
                    entry.file_path,
                    entry.file_size,
                    entry.file_mtime,
                )
            if cached_hash is not None:
                unchanged_files += 1
                continue
            candidate_entries.append(entry)

        indexed_hashes = indexer.get_file_hashes([entry.file_path for entry in candidate_entries])
        changed_results: list[ExtractionResult] = []

        if show_progress and candidate_entries:
            print(f"Extracting symbols from {len(candidate_entries)} candidate files...")

        candidate_paths = {entry.file_path for entry in candidate_entries}
        if candidate_entries and (
            len(candidate_entries) == len(scan_entries)
            or len(candidate_entries) > 100
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
            for entry in candidate_entries:
                extracted_candidates.extend(self.extract_symbols(entry.file_path))

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

        if show_progress and unchanged_files > 0:
            print(f"Skipping {unchanged_files} files (already indexed with same hash)")

        if not changed_results:
            if show_progress:
                print("No file changes detected; keeping active collection unchanged")
            return IndexStats(
                total_chunks=0,
                upserted_points=0,
                failed_points=0,
                total_tokens=0,
                duration_seconds=time.time() - started_at,
                llm_cost_usd=0.0,
                embedding_cost_usd=0.0,
                descriptions_enabled=generate_descriptions,
            )

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

        chunks = self.chunker.build_chunks(changed_results)

        if fast:
            generate_descriptions = False
            min_chunk_bytes = max(min_chunk_bytes, 200)
            max_chunks_per_file = min(max_chunks_per_file, 180)
            skip_minified = True
            if max_total_chunks is None:
                max_total_chunks = 50000

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

        if show_progress:
            print(
                "Chunk filtering: "
                f"kept {filter_stats.kept_chunks}/{filter_stats.input_chunks}, "
                f"small={filter_stats.skipped_small}, "
                f"minified={filter_stats.skipped_minified}, "
                f"capped={filter_stats.skipped_per_file_cap}"
            )

        if not chunks:
            file_cache.save()
            if incremental_resume:
                if show_progress:
                    print("No chunks to index after filtering")
                return IndexStats(
                    total_chunks=0,
                    upserted_points=0,
                    failed_points=0,
                    total_tokens=0,
                    duration_seconds=time.time() - started_at,
                    llm_cost_usd=0.0,
                    embedding_cost_usd=0.0,
                    skipped_small_chunks=filter_stats.skipped_small,
                    skipped_minified_chunks=filter_stats.skipped_minified,
                    skipped_capped_chunks=filter_stats.skipped_per_file_cap,
                    files_capped=filter_stats.files_capped,
                    descriptions_enabled=generate_descriptions,
                )

            shadow_name = collection.create_shadow()
            from engine.src.indexer import QdrantIndexer
            shadow_indexer = QdrantIndexer(
                client=collection.client,
                collection_name=shadow_name,
                batch_size=self._config.qdrant.upsert_batch_size,
                upsert_concurrency=self._config.qdrant.upsert_concurrency,
            )
            old_collection = collection.resolve_alias()
            if old_collection is not None and not force_refresh:
                copied_points = collection.copy_points(
                    source_collection=old_collection,
                    target_collection=shadow_name,
                    batch_size=self._config.qdrant.upsert_batch_size,
                )
                if show_progress:
                    print(f"Copied {copied_points} existing points into {shadow_name}")

                changed_paths = sorted({result.file_path for result in changed_results})
                if changed_paths:
                    shadow_indexer.delete_by_files(changed_paths)
                    if show_progress:
                        print(f"Removed stale points for {len(changed_paths)} changed files from {shadow_name}")

            old_collection = collection.swap_alias(shadow_name)
            if old_collection is not None:
                try:
                    collection.client.delete_collection(old_collection)
                except Exception:
                    pass
            collection.cleanup_old_versions()

            if show_progress:
                print("Applied filtered deletions with no replacement chunks")

            return IndexStats(
                total_chunks=0,
                upserted_points=0,
                failed_points=0,
                total_tokens=0,
                duration_seconds=time.time() - started_at,
                llm_cost_usd=0.0,
                embedding_cost_usd=0.0,
                skipped_small_chunks=filter_stats.skipped_small,
                skipped_minified_chunks=filter_stats.skipped_minified,
                skipped_capped_chunks=filter_stats.skipped_per_file_cap,
                files_capped=filter_stats.files_capped,
                descriptions_enabled=generate_descriptions,
            )

        from engine.src.dedup import deduplicate_chunks, expand_descriptions, expand_embeddings
        dedup_result = deduplicate_chunks(chunks)

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

        shadow_name = collection.create_shadow()
        if show_progress:
            print(f"Created shadow collection: {shadow_name}")

        from engine.src.indexer import QdrantIndexer
        shadow_indexer = QdrantIndexer(
            client=collection.client,
            collection_name=shadow_name,
            batch_size=self._config.qdrant.upsert_batch_size,
            upsert_concurrency=self._config.qdrant.upsert_concurrency,
        )

        old_collection = collection.resolve_alias()
        if old_collection is not None and not force_refresh:
            copied_points = collection.copy_points(
                source_collection=old_collection,
                target_collection=shadow_name,
                batch_size=self._config.qdrant.upsert_batch_size,
            )
            if show_progress:
                print(f"Copied {copied_points} existing points into {shadow_name}")

            changed_paths = sorted({result.file_path for result in changed_results})
            if changed_paths:
                shadow_indexer.delete_by_files(changed_paths)
                if show_progress:
                    print(f"Removed stale points for {len(changed_paths)} changed files from {shadow_name}")

        if generate_descriptions:
            if show_progress:
                print(f"Generating descriptions for {dedup_result.unique_count} unique chunks...")

            unique_descriptions = self.describer.generate_batch(dedup_result.unique_chunks)
            descriptions = expand_descriptions(unique_descriptions, dedup_result)
            llm_cost = sum(d.cost_usd for d in unique_descriptions)

            if show_progress:
                total_tokens = sum(d.token_count for d in unique_descriptions)
                print(f"Generated descriptions ({total_tokens} tokens, ${llm_cost:.4f})")
        else:
            from engine.src.describer import build_fallback_descriptions

            unique_descriptions = build_fallback_descriptions(dedup_result.unique_chunks)
            descriptions = expand_descriptions(unique_descriptions, dedup_result)
            llm_cost = 0.0
            if show_progress:
                print("Skipping description generation (fast/no-descriptions mode)")

        if show_progress:
            print("Generating embeddings...")

        unique_embedded, embedding_cost = self.embedder.embed_batch(
            dedup_result.unique_chunks,
            unique_descriptions,
            concurrency_override=embedding_concurrency,
            max_retries_override=embedding_max_retries,
            batch_size_override=embedding_batch_size,
            adaptive_batching_override=embedding_adaptive_batching,
        )
        embed_stats = self.embedder.last_run_stats
        embedded_chunks = expand_embeddings(unique_embedded, chunks, descriptions, dedup_result)

        if show_progress:
            print(f"Upserting {len(embedded_chunks)} chunks to {shadow_name}...")

        upsert_stats = shadow_indexer.upsert_chunks(embedded_chunks)

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

        stats = IndexStats(
            total_chunks=upsert_stats.total_chunks,
            upserted_points=upsert_stats.upserted_points,
            failed_points=upsert_stats.failed_points,
            total_tokens=upsert_stats.total_tokens,
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

        missing = []
        if self._config.qdrant is None:
            missing.append("qdrant")
        if self._config.code_embedding is None:
            missing.append("code_embedding")
        if self._config.desc_embedding is None:
            missing.append("desc_embedding")
        if self._config.llm is None and generate_descriptions:
            missing.append("llm")
        if missing:
            raise RuntimeError(
                f"Cannot refresh: disabled subsystems: {', '.join(missing)}. "
                "Refreshing requires qdrant, code_embedding, desc_embedding, and llm."
            )

        resolved_paths = [str(Path(p).resolve()) for p in file_paths]

        if not resolved_paths:
            raise ValueError("No file paths provided")

        detected_project = detect_project_name(Path(resolved_paths[0]).parent, manual_override=project_name)
        indexer = self._get_indexer(detected_project)
        file_cache = self._get_file_cache(Path(resolved_paths[0]).parent)

        if show_progress:
            print(f"Refreshing files in project: {detected_project}")
            print(f"Checking mtime cache for {len(resolved_paths)} files...")

        paths_to_extract: list[str] = []
        mtime_skipped = 0
        for fp in resolved_paths:
            cached_hash = file_cache.is_unchanged(fp)
            if cached_hash is not None:
                mtime_skipped += 1
            else:
                paths_to_extract.append(fp)

        if show_progress and mtime_skipped > 0:
            print(f"Skipping {mtime_skipped} files (mtime+size unchanged)")

        if not paths_to_extract:
            if show_progress:
                print("All files unchanged (mtime cache hit)")
            return IndexStats(
                total_chunks=0,
                upserted_points=0,
                failed_points=0,
                total_tokens=0,
                duration_seconds=time.time() - started_at,
                llm_cost_usd=0.0,
                embedding_cost_usd=0.0,
                descriptions_enabled=generate_descriptions,
            )

        if not paths_to_extract:
            file_cache.save()
            if show_progress:
                print("No files to refresh after filtering")
            return IndexStats(
                total_chunks=0,
                upserted_points=0,
                failed_points=0,
                total_tokens=0,
                duration_seconds=time.time() - started_at,
                llm_cost_usd=0.0,
                embedding_cost_usd=0.0,
                descriptions_enabled=generate_descriptions,
            )

        if show_progress:
            print(f"Analyzing changes in {len(paths_to_extract)} files...")

        if fast:
            generate_descriptions = False
            min_chunk_bytes = max(min_chunk_bytes, 200)
            max_chunks_per_file = min(max_chunks_per_file, 180)
            skip_minified = True
            if max_total_chunks is None:
                max_total_chunks = 50000

        all_chunks_to_reindex: list[CodeChunk] = []
        all_identities_to_delete: list = []
        total_unchanged = 0
        total_skipped_small = 0
        total_skipped_minified = 0
        total_skipped_capped = 0
        total_files_capped = 0
        old_chunks_by_file = indexer.get_chunks_by_files(paths_to_extract)

        for file_path in paths_to_extract:
            old_chunks = old_chunks_by_file.get(file_path, [])

            results = self.extract_symbols(file_path)
            new_chunks = self.chunker.build_chunks(results)

            for result in results:
                file_cache.update_from_extraction(
                    result.file_path, result.file_hash, result.file_size, result.file_mtime,
                )

            filtered_chunks, filter_stats = filter_chunks(
                new_chunks,
                ChunkFilterConfig(
                    min_chunk_bytes=max(1, int(min_chunk_bytes)),
                    max_chunks_per_file=max(1, int(max_chunks_per_file)),
                    skip_minified=bool(skip_minified),
                ),
            )
            diff = ChunkDiffer.diff(old_chunks, filtered_chunks)

            all_chunks_to_reindex.extend(diff.needs_reindex)
            all_identities_to_delete.extend(diff.needs_deletion)
            total_unchanged += len(diff.unchanged)
            total_skipped_small += filter_stats.skipped_small
            total_skipped_minified += filter_stats.skipped_minified
            total_skipped_capped += filter_stats.skipped_per_file_cap
            total_files_capped += filter_stats.files_capped

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
            return IndexStats(
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

        from engine.src.dedup import deduplicate_chunks, expand_descriptions, expand_embeddings
        dedup_result = deduplicate_chunks(all_chunks_to_reindex)

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

            unique_descriptions = self.describer.generate_batch(dedup_result.unique_chunks)
            descriptions = expand_descriptions(unique_descriptions, dedup_result)
            llm_cost = sum(d.cost_usd for d in unique_descriptions)

            if show_progress:
                total_tokens = sum(d.token_count for d in unique_descriptions)
                print(f"Generated descriptions ({total_tokens} tokens, ${llm_cost:.4f})")
        else:
            from engine.src.describer import build_fallback_descriptions

            unique_descriptions = build_fallback_descriptions(dedup_result.unique_chunks)
            descriptions = expand_descriptions(unique_descriptions, dedup_result)
            llm_cost = 0.0
            if show_progress:
                print("Skipping description generation (fast/no-descriptions mode)")

        if show_progress:
            print("Generating embeddings...")

        unique_embedded, embedding_cost = self.embedder.embed_batch(
            dedup_result.unique_chunks,
            unique_descriptions,
            concurrency_override=embedding_concurrency,
            max_retries_override=embedding_max_retries,
            batch_size_override=embedding_batch_size,
            adaptive_batching_override=embedding_adaptive_batching,
        )
        embed_stats = self.embedder.last_run_stats
        embedded_chunks = expand_embeddings(
            unique_embedded, all_chunks_to_reindex, descriptions, dedup_result,
        )

        if show_progress:
            print(f"Upserting {len(embedded_chunks)} chunks to Qdrant...")

        stats = indexer.upsert_chunks(embedded_chunks)

        stats = IndexStats(
            total_chunks=stats.total_chunks,
            upserted_points=stats.upserted_points,
            failed_points=stats.failed_points,
            total_tokens=stats.total_tokens,
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
            if stats.failed_points > 0:
                print(f"Failed: {stats.failed_points} chunks")

        return stats

    def status(self) -> dict:
        """
        Get engine status: connection health, collection info, provider details.

        Returns: dict — Status report.
        """
        result = {}

        if self._config.qdrant is not None:
            alive = self._conn.is_alive() if self._conn else False
            result["qdrant"] = {
                "url": self._config.qdrant.url,
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
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        qdrant_info = "disabled" if self._config.qdrant is None else self._config.qdrant.url
        return f"QuickContext(qdrant={qdrant_info!r})"
