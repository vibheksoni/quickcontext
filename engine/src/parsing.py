from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import os

from engine.src.pipe import PIPE_NAME, PipeClient, PipeProtocolError


@dataclass(frozen=True, slots=True)
class ExtractedSymbol:
    """
    Parsed symbol returned by the Rust extraction service.

    name: str — Symbol name.
    kind: str — Symbol kind in snake_case.
    language: str — Language identifier.
    file_path: str — Source file path.
    line_start: int — Start line (0-based from Rust output).
    line_end: int — End line (0-based from Rust output).
    byte_start: int — Start byte offset.
    byte_end: int — End byte offset.
    source: str — Raw source text for the symbol.
    signature: Optional[str] — Signature text if available.
    docstring: Optional[str] — Docstring/comment text if available.
    params: Optional[str] — Parameter text capture if available.
    return_type: Optional[str] — Return type text capture if available.
    parent: Optional[str] — Parent container name if available.
    visibility: Optional[str] — Visibility modifier if available.
    role: Optional[str] — Architectural role classification (definition/entrypoint/orchestration/logic/utility/test/configuration).
    """

    name: str
    kind: str
    language: str
    file_path: str
    line_start: int
    line_end: int
    byte_start: int
    byte_end: int
    source: str
    signature: Optional[str]
    docstring: Optional[str]
    params: Optional[str]
    return_type: Optional[str]
    parent: Optional[str]
    visibility: Optional[str]
    role: Optional[str]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ExtractedSymbol":
        """
        Build a typed symbol object from Rust JSON payload.

        data: dict[str, Any] — Symbol dictionary from pipe response.
        Returns: ExtractedSymbol — Parsed symbol instance.
        """
        return ExtractedSymbol(
            name=str(data.get("name", "")),
            kind=str(data.get("kind", "")),
            language=str(data.get("language", "")),
            file_path=str(data.get("file_path", "")),
            line_start=int(data.get("line_start", 0)),
            line_end=int(data.get("line_end", 0)),
            byte_start=int(data.get("byte_start", 0)),
            byte_end=int(data.get("byte_end", 0)),
            source=str(data.get("source", "")),
            signature=data.get("signature"),
            docstring=data.get("docstring"),
            params=data.get("params"),
            return_type=data.get("return_type"),
            parent=data.get("parent"),
            visibility=data.get("visibility"),
            role=data.get("role"),
        )


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """
    Per-file extraction result returned by the Rust extraction service.

    file_path: str — Parsed file path.
    language: str — Language identifier.
    symbols: list[ExtractedSymbol] — Symbols extracted from the file.
    errors: list[str] — Non-fatal extraction errors.
    file_hash: Optional[str] — SHA256 hex digest of file content (computed by Rust during parse).
    file_size: Optional[int] — File size in bytes at extraction time.
    file_mtime: Optional[int] — File mtime as Unix epoch seconds at extraction time.
    """

    file_path: str
    language: str
    symbols: list[ExtractedSymbol]
    errors: list[str]
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    file_mtime: Optional[int] = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ExtractionResult":
        """
        Build a typed extraction result from Rust JSON payload.

        data: dict[str, Any] — Result dictionary from pipe response.
        Returns: ExtractionResult — Parsed extraction result instance.
        """
        symbols_raw = data.get("symbols", [])
        symbols = [
            ExtractedSymbol.from_dict(item)
            for item in symbols_raw
            if isinstance(item, dict)
        ]
        errors_raw = data.get("errors", [])
        errors = [str(item) for item in errors_raw] if isinstance(errors_raw, list) else []

        file_size_raw = data.get("file_size")
        file_mtime_raw = data.get("file_mtime")

        return ExtractionResult(
            file_path=str(data.get("file_path", "")),
            language=str(data.get("language", "")),
            symbols=symbols,
            errors=errors,
            file_hash=data.get("file_hash"),
            file_size=int(file_size_raw) if file_size_raw is not None else None,
            file_mtime=int(file_mtime_raw) if file_mtime_raw is not None else None,
        )


@dataclass(frozen=True, slots=True)
class CompactSymbol:
    """
    Lightweight symbol without source/docstring/params/return_type.

    name: str — Symbol name.
    kind: str — Symbol kind in snake_case.
    language: str — Language identifier.
    file_path: str — Source file path.
    line_start: int — Start line (0-based from Rust output).
    line_end: int — End line (0-based from Rust output).
    signature: Optional[str] — Signature text if available.
    parent: Optional[str] — Parent container name if available.
    visibility: Optional[str] — Visibility modifier if available.
    role: Optional[str] — Architectural role classification.
    """

    name: str
    kind: str
    language: str
    file_path: str
    line_start: int
    line_end: int
    signature: Optional[str]
    parent: Optional[str]
    visibility: Optional[str]
    role: Optional[str]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CompactSymbol":
        """
        Build a compact symbol from Rust JSON payload.

        data: dict[str, Any] — Symbol dictionary from pipe response.
        Returns: CompactSymbol — Parsed compact symbol instance.
        """
        return CompactSymbol(
            name=str(data.get("name", "")),
            kind=str(data.get("kind", "")),
            language=str(data.get("language", "")),
            file_path=str(data.get("file_path", "")),
            line_start=int(data.get("line_start", 0)),
            line_end=int(data.get("line_end", 0)),
            signature=data.get("signature"),
            parent=data.get("parent"),
            visibility=data.get("visibility"),
            role=data.get("role"),
        )


@dataclass(frozen=True, slots=True)
class CompactExtractionResult:
    """
    Per-file compact extraction result (no source/docstring/params).

    file_path: str — Parsed file path.
    language: str — Language identifier.
    symbols: list[CompactSymbol] — Compact symbols extracted from the file.
    symbol_count: int — Number of symbols in this file.
    file_hash: Optional[str] — SHA256 hex digest of file content.
    file_size: Optional[int] — File size in bytes.
    file_mtime: Optional[int] — File mtime as Unix epoch seconds.
    """

    file_path: str
    language: str
    symbols: list[CompactSymbol]
    symbol_count: int
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    file_mtime: Optional[int] = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CompactExtractionResult":
        """
        Build a compact extraction result from Rust JSON payload.

        data: dict[str, Any] — Result dictionary from pipe response.
        Returns: CompactExtractionResult — Parsed compact result instance.
        """
        symbols_raw = data.get("symbols", [])
        symbols = [
            CompactSymbol.from_dict(item)
            for item in symbols_raw
            if isinstance(item, dict)
        ]
        file_size_raw = data.get("file_size")
        file_mtime_raw = data.get("file_mtime")
        return CompactExtractionResult(
            file_path=str(data.get("file_path", "")),
            language=str(data.get("language", "")),
            symbols=symbols,
            symbol_count=int(data.get("symbol_count", len(symbols))),
            file_hash=data.get("file_hash"),
            file_size=int(file_size_raw) if file_size_raw is not None else None,
            file_mtime=int(file_mtime_raw) if file_mtime_raw is not None else None,
        )


@dataclass(frozen=True, slots=True)
class FileScanEntry:
    """
    Lightweight file metadata for supported source files.

    file_path: str — Absolute file path.
    language: str — Detected language name.
    file_size: int — File size in bytes.
    file_mtime: int — File mtime as Unix epoch seconds.
    """

    file_path: str
    language: str
    file_size: int
    file_mtime: int

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "FileScanEntry":
        return FileScanEntry(
            file_path=str(data.get("file_path", "")),
            language=str(data.get("language", "")),
            file_size=int(data.get("file_size", 0)),
            file_mtime=int(data.get("file_mtime", 0)),
        )


@dataclass(frozen=True, slots=True)
class ExtractStats:
    """
    Aggregate extraction statistics returned by compact/stats_only mode.

    total_files: int — Number of files processed.
    total_symbols: int — Total symbols extracted across all files.
    languages: dict[str, int] — Per-language symbol counts.
    duration_ms: int — Extraction duration in milliseconds.
    """

    total_files: int
    total_symbols: int
    languages: dict[str, int]
    duration_ms: int

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ExtractStats":
        """
        Build extraction stats from Rust JSON payload.

        data: dict[str, Any] — Stats dictionary from pipe response.
        Returns: ExtractStats — Parsed stats instance.
        """
        return ExtractStats(
            total_files=int(data.get("total_files", 0)),
            total_symbols=int(data.get("total_symbols", 0)),
            languages=dict(data.get("languages", {})),
            duration_ms=int(data.get("duration_ms", 0)),
        )


@dataclass(frozen=True, slots=True)
class ExtractSymbolResult:
    """
    Result of extracting a specific symbol by name from a single file.

    file_path: str — Source file path.
    language: str — Language identifier.
    query: str — Original symbol query string.
    symbols: list[ExtractedSymbol] — Matched symbols.
    total_matches: int — Number of matched symbols.
    """

    file_path: str
    language: str
    query: str
    symbols: list[ExtractedSymbol]
    total_matches: int

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ExtractSymbolResult":
        """
        Build a typed result from Rust JSON payload.

        data: dict[str, Any] — Result dictionary from pipe response.
        Returns: ExtractSymbolResult — Parsed result instance.
        """
        symbols_raw = data.get("symbols", [])
        symbols = [
            ExtractedSymbol.from_dict(item)
            for item in symbols_raw
            if isinstance(item, dict)
        ]
        return ExtractSymbolResult(
            file_path=str(data.get("file_path", "")),
            language=str(data.get("language", "")),
            query=str(data.get("query", "")),
            symbols=symbols,
            total_matches=int(data.get("total_matches", 0)),
        )


@dataclass(frozen=True, slots=True)
class GrepMatch:
    """
    Single literal grep match returned by the Rust grep service.

    file_path: str — Matched file path.
    line_number: int — 1-based line number of the match.
    column_start: int — 1-based start column of the match.
    column_end: int — 1-based end column of the match.
    line: str — Full matched line text.
    context_before: tuple[str, ...] — Lines before the match (empty if no context).
    context_after: tuple[str, ...] — Lines after the match (empty if no context).
    """

    file_path: str
    line_number: int
    column_start: int
    column_end: int
    line: str
    context_before: tuple[str, ...] = ()
    context_after: tuple[str, ...] = ()

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "GrepMatch":
        """
        Build a typed grep match from Rust JSON payload.

        data: dict[str, Any] — Match dictionary from pipe response.
        Returns: GrepMatch — Parsed grep match instance.
        """
        return GrepMatch(
            file_path=str(data.get("file_path", "")),
            line_number=int(data.get("line_number", 0)),
            column_start=int(data.get("column_start", 0)),
            column_end=int(data.get("column_end", 0)),
            line=str(data.get("line", "")),
            context_before=tuple(data.get("context_before", [])),
            context_after=tuple(data.get("context_after", [])),
        )


@dataclass(frozen=True, slots=True)
class GrepResult:
    """
    Result payload for literal grep search.

    matches: list[GrepMatch] — Matched lines.
    searched_files: int — Number of files scanned.
    duration_ms: int — End-to-end search duration in milliseconds.
    truncated: bool — True when result limit was reached.
    """

    matches: list[GrepMatch]
    searched_files: int
    duration_ms: int
    truncated: bool

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "GrepResult":
        """
        Build a typed grep result from Rust JSON payload.

        data: dict[str, Any] — Grep result dictionary from pipe response.
        Returns: GrepResult — Parsed grep result instance.
        """
        matches_raw = data.get("matches", [])
        matches = [
            GrepMatch.from_dict(item)
            for item in matches_raw
            if isinstance(item, dict)
        ]
        return GrepResult(
            matches=matches,
            searched_files=int(data.get("searched_files", 0)),
            duration_ms=int(data.get("duration_ms", 0)),
            truncated=bool(data.get("truncated", False)),
        )


@dataclass(frozen=True, slots=True)
class SymbolLookupItem:
    """
    Symbol lookup row returned by Rust symbol index.

    name: str — Symbol name.
    kind: str — Symbol kind label.
    language: str — Language identifier.
    file_path: str — File containing the symbol.
    line_start: int — Start line (0-based from Rust).
    line_end: int — End line (0-based from Rust).
    parent: Optional[str] — Parent symbol/container.
    signature: Optional[str] — Symbol signature text.
    """

    name: str
    kind: str
    language: str
    file_path: str
    line_start: int
    line_end: int
    parent: Optional[str]
    signature: Optional[str]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SymbolLookupItem":
        """
        Build typed symbol lookup item from Rust JSON payload.

        data: dict[str, Any] — Symbol lookup row from pipe response.
        Returns: SymbolLookupItem — Parsed symbol lookup item.
        """
        return SymbolLookupItem(
            name=str(data.get("name", "")),
            kind=str(data.get("kind", "")),
            language=str(data.get("language", "")),
            file_path=str(data.get("file_path", "")),
            line_start=int(data.get("line_start", 0)),
            line_end=int(data.get("line_end", 0)),
            parent=data.get("parent"),
            signature=data.get("signature"),
        )


@dataclass(frozen=True, slots=True)
class SymbolLookupResult:
    """
    Symbol lookup result payload.

    project_root: str — Resolved project root path.
    results: list[SymbolLookupItem] — Matched symbols.
    indexed_files: int — File count used in index.
    indexed_symbols: int — Symbol count used in index.
    from_cache: bool — True when served from in-memory cache.
    """

    project_root: str
    results: list[SymbolLookupItem]
    indexed_files: int
    indexed_symbols: int
    from_cache: bool

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SymbolLookupResult":
        """
        Build typed symbol lookup result from Rust JSON payload.

        data: dict[str, Any] — Symbol lookup result dictionary.
        Returns: SymbolLookupResult — Parsed symbol lookup result.
        """
        results_raw = data.get("results", [])
        results = [
            SymbolLookupItem.from_dict(item)
            for item in results_raw
            if isinstance(item, dict)
        ]
        return SymbolLookupResult(
            project_root=str(data.get("project_root", "")),
            results=results,
            indexed_files=int(data.get("indexed_files", 0)),
            indexed_symbols=int(data.get("indexed_symbols", 0)),
            from_cache=bool(data.get("from_cache", False)),
        )


@dataclass(frozen=True, slots=True)
class CallerItem:
    """
    Caller relationship row returned by Rust call index.

    callee_name: str — Target callee symbol.
    caller_name: str — Caller symbol name.
    caller_kind: str — Caller symbol kind label.
    caller_language: str — Caller language identifier.
    caller_file_path: str — File path of caller.
    caller_line: int — 1-based call site line.
    """

    callee_name: str
    caller_name: str
    caller_kind: str
    caller_language: str
    caller_file_path: str
    caller_line: int

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CallerItem":
        """
        Build typed caller row from Rust JSON payload.

        data: dict[str, Any] — Caller row dictionary from pipe response.
        Returns: CallerItem — Parsed caller row.
        """
        return CallerItem(
            callee_name=str(data.get("callee_name", "")),
            caller_name=str(data.get("caller_name", "")),
            caller_kind=str(data.get("caller_kind", "")),
            caller_language=str(data.get("caller_language", "")),
            caller_file_path=str(data.get("caller_file_path", "")),
            caller_line=int(data.get("caller_line", 0)),
        )


@dataclass(frozen=True, slots=True)
class CallerLookupResult:
    """
    Caller lookup result payload.

    project_root: str — Resolved project root path.
    symbol: str — Requested symbol.
    callers: list[CallerItem] — Caller rows.
    indexed_files: int — File count used in index.
    indexed_symbols: int — Symbol count used in index.
    from_cache: bool — True when served from in-memory cache.
    """

    project_root: str
    symbol: str
    callers: list[CallerItem]
    indexed_files: int
    indexed_symbols: int
    from_cache: bool

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CallerLookupResult":
        """
        Build typed caller lookup result from Rust JSON payload.

        data: dict[str, Any] — Caller lookup result dictionary.
        Returns: CallerLookupResult — Parsed caller lookup result.
        """
        callers_raw = data.get("callers", [])
        callers = [
            CallerItem.from_dict(item)
            for item in callers_raw
            if isinstance(item, dict)
        ]
        return CallerLookupResult(
            project_root=str(data.get("project_root", "")),
            symbol=str(data.get("symbol", "")),
            callers=callers,
            indexed_files=int(data.get("indexed_files", 0)),
            indexed_symbols=int(data.get("indexed_symbols", 0)),
            from_cache=bool(data.get("from_cache", False)),
        )


@dataclass(frozen=True, slots=True)
class ImportEdge:
    """
    Single import dependency edge in the file-level graph.

    source_file: str — File that contains the import statement.
    target_file: str — File being imported (resolved to disk path).
    import_statement: str — Raw import statement text.
    module_path: str — Parsed module/package path from the statement.
    language: str — Language identifier.
    line: int — 0-based line number of the import statement.
    """

    source_file: str
    target_file: str
    import_statement: str
    module_path: str
    language: str
    line: int

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ImportEdge":
        """
        Build typed import edge from Rust JSON payload.

        data: dict[str, Any] — Edge dictionary from pipe response.
        Returns: ImportEdge — Parsed import edge.
        """
        return ImportEdge(
            source_file=str(data.get("source_file", "")),
            target_file=str(data.get("target_file", "")),
            import_statement=str(data.get("import_statement", "")),
            module_path=str(data.get("module_path", "")),
            language=str(data.get("language", "")),
            line=int(data.get("line", 0)),
        )


@dataclass(frozen=True, slots=True)
class ImportGraphResult:
    """
    Import graph query result payload.

    file_path: str — Queried file path.
    project_root: str — Resolved project root.
    edges: list[ImportEdge] — Import dependency edges.
    total_files: int — Total files scanned in project.
    total_imports: int — Total resolved import edges in project.
    duration_ms: int — Graph build time in milliseconds.
    """

    file_path: str
    project_root: str
    edges: list[ImportEdge]
    total_files: int
    total_imports: int
    duration_ms: int

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ImportGraphResult":
        """
        Build typed import graph result from Rust JSON payload.

        data: dict[str, Any] — Import graph result dictionary.
        Returns: ImportGraphResult — Parsed import graph result.
        """
        edges_raw = data.get("edges", [])
        edges = [
            ImportEdge.from_dict(item)
            for item in edges_raw
            if isinstance(item, dict)
        ]
        return ImportGraphResult(
            file_path=str(data.get("file_path", "")),
            project_root=str(data.get("project_root", "")),
            edges=edges,
            total_files=int(data.get("total_files", 0)),
            total_imports=int(data.get("total_imports", 0)),
            duration_ms=int(data.get("duration_ms", 0)),
        )


@dataclass(frozen=True, slots=True)
class SkeletonSymbol:
    """
    Symbol entry in a skeleton file node.

    name: str — Symbol name.
    kind: str — Symbol kind label.
    signature: Optional[str] — Cleaned signature text.
    parent: Optional[str] — Parent container name.
    visibility: Optional[str] — Visibility modifier.
    line_start: int — Start line (0-based).
    line_end: int — End line (0-based).
    """

    name: str
    kind: str
    signature: Optional[str]
    parent: Optional[str]
    visibility: Optional[str]
    line_start: int
    line_end: int

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SkeletonSymbol":
        """
        Build typed skeleton symbol from Rust JSON payload.

        data: dict[str, Any] — Symbol dictionary from skeleton response.
        Returns: SkeletonSymbol — Parsed skeleton symbol.
        """
        return SkeletonSymbol(
            name=str(data.get("name", "")),
            kind=str(data.get("kind", "")),
            signature=data.get("signature"),
            parent=data.get("parent"),
            visibility=data.get("visibility"),
            line_start=int(data.get("line_start", 0)),
            line_end=int(data.get("line_end", 0)),
        )


@dataclass(frozen=True, slots=True)
class SkeletonResult:
    """
    Skeleton generation result payload.

    root: dict — Hierarchical skeleton tree (directory/file/collapsed nodes).
    total_files: int — Total files in skeleton.
    total_symbols: int — Total symbols after filtering.
    total_directories: int — Total directory nodes.
    duration_ms: int — Generation time in milliseconds.
    markdown: Optional[str] — Rendered markdown when format="markdown".
    """

    root: Optional[dict]
    total_files: int
    total_symbols: int
    total_directories: int
    duration_ms: int
    markdown: Optional[str]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SkeletonResult":
        """
        Build typed skeleton result from Rust JSON payload.

        data: dict[str, Any] — Skeleton result dictionary from pipe response.
        Returns: SkeletonResult — Parsed skeleton result.
        """
        return SkeletonResult(
            root=data.get("root"),
            total_files=int(data.get("total_files", 0)),
            total_symbols=int(data.get("total_symbols", 0)),
            total_directories=int(data.get("total_directories", 0)),
            duration_ms=int(data.get("duration_ms", 0)),
            markdown=data.get("markdown"),
        )


@dataclass(frozen=True, slots=True)
class TextSearchMatch:
    """
    Single BM25-ranked text search match.

    file_path: str — Matched file path.
    score: float — BM25 relevance score.
    matched_terms: list[str] — Query terms found in the file.
    snippet: str — Context snippet around first match.
    snippet_line_start: int — 1-based start line of snippet.
    snippet_line_end: int — End line of snippet.
    language: str — Detected language identifier.
    """

    file_path: str
    score: float
    matched_terms: list[str]
    snippet: str
    snippet_line_start: int
    snippet_line_end: int
    language: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TextSearchMatch":
        """
        Build typed text search match from Rust JSON payload.

        data: dict[str, Any] — Match dictionary from pipe response.
        Returns: TextSearchMatch — Parsed text search match.
        """
        return TextSearchMatch(
            file_path=str(data.get("file_path", "")),
            score=float(data.get("score", 0.0)),
            matched_terms=list(data.get("matched_terms", [])),
            snippet=str(data.get("snippet", "")),
            snippet_line_start=int(data.get("snippet_line_start", 0)),
            snippet_line_end=int(data.get("snippet_line_end", 0)),
            language=str(data.get("language", "")),
        )


@dataclass(frozen=True, slots=True)
class TextSearchResult:
    """
    BM25 full-text search result payload.

    matches: list[TextSearchMatch] — Ranked matches.
    searched_files: int — Number of files scanned.
    query_expr: str — Debug representation of parsed query expression.
    duration_ms: int — Search duration in milliseconds.
    truncated: bool — True when result limit was reached.
    """

    matches: list[TextSearchMatch]
    searched_files: int
    query_expr: str
    duration_ms: int
    truncated: bool

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TextSearchResult":
        """
        Build typed text search result from Rust JSON payload.

        data: dict[str, Any] — Text search result dictionary.
        Returns: TextSearchResult — Parsed text search result.
        """
        matches_raw = data.get("matches", [])
        matches = [
            TextSearchMatch.from_dict(item)
            for item in matches_raw
            if isinstance(item, dict)
        ]
        return TextSearchResult(
            matches=matches,
            searched_files=int(data.get("searched_files", 0)),
            query_expr=str(data.get("query_expr", "")),
            duration_ms=int(data.get("duration_ms", 0)),
            truncated=bool(data.get("truncated", False)),
        )


@dataclass(frozen=True, slots=True)
class ProtocolInputField:
    """
    Input field discovered for an extracted protocol contract.

    name: str — Field name/path from request payload.
    required: bool — Heuristic required flag.
    """

    name: str
    required: bool

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ProtocolInputField":
        """
        Build typed protocol input field from Rust JSON payload.

        data: dict[str, Any] — Input field dictionary.
        Returns: ProtocolInputField — Parsed input field.
        """
        return ProtocolInputField(
            name=str(data.get("name", "")),
            required=bool(data.get("required", False)),
        )


@dataclass(frozen=True, slots=True)
class ProtocolOutputField:
    """
    Output field/path discovered for an extracted protocol contract.

    path: str — Dot path into response object.
    """

    path: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ProtocolOutputField":
        """
        Build typed protocol output field from Rust JSON payload.

        data: dict[str, Any] — Output field dictionary.
        Returns: ProtocolOutputField — Parsed output field.
        """
        return ProtocolOutputField(path=str(data.get("path", "")))


@dataclass(frozen=True, slots=True)
class ProtocolEvidence:
    """
    Evidence snippet linked to a protocol contract.

    kind: str — Evidence category (transport_call, request_builder, graphql_operation).
    line: int — 1-based source line.
    text: str — Trimmed source snippet.
    """

    kind: str
    line: int
    text: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ProtocolEvidence":
        """
        Build typed protocol evidence from Rust JSON payload.

        data: dict[str, Any] — Evidence dictionary.
        Returns: ProtocolEvidence — Parsed evidence item.
        """
        return ProtocolEvidence(
            kind=str(data.get("kind", "")),
            line=int(data.get("line", 0)),
            text=str(data.get("text", "")),
        )


@dataclass(frozen=True, slots=True)
class ProtocolContract:
    """
    Extracted protocol request/response contract.

    file_path: str — Source file containing the contract evidence.
    operation: str — Canonical operation label.
    transport: str — Transport type (graphql/rest_post).
    endpoint: Optional[str] — Endpoint path for REST-like contracts.
    operation_name: Optional[str] — GraphQL operation name when available.
    input_fields: list[ProtocolInputField] — Request input fields.
    output_fields: list[ProtocolOutputField] — Response output fields.
    evidence: list[ProtocolEvidence] — Evidence snippets with line anchors.
    matched_terms: list[str] — Query terms matched in local context.
    score: float — Ranking score.
    confidence: float — Heuristic confidence in [0, 1].
    """

    file_path: str
    operation: str
    transport: str
    endpoint: Optional[str]
    operation_name: Optional[str]
    input_fields: list[ProtocolInputField]
    output_fields: list[ProtocolOutputField]
    evidence: list[ProtocolEvidence]
    matched_terms: list[str]
    score: float
    confidence: float

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ProtocolContract":
        """
        Build typed protocol contract from Rust JSON payload.

        data: dict[str, Any] — Contract dictionary.
        Returns: ProtocolContract — Parsed protocol contract.
        """
        input_raw = data.get("input_fields", [])
        output_raw = data.get("output_fields", [])
        evidence_raw = data.get("evidence", [])

        return ProtocolContract(
            file_path=str(data.get("file_path", "")),
            operation=str(data.get("operation", "")),
            transport=str(data.get("transport", "")),
            endpoint=data.get("endpoint"),
            operation_name=data.get("operation_name"),
            input_fields=[
                ProtocolInputField.from_dict(item)
                for item in input_raw
                if isinstance(item, dict)
            ],
            output_fields=[
                ProtocolOutputField.from_dict(item)
                for item in output_raw
                if isinstance(item, dict)
            ],
            evidence=[
                ProtocolEvidence.from_dict(item)
                for item in evidence_raw
                if isinstance(item, dict)
            ],
            matched_terms=list(data.get("matched_terms", [])),
            score=float(data.get("score", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class ProtocolSearchResult:
    """
    Protocol search result payload.

    contracts: list[ProtocolContract] — Ranked protocol contracts.
    searched_files: int — Number of files scanned.
    duration_ms: int — Search duration in milliseconds.
    truncated: bool — True when result limit was reached.
    """

    contracts: list[ProtocolContract]
    searched_files: int
    duration_ms: int
    truncated: bool

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ProtocolSearchResult":
        """
        Build typed protocol search result from Rust JSON payload.

        data: dict[str, Any] — Protocol search result dictionary.
        Returns: ProtocolSearchResult — Parsed protocol search result.
        """
        contracts_raw = data.get("contracts", [])
        contracts = [
            ProtocolContract.from_dict(item)
            for item in contracts_raw
            if isinstance(item, dict)
        ]
        return ProtocolSearchResult(
            contracts=contracts,
            searched_files=int(data.get("searched_files", 0)),
            duration_ms=int(data.get("duration_ms", 0)),
            truncated=bool(data.get("truncated", False)),
        )


@dataclass(frozen=True, slots=True)
class PatternMatchCapture:
    """
    Single metavariable capture from an AST pattern match.

    name: str — Metavariable name (e.g. "NAME", "ARGS").
    text: str — Captured source text.
    """

    name: str
    text: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PatternMatchCapture":
        """
        Build typed capture from Rust JSON payload.

        data: dict[str, Any] — Capture dictionary from pipe response.
        Returns: PatternMatchCapture — Parsed capture.
        """
        return PatternMatchCapture(
            name=str(data.get("name", "")),
            text=str(data.get("text", "")),
        )


@dataclass(frozen=True, slots=True)
class PatternMatchItem:
    """
    Single AST pattern match result.

    file_path: str — File containing the match.
    language: str — Language identifier.
    matched_text: str — Full matched source text.
    line_start: int — 1-based start line.
    column_start: int — 0-based start column.
    line_end: int — 1-based end line.
    column_end: int — 0-based end column.
    captures: list[PatternMatchCapture] — Metavariable captures.
    """

    file_path: str
    language: str
    matched_text: str
    line_start: int
    column_start: int
    line_end: int
    column_end: int
    captures: list[PatternMatchCapture]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PatternMatchItem":
        """
        Build typed pattern match item from Rust JSON payload.

        data: dict[str, Any] — Match dictionary from pipe response.
        Returns: PatternMatchItem — Parsed pattern match item.
        """
        captures_raw = data.get("captures", [])
        captures = [
            PatternMatchCapture.from_dict(item)
            for item in captures_raw
            if isinstance(item, dict)
        ]
        return PatternMatchItem(
            file_path=str(data.get("file_path", "")),
            language=str(data.get("language", "")),
            matched_text=str(data.get("matched_text", "")),
            line_start=int(data.get("line_start", 0)),
            column_start=int(data.get("column_start", 0)),
            line_end=int(data.get("line_end", 0)),
            column_end=int(data.get("column_end", 0)),
            captures=captures,
        )


@dataclass(frozen=True, slots=True)
class PatternMatchResult:
    """
    AST pattern search result payload.

    matches: list[PatternMatchItem] — Pattern matches found.
    searched_files: int — Number of files scanned.
    duration_ms: int — Search duration in milliseconds.
    truncated: bool — True when result limit was reached.
    """

    matches: list[PatternMatchItem]
    searched_files: int
    duration_ms: int
    truncated: bool

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "PatternMatchResult":
        """
        Build typed pattern match result from Rust JSON payload.

        data: dict[str, Any] — Pattern match result dictionary.
        Returns: PatternMatchResult — Parsed pattern match result.
        """
        matches_raw = data.get("matches", [])
        matches = [
            PatternMatchItem.from_dict(item)
            for item in matches_raw
            if isinstance(item, dict)
        ]
        return PatternMatchResult(
            matches=matches,
            searched_files=int(data.get("searched_files", 0)),
            duration_ms=int(data.get("duration_ms", 0)),
            truncated=bool(data.get("truncated", False)),
        )


@dataclass(frozen=True, slots=True)
class RewriteEdit:
    """
    Single byte-offset edit produced by pattern rewrite.

    line_start: int — Start line (1-based).
    column_start: int — Start column (1-based).
    line_end: int — End line (1-based).
    column_end: int — End column (1-based).
    byte_start: int — Start byte offset.
    byte_end: int — End byte offset.
    original: str — Original matched text.
    replacement: str — Replacement text after substitution.
    """

    line_start: int
    column_start: int
    line_end: int
    column_end: int
    byte_start: int
    byte_end: int
    original: str
    replacement: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RewriteEdit":
        """
        Build typed rewrite edit from Rust JSON payload.

        data: dict[str, Any] — Edit dictionary.
        Returns: RewriteEdit — Parsed edit.
        """
        return RewriteEdit(
            line_start=int(data.get("line_start", 0)),
            column_start=int(data.get("column_start", 0)),
            line_end=int(data.get("line_end", 0)),
            column_end=int(data.get("column_end", 0)),
            byte_start=int(data.get("byte_start", 0)),
            byte_end=int(data.get("byte_end", 0)),
            original=str(data.get("original", "")),
            replacement=str(data.get("replacement", "")),
        )


@dataclass(frozen=True, slots=True)
class RewriteFileResult:
    """
    Rewrite results for a single file.

    file_path: str — Path to the rewritten file.
    edits: list[RewriteEdit] — Edits applied to this file.
    rewritten_source: Optional[str] — Full rewritten source (dry_run only).
    """

    file_path: str
    edits: list[RewriteEdit]
    rewritten_source: Optional[str]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RewriteFileResult":
        """
        Build typed file rewrite result from Rust JSON payload.

        data: dict[str, Any] — File result dictionary.
        Returns: RewriteFileResult — Parsed file result.
        """
        edits_raw = data.get("edits", [])
        edits = [
            RewriteEdit.from_dict(e)
            for e in edits_raw
            if isinstance(e, dict)
        ]
        return RewriteFileResult(
            file_path=str(data.get("file_path", "")),
            edits=edits,
            rewritten_source=data.get("rewritten_source"),
        )


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """
    Full pattern rewrite result payload.

    files: list[RewriteFileResult] — Per-file rewrite results.
    searched_files: int — Number of files scanned.
    total_edits: int — Total edits across all files.
    dry_run: bool — True when edits were computed but not written.
    duration_ms: int — Rewrite duration in milliseconds.
    """

    files: list[RewriteFileResult]
    searched_files: int
    total_edits: int
    dry_run: bool
    duration_ms: int

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RewriteResult":
        """
        Build typed rewrite result from Rust JSON payload.

        data: dict[str, Any] — Rewrite result dictionary.
        Returns: RewriteResult — Parsed rewrite result.
        """
        files_raw = data.get("files", [])
        files = [
            RewriteFileResult.from_dict(f)
            for f in files_raw
            if isinstance(f, dict)
        ]
        return RewriteResult(
            files=files,
            searched_files=int(data.get("searched_files", 0)),
            total_edits=int(data.get("total_edits", 0)),
            dry_run=bool(data.get("dry_run", True)),
            duration_ms=int(data.get("duration_ms", 0)),
        )




@dataclass(frozen=True, slots=True)
class FileReadLine:
    """
    Single line row from file_read result.

    line_number: int — 1-based line number.
    text: str — Line text content.
    """

    line_number: int
    text: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "FileReadLine":
        return FileReadLine(
            line_number=int(data.get("line_number", 0)),
            text=str(data.get("text", "")),
        )


@dataclass(frozen=True, slots=True)
class FileReadResult:
    """
    Result payload for line-aware file reads.

    file_path: str — Read file path.
    line_start: int — Requested or resolved start line (1-based).
    line_end: int — Resolved end line (1-based).
    total_lines: int — Total lines in file.
    truncated: bool — True when max_bytes truncation occurred.
    duration_ms: int — Read duration in milliseconds.
    lines: list[FileReadLine] — Returned line rows.
    """

    file_path: str
    line_start: int
    line_end: int
    total_lines: int
    truncated: bool
    duration_ms: int
    lines: list[FileReadLine]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "FileReadResult":
        lines_raw = data.get("lines", [])
        lines = [
            FileReadLine.from_dict(item)
            for item in lines_raw
            if isinstance(item, dict)
        ]
        return FileReadResult(
            file_path=str(data.get("file_path", "")),
            line_start=int(data.get("line_start", 0)),
            line_end=int(data.get("line_end", 0)),
            total_lines=int(data.get("total_lines", 0)),
            truncated=bool(data.get("truncated", False)),
            duration_ms=int(data.get("duration_ms", 0)),
            lines=lines,
        )


@dataclass(frozen=True, slots=True)
class FileEditResult:
    """
    Result payload for file_edit.

    file_path: str — Edited file path.
    applied: bool — True when edit operation succeeded.
    dry_run: bool — True when no write occurred.
    before_hash: str — SHA256 hash before edit.
    after_hash: str — SHA256 hash after edit.
    duration_ms: int — Edit duration in milliseconds.
    edit_id: Optional[str] — Undo record ID when record_undo is enabled.
    updated_text: Optional[str] — Full resulting file text in dry_run mode.
    """

    file_path: str
    applied: bool
    dry_run: bool
    before_hash: str
    after_hash: str
    duration_ms: int
    edit_id: Optional[str]
    updated_text: Optional[str]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "FileEditResult":
        return FileEditResult(
            file_path=str(data.get("file_path", "")),
            applied=bool(data.get("applied", False)),
            dry_run=bool(data.get("dry_run", False)),
            before_hash=str(data.get("before_hash", "")),
            after_hash=str(data.get("after_hash", "")),
            duration_ms=int(data.get("duration_ms", 0)),
            edit_id=data.get("edit_id"),
            updated_text=data.get("updated_text"),
        )


@dataclass(frozen=True, slots=True)
class FileEditRevertResult:
    """
    Result payload for file_edit_revert.

    file_path: str — Reverted file path.
    reverted: bool — True when revert succeeded.
    before_hash: str — SHA256 hash before revert.
    after_hash: str — SHA256 hash after revert.
    duration_ms: int — Revert duration in milliseconds.
    edit_id: str — Reverted edit ID.
    """

    file_path: str
    reverted: bool
    before_hash: str
    after_hash: str
    duration_ms: int
    edit_id: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "FileEditRevertResult":
        return FileEditRevertResult(
            file_path=str(data.get("file_path", "")),
            reverted=bool(data.get("reverted", False)),
            before_hash=str(data.get("before_hash", "")),
            after_hash=str(data.get("after_hash", "")),
            duration_ms=int(data.get("duration_ms", 0)),
            edit_id=str(data.get("edit_id", "")),
        )


@dataclass
class SymbolEditResult:
    """Result payload for symbol_edit."""

    file_path: str
    symbol_name: str
    applied: bool
    dry_run: bool
    before_hash: str
    after_hash: str
    line_start: int
    line_end: int
    duration_ms: int
    edit_id: str | None
    updated_text: str | None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SymbolEditResult":
        return SymbolEditResult(
            file_path=str(data.get("file_path", "")),
            symbol_name=str(data.get("symbol_name", "")),
            applied=bool(data.get("applied", False)),
            dry_run=bool(data.get("dry_run", False)),
            before_hash=str(data.get("before_hash", "")),
            after_hash=str(data.get("after_hash", "")),
            line_start=int(data.get("line_start", 0)),
            line_end=int(data.get("line_end", 0)),
            duration_ms=int(data.get("duration_ms", 0)),
            edit_id=data.get("edit_id"),
            updated_text=data.get("updated_text"),
        )


@dataclass(frozen=True, slots=True)
class TraceNode:
    """
    Single node in a call graph trace.

    name: str — Symbol name.
    kind: str — Symbol kind label.
    language: str — Language identifier.
    file_path: str — File containing the symbol.
    line_start: int — Start line (0-based).
    line_end: int — End line (0-based).
    depth: int — BFS depth from root symbol.
    """

    name: str
    kind: str
    language: str
    file_path: str
    line_start: int
    line_end: int
    depth: int

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TraceNode":
        """
        Build typed trace node from Rust JSON payload.

        data: dict[str, Any] — Node dictionary from pipe response.
        Returns: TraceNode — Parsed trace node.
        """
        return TraceNode(
            name=str(data.get("name", "")),
            kind=str(data.get("kind", "")),
            language=str(data.get("language", "")),
            file_path=str(data.get("file_path", "")),
            line_start=int(data.get("line_start", 0)),
            line_end=int(data.get("line_end", 0)),
            depth=int(data.get("depth", 0)),
        )


@dataclass(frozen=True, slots=True)
class TraceEdge:
    """
    Directed edge in a call graph trace.

    from_name: str — Caller symbol name.
    to_name: str — Callee symbol name.
    call_line: int — 1-based line of the call site.
    file_path: str — File containing the call site.
    """

    from_name: str
    to_name: str
    call_line: int
    file_path: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TraceEdge":
        """
        Build typed trace edge from Rust JSON payload.

        data: dict[str, Any] — Edge dictionary from pipe response.
        Returns: TraceEdge — Parsed trace edge.
        """
        return TraceEdge(
            from_name=str(data.get("from_name", "")),
            to_name=str(data.get("to_name", "")),
            call_line=int(data.get("call_line", 0)),
            file_path=str(data.get("file_path", "")),
        )


@dataclass(frozen=True, slots=True)
class CallGraphTraceResult:
    """
    Call graph BFS trace result payload.

    root_symbol: str — Starting symbol name.
    direction: str — Trace direction (upstream/downstream/both).
    max_depth: int — Maximum BFS depth used.
    nodes: list[TraceNode] — Discovered nodes at each depth.
    edges: list[TraceEdge] — Call edges traversed.
    cycles_detected: list[str] — Cycle descriptions if any.
    indexed_files: int — File count used in index.
    indexed_symbols: int — Symbol count used in index.
    from_cache: bool — True when served from in-memory cache.
    """

    root_symbol: str
    direction: str
    max_depth: int
    nodes: list[TraceNode]
    edges: list[TraceEdge]
    cycles_detected: list[str]
    indexed_files: int
    indexed_symbols: int
    from_cache: bool

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CallGraphTraceResult":
        """
        Build typed call graph trace result from Rust JSON payload.

        data: dict[str, Any] — Trace result dictionary.
        Returns: CallGraphTraceResult — Parsed trace result.
        """
        nodes_raw = data.get("nodes", [])
        nodes = [
            TraceNode.from_dict(item)
            for item in nodes_raw
            if isinstance(item, dict)
        ]
        edges_raw = data.get("edges", [])
        edges = [
            TraceEdge.from_dict(item)
            for item in edges_raw
            if isinstance(item, dict)
        ]
        return CallGraphTraceResult(
            root_symbol=str(data.get("root_symbol", "")),
            direction=str(data.get("direction", "")),
            max_depth=int(data.get("max_depth", 0)),
            nodes=nodes,
            edges=edges,
            cycles_detected=list(data.get("cycles_detected", [])),
            indexed_files=int(data.get("indexed_files", 0)),
            indexed_symbols=int(data.get("indexed_symbols", 0)),
            from_cache=bool(data.get("from_cache", False)),
        )


class RustParserService:
    """
    Python wrapper around the Rust extraction service transport.

    _client: PipeClient — Low-level transport client.
    """

    def __init__(self, pipe_name: str = PIPE_NAME, service_path: Optional[str] = None):
        """
        pipe_name: str — Transport endpoint path.
        service_path: Optional[str] — Optional explicit service binary path.
        """
        resolved_service_path = service_path or _default_service_path()
        self._client = PipeClient(pipe_name=pipe_name, service_path=resolved_service_path)

    @property
    def connected(self) -> bool:
        """
        Returns: bool — True if the transport handle is currently connected.
        """
        return self._client.connected

    def connect(self, timeout_ms: int = 5000) -> None:
        """
        Connect to the Rust service transport.

        timeout_ms: int — Connection timeout in milliseconds.
        """
        self._client.connect(timeout_ms=timeout_ms)

    def close(self) -> None:
        """
        Close the service transport connection.
        """
        self._client.close()

    def ping(self, ensure_server: bool = True, timeout_ms: int = 5000) -> str:
        """
        Ping the Rust service to validate connectivity.

        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: str — Expected value is "pong".
        """
        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)
        return self._client.ping()

    def extract(self, path: str | Path, ensure_server: bool = True, timeout_ms: int = 10000) -> list[ExtractionResult]:
        """
        Extract parsed symbols for a file or directory path.

        path: str | Path — File or directory to parse.
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: list[ExtractionResult] — One result per parsed file.
        """
        input_path = Path(path).resolve()

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        try:
            raw = self._client.extract(str(input_path))
            return [ExtractionResult.from_dict(item) for item in raw if isinstance(item, dict)]
        except PipeProtocolError as exc:
            if "frame too large" not in str(exc).lower() or not input_path.is_dir():
                raise

            results: list[ExtractionResult] = []
            for child in sorted(input_path.iterdir(), key=lambda p: p.name.lower()):
                if child.is_dir() or child.is_file():
                    child_raw = self._client.extract(str(child))
                    results.extend(
                        ExtractionResult.from_dict(item)
                        for item in child_raw
                        if isinstance(item, dict)
                    )
            return results

    def scan_files(
        self,
        path: str | Path,
        respect_gitignore: bool = True,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> list[FileScanEntry]:
        """
        List supported source files with lightweight metadata.

        path: str | Path â€” File or directory to scan.
        respect_gitignore: bool â€” Apply ignore rules when True.
        ensure_server: bool â€” Auto-start service if unavailable.
        timeout_ms: int â€” Startup/connect timeout in milliseconds.
        Returns: list[FileScanEntry] â€” Supported files with size and mtime.
        """
        input_path = Path(path).resolve()

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.scan_files(str(input_path), respect_gitignore=respect_gitignore)
        return [FileScanEntry.from_dict(item) for item in raw if isinstance(item, dict)]

    def extract_compact(
        self,
        path: str | Path,
        respect_gitignore: bool = True,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> tuple[list[CompactExtractionResult], ExtractStats]:
        """
        Extract symbols in compact mode (no source/docstring/params).

        path: str | Path — Directory to parse.
        respect_gitignore: bool — Honor .gitignore rules.
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: tuple[list[CompactExtractionResult], ExtractStats] — Compact results and stats.
        """
        input_path = str(Path(path).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.extract(input_path, compact=True, respect_gitignore=respect_gitignore)
        results = [
            CompactExtractionResult.from_dict(item)
            for item in raw.get("results", [])
            if isinstance(item, dict)
        ]
        stats = ExtractStats.from_dict(raw.get("stats", {}))
        return results, stats

    def extract_stats(
        self,
        path: str | Path,
        respect_gitignore: bool = True,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> ExtractStats:
        """
        Extract aggregate statistics only (no symbol data).

        path: str | Path — Directory to parse.
        respect_gitignore: bool — Honor .gitignore rules.
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: ExtractStats — Aggregate extraction statistics.
        """
        input_path = str(Path(path).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.extract(input_path, stats_only=True, respect_gitignore=respect_gitignore)
        return ExtractStats.from_dict(raw)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: ExtractSymbolResult — Matched symbols with metadata.
        """
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.extract_symbol(input_file, symbol)
        return ExtractSymbolResult.from_dict(raw)

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
        Read a file with optional line range slicing.

        file: str | Path — Absolute path to the source file.
        start_line: int | None — 1-based start line. Uses 1 when None.
        end_line: int | None — 1-based end line. Reads to EOF when None.
        max_bytes: int | None — Optional byte cap for returned content.
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: FileReadResult — Typed file read payload.
        """
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.file_read(
            file=input_file,
            start_line=start_line,
            end_line=end_line,
            max_bytes=max_bytes,
        )
        return FileReadResult.from_dict(raw)

    def file_edit(
        self,
        file: str | Path,
        mode: str,
        edits: list[dict[str, Any]] | None = None,
        text: str | None = None,
        dry_run: bool = False,
        expected_hash: str | None = None,
        record_undo: bool = True,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> FileEditResult:
        """
        Apply a line-based file edit operation.

        file: str | Path — Absolute path to the source file.
        mode: str — Edit mode: append|insert|replace|delete|batch.
        edits: list[dict[str, Any]] | None — Edit rows with start_line/end_line/text.
        text: str | None — Fallback text for modes that require text.
        dry_run: bool — Compute result without writing file.
        expected_hash: str | None — Optimistic hash guard.
        record_undo: bool — Persist undo record for revert calls.
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: FileEditResult — Typed file edit payload.
        """
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.file_edit(
            file=input_file,
            mode=mode,
            edits=edits,
            text=text,
            dry_run=dry_run,
            expected_hash=expected_hash,
            record_undo=record_undo,
        )
        return FileEditResult.from_dict(raw)

    def file_edit_revert(
        self,
        edit_id: str,
        dry_run: bool = False,
        expected_hash: str | None = None,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> FileEditRevertResult:
        """
        Revert a previous file_edit operation by undo edit ID.

        edit_id: str — Undo record ID returned by file_edit.
        dry_run: bool — Simulate revert without writing file.
        expected_hash: str | None — Optional pre-revert hash guard.
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: FileEditRevertResult — Typed revert result payload.
        """
        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.file_edit_revert(
            edit_id=edit_id,
            dry_run=dry_run,
            expected_hash=expected_hash,
        )
        return FileEditRevertResult.from_dict(raw)

    def symbol_edit(
        self,
        file: str | Path,
        symbol_name: str,
        new_source: str,
        dry_run: bool = False,
        expected_hash: str | None = None,
        record_undo: bool = True,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> SymbolEditResult:
        """Edit a symbol by name using AST extraction."""
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.symbol_edit(
            file=input_file,
            symbol_name=symbol_name,
            new_source=new_source,
            dry_run=dry_run,
            expected_hash=expected_hash,
            record_undo=record_undo,
        )
        return SymbolEditResult.from_dict(raw)

    def grep(
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
        Perform fast literal grep via the Rust service transport.

        query: str — Literal text pattern to search.
        path: str | Path | None — Optional file/directory path. Uses current directory when None.
        respect_gitignore: bool — Apply .gitignore/.ignore/.quick-ignore when True.
        limit: int — Maximum number of matches returned.
        before_context: int — Number of lines before each match.
        after_context: int — Number of lines after each match.
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: GrepResult — Typed grep result payload.
        """
        input_path = str(Path(path).resolve()) if path is not None else str(Path.cwd().resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.grep(
            query=query,
            path=input_path,
            respect_gitignore=respect_gitignore,
            limit=limit,
            before_context=max(0, int(before_context)),
            after_context=max(0, int(after_context)),
        )
        return GrepResult.from_dict(raw)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: SymbolLookupResult — Typed symbol lookup payload.
        """
        input_path = str(Path(path).resolve()) if path is not None else str(Path.cwd().resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.symbol_lookup(
            query=query,
            path=input_path,
            respect_gitignore=respect_gitignore,
            limit=limit,
            intent_mode=intent_mode,
            intent_level=intent_level,
        )
        return SymbolLookupResult.from_dict(raw)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: CallerLookupResult — Typed caller lookup payload.
        """
        input_path = str(Path(path).resolve()) if path is not None else str(Path.cwd().resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.find_callers(
            symbol=symbol,
            path=input_path,
            respect_gitignore=respect_gitignore,
            limit=limit,
        )
        return CallerLookupResult.from_dict(raw)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: CallGraphTraceResult — Typed call graph trace payload.
        """
        input_path = str(Path(path).resolve()) if path is not None else str(Path.cwd().resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.trace_call_graph(
            symbol=symbol,
            path=input_path,
            respect_gitignore=respect_gitignore,
            direction=direction,
            max_depth=max_depth,
        )
        return CallGraphTraceResult.from_dict(raw)

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
        Generate a repo skeleton for a file or directory via the Rust service.

        path: str | Path — File or directory to skeletonize.
        max_depth: int | None — Max directory recursion depth. None uses server default (20).
        include_signatures: bool — Include function/class signatures.
        include_line_numbers: bool — Include line number ranges.
        collapse_threshold: int — Collapse dirs with fewer files than this.
        respect_gitignore: bool — Apply .gitignore/.ignore/.quick-ignore when True.
        format: str — Output format: "json" or "markdown".
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: SkeletonResult — Typed skeleton result payload.
        """
        input_path = str(Path(path).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.skeleton(
            path=input_path,
            max_depth=max_depth,
            include_signatures=include_signatures,
            include_line_numbers=include_line_numbers,
            collapse_threshold=collapse_threshold,
            respect_gitignore=respect_gitignore,
            format=format,
        )
        return SkeletonResult.from_dict(raw)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: ImportGraphResult — Typed import graph payload.
        """
        input_file = str(Path(file).resolve())
        input_path = str(Path(path).resolve()) if path is not None else str(Path.cwd().resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.import_graph(
            file=input_file,
            path=input_path,
            respect_gitignore=respect_gitignore,
        )
        return ImportGraphResult.from_dict(raw)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: ImportGraphResult — Typed import graph payload.
        """
        input_file = str(Path(file).resolve())
        input_path = str(Path(path).resolve()) if path is not None else str(Path.cwd().resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.find_importers(
            file=input_file,
            path=input_path,
            respect_gitignore=respect_gitignore,
        )
        return ImportGraphResult.from_dict(raw)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: TextSearchResult — Typed BM25 search result payload.
        """
        input_path = str(Path(path).resolve()) if path is not None else str(Path.cwd().resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.text_search(
            query=query,
            path=input_path,
            respect_gitignore=respect_gitignore,
            limit=limit,
            intent_mode=intent_mode,
            intent_level=intent_level,
        )
        return TextSearchResult.from_dict(raw)


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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: PatternMatchResult — Typed pattern match result payload.
        """
        input_path = str(Path(path).resolve()) if path is not None else str(Path.cwd().resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.pattern_search(
            pattern=pattern,
            language=language,
            path=input_path,
            respect_gitignore=respect_gitignore,
            limit=limit,
        )
        return PatternMatchResult.from_dict(raw)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: ProtocolSearchResult — Typed protocol extraction result payload.
        """
        input_path = str(Path(path).resolve()) if path is not None else str(Path.cwd().resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.protocol_search(
            query=query,
            path=input_path,
            respect_gitignore=respect_gitignore,
            limit=limit,
            context_radius=context_radius,
            min_score=min_score,
            include_markers=include_markers,
            exclude_markers=exclude_markers,
            max_input_fields=max_input_fields,
            max_output_fields=max_output_fields,
        )
        return ProtocolSearchResult.from_dict(raw)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: RewriteResult — Typed rewrite result payload.
        """
        input_path = str(Path(path).resolve()) if path is not None else str(Path.cwd().resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        raw = self._client.pattern_rewrite(
            pattern=pattern,
            replacement=replacement,
            language=language,
            path=input_path,
            respect_gitignore=respect_gitignore,
            limit=limit,
            dry_run=dry_run,
        )
        return RewriteResult.from_dict(raw)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP definition location(s).
        """
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        return self._client.lsp_definition(input_file, line, character)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP reference locations.
        """
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        return self._client.lsp_references(input_file, line, character, include_declaration)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP hover contents (type info, docs).
        """
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        return self._client.lsp_hover(input_file, line, character)

    def lsp_symbols(
        self,
        file: str | Path,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Get document symbols (outline) for a file via LSP.

        file: str | Path — Absolute path to the source file.
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP hierarchical symbol tree.
        """
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        return self._client.lsp_symbols(input_file)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP text edits to apply.
        """
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        return self._client.lsp_format(input_file, tab_size, insert_spaces)

    def lsp_diagnostics(
        self,
        file: str | Path,
        ensure_server: bool = True,
        timeout_ms: int = 10000,
    ) -> dict:
        """
        Request diagnostics for a file via LSP.

        file: str | Path — Absolute path to the source file.
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP diagnostic request status.
        """
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        return self._client.lsp_diagnostics(input_file)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP completion items.
        """
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        return self._client.lsp_completion(input_file, line, character)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP workspace edit (file changes).
        """
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        return self._client.lsp_rename(input_file, line, character, new_name)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP rename range and placeholder text.
        """
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        return self._client.lsp_prepare_rename(input_file, line, character)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP available code actions.
        """
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        return self._client.lsp_code_actions(
            input_file, start_line, start_character,
            end_line, end_character, diagnostics,
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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP active signature and parameter info.
        """
        input_file = str(Path(file).resolve())

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        return self._client.lsp_signature_help(input_file, line, character)

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
        ensure_server: bool — Auto-start service if unavailable.
        timeout_ms: int — Startup/connect timeout in milliseconds.
        Returns: dict — LSP matching workspace symbols.
        """
        input_file = str(Path(file).resolve()) if file is not None else None

        if ensure_server:
            self._client.ensure_server(timeout_ms=timeout_ms)
        elif not self.connected:
            self.connect(timeout_ms=timeout_ms)

        return self._client.lsp_workspace_symbols(query, input_file)


def _default_service_path() -> Optional[str]:
    """
    Resolve default Rust service binary path.

    Returns: Optional[str] — Absolute path to quickcontext-service binary if found.
    """
    env_path = os.environ.get("QC_SERVICE_PATH")
    if env_path:
        return env_path

    module_file = Path(__file__).resolve()
    root = module_file.parents[2]

    candidates = [
        root / "service" / "target" / "release" / "quickcontext-service.exe",
        root / "service" / "target" / "release" / "quickcontext-service",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return None
