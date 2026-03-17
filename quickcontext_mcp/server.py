from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Literal
import os
import uuid

from fastmcp import FastMCP
from fastmcp.dependencies import CurrentContext
from fastmcp.server.context import Context
from pydantic import BaseModel, Field

from engine.sdk import QuickContext
from engine.src.collection import CollectionManager
from engine.src.config import EngineConfig
from engine.src.project import detect_project_name


SERVER_INSTRUCTIONS = """
quickcontext is a local code-context MCP server backed by the QuickContext Python SDK and Rust service.

Use `project_info` first when you need to understand the current path, index state, or available folder scopes.
Use `search` as the default retrieval tool. Prefer `mode="auto"` unless you explicitly want lexical or bundle behavior.
Use `grep` for exact literals and `symbol_lookup` for precise identifiers, functions, classes, or methods.
Use `index` before semantic retrieval when the target codebase is not yet indexed or when the index should be refreshed.
Pass `path` whenever the target repository is outside the current server working directory.
Keep tool calls scoped and incremental: start with a narrow folder or project path when possible.
""".strip()


SearchMode = Literal["auto", "text", "semantic", "bundle"]
IndexStatus = Literal["queued", "running", "completed", "failed", "already_running"]


class PhaseTimings(BaseModel):
    scan_seconds: float = 0.0
    artifact_profile_seconds: float = 0.0
    extract_seconds: float = 0.0
    chunk_build_seconds: float = 0.0
    filter_seconds: float = 0.0
    dedup_seconds: float = 0.0
    description_seconds: float = 0.0
    embedding_seconds: float = 0.0
    point_build_seconds: float = 0.0
    upsert_seconds: float = 0.0


class IndexStatsView(BaseModel):
    total_chunks: int
    upserted_points: int
    failed_points: int
    total_tokens: int
    duration_seconds: float
    llm_cost_usd: float
    embedding_cost_usd: float
    skipped_small_chunks: int
    skipped_minified_chunks: int
    skipped_capped_chunks: int
    files_capped: int
    descriptions_enabled: bool
    embedding_requests: int
    embedding_retries: int
    embedding_failed_requests: int
    embedding_input_count: int
    embedding_final_batch_size: int
    embedding_batch_shrink_events: int
    embedding_batch_grow_events: int
    phase_timings: PhaseTimings


class IndexRunView(BaseModel):
    run_id: str
    status: IndexStatus
    path: str
    project_name: str
    reindex: bool
    fast: bool
    attached_to_existing: bool = False
    duplicate_of: str | None = None
    message: str = ""
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    stats: IndexStatsView | None = None


class IndexStatusResponse(BaseModel):
    runs: list[IndexRunView]


class ProjectFolder(BaseModel):
    relative_path: str
    absolute_path: str


class ProjectSummary(BaseModel):
    project_name: str
    indexed: bool
    real_collection: str | None = None
    points_count: int | None = None
    indexed_vectors_count: int | None = None
    status: str | None = None


class ProjectInfoResponse(BaseModel):
    path: str
    project_name: str
    indexed: bool
    qdrant_enabled: bool
    qdrant_available: bool
    parser_connected: bool
    cache_entries: int
    cache_disk_bytes: int
    cache_path: str
    collection: ProjectSummary | None = None
    active_index_runs: list[IndexRunView] = Field(default_factory=list)
    folders: list[ProjectFolder] = Field(default_factory=list)
    search_modes: list[SearchMode] = Field(default_factory=lambda: ["auto", "text", "semantic", "bundle"])


class ListProjectsResponse(BaseModel):
    qdrant_available: bool
    projects: list[ProjectSummary]
    active_index_runs: list[IndexRunView] = Field(default_factory=list)


class SearchHit(BaseModel):
    kind: Literal["semantic", "text", "symbol", "grep"]
    file_path: str
    language: str | None = None
    symbol_name: str | None = None
    symbol_kind: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    score: float | None = None
    signature: str | None = None
    description: str | None = None
    snippet: str | None = None
    path_context: str | None = None
    matched_terms: list[str] = Field(default_factory=list)


class RelatedFileView(BaseModel):
    file_path: str
    distance: int
    relations: list[dict[str, Any]] = Field(default_factory=list)


class RelatedCallerView(BaseModel):
    symbol: str
    caller_name: str
    caller_kind: str
    caller_file_path: str
    caller_line: int
    language: str | None = None


class SearchResponse(BaseModel):
    query: str
    path: str
    project_name: str
    mode_requested: SearchMode
    mode_used: str
    indexed: bool
    results: list[SearchHit] = Field(default_factory=list)
    related_files: list[RelatedFileView] = Field(default_factory=list)
    related_callers: list[RelatedCallerView] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class GrepHit(BaseModel):
    file_path: str
    line_number: int
    column_start: int
    column_end: int
    line: str
    context_before: list[str] = Field(default_factory=list)
    context_after: list[str] = Field(default_factory=list)


class GrepResponse(BaseModel):
    query: str
    path: str
    searched_files: int
    duration_ms: int
    truncated: bool
    matches: list[GrepHit] = Field(default_factory=list)


class SymbolHit(BaseModel):
    name: str
    kind: str
    language: str
    file_path: str
    line_start: int
    line_end: int
    parent: str | None = None
    signature: str | None = None


class SymbolLookupResponse(BaseModel):
    query: str
    path: str
    project_name: str
    indexed_files: int
    indexed_symbols: int
    from_cache: bool
    results: list[SymbolHit] = Field(default_factory=list)


class CapabilitiesResource(BaseModel):
    product: str = "quickcontext"
    server: str = "quickcontext-mcp"
    primary_sdk: str = "engine.sdk.QuickContext"
    default_search_mode: SearchMode = "auto"
    search_modes: list[SearchMode] = Field(default_factory=lambda: ["auto", "text", "semantic", "bundle"])
    index_behavior: str = "thin SDK wrapper with duplicate suppression and resumable full indexing"
    transport_guidance: str = "Use stdio for local desktop clients and streamable HTTP for long-lived multi-client deployments."
    notes: list[str] = Field(
        default_factory=lambda: [
            "Semantic retrieval requires an indexed project and working Qdrant plus embedding configuration.",
            "Text, grep, and symbol lookup routes work through the Rust service and do not require vector search.",
            "Repeated index calls for the same path/project attach to the existing active run instead of starting another index operation.",
        ]
    )


class ProjectsResource(BaseModel):
    projects: list[ProjectSummary]


class IndexJobsResource(BaseModel):
    runs: list[IndexRunView]


@dataclass
class _IndexRunRecord:
    run_id: str
    key: str
    path: str
    project_name: str
    reindex: bool
    fast: bool
    status: IndexStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    message: str = ""
    error: str | None = None
    stats: dict[str, Any] | None = None


class _IndexRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._runs: dict[str, _IndexRunRecord] = {}
        self._active_by_key: dict[str, str] = {}
        self._latest_by_key: dict[str, str] = {}
        self._order: list[str] = []

    def start_or_attach(
        self,
        *,
        path: str,
        project_name: str,
        reindex: bool,
        fast: bool,
    ) -> tuple[_IndexRunRecord, bool]:
        key = self._make_key(path=path, project_name=project_name)
        with self._lock:
            active_id = self._active_by_key.get(key)
            if active_id:
                return self._runs[active_id], True

            run_id = uuid.uuid4().hex[:12]
            record = _IndexRunRecord(
                run_id=run_id,
                key=key,
                path=path,
                project_name=project_name,
                reindex=reindex,
                fast=fast,
                status="queued",
                created_at=_now_iso(),
                message="Index queued",
            )
            self._runs[run_id] = record
            self._active_by_key[key] = run_id
            self._latest_by_key[key] = run_id
            self._order.append(run_id)
            self._prune_locked()
            return record, False

    def mark_running(self, run_id: str, message: str) -> None:
        with self._lock:
            record = self._runs[run_id]
            record.status = "running"
            record.started_at = record.started_at or _now_iso()
            record.message = message

    def mark_completed(self, run_id: str, stats: dict[str, Any]) -> None:
        with self._lock:
            record = self._runs[run_id]
            record.status = "completed"
            record.message = "Index complete"
            record.stats = stats
            record.finished_at = _now_iso()
            self._active_by_key.pop(record.key, None)

    def mark_failed(self, run_id: str, error: str) -> None:
        with self._lock:
            record = self._runs[run_id]
            record.status = "failed"
            record.error = error
            record.message = "Index failed"
            record.finished_at = _now_iso()
            self._active_by_key.pop(record.key, None)

    def get_by_run_id(self, run_id: str) -> _IndexRunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def get_for_target(self, *, path: str, project_name: str) -> list[_IndexRunRecord]:
        key = self._make_key(path=path, project_name=project_name)
        with self._lock:
            latest_id = self._latest_by_key.get(key)
            if latest_id is None:
                return []
            latest = self._runs.get(latest_id)
            return [latest] if latest is not None else []

    def list_recent(self, limit: int = 20) -> list[_IndexRunRecord]:
        with self._lock:
            run_ids = list(reversed(self._order[-limit:]))
            return [self._runs[run_id] for run_id in run_ids if run_id in self._runs]

    def list_active_for_target(self, *, path: str, project_name: str) -> list[_IndexRunRecord]:
        key = self._make_key(path=path, project_name=project_name)
        with self._lock:
            run_id = self._active_by_key.get(key)
            if run_id is None:
                return []
            record = self._runs.get(run_id)
            return [record] if record is not None else []

    def _prune_locked(self) -> None:
        while len(self._order) > 128:
            oldest_id = self._order.pop(0)
            record = self._runs.get(oldest_id)
            if record and record.status in {"queued", "running"}:
                self._order.append(oldest_id)
                break
            self._runs.pop(oldest_id, None)

    @staticmethod
    def _make_key(*, path: str, project_name: str) -> str:
        return f"{project_name}|{path.lower()}"


_INDEX_REGISTRY = _IndexRegistry()
_CONFIG_LOCK = Lock()
_CONFIG_CACHE: tuple[str, EngineConfig] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_config_path() -> str:
    configured = os.environ.get("QC_MCP_CONFIG")
    if configured:
        return str(Path(configured).resolve())

    for candidate in ("quickcontext.json", ".quickcontext.json"):
        path = Path(candidate).resolve()
        if path.exists():
            return str(path)

    return ""


def _load_config() -> EngineConfig:
    global _CONFIG_CACHE

    config_path = _resolve_config_path()
    with _CONFIG_LOCK:
        if _CONFIG_CACHE and _CONFIG_CACHE[0] == config_path:
            return _CONFIG_CACHE[1]

        if config_path:
            config = EngineConfig.from_json(config_path)
        else:
            config = EngineConfig.auto()

        _CONFIG_CACHE = (config_path, config)
        return config


def _normalize_scope_path(path: str) -> Path:
    return Path(path or ".").expanduser().resolve()


def _project_anchor(path: Path) -> Path:
    return path.parent if path.is_file() else path


def _truncate_text(text: str | None, *, max_chars: int = 1800, max_lines: int = 60) -> str | None:
    if text is None:
        return None
    lines = text.splitlines()
    if len(lines) > max_lines:
        text = "\n".join(lines[:max_lines]) + "\n..."
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _display_path(value: str | Path) -> str:
    return str(value).replace("\\\\?\\", "")


def _friendly_error_message(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if "6333" in lowered or "6334" in lowered or "connection refused" in lowered:
        return "Qdrant is configured but unreachable at localhost:6333/6334. Start Qdrant or update quickcontext.json before indexing or vector search."
    return message


def _qdrant_available(config: EngineConfig) -> bool:
    if config.qdrant is None:
        return False
    try:
        with QuickContext(config) as qc:
            qc.connect(verify=True)
        return True
    except Exception:
        return False


def _phase_timings_from_stats(stats: dict[str, Any]) -> PhaseTimings:
    return PhaseTimings(
        scan_seconds=float(stats.get("scan_stage_duration_seconds", 0.0)),
        artifact_profile_seconds=float(stats.get("artifact_profile_stage_duration_seconds", 0.0)),
        extract_seconds=float(stats.get("extract_stage_duration_seconds", 0.0)),
        chunk_build_seconds=float(stats.get("chunk_build_stage_duration_seconds", 0.0)),
        filter_seconds=float(stats.get("filter_stage_duration_seconds", 0.0)),
        dedup_seconds=float(stats.get("dedup_stage_duration_seconds", 0.0)),
        description_seconds=float(stats.get("description_stage_duration_seconds", 0.0)),
        embedding_seconds=float(stats.get("embedding_stage_duration_seconds", 0.0)),
        point_build_seconds=float(stats.get("point_build_stage_duration_seconds", 0.0)),
        upsert_seconds=float(stats.get("upsert_stage_duration_seconds", 0.0)),
    )


def _index_stats_view(stats: dict[str, Any] | None) -> IndexStatsView | None:
    if not stats:
        return None
    return IndexStatsView(
        total_chunks=int(stats.get("total_chunks", 0)),
        upserted_points=int(stats.get("upserted_points", 0)),
        failed_points=int(stats.get("failed_points", 0)),
        total_tokens=int(stats.get("total_tokens", 0)),
        duration_seconds=float(stats.get("duration_seconds", 0.0)),
        llm_cost_usd=float(stats.get("llm_cost_usd", 0.0)),
        embedding_cost_usd=float(stats.get("embedding_cost_usd", 0.0)),
        skipped_small_chunks=int(stats.get("skipped_small_chunks", 0)),
        skipped_minified_chunks=int(stats.get("skipped_minified_chunks", 0)),
        skipped_capped_chunks=int(stats.get("skipped_capped_chunks", 0)),
        files_capped=int(stats.get("files_capped", 0)),
        descriptions_enabled=bool(stats.get("descriptions_enabled", False)),
        embedding_requests=int(stats.get("embedding_requests", 0)),
        embedding_retries=int(stats.get("embedding_retries", 0)),
        embedding_failed_requests=int(stats.get("embedding_failed_requests", 0)),
        embedding_input_count=int(stats.get("embedding_input_count", 0)),
        embedding_final_batch_size=int(stats.get("embedding_final_batch_size", 0)),
        embedding_batch_shrink_events=int(stats.get("embedding_batch_shrink_events", 0)),
        embedding_batch_grow_events=int(stats.get("embedding_batch_grow_events", 0)),
        phase_timings=_phase_timings_from_stats(stats),
    )


def _index_run_view(record: _IndexRunRecord, *, attached: bool = False, duplicate_of: str | None = None) -> IndexRunView:
    return IndexRunView(
        run_id=record.run_id,
        status=record.status,
        path=_display_path(record.path),
        project_name=record.project_name,
        reindex=record.reindex,
        fast=record.fast,
        attached_to_existing=attached,
        duplicate_of=duplicate_of,
        message=record.message,
        error=record.error,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        stats=_index_stats_view(record.stats),
    )


def _related_file_view(item: dict[str, Any]) -> RelatedFileView:
    normalized = dict(item)
    normalized["file_path"] = _display_path(normalized.get("file_path", ""))
    return RelatedFileView.model_validate(normalized)


def _related_caller_view(item: dict[str, Any]) -> RelatedCallerView:
    normalized = dict(item)
    normalized["caller_file_path"] = _display_path(normalized.get("caller_file_path", ""))
    return RelatedCallerView.model_validate(normalized)


def _list_project_folders(project_root: str | Path, max_depth: int = 5, limit: int = 200) -> list[ProjectFolder]:
    root = Path(project_root).resolve()
    if not root.exists() or not root.is_dir():
        return []

    folders: list[ProjectFolder] = [ProjectFolder(relative_path=".", absolute_path=str(root))]
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
            ProjectFolder(
                relative_path=str(relative).replace("\\", "/"),
                absolute_path=str(candidate.resolve()),
            )
        )
        if len(folders) >= limit:
            break

    return folders


def _project_catalog(config: EngineConfig) -> dict[str, ProjectSummary]:
    if config.qdrant is None:
        return {}

    try:
        with QuickContext(config) as qc:
            manager = CollectionManager(
                client=qc.connection.client,
                config=config,
                collection_name="_quickcontext_mcp_dummy",
            )
            catalog = {}
            for item in manager.list_all_projects():
                catalog[item["name"]] = ProjectSummary(
                    project_name=item["name"],
                    indexed=True,
                    real_collection=item.get("real_collection"),
                    points_count=int(item.get("points_count", 0)),
                    indexed_vectors_count=None,
                    status=item.get("status"),
                )
            return catalog
    except Exception:
        return {}


def _project_summary_for_path(path: Path, project_name: str, config: EngineConfig) -> ProjectSummary | None:
    summary = _project_catalog(config).get(project_name)
    if summary is not None:
        return summary
    return ProjectSummary(project_name=project_name, indexed=False)


def _semantic_hit(result: Any) -> SearchHit:
    return SearchHit(
        kind="semantic",
        file_path=_display_path(getattr(result, "file_path", "")),
        language=getattr(result, "language", None),
        symbol_name=getattr(result, "symbol_name", None),
        symbol_kind=getattr(result, "symbol_kind", None),
        line_start=getattr(result, "line_start", None),
        line_end=getattr(result, "line_end", None),
        score=float(getattr(result, "score", 0.0)),
        signature=getattr(result, "signature", None),
        description=getattr(result, "description", None),
        snippet=_truncate_text(getattr(result, "source", None)),
        path_context=getattr(result, "path_context", None),
    )


def _text_hit(match: Any) -> SearchHit:
    return SearchHit(
        kind="text",
        file_path=_display_path(getattr(match, "file_path", "")),
        language=getattr(match, "language", None),
        line_start=getattr(match, "snippet_line_start", None),
        line_end=getattr(match, "snippet_line_end", None),
        score=float(getattr(match, "score", 0.0)),
        snippet=_truncate_text(getattr(match, "snippet", None), max_chars=1200, max_lines=32),
        matched_terms=list(getattr(match, "matched_terms", [])),
    )


def _search_impl(
    *,
    query: str,
    path: str,
    project_name: str | None,
    mode: SearchMode,
    limit: int,
    path_prefix: str | None,
) -> SearchResponse:
    scope = _normalize_scope_path(path)
    anchor = _project_anchor(scope)
    config = _load_config()
    resolved_project = detect_project_name(anchor, manual_override=project_name)
    summary = _project_summary_for_path(anchor, resolved_project, config)
    indexed = bool(summary and summary.indexed)

    with QuickContext(config) as qc:
        if mode == "text":
            result = qc.text_search(
                query=query,
                path=scope,
                limit=limit,
                intent_mode=True,
                intent_level=2,
            )
            return SearchResponse(
                query=query,
                path=_display_path(scope),
                project_name=resolved_project,
                mode_requested=mode,
                mode_used="text",
                indexed=indexed,
                results=[_text_hit(item) for item in result.matches],
            )

        if mode == "semantic":
            try:
                results = qc.semantic_search(
                    query=query,
                    mode="hybrid",
                    limit=limit,
                    project_name=resolved_project,
                    path=scope,
                    path_prefix=path_prefix,
                )
            except Exception as exc:
                return SearchResponse(
                    query=query,
                    path=str(scope),
                    project_name=resolved_project,
                    mode_requested=mode,
                    mode_used="semantic",
                    indexed=indexed,
                    error=_friendly_error_message(exc),
                    warnings=["Semantic retrieval requires a live vector index for the target project."],
                )
            return SearchResponse(
                query=query,
                path=_display_path(scope),
                project_name=resolved_project,
                mode_requested=mode,
                mode_used="semantic",
                indexed=indexed,
                results=[_semantic_hit(item) for item in results],
            )

        if mode == "bundle":
            try:
                payload = qc.semantic_search_bundle(
                    query=query,
                    mode="hybrid",
                    limit=limit,
                    project_name=resolved_project,
                    path=scope,
                    path_prefix=path_prefix,
                    related_file_limit=max(limit, 6),
                )
            except Exception as exc:
                return SearchResponse(
                    query=query,
                    path=str(scope),
                    project_name=resolved_project,
                    mode_requested=mode,
                    mode_used="bundle",
                    indexed=indexed,
                    error=_friendly_error_message(exc),
                    warnings=["Bundle retrieval requires a live vector index for the target project."],
                )
            return SearchResponse(
                query=query,
                path=_display_path(scope),
                project_name=resolved_project,
                mode_requested=mode,
                mode_used=str(payload.get("mode", "bundle")),
                indexed=indexed,
                results=[_semantic_hit(item) for item in payload.get("results", [])],
                related_files=[_related_file_view(item) for item in payload.get("related_files", [])],
                related_callers=[_related_caller_view(item) for item in payload.get("related_callers", [])],
            )

        payload = qc.retrieve_context_auto(
            query=query,
            project_name=resolved_project,
            path=scope,
            limit=limit,
            path_prefix=path_prefix,
            related_file_limit=max(limit, 6),
        )
        warnings: list[str] = []
        if payload.get("mode") == "text_primary" and not indexed:
            warnings.append("Vector index not available for this project, so auto mode used Rust text retrieval.")
        return SearchResponse(
            query=query,
            path=_display_path(scope),
            project_name=resolved_project,
            mode_requested=mode,
            mode_used=str(payload.get("mode", "auto")),
            indexed=indexed,
            results=[_semantic_hit(item) for item in payload.get("results", [])],
            related_files=[_related_file_view(item) for item in payload.get("related_files", [])],
            related_callers=[_related_caller_view(item) for item in payload.get("related_callers", [])],
            warnings=warnings,
        )


def _project_info_impl(*, path: str, project_name: str | None) -> ProjectInfoResponse:
    scope = _normalize_scope_path(path)
    anchor = _project_anchor(scope)
    config = _load_config()
    resolved_project = detect_project_name(anchor, manual_override=project_name)
    qdrant_available = _qdrant_available(config)

    with QuickContext(config) as qc:
        status = qc.status()
        cache = qc.cache_status(anchor)
        summary = _project_summary_for_path(anchor, resolved_project, config)

    active_runs = [
        _index_run_view(record)
        for record in _INDEX_REGISTRY.list_active_for_target(path=str(anchor), project_name=resolved_project)
    ]

    return ProjectInfoResponse(
        path=_display_path(anchor),
        project_name=resolved_project,
        indexed=bool(summary and summary.indexed),
        qdrant_enabled=config.qdrant is not None,
        qdrant_available=qdrant_available,
        parser_connected=bool(status.get("parser", {}).get("connected", False)),
        cache_entries=int(cache.get("entries", 0)),
        cache_disk_bytes=int(cache.get("disk_bytes", 0)),
        cache_path=str(cache.get("cache_path", "")),
        collection=summary,
        active_index_runs=active_runs,
        folders=_list_project_folders(anchor),
    )


def _list_projects_impl() -> ListProjectsResponse:
    config = _load_config()
    catalog = sorted(_project_catalog(config).values(), key=lambda item: item.project_name.lower())
    return ListProjectsResponse(
        qdrant_available=_qdrant_available(config),
        projects=catalog,
        active_index_runs=[
            _index_run_view(record)
            for record in _INDEX_REGISTRY.list_recent(limit=32)
            if record.status in {"queued", "running"}
        ],
    )


def _grep_impl(
    *,
    query: str,
    path: str,
    limit: int,
    before_context: int,
    after_context: int,
) -> GrepResponse:
    scope = _normalize_scope_path(path)
    config = _load_config()
    with QuickContext(config) as qc:
        result = qc.grep_text(
            query=query,
            path=scope,
            limit=limit,
            before_context=before_context,
            after_context=after_context,
        )
    return GrepResponse(
        query=query,
        path=_display_path(scope),
        searched_files=int(result.searched_files),
        duration_ms=int(result.duration_ms),
        truncated=bool(result.truncated),
        matches=[
            GrepHit(
                file_path=_display_path(item.file_path),
                line_number=item.line_number,
                column_start=item.column_start,
                column_end=item.column_end,
                line=item.line,
                context_before=list(item.context_before),
                context_after=list(item.context_after),
            )
            for item in result.matches
        ],
    )


def _symbol_lookup_impl(*, query: str, path: str, project_name: str | None, limit: int) -> SymbolLookupResponse:
    scope = _normalize_scope_path(path)
    anchor = _project_anchor(scope)
    config = _load_config()
    resolved_project = detect_project_name(anchor, manual_override=project_name)
    with QuickContext(config) as qc:
        result = qc.symbol_lookup(
            query=query,
            path=scope,
            limit=limit,
            intent_mode=True,
            intent_level=2,
        )
    return SymbolLookupResponse(
        query=query,
        path=_display_path(scope),
        project_name=resolved_project,
        indexed_files=int(result.indexed_files),
        indexed_symbols=int(result.indexed_symbols),
        from_cache=bool(result.from_cache),
        results=[
            SymbolHit(
                name=item.name,
                kind=item.kind,
                language=item.language,
                file_path=_display_path(item.file_path),
                line_start=item.line_start,
                line_end=item.line_end,
                parent=item.parent,
                signature=item.signature,
            )
            for item in result.results
        ],
    )


def _index_impl(*, path: str, project_name: str | None, reindex: bool, fast: bool) -> IndexRunView:
    scope = _normalize_scope_path(path)
    if not scope.exists():
        failed = _IndexRunRecord(
            run_id=uuid.uuid4().hex[:12],
            key="missing",
            path=str(scope),
            project_name=project_name or detect_project_name(scope.parent if scope.parent.exists() else Path.cwd(), manual_override=None),
            reindex=reindex,
            fast=fast,
            status="failed",
            created_at=_now_iso(),
            finished_at=_now_iso(),
            error=f"Path does not exist: {_display_path(scope)}",
            message="Index failed",
        )
        return _index_run_view(failed)

    anchor = _project_anchor(scope)
    resolved_project = detect_project_name(anchor, manual_override=project_name)
    record, attached = _INDEX_REGISTRY.start_or_attach(
        path=str(anchor),
        project_name=resolved_project,
        reindex=reindex,
        fast=fast,
    )
    if attached:
        return _index_run_view(record, attached=True, duplicate_of=record.run_id)

    _INDEX_REGISTRY.mark_running(record.run_id, "Indexing directory")
    config = _load_config()
    try:
        with QuickContext(config) as qc:
            stats = qc.index_directory(
                directory=anchor,
                force_refresh=reindex,
                show_progress=False,
                project_name=resolved_project,
                fast=fast,
            )
    except Exception as exc:
        _INDEX_REGISTRY.mark_failed(record.run_id, _friendly_error_message(exc))
        current = _INDEX_REGISTRY.get_by_run_id(record.run_id)
        return _index_run_view(current or record)

    _INDEX_REGISTRY.mark_completed(record.run_id, asdict(stats))
    current = _INDEX_REGISTRY.get_by_run_id(record.run_id)
    return _index_run_view(current or record)


mcp = FastMCP(
    name="quickcontext",
    instructions=SERVER_INSTRUCTIONS,
)
server = mcp
app = mcp


@mcp.tool(
    name="project_info",
    description="Inspect one local project path, including detected project name, current index state, cache state, active index runs, and useful folder scopes for narrowing later searches.",
    tags={"projects", "read", "discovery"},
    annotations={
        "title": "Project Info",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    meta={
        "recommended": True,
        "domain": "code-retrieval",
        "pathScoped": True,
    },
)
async def project_info(
    path: Annotated[str, Field(description="Absolute or relative file or directory path to inspect.")] = ".",
    project_name: Annotated[str | None, Field(description="Optional project collection override.")] = None,
    ctx: Context = CurrentContext(),
) -> ProjectInfoResponse:
    await ctx.info(f"Inspecting project state for {path}")
    return await asyncio.to_thread(_project_info_impl, path=path, project_name=project_name)


@mcp.tool(
    name="list_projects",
    description="List currently indexed QuickContext projects visible through Qdrant, plus any active indexing runs.",
    tags={"projects", "read", "discovery"},
    annotations={
        "title": "List Indexed Projects",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    meta={
        "recommended": True,
        "domain": "code-retrieval",
    },
)
async def list_projects(ctx: Context = CurrentContext()) -> ListProjectsResponse:
    await ctx.info("Listing indexed projects")
    return await asyncio.to_thread(_list_projects_impl)


@mcp.tool(
    name="index",
    description="Index a local project directory for semantic retrieval. Repeated calls for the same project/path attach to the existing active run instead of starting another index operation.",
    tags={"indexing", "write"},
    annotations={
        "title": "Index Project",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    meta={
        "recommended": True,
        "domain": "code-retrieval",
        "pathScoped": True,
        "duplicateSuppression": True,
    },
)
async def index(
    path: Annotated[str, Field(description="Absolute or relative directory path to index.")] = ".",
    reindex: Annotated[bool, Field(description="When true, force-refresh already indexed files instead of trusting unchanged-file cache.")] = True,
    fast: Annotated[bool, Field(description="When true, skip description generation and use the fast indexing profile.")] = True,
    project_name: Annotated[str | None, Field(description="Optional project collection override.")] = None,
    ctx: Context = CurrentContext(),
) -> IndexRunView:
    await ctx.info(f"Index request for {path}")
    await ctx.report_progress(0, 2)
    await ctx.report_progress(1, 2)
    result = await asyncio.to_thread(
        _index_impl,
        path=path,
        project_name=project_name,
        reindex=reindex,
        fast=fast,
    )
    await ctx.report_progress(2, 2)
    return result


@mcp.tool(
    name="index_status",
    description="Check the current or last known indexing run by run ID or by path/project. Call this after `index` when you want to inspect state without starting new work.",
    tags={"indexing", "read"},
    annotations={
        "title": "Index Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    meta={
        "domain": "code-retrieval",
        "pathScoped": True,
    },
)
async def index_status(
    run_id: Annotated[str | None, Field(description="Specific QuickContext MCP run ID to inspect.")] = None,
    path: Annotated[str | None, Field(description="Project path to inspect when run_id is not known.")] = None,
    project_name: Annotated[str | None, Field(description="Optional project collection override used with `path`.")] = None,
    ctx: Context = CurrentContext(),
) -> IndexStatusResponse:
    await ctx.info("Inspecting index run state")

    if run_id:
        record = _INDEX_REGISTRY.get_by_run_id(run_id)
        return IndexStatusResponse(runs=[] if record is None else [_index_run_view(record)])

    if path:
        scope = _normalize_scope_path(path)
        anchor = _project_anchor(scope)
        resolved_project = detect_project_name(anchor, manual_override=project_name)
        return IndexStatusResponse(
            runs=[
                _index_run_view(record)
                for record in _INDEX_REGISTRY.get_for_target(path=str(anchor), project_name=resolved_project)
            ]
        )

    return IndexStatusResponse(runs=[_index_run_view(record) for record in _INDEX_REGISTRY.list_recent()])


@mcp.tool(
    name="search",
    description="Search indexed code and related context. Use mode='auto' as the default AI-facing retrieval route, mode='text' for lexical Rust text search, mode='semantic' for direct vector retrieval, and mode='bundle' for deeper cross-file semantic expansion.",
    tags={"search", "read"},
    annotations={
        "title": "Search Codebase",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    meta={
        "recommended": True,
        "domain": "code-retrieval",
        "pathScoped": True,
    },
)
async def search(
    query: Annotated[str, Field(description="Natural-language or code-oriented query to search for.")],
    path: Annotated[str, Field(description="Absolute or relative repository root, directory, or file scope to search.")] = ".",
    mode: Annotated[SearchMode, Field(description="Retrieval strategy: auto, text, semantic, or bundle.")] = "auto",
    project_name: Annotated[str | None, Field(description="Optional project collection override.")] = None,
    path_prefix: Annotated[str | None, Field(description="Optional folder prefix inside the project for narrower search scope.")] = None,
    limit: Annotated[int, Field(description="Maximum number of primary results to return.", ge=1, le=20)] = 8,
    ctx: Context = CurrentContext(),
) -> SearchResponse:
    await ctx.info(f"Searching {path} with mode={mode}")
    return await asyncio.to_thread(
        _search_impl,
        query=query,
        path=path,
        project_name=project_name,
        mode=mode,
        limit=limit,
        path_prefix=path_prefix,
    )


@mcp.tool(
    name="grep",
    description="Run exact literal grep through the Rust service. Use this for exact text, filenames, log messages, or copy-pasted code fragments when semantic retrieval is not the right fit.",
    tags={"search", "read", "lexical"},
    annotations={
        "title": "Literal Grep",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    meta={
        "domain": "code-retrieval",
        "pathScoped": True,
    },
)
async def grep(
    query: Annotated[str, Field(description="Exact literal text to grep for.")],
    path: Annotated[str, Field(description="Absolute or relative repository root, directory, or file scope to grep.")] = ".",
    limit: Annotated[int, Field(description="Maximum number of matches to return.", ge=1, le=200)] = 40,
    before_context: Annotated[int, Field(description="Number of leading context lines to include before each match.", ge=0, le=10)] = 1,
    after_context: Annotated[int, Field(description="Number of trailing context lines to include after each match.", ge=0, le=10)] = 1,
    ctx: Context = CurrentContext(),
) -> GrepResponse:
    await ctx.info(f"Grep search for {query}")
    return await asyncio.to_thread(
        _grep_impl,
        query=query,
        path=path,
        limit=limit,
        before_context=before_context,
        after_context=after_context,
    )


@mcp.tool(
    name="symbol_lookup",
    description="Look up exact or near-exact symbols through the Rust symbol index. Use this before semantic search when the question names a specific function, class, method, or identifier.",
    tags={"search", "read", "symbols"},
    annotations={
        "title": "Symbol Lookup",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    meta={
        "domain": "code-retrieval",
        "pathScoped": True,
    },
)
async def symbol_lookup(
    query: Annotated[str, Field(description="Symbol or identifier to look up.")],
    path: Annotated[str, Field(description="Absolute or relative repository root, directory, or file scope to inspect.")] = ".",
    project_name: Annotated[str | None, Field(description="Optional project collection override.")] = None,
    limit: Annotated[int, Field(description="Maximum number of symbols to return.", ge=1, le=50)] = 12,
    ctx: Context = CurrentContext(),
) -> SymbolLookupResponse:
    await ctx.info(f"Looking up symbol {query}")
    return await asyncio.to_thread(
        _symbol_lookup_impl,
        query=query,
        path=path,
        project_name=project_name,
        limit=limit,
    )


@mcp.resource(
    "quickcontext://capabilities",
    title="QuickContext Capabilities",
    description="Stable capability summary for the QuickContext MCP wrapper.",
    tags={"capabilities", "discovery"},
    meta={"domain": "code-retrieval"},
)
def capabilities_resource() -> CapabilitiesResource:
    return CapabilitiesResource()


@mcp.resource(
    "quickcontext://projects",
    title="Indexed Projects",
    description="Indexed QuickContext projects currently visible through the active configuration.",
    tags={"projects", "discovery"},
    meta={"domain": "code-retrieval"},
)
def projects_resource() -> ProjectsResource:
    return ProjectsResource(projects=_list_projects_impl().projects)


@mcp.resource(
    "quickcontext://jobs",
    title="Index Jobs",
    description="Recent QuickContext indexing runs tracked by the MCP wrapper.",
    tags={"indexing", "discovery"},
    meta={"domain": "code-retrieval"},
)
def jobs_resource() -> IndexJobsResource:
    return IndexJobsResource(runs=[_index_run_view(record) for record in _INDEX_REGISTRY.list_recent()])


@mcp.prompt(
    name="quickcontext_search_playbook",
    description="Reusable instructions for deciding which QuickContext MCP tool to use for a codebase task.",
    tags={"guidance", "search"},
    meta={"domain": "code-retrieval"},
)
def quickcontext_search_playbook(
    goal: Annotated[str, Field(description="The coding or code-understanding goal the assistant is trying to solve.")],
    path: Annotated[str, Field(description="Absolute or relative project path the assistant is working in.")] = ".",
    project_name: Annotated[str | None, Field(description="Optional project collection override.")] = None,
) -> str:
    resolved_path = str(_project_anchor(_normalize_scope_path(path)))
    resolved_project = detect_project_name(Path(resolved_path), manual_override=project_name)
    return (
        f"Goal: {goal}\n"
        f"Project path: {resolved_path}\n"
        f"Project name: {resolved_project}\n\n"
        "Playbook:\n"
        "1. Call project_info first to confirm index state and useful folder scopes.\n"
        "2. If semantic retrieval is needed and the project is not indexed, call index.\n"
        "3. Use search(mode='auto') as the default retrieval tool.\n"
        "4. Use grep for exact literals and symbol_lookup for exact identifiers.\n"
        "5. Use search(mode='bundle') for architectural questions that likely need related files.\n"
        "6. Keep follow-up searches path-scoped when the codebase is large or noisy."
    )


def main() -> None:
    transport = os.environ.get("QC_MCP_TRANSPORT", "stdio").strip().lower() or "stdio"
    if transport == "streamable-http":
        transport = "http"

    run_kwargs: dict[str, Any] = {}
    if transport in {"http", "sse"}:
        run_kwargs["host"] = os.environ.get("QC_MCP_HOST", "127.0.0.1")
        run_kwargs["port"] = int(os.environ.get("QC_MCP_PORT", "8000"))
        run_kwargs["path"] = os.environ.get("QC_MCP_HTTP_PATH", "/mcp/")
        run_kwargs["stateless_http"] = _env_flag("QC_MCP_STATELESS_HTTP", False)

    mcp.run(transport=transport, **run_kwargs)
