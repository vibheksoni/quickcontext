from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ProjectFolderInfo:
    relative_path: str
    absolute_path: str


@dataclass(frozen=True, slots=True)
class ProjectCollectionInfo:
    project_name: str
    indexed: bool
    real_collection: str | None = None
    points_count: int | None = None
    indexed_vectors_count: int | None = None
    segments_count: int | None = None
    status: str | None = None
    vectors: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProjectInfo:
    path: str
    project_name: str
    indexed: bool
    qdrant_enabled: bool
    qdrant_available: bool
    parser_connected: bool
    cache_entries: int
    cache_disk_bytes: int
    cache_path: str
    collection: ProjectCollectionInfo
    folders: list[ProjectFolderInfo] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class IndexOperationSnapshot:
    operation_id: str
    kind: str
    path: str
    project_name: str
    status: str
    current_stage: str
    message: str
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    files_discovered: int = 0
    files_unchanged: int = 0
    files_deleted: int = 0
    files_planned: int = 0
    files_analyzed: int = 0
    files_indexed: int = 0
    files_remaining: int = 0
    artifact_downgraded_files: int = 0
    symbols_extracted: int = 0
    chunks_built: int = 0
    chunks_kept: int = 0
    duplicate_chunks: int = 0
    description_chunks_completed: int = 0
    embedding_chunks_completed: int = 0
    points_upserted: int = 0
    points_failed: int = 0
    final_stats: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class LspInstallStepInfo:
    manager: str
    command: str
    note: str | None = None


@dataclass(frozen=True, slots=True)
class LspServerSetupInfo:
    name: str
    language_id: str
    binary: str
    installed: bool
    detection_reasons: list[str] = field(default_factory=list)
    auto_install_supported: bool = False
    install_steps: list[LspInstallStepInfo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LspSetupPlanInfo:
    target_path: str
    platform: str
    servers: list[LspServerSetupInfo] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LspServerCheckInfo:
    name: str
    language_id: str
    binary: str
    status: str
    installed: bool
    auto_install_supported: bool
    detection_reasons: list[str] = field(default_factory=list)
    install_steps: list[LspInstallStepInfo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    probe_command: str | None = None
    probe_message: str | None = None


@dataclass(frozen=True, slots=True)
class LspCheckPlanInfo:
    target_path: str
    platform: str
    servers: list[LspServerCheckInfo] = field(default_factory=list)
