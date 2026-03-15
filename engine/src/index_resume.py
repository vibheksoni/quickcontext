from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ResumeFile:
    file_path: str
    file_hash: str | None
    chunk_count: int


@dataclass(frozen=True)
class ResumeState:
    manifest_version: int
    project_name: str
    directory: str
    shadow_collection: str
    source_collection: str | None
    settings_fingerprint: str
    copied_points: bool
    deleted_changed_paths: bool
    files: list[ResumeFile]
    delete_only_files: list[str]


def manifest_path(directory: str | Path, project_name: str) -> Path:
    root = Path(directory).resolve()
    state_dir = root / ".quickcontext"
    state_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(project_name.encode("utf-8")).hexdigest()[:12]
    return state_dir / f"index_resume_{digest}.json"


def load_state(directory: str | Path, project_name: str) -> ResumeState | None:
    path = manifest_path(directory, project_name)
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_version") != _MANIFEST_VERSION:
        return None

    files = [
        ResumeFile(
            file_path=str(item["file_path"]),
            file_hash=item.get("file_hash"),
            chunk_count=int(item["chunk_count"]),
        )
        for item in payload.get("files", [])
    ]
    return ResumeState(
        manifest_version=_MANIFEST_VERSION,
        project_name=str(payload["project_name"]),
        directory=str(payload["directory"]),
        shadow_collection=str(payload["shadow_collection"]),
        source_collection=payload.get("source_collection"),
        settings_fingerprint=str(payload["settings_fingerprint"]),
        copied_points=bool(payload.get("copied_points", False)),
        deleted_changed_paths=bool(payload.get("deleted_changed_paths", False)),
        files=files,
        delete_only_files=sorted(str(item) for item in payload.get("delete_only_files", [])),
    )


def save_state(directory: str | Path, state: ResumeState) -> None:
    path = manifest_path(directory, state.project_name)
    payload: dict[str, Any] = asdict(state)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def clear_state(directory: str | Path, project_name: str) -> None:
    path = manifest_path(directory, project_name)
    if path.exists():
        path.unlink()


def build_settings_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
