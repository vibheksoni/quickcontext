from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CACHE_DIR = ".quickcontext"
CACHE_FILENAME = "file_cache.json"
CACHE_VERSION = 1


@dataclass(slots=True)
class FileSignature:
    """
    Cached file signature for change detection.

    mtime: int — File modification time as Unix epoch seconds.
    size: int — File size in bytes.
    file_hash: str — SHA256 hex digest of file content.
    cached_at: float — Timestamp when this entry was cached.
    """

    mtime: int
    size: int
    file_hash: str
    cached_at: float

    def to_dict(self) -> dict:
        return {
            "mtime": self.mtime,
            "size": self.size,
            "file_hash": self.file_hash,
            "cached_at": self.cached_at,
        }

    @staticmethod
    def from_dict(data: dict) -> "FileSignature":
        return FileSignature(
            mtime=int(data["mtime"]),
            size=int(data["size"]),
            file_hash=str(data["file_hash"]),
            cached_at=float(data.get("cached_at", 0.0)),
        )


class FileSignatureCache:
    """
    Mtime+size based file signature cache for skipping unchanged files.

    Stores (mtime, size, sha256_hash) per file path. When a file's mtime and size
    match the cached values, we know the content hash hasn't changed — no need to
    re-read, re-parse, or re-hash the file.

    Persisted as JSON at <project_root>/.quickcontext/file_cache.json.

    _project_root: Path — Project root directory.
    _entries: dict[str, FileSignature] — Cached file signatures keyed by normalized path.
    _dirty: bool — True when in-memory state differs from disk.
    """

    def __init__(self, project_root: str | Path):
        """
        project_root: str | Path — Project root directory for cache storage.
        """
        self._project_root = Path(project_root).resolve()
        self._entries: dict[str, FileSignature] = {}
        self._dirty = False
        self._load()

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def project_root(self) -> Path:
        return self._project_root

    def _cache_path(self) -> Path:
        return self._project_root / CACHE_DIR / CACHE_FILENAME

    def _normalize_path(self, file_path: str | Path) -> str:
        return str(Path(file_path).resolve())

    def _load(self) -> None:
        """Load cache from disk. Silently starts empty on any error."""
        cache_file = self._cache_path()
        if not cache_file.exists():
            return

        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            if raw.get("version") != CACHE_VERSION:
                return
            entries = raw.get("entries", {})
            for path_key, entry_data in entries.items():
                self._entries[path_key] = FileSignature.from_dict(entry_data)
        except Exception:
            self._entries.clear()

    def save(self) -> None:
        """Persist cache to disk if dirty."""
        if not self._dirty:
            return

        cache_file = self._cache_path()
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": CACHE_VERSION,
            "project_root": str(self._project_root),
            "saved_at": time.time(),
            "entries": {k: v.to_dict() for k, v in self._entries.items()},
        }

        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        tmp.replace(cache_file)
        self._dirty = False

    def get(self, file_path: str | Path) -> Optional[FileSignature]:
        """
        Look up cached signature for a file.

        file_path: str | Path — Absolute or relative file path.
        Returns: Optional[FileSignature] — Cached signature or None.
        """
        key = self._normalize_path(file_path)
        return self._entries.get(key)

    def put(self, file_path: str | Path, mtime: int, size: int, file_hash: str) -> None:
        """
        Store or update a file signature in the cache.

        file_path: str | Path — File path.
        mtime: int — File mtime as Unix epoch seconds.
        size: int — File size in bytes.
        file_hash: str — SHA256 hex digest of file content.
        """
        key = self._normalize_path(file_path)
        self._entries[key] = FileSignature(
            mtime=mtime,
            size=size,
            file_hash=file_hash,
            cached_at=time.time(),
        )
        self._dirty = True

    def remove(self, file_path: str | Path) -> bool:
        """
        Remove a file from the cache.

        file_path: str | Path — File path to remove.
        Returns: bool — True if the entry existed and was removed.
        """
        key = self._normalize_path(file_path)
        if key in self._entries:
            del self._entries[key]
            self._dirty = True
            return True
        return False

    def is_unchanged(self, file_path: str | Path) -> Optional[str]:
        """
        Check if a file is unchanged based on mtime+size.

        file_path: str | Path — File path to check.
        Returns: Optional[str] — Cached file_hash if unchanged, None if changed or not cached.
        """
        key = self._normalize_path(file_path)
        cached = self._entries.get(key)
        if cached is None:
            return None

        try:
            stat = os.stat(file_path)
        except OSError:
            return None

        current_mtime = int(stat.st_mtime)
        current_size = stat.st_size

        if current_mtime == cached.mtime and current_size == cached.size:
            return cached.file_hash

        return None

    def is_unchanged_from_metadata(
        self,
        file_path: str | Path,
        file_size: Optional[int],
        file_mtime: Optional[int],
    ) -> Optional[str]:
        """
        Check if a file is unchanged using externally provided metadata.

        file_path: str | Path — File path to check.
        file_size: Optional[int] — Current file size in bytes.
        file_mtime: Optional[int] — Current file mtime as Unix epoch seconds.
        Returns: Optional[str] — Cached file_hash if unchanged, None otherwise.
        """
        if file_size is None or file_mtime is None:
            return None

        key = self._normalize_path(file_path)
        cached = self._entries.get(key)
        if cached is None:
            return None

        if int(file_mtime) == cached.mtime and int(file_size) == cached.size:
            return cached.file_hash

        return None

    def update_from_extraction(self, file_path: str, file_hash: Optional[str], file_size: Optional[int], file_mtime: Optional[int]) -> None:
        """
        Update cache from Rust extraction result metadata.

        file_path: str — File path from extraction result.
        file_hash: Optional[str] — SHA256 hash computed by Rust.
        file_size: Optional[int] — File size from Rust.
        file_mtime: Optional[int] — File mtime from Rust.
        """
        if file_hash is None or file_size is None or file_mtime is None:
            return
        self.put(file_path, file_mtime, file_size, file_hash)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._entries.clear()
        self._dirty = True

    def stats(self) -> dict:
        """
        Returns: dict — Cache statistics.
        """
        cache_file = self._cache_path()
        disk_size = cache_file.stat().st_size if cache_file.exists() else 0

        return {
            "project_root": str(self._project_root),
            "entries": len(self._entries),
            "disk_bytes": disk_size,
            "cache_path": str(cache_file),
        }
