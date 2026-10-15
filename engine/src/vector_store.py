"""
Embedded vector store: SQLite for durable storage, numpy for exact in-memory search.

No server process is involved. Each collection's vectors are loaded once into a
contiguous float32 matrix and kept current through a per-collection revision
counter, so searches cost one tiny revision lookup plus one matrix product.
Writes from this or any other process are picked up incrementally via row
revisions and deletion tombstones instead of full reloads.
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Iterator, Mapping, Optional, Sequence
import uuid

import numpy as np

from engine.src.config import DEFAULT_KEYWORD_FIELDS, CollectionVectorConfig, StoreConfig
from engine.src.keywords import STOP_WORDS


SCHEMA_VERSION = 1
SUPPORTED_DISTANCES = ("cosine", "dot", "euclid")
TOMBSTONE_PRUNE_THRESHOLD = 50_000
MASK_CACHE_SIZE = 64
_SQL_CHUNK = 500
_FTS_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS collections (
    name TEXT PRIMARY KEY,
    uid TEXT NOT NULL,
    kind TEXT NOT NULL,
    vectors TEXT NOT NULL,
    keyword_fields TEXT NOT NULL,
    text_field TEXT NOT NULL,
    fulltext INTEGER NOT NULL DEFAULT 0,
    revision INTEGER NOT NULL DEFAULT 0,
    tombstone_floor INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS aliases (
    alias TEXT PRIMARY KEY,
    collection TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS points (
    collection TEXT NOT NULL,
    id TEXT NOT NULL,
    rev INTEGER NOT NULL,
    file_path TEXT,
    file_hash TEXT,
    path_key TEXT,
    keys TEXT,
    payload TEXT NOT NULL,
    body TEXT,
    vectors BLOB,
    UNIQUE (collection, id)
);
CREATE INDEX IF NOT EXISTS points_by_file ON points (collection, file_path);
CREATE INDEX IF NOT EXISTS points_by_rev ON points (collection, rev);
CREATE TABLE IF NOT EXISTS tombstones (
    collection TEXT NOT NULL,
    id TEXT NOT NULL,
    rev INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS tombstones_by_rev ON tombstones (collection, rev);
CREATE VIRTUAL TABLE IF NOT EXISTS points_fts USING fts5 (body, tokenize = 'unicode61 remove_diacritics 2');
"""


class CollectionNotFoundError(KeyError):
    """
    Raised when a collection or alias does not exist.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name

    def __str__(self) -> str:
        return f"Collection not found: {self.name}"


def normalize_path_key(path: str | None) -> str:
    """
    Normalize a file path for case-insensitive, separator-agnostic prefix matching.

    The key is wrapped in slashes so a scope like "engine/src" matches any
    whole-segment run inside an absolute path.
    """
    if not path:
        return ""
    normalized = str(path).replace("\\\\?\\", "").replace("\\", "/").strip("/").lower()
    return f"/{normalized}/" if normalized else ""


def normalize_path_prefix(prefix: str | None) -> str | None:
    """
    Normalize a user-supplied path scope to the form used by normalize_path_key.
    """
    if not prefix:
        return None
    normalized = str(prefix).replace("\\", "/").strip().strip("/").lower()
    return normalized or None


@dataclass(frozen=True, slots=True)
class SearchFilter:
    """
    Conjunctive filter over keyword fields and an optional path scope.

    equals: tuple[(field, value)] — value may be a tuple meaning "any of".
        List-valued fields (e.g. tags) match when any element matches.
    path_prefix: Optional[str] — Normalized path scope (see normalize_path_prefix).
    """

    equals: tuple[tuple[str, Any], ...] = ()
    path_prefix: Optional[str] = None

    @staticmethod
    def build(path_prefix: str | None = None, **equals: Any) -> Optional["SearchFilter"]:
        """
        Build a filter from keyword arguments, dropping None values.

        Returns None when nothing would be filtered.
        """
        items = []
        for key, value in sorted(equals.items()):
            if value is None:
                continue
            if isinstance(value, (list, tuple, set, frozenset)):
                value = tuple(sorted(str(item) for item in value))
                if not value:
                    continue
            items.append((key, value))
        prefix = normalize_path_prefix(path_prefix)
        if not items and prefix is None:
            return None
        return SearchFilter(equals=tuple(items), path_prefix=prefix)

    def without_path(self) -> Optional["SearchFilter"]:
        """
        Return the same filter with the path scope removed, or None if empty.
        """
        if not self.equals:
            return None
        return SearchFilter(equals=self.equals, path_prefix=None)


@dataclass(frozen=True, slots=True)
class PointRecord:
    """
    One point to write.

    id: str — Stable point identifier, unique within a collection.
    payload: dict — JSON-serializable metadata. The collection's text field is
        stored separately and only loaded when explicitly requested.
    vectors: Mapping[str, Sequence[float]] — One vector per collection vector space.
    search_text: Optional[str] — Text to index for full-text search instead of the
        text field alone (e.g. text plus its source and tags: "contextual BM25").
    """

    id: str
    payload: dict
    vectors: Mapping[str, Any] = field(default_factory=dict)
    search_text: Optional[str] = None


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """
    One vector search against a named vector space.
    """

    vector: Any
    using: str
    limit: int
    filter: Optional[SearchFilter] = None
    with_payload: bool | Sequence[str] = True


@dataclass(frozen=True, slots=True)
class SearchPoint:
    id: str
    score: float
    payload: dict


@dataclass(frozen=True, slots=True)
class SearchRecord:
    id: str
    payload: dict


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    name: str
    uid: str
    kind: str
    vectors: tuple[CollectionVectorConfig, ...]
    keyword_fields: tuple[str, ...]
    text_field: str
    fulltext: bool
    revision: int
    created_at: float
    updated_at: float
    points_count: int = 0

    def vector_spec(self, name: str) -> CollectionVectorConfig | None:
        for spec in self.vectors:
            if spec.name == name:
                return spec
        return None

    def to_dict(self, alias: str | None = None) -> dict:
        """
        Collection metadata in the shape the SDK and CLI report.
        """
        return {
            "name": alias or self.name,
            "real_collection": self.name,
            "kind": self.kind,
            "points_count": self.points_count,
            "indexed_vectors_count": self.points_count if self.vectors else 0,
            "segments_count": 1,
            "status": "green",
            "fulltext": self.fulltext,
            "revision": self.revision,
            "vectors": {
                spec.name: {"size": spec.dimension, "distance": spec.distance}
                for spec in self.vectors
            },
        }


class _Snapshot:
    """
    Immutable in-memory view of one collection at one revision.

    Vector rows live in growable buffers; a snapshot only ever reads rows
    [0, size). Deleted or replaced rows stay in place but are masked out via
    `alive` until enough of them accumulate to justify compaction, which keeps
    incremental updates proportional to the number of changed rows.
    """

    __slots__ = (
        "uid",
        "revision",
        "spec",
        "ids",
        "index",
        "buffers",
        "matrices",
        "alive",
        "dead",
        "keys_raw",
        "path_keys",
        "_parsed_keys",
        "_fields",
        "_masks",
    )

    def __init__(
        self,
        uid: str,
        revision: int,
        spec: tuple[CollectionVectorConfig, ...],
        ids: list[str],
        index: dict[str, int],
        buffers: dict[str, np.ndarray],
        alive: np.ndarray | None,
        dead: int,
        keys_raw: list[str | None],
        path_keys: list[str],
    ):
        self.uid = uid
        self.revision = revision
        self.spec = spec
        self.ids = ids
        self.index = index
        self.buffers = buffers
        self.matrices = {name: buffer[:len(ids)] for name, buffer in buffers.items()}
        self.alive = alive
        self.dead = dead
        self.keys_raw = keys_raw
        self.path_keys = path_keys
        self._parsed_keys: list[dict | None] | None = None
        self._fields: dict[str, tuple[np.ndarray, bool]] = {}
        self._masks: OrderedDict[SearchFilter | None, np.ndarray] = OrderedDict()

    @property
    def size(self) -> int:
        return len(self.ids)

    @property
    def live_count(self) -> int:
        return len(self.ids) - self.dead

    def inherit_fields(self, previous: "_Snapshot", appended_keys: list[str | None]) -> None:
        """
        Reuse keyword-field arrays already decoded by the previous snapshot,
        decoding only the appended rows.
        """
        if not previous._fields or previous.size + len(appended_keys) != self.size:
            return
        parsed = json.loads("[" + ",".join(raw or "null" for raw in appended_keys) + "]")
        for name, (old_values, old_multi) in previous._fields.items():
            tail, multi = _decode_field(parsed, name)
            self._fields[name] = (np.concatenate([old_values, tail]), old_multi or multi)

    def field_values(self, name: str) -> tuple[np.ndarray, bool]:
        """
        Return (values, multi) for a keyword field; multi is True when any row holds a list.
        """
        cached = self._fields.get(name)
        if cached is not None:
            return cached
        if self._parsed_keys is None:
            self._parsed_keys = json.loads("[" + ",".join(raw or "null" for raw in self.keys_raw) + "]")
        values, multi = _decode_field(self._parsed_keys, name)
        self._fields[name] = (values, multi)
        return values, multi

    def mask(self, flt: SearchFilter | None) -> np.ndarray | None:
        """
        Boolean row mask for live rows matching the filter, or None for "all rows".
        """
        if flt is None and self.alive is None:
            return None
        cached = self._masks.get(flt)
        if cached is not None:
            self._masks.move_to_end(flt)
            return cached

        mask = np.ones(self.size, dtype=bool) if self.alive is None else self.alive.copy()
        if flt is not None:
            for name, expected in flt.equals:
                values, multi = self.field_values(name)
                options = expected if isinstance(expected, tuple) else (expected,)
                if multi:
                    allowed = {str(option) for option in options}
                    mask &= np.fromiter(
                        (_value_matches(value, allowed) for value in values),
                        dtype=bool,
                        count=self.size,
                    )
                else:
                    matched = np.zeros(self.size, dtype=bool)
                    for option in options:
                        matched |= values == str(option)
                    mask &= matched
            if flt.path_prefix:
                needle = f"/{flt.path_prefix}/"
                mask &= np.fromiter((needle in key for key in self.path_keys), dtype=bool, count=self.size)

        self._masks[flt] = mask
        while len(self._masks) > MASK_CACHE_SIZE:
            self._masks.popitem(last=False)
        return mask


def _decode_field(parsed_keys: list[dict | None], name: str) -> tuple[np.ndarray, bool]:
    values = np.empty(len(parsed_keys), dtype=object)
    multi = False
    for idx, keys in enumerate(parsed_keys):
        if keys:
            value = keys.get(name)
            if isinstance(value, list):
                multi = True
                values[idx] = tuple(str(item) for item in value)
            elif value is not None:
                values[idx] = str(value)
    return values, multi


def _value_matches(value: Any, allowed: set) -> bool:
    if value is None:
        return False
    if isinstance(value, tuple):
        return any(item in allowed for item in value)
    return value in allowed


def _is_keyword_value(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool)):
        return True
    return isinstance(value, (list, tuple)) and all(isinstance(item, (str, int, float, bool)) for item in value)


def _keyword_values(payload: dict, keyword_fields: Sequence[str]) -> dict:
    """
    Extract filterable values from a payload; "*" selects every scalar or
    scalar-list top-level field.
    """
    if "*" in keyword_fields:
        return {name: value for name, value in payload.items() if value is not None and _is_keyword_value(value)}
    return {
        name: payload[name]
        for name in keyword_fields
        if payload.get(name) is not None and _is_keyword_value(payload[name])
    }


def _fts_query(text: str) -> str | None:
    """
    Build an OR query of distinct terms. Stopwords are dropped (unless nothing else
    remains) because they carry almost no BM25 weight but match most documents,
    which is what dominates full-text latency.
    """
    tokens: list[str] = []
    for token in _FTS_TOKEN_RE.findall(text or ""):
        lowered = token.lower()
        if len(lowered) > 1 and lowered not in tokens:
            tokens.append(lowered)
    if not tokens:
        return None
    content = [token for token in tokens if token not in STOP_WORDS]
    return " OR ".join(f'"{token}"' for token in (content or tokens)[:32])


def _chunks(items: Sequence[Any], size: int = _SQL_CHUNK) -> Iterator[Sequence[Any]]:
    for offset in range(0, len(items), size):
        yield items[offset:offset + size]


class VectorStore:
    """
    Thread-safe embedded vector store shared by indexing, search, and agent memory.

    One SQLite connection per store instance (serialized with a lock); WAL mode
    lets other processes read while one writes.
    """

    def __init__(self, config: StoreConfig | None = None):
        self._config = config or StoreConfig()
        self._path = self._config.resolved_path
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._snapshots: OrderedDict[str, _Snapshot] = OrderedDict()
        self._snapshot_lock = threading.Lock()
        self._load_locks: dict[str, threading.Lock] = {}

    @property
    def path(self) -> str:
        return self._path

    @property
    def config(self) -> StoreConfig:
        return self._config

    def open(self) -> "VectorStore":
        """
        Open the database (creating it and its schema if needed).
        """
        self._connection()
        return self

    def is_open(self) -> bool:
        return self._conn is not None

    def ping(self) -> bool:
        """
        Return True when the database can be opened and queried.
        """
        try:
            with self._lock:
                self._connection().execute("SELECT 1").fetchone()
            return True
        except Exception:
            return False

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None
        with self._snapshot_lock:
            self._snapshots.clear()

    def __enter__(self) -> "VectorStore":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"VectorStore(path={self._path!r}, open={self.is_open()})"

    def _connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        with self._lock:
            if self._conn is not None:
                return self._conn
            if self._path != ":memory:":
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                self._path,
                timeout=max(0.0, float(self._config.busy_timeout_seconds)),
                isolation_level=None,
                check_same_thread=False,
            )
            conn.execute("PRAGMA page_size=16384")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA wal_autocheckpoint=8192")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-65536")
            conn.execute("PRAGMA mmap_size=268435456")
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn = conn
            return conn

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = self._connection()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def _read(self, sql: str, params: Sequence[Any] = ()) -> list[tuple]:
        with self._lock:
            return self._connection().execute(sql, params).fetchall()

    def _row_to_info(self, row: tuple, points_count: int = 0) -> CollectionInfo:
        (name, uid, kind, vectors, keyword_fields, text_field, fulltext, revision, _floor, created_at, updated_at) = row
        return CollectionInfo(
            name=name,
            uid=uid,
            kind=kind,
            vectors=tuple(CollectionVectorConfig(**item) for item in json.loads(vectors)),
            keyword_fields=tuple(json.loads(keyword_fields)),
            text_field=text_field,
            fulltext=bool(fulltext),
            revision=int(revision),
            created_at=float(created_at),
            updated_at=float(updated_at),
            points_count=points_count,
        )

    _COLLECTION_COLUMNS = (
        "name, uid, kind, vectors, keyword_fields, text_field, fulltext, "
        "revision, tombstone_floor, created_at, updated_at"
    )

    def _collection_row(self, conn: sqlite3.Connection, name: str) -> tuple | None:
        return conn.execute(
            f"SELECT {self._COLLECTION_COLUMNS} FROM collections WHERE name = ?",
            (name,),
        ).fetchone()

    def _require(self, conn: sqlite3.Connection, name: str) -> CollectionInfo:
        row = self._collection_row(conn, name)
        if row is None:
            raise CollectionNotFoundError(name)
        return self._row_to_info(row)

    def resolve(self, name: str) -> str:
        """
        Resolve an alias to its collection; plain collection names resolve to themselves.
        """
        rows = self._read("SELECT collection FROM aliases WHERE alias = ?", (name,))
        return rows[0][0] if rows else name

    def create_collection(
        self,
        name: str,
        vectors: Sequence[CollectionVectorConfig] = (),
        *,
        kind: str = "code",
        keyword_fields: Sequence[str] = DEFAULT_KEYWORD_FIELDS,
        text_field: str = "source",
        fulltext: bool = False,
    ) -> CollectionInfo:
        """
        Create a collection. Raises ValueError if the name or alias already exists.
        """
        names = [spec.name for spec in vectors]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate vector space names: {names}")
        for spec in vectors:
            if spec.distance not in SUPPORTED_DISTANCES:
                raise ValueError(
                    f"Unknown distance metric: {spec.distance!r}. Supported: {list(SUPPORTED_DISTANCES)}"
                )
            if int(spec.dimension) <= 0:
                raise ValueError(f"Vector space {spec.name!r} must have a positive dimension")

        now = time.time()
        with self._write() as conn:
            if self._collection_row(conn, name) is not None:
                raise ValueError(f"Collection already exists: {name}")
            if conn.execute("SELECT 1 FROM aliases WHERE alias = ?", (name,)).fetchone():
                raise ValueError(f"Name is already used by an alias: {name}")
            conn.execute(
                f"INSERT INTO collections ({self._COLLECTION_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)",
                (
                    name,
                    uuid.uuid4().hex,
                    kind,
                    json.dumps([
                        {"name": spec.name, "dimension": int(spec.dimension), "distance": spec.distance}
                        for spec in vectors
                    ]),
                    json.dumps(list(keyword_fields)),
                    text_field,
                    1 if fulltext else 0,
                    now,
                    now,
                ),
            )
            return self._require(conn, name)

    def get_collection(self, name: str, *, with_count: bool = True) -> CollectionInfo | None:
        """
        Return collection metadata by collection name or alias, or None.
        """
        real = self.resolve(name)
        with self._lock:
            conn = self._connection()
            row = self._collection_row(conn, real)
            if row is None:
                return None
            count = 0
            if with_count:
                count = conn.execute("SELECT COUNT(*) FROM points WHERE collection = ?", (real,)).fetchone()[0]
            return self._row_to_info(row, points_count=int(count))

    def collection_exists(self, name: str) -> bool:
        """
        Return True if a real collection (not an alias) with this name exists.
        """
        return bool(self._read("SELECT 1 FROM collections WHERE name = ?", (name,)))

    def list_collections(self) -> list[CollectionInfo]:
        with self._lock:
            conn = self._connection()
            counts = dict(conn.execute("SELECT collection, COUNT(*) FROM points GROUP BY collection").fetchall())
            rows = conn.execute(f"SELECT {self._COLLECTION_COLUMNS} FROM collections ORDER BY name").fetchall()
        return [self._row_to_info(row, points_count=int(counts.get(row[0], 0))) for row in rows]

    def delete_collection(self, name: str) -> bool:
        """
        Delete a real collection, its points, full-text rows, and any aliases to it.
        """
        with self._write() as conn:
            if self._collection_row(conn, name) is None:
                return False
            conn.execute(
                "DELETE FROM points_fts WHERE rowid IN (SELECT rowid FROM points WHERE collection = ?)",
                (name,),
            )
            conn.execute("DELETE FROM points WHERE collection = ?", (name,))
            conn.execute("DELETE FROM tombstones WHERE collection = ?", (name,))
            conn.execute("DELETE FROM aliases WHERE collection = ?", (name,))
            conn.execute("DELETE FROM collections WHERE name = ?", (name,))
        self._drop_snapshot(name)
        return True

    def list_aliases(self) -> dict[str, str]:
        return dict(self._read("SELECT alias, collection FROM aliases ORDER BY alias"))

    def get_alias(self, alias: str) -> str | None:
        rows = self._read("SELECT collection FROM aliases WHERE alias = ?", (alias,))
        return rows[0][0] if rows else None

    def set_alias(self, alias: str, collection: str) -> str | None:
        """
        Atomically point an alias at a collection. Returns the previous target.
        """
        with self._write() as conn:
            if self._collection_row(conn, collection) is None:
                raise CollectionNotFoundError(collection)
            if self._collection_row(conn, alias) is not None:
                raise ValueError(f"Alias name collides with a collection: {alias}")
            previous = conn.execute("SELECT collection FROM aliases WHERE alias = ?", (alias,)).fetchone()
            conn.execute("INSERT OR REPLACE INTO aliases (alias, collection) VALUES (?, ?)", (alias, collection))
        return previous[0] if previous else None

    def delete_alias(self, alias: str) -> bool:
        with self._write() as conn:
            cursor = conn.execute("DELETE FROM aliases WHERE alias = ?", (alias,))
            return cursor.rowcount > 0

    def _bump_revision(self, conn: sqlite3.Connection, name: str) -> int:
        conn.execute(
            "UPDATE collections SET revision = revision + 1, updated_at = ? WHERE name = ?",
            (time.time(), name),
        )
        return int(conn.execute("SELECT revision FROM collections WHERE name = ?", (name,)).fetchone()[0])

    def _encode_vectors(self, info: CollectionInfo, points: Sequence[PointRecord]) -> list[bytes | None]:
        if not info.vectors:
            return [None] * len(points)
        blocks = []
        for spec in info.vectors:
            try:
                matrix = np.asarray([point.vectors[spec.name] for point in points], dtype=np.float32)
            except KeyError as exc:
                raise ValueError(f"Point is missing vector {spec.name!r} for collection {info.name!r}") from exc
            if matrix.ndim != 2 or matrix.shape[1] != spec.dimension:
                got = matrix.shape[1] if matrix.ndim == 2 else matrix.shape
                raise ValueError(
                    f"Vector {spec.name!r} has dimension {got}, but collection {info.name!r} "
                    f"expects {spec.dimension}. Re-index with a matching embedding config."
                )
            if spec.distance == "cosine":
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                matrix = matrix / norms
            blocks.append(matrix)
        combined = np.ascontiguousarray(np.hstack(blocks), dtype=np.float32)
        return [row.tobytes() for row in combined]

    def upsert(self, collection: str, points: Sequence[PointRecord]) -> int:
        """
        Insert or replace points in one transaction. Returns the number written.
        """
        if not points:
            return 0
        real = self.resolve(collection)
        with self._lock:
            info = self._require(self._connection(), real)
        blobs = self._encode_vectors(info, points)

        rows = []
        for point, blob in zip(points, blobs):
            payload = dict(point.payload)
            body = payload.pop(info.text_field, None)
            file_path = payload.get("file_path")
            keys = _keyword_values(payload, info.keyword_fields)
            rows.append((
                real,
                str(point.id),
                file_path,
                payload.get("file_hash"),
                normalize_path_key(file_path),
                json.dumps(keys, separators=(",", ":")) if keys else None,
                json.dumps(payload, separators=(",", ":"), default=str),
                body if body is None else str(body),
                blob,
            ))

        with self._write() as conn:
            revision = self._bump_revision(conn, real)
            ids = [row[1] for row in rows]
            if info.fulltext:
                self._delete_fts_for_ids(conn, real, ids)
            conn.executemany(
                "INSERT OR REPLACE INTO points "
                "(collection, id, rev, file_path, file_hash, path_key, keys, payload, body, vectors) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(row[0], row[1], revision, *row[2:]) for row in rows],
            )
            if info.fulltext:
                fts_text = {
                    str(point.id): point.search_text if point.search_text is not None else row[7]
                    for point, row in zip(points, rows)
                }
                for batch in _chunks(ids):
                    placeholders = ",".join("?" * len(batch))
                    rowids = conn.execute(
                        f"SELECT rowid, id FROM points WHERE collection = ? AND id IN ({placeholders})",
                        (real, *batch),
                    ).fetchall()
                    conn.executemany(
                        "INSERT INTO points_fts (rowid, body) VALUES (?, ?)",
                        [(rowid, fts_text[point_id]) for rowid, point_id in rowids if fts_text.get(point_id)],
                    )
        return len(rows)

    def _delete_fts_for_ids(self, conn: sqlite3.Connection, real: str, ids: Sequence[str]) -> None:
        for batch in _chunks(ids):
            placeholders = ",".join("?" * len(batch))
            conn.execute(
                "DELETE FROM points_fts WHERE rowid IN "
                f"(SELECT rowid FROM points WHERE collection = ? AND id IN ({placeholders}))",
                (real, *batch),
            )

    def _delete_where(self, collection: str, where: str, params: Sequence[Any]) -> int:
        real = self.resolve(collection)
        with self._write() as conn:
            info = self._require(conn, real)
            revision = self._bump_revision(conn, real)
            conn.execute(
                f"INSERT INTO tombstones (collection, id, rev) SELECT collection, id, ? FROM points "
                f"WHERE collection = ? AND {where}",
                (revision, real, *params),
            )
            if info.fulltext:
                conn.execute(
                    f"DELETE FROM points_fts WHERE rowid IN (SELECT rowid FROM points WHERE collection = ? AND {where})",
                    (real, *params),
                )
            deleted = conn.execute(
                f"DELETE FROM points WHERE collection = ? AND {where}",
                (real, *params),
            ).rowcount
            self._maybe_prune_tombstones(conn, real, revision)
        return int(deleted)

    def _maybe_prune_tombstones(self, conn: sqlite3.Connection, real: str, revision: int) -> None:
        count = conn.execute("SELECT COUNT(*) FROM tombstones WHERE collection = ?", (real,)).fetchone()[0]
        if count <= TOMBSTONE_PRUNE_THRESHOLD:
            return
        conn.execute("DELETE FROM tombstones WHERE collection = ?", (real,))
        conn.execute("UPDATE collections SET tombstone_floor = ? WHERE name = ?", (revision, real))

    def delete_points(self, collection: str, ids: Sequence[str]) -> int:
        total = 0
        id_list = [str(item) for item in ids]
        for batch in _chunks(id_list):
            placeholders = ",".join("?" * len(batch))
            total += self._delete_where(collection, f"id IN ({placeholders})", batch)
        return total

    def delete_by_files(self, collection: str, file_paths: Sequence[str]) -> int:
        total = 0
        paths = [str(item) for item in file_paths]
        for batch in _chunks(paths):
            placeholders = ",".join("?" * len(batch))
            total += self._delete_where(collection, f"file_path IN ({placeholders})", batch)
        return total

    def delete_by_filter(self, collection: str, flt: SearchFilter) -> int:
        where, params = self._filter_sql(flt)
        if not where:
            raise ValueError("Refusing to delete with an empty filter; use delete_collection instead")
        return self._delete_where(collection, where, params)

    def _filter_sql(self, flt: SearchFilter | None) -> tuple[str, list[Any]]:
        if flt is None:
            return "", []
        clauses: list[str] = []
        params: list[Any] = []
        for name, expected in flt.equals:
            values = expected if isinstance(expected, tuple) else (expected,)
            placeholders = ",".join("?" * len(values))
            clauses.append(
                "EXISTS (SELECT 1 FROM json_each(points.keys, ?) AS item "
                f"WHERE CAST(item.value AS TEXT) IN ({placeholders}))"
            )
            params.append(f'$."{name}"')
            params.extend(str(value) for value in values)
        if flt.path_prefix:
            clauses.append("instr(points.path_key, ?) > 0")
            params.append(f"/{flt.path_prefix}/")
        return " AND ".join(clauses), params

    def copy_points(self, source: str, target: str) -> int:
        """
        Copy every point from one collection into another with an identical layout.
        """
        source_real = self.resolve(source)
        target_real = self.resolve(target)
        with self._write() as conn:
            source_info = self._require(conn, source_real)
            target_info = self._require(conn, target_real)
            if source_info.vectors != target_info.vectors or source_info.text_field != target_info.text_field:
                raise ValueError(
                    f"Cannot copy points: {source_real!r} and {target_real!r} have different vector layouts"
                )
            revision = self._bump_revision(conn, target_real)
            copied = conn.execute(
                "INSERT OR REPLACE INTO points "
                "(collection, id, rev, file_path, file_hash, path_key, keys, payload, body, vectors) "
                "SELECT ?, id, ?, file_path, file_hash, path_key, keys, payload, body, vectors "
                "FROM points WHERE collection = ?",
                (target_real, revision, source_real),
            ).rowcount
            if target_info.fulltext:
                conn.execute(
                    "INSERT INTO points_fts (rowid, body) SELECT rowid, body FROM points "
                    "WHERE collection = ? AND body IS NOT NULL",
                    (target_real,),
                )
        return int(copied)

    def file_states(self, collection: str, file_paths: Sequence[str]) -> dict[str, tuple[str | None, int]]:
        """
        Return {file_path: (file_hash, point_count)} for files that have points.
        """
        real = self.resolve(collection)
        out: dict[str, tuple[str | None, int]] = {}
        paths = list(dict.fromkeys(str(item) for item in file_paths))
        for batch in _chunks(paths):
            placeholders = ",".join("?" * len(batch))
            for file_path, file_hash, count in self._read(
                "SELECT file_path, MAX(file_hash), COUNT(*) FROM points "
                f"WHERE collection = ? AND file_path IN ({placeholders}) GROUP BY file_path",
                (real, *batch),
            ):
                out[file_path] = (file_hash, int(count))
        return out

    def records_by_files(self, collection: str, file_paths: Sequence[str]) -> list[SearchRecord]:
        """
        Return full records (payload plus text field) for every point in the given files.
        """
        real = self.resolve(collection)
        info = self.get_collection(real, with_count=False)
        if info is None:
            return []
        records: list[SearchRecord] = []
        paths = list(dict.fromkeys(str(item) for item in file_paths))
        for batch in _chunks(paths):
            placeholders = ",".join("?" * len(batch))
            for point_id, payload, body in self._read(
                "SELECT id, payload, body FROM points "
                f"WHERE collection = ? AND file_path IN ({placeholders}) ORDER BY rowid",
                (real, *batch),
            ):
                records.append(SearchRecord(id=point_id, payload=self._decode_payload(payload, body, info, True)))
        return records

    def scroll(
        self,
        collection: str,
        *,
        flt: SearchFilter | None = None,
        limit: int = 100,
        offset: int = 0,
        with_payload: bool | Sequence[str] = True,
    ) -> list[SearchRecord]:
        """
        Page through points in insertion order, optionally filtered.
        """
        real = self.resolve(collection)
        info = self.get_collection(real, with_count=False)
        if info is None:
            return []
        where, params = self._filter_sql(flt)
        include_body = self._wants_body(with_payload, info)
        rows = self._read(
            f"SELECT id, payload, {'body' if include_body else 'NULL'} FROM points "
            f"WHERE collection = ?{' AND ' + where if where else ''} ORDER BY rowid LIMIT ? OFFSET ?",
            (real, *params, max(0, int(limit)), max(0, int(offset))),
        )
        return [
            SearchRecord(id=point_id, payload=self._decode_payload(payload, body, info, with_payload))
            for point_id, payload, body in rows
        ]

    def count(self, collection: str, flt: SearchFilter | None = None) -> int:
        real = self.resolve(collection)
        where, params = self._filter_sql(flt)
        rows = self._read(
            f"SELECT COUNT(*) FROM points WHERE collection = ?{' AND ' + where if where else ''}",
            (real, *params),
        )
        return int(rows[0][0])

    def retrieve(
        self,
        collection: str,
        ids: Sequence[str],
        with_payload: bool | Sequence[str] = True,
    ) -> list[SearchRecord]:
        """
        Fetch points by ID, preserving the requested order and skipping missing IDs.
        """
        real = self.resolve(collection)
        info = self.get_collection(real, with_count=False)
        if info is None or not ids:
            return []
        found = self._fetch_payloads(real, info, [str(item) for item in ids], with_payload)
        return [SearchRecord(id=str(point_id), payload=found[str(point_id)]) for point_id in ids if str(point_id) in found]

    def _wants_body(self, with_payload: bool | Sequence[str], info: CollectionInfo) -> bool:
        if with_payload is True:
            return True
        if not with_payload:
            return False
        return info.text_field in with_payload

    def _decode_payload(
        self,
        payload_json: str,
        body: str | None,
        info: CollectionInfo,
        with_payload: bool | Sequence[str],
    ) -> dict:
        if not with_payload:
            return {}
        payload = json.loads(payload_json)
        if body is not None and self._wants_body(with_payload, info):
            payload[info.text_field] = body
        if with_payload is True:
            return payload
        return {key: payload[key] for key in with_payload if key in payload}

    def _fetch_payloads(
        self,
        real: str,
        info: CollectionInfo,
        ids: Sequence[str],
        with_payload: bool | Sequence[str],
    ) -> dict[str, dict]:
        if not with_payload:
            return {point_id: {} for point_id in ids}
        include_body = self._wants_body(with_payload, info)
        out: dict[str, dict] = {}
        unique = list(dict.fromkeys(ids))
        for batch in _chunks(unique):
            placeholders = ",".join("?" * len(batch))
            for point_id, payload, body in self._read(
                f"SELECT id, payload, {'body' if include_body else 'NULL'} FROM points "
                f"WHERE collection = ? AND id IN ({placeholders})",
                (real, *batch),
            ):
                out[point_id] = self._decode_payload(payload, body, info, with_payload)
        return out

    def _drop_snapshot(self, name: str) -> None:
        with self._snapshot_lock:
            self._snapshots.pop(name, None)

    def _load_lock(self, name: str) -> threading.Lock:
        with self._snapshot_lock:
            lock = self._load_locks.get(name)
            if lock is None:
                lock = threading.Lock()
                self._load_locks[name] = lock
            return lock

    def _snapshot(self, real: str) -> tuple[_Snapshot, CollectionInfo]:
        with self._load_lock(real):
            with self._lock:
                conn = self._connection()
                row = self._collection_row(conn, real)
                if row is None:
                    raise CollectionNotFoundError(real)
                info = self._row_to_info(row)
                floor = int(row[8])
                with self._snapshot_lock:
                    current = self._snapshots.get(real)
                if (
                    current is not None
                    and current.uid == info.uid
                    and current.revision == info.revision
                ):
                    with self._snapshot_lock:
                        self._snapshots.move_to_end(real)
                    return current, info

                if (
                    current is not None
                    and current.uid == info.uid
                    and current.spec == info.vectors
                    and current.revision >= floor
                ):
                    snapshot = self._apply_delta(conn, real, info, current)
                else:
                    snapshot = self._load_full(conn, real, info)

            with self._snapshot_lock:
                self._snapshots[real] = snapshot
                self._snapshots.move_to_end(real)
                while len(self._snapshots) > max(1, int(self._config.max_cached_collections)):
                    self._snapshots.popitem(last=False)
            return snapshot, info

    def _decode_rows(self, info: CollectionInfo, blobs: list[bytes | None], capacity: int) -> dict[str, np.ndarray]:
        """
        Decode concatenated per-point vector blobs into one buffer per vector space.
        """
        if not info.vectors:
            return {}
        total = sum(spec.dimension for spec in info.vectors)
        if blobs:
            combined = np.frombuffer(b"".join(blob or b"" for blob in blobs), dtype=np.float32).reshape(len(blobs), total)
        else:
            combined = np.empty((0, total), dtype=np.float32)
        buffers: dict[str, np.ndarray] = {}
        offset = 0
        for spec in info.vectors:
            buffer = np.empty((max(capacity, len(blobs)), spec.dimension), dtype=np.float32)
            buffer[:len(blobs)] = combined[:, offset:offset + spec.dimension]
            buffers[spec.name] = buffer
            offset += spec.dimension
        return buffers

    def _load_full(self, conn: sqlite3.Connection, real: str, info: CollectionInfo) -> _Snapshot:
        rows = conn.execute(
            "SELECT id, keys, path_key, vectors FROM points WHERE collection = ? ORDER BY rowid",
            (real,),
        ).fetchall()
        ids = [row[0] for row in rows]
        return _Snapshot(
            uid=info.uid,
            revision=info.revision,
            spec=info.vectors,
            ids=ids,
            index={point_id: idx for idx, point_id in enumerate(ids)},
            buffers=self._decode_rows(info, [row[3] for row in rows], len(rows) + max(256, len(rows) // 16)),
            alive=None,
            dead=0,
            keys_raw=[row[1] for row in rows],
            path_keys=[row[2] or "" for row in rows],
        )

    def _apply_delta(self, conn: sqlite3.Connection, real: str, info: CollectionInfo, current: _Snapshot) -> _Snapshot:
        changed = conn.execute(
            "SELECT id, keys, path_key, vectors FROM points WHERE collection = ? AND rev > ? ORDER BY rowid",
            (real, current.revision),
        ).fetchall()
        removed = {
            row[0]
            for row in conn.execute(
                "SELECT id FROM tombstones WHERE collection = ? AND rev > ?",
                (real, current.revision),
            ).fetchall()
        }
        removed.update(row[0] for row in changed)
        dead_rows = [current.index[point_id] for point_id in removed if point_id in current.index]

        old_size = current.size
        new_size = old_size + len(changed)
        dead = current.dead + len(dead_rows)

        alive = current.alive
        if dead_rows:
            alive = np.ones(old_size, dtype=bool) if alive is None else alive.copy()
            alive[dead_rows] = False
        if alive is not None and changed:
            alive = np.concatenate([alive, np.ones(len(changed), dtype=bool)])

        appended = self._decode_rows(info, [row[3] for row in changed], len(changed))
        buffers: dict[str, np.ndarray] = {}
        for name, buffer in current.buffers.items():
            if new_size > buffer.shape[0]:
                grown = np.empty((max(new_size + 1024, int(new_size * 1.25)), buffer.shape[1]), dtype=np.float32)
                grown[:old_size] = buffer[:old_size]
                buffer = grown
            if changed:
                buffer[old_size:new_size] = appended[name][:len(changed)]
            buffers[name] = buffer

        index = dict(current.index)
        for row in dead_rows:
            index.pop(current.ids[row], None)
        new_ids = [row[0] for row in changed]
        for offset, point_id in enumerate(new_ids):
            index[point_id] = old_size + offset

        snapshot = _Snapshot(
            uid=info.uid,
            revision=info.revision,
            spec=info.vectors,
            ids=current.ids + new_ids,
            index=index,
            buffers=buffers,
            alive=alive,
            dead=dead,
            keys_raw=current.keys_raw + [row[1] for row in changed],
            path_keys=current.path_keys + [row[2] or "" for row in changed],
        )
        snapshot.inherit_fields(current, [row[1] for row in changed])
        if dead > max(1024, new_size // 5):
            return self._compact(snapshot)
        return snapshot

    def _compact(self, snapshot: _Snapshot) -> _Snapshot:
        keep = np.flatnonzero(snapshot.alive) if snapshot.alive is not None else np.arange(snapshot.size)
        ids = [snapshot.ids[idx] for idx in keep]
        compacted = _Snapshot(
            uid=snapshot.uid,
            revision=snapshot.revision,
            spec=snapshot.spec,
            ids=ids,
            index={point_id: idx for idx, point_id in enumerate(ids)},
            buffers={name: matrix[keep] for name, matrix in snapshot.matrices.items()},
            alive=None,
            dead=0,
            keys_raw=[snapshot.keys_raw[idx] for idx in keep],
            path_keys=[snapshot.path_keys[idx] for idx in keep],
        )
        for name, (values, multi) in snapshot._fields.items():
            compacted._fields[name] = (values[keep], multi)
        return compacted

    def vectors(self, collection: str, using: str, ids: Sequence[str]) -> dict[str, np.ndarray]:
        """
        Return stored vectors (normalized for cosine spaces) for the given IDs from
        the in-memory snapshot; missing IDs are omitted.
        """
        real = self.resolve(collection)
        snapshot, info = self._snapshot(real)
        if info.vector_spec(using) is None:
            raise ValueError(f"Collection {info.name!r} has no vector space {using!r}")
        matrix = snapshot.matrices[using]
        out: dict[str, np.ndarray] = {}
        for point_id in ids:
            row = snapshot.index.get(str(point_id))
            if row is not None:
                out[str(point_id)] = matrix[row]
        return out

    def warm(self, collection: str) -> int:
        """
        Load a collection's vectors into memory ahead of the first query. Returns point count.
        """
        real = self.resolve(collection)
        info = self.get_collection(real, with_count=False)
        if info is None or not info.vectors:
            return 0
        snapshot, _ = self._snapshot(real)
        return snapshot.live_count

    def search(self, collection: str, request: SearchRequest) -> list[SearchPoint]:
        return self.search_batch(collection, [request])[0]

    def search_batch(self, collection: str, requests: Sequence[SearchRequest]) -> list[list[SearchPoint]]:
        """
        Run several exact vector searches against one collection snapshot.

        Raises: CollectionNotFoundError — If the collection or alias does not exist.
        """
        if not requests:
            return []
        real = self.resolve(collection)
        snapshot, info = self._snapshot(real)

        ranked: list[list[tuple[str, float]]] = [
            self._rank(snapshot, info, request) for request in requests
        ]

        wanted: dict[str, None] = {}
        merged_payload: bool | set[str] = False
        for request, hits in zip(requests, ranked):
            if not hits or not request.with_payload:
                continue
            for point_id, _ in hits:
                wanted[point_id] = None
            if request.with_payload is True or merged_payload is True:
                merged_payload = True
            else:
                merged_payload = (merged_payload or set()) | set(request.with_payload)

        payloads: dict[str, dict] = {}
        if wanted:
            fields: bool | list[str] = True if merged_payload is True else sorted(merged_payload)
            payloads = self._fetch_payloads(real, info, list(wanted), fields)

        out: list[list[SearchPoint]] = []
        for request, hits in zip(requests, ranked):
            results = []
            for point_id, score in hits:
                payload = payloads.get(point_id, {}) if request.with_payload else {}
                if request.with_payload and request.with_payload is not True:
                    payload = {key: payload[key] for key in request.with_payload if key in payload}
                results.append(SearchPoint(id=point_id, score=score, payload=payload))
            out.append(results)
        return out

    def _rank(self, snapshot: _Snapshot, info: CollectionInfo, request: SearchRequest) -> list[tuple[str, float]]:
        spec = info.vector_spec(request.using)
        if spec is None:
            raise ValueError(
                f"Collection {info.name!r} has no vector space {request.using!r} "
                f"(available: {[item.name for item in info.vectors]})"
            )
        query = np.asarray(request.vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != spec.dimension:
            raise ValueError(
                f"Query vector has dimension {query.shape[0]}, but {info.name!r}.{spec.name} "
                f"expects {spec.dimension}. Check that the embedding config matches the indexed collection."
            )
        limit = max(0, int(request.limit))
        if limit == 0 or snapshot.live_count == 0:
            return []

        if spec.distance == "cosine":
            norm = float(np.linalg.norm(query))
            if norm > 0:
                query = query / norm

        matrix = snapshot.matrices[spec.name]
        mask = snapshot.mask(request.filter)
        rows: np.ndarray | None = None
        if mask is not None:
            selected = int(np.count_nonzero(mask))
            if selected == 0:
                return []
            if selected * 4 < mask.shape[0]:
                rows = np.flatnonzero(mask)
                matrix = matrix[rows]
                mask = None

        if spec.distance == "euclid":
            scores = -np.sqrt(np.maximum(
                np.einsum("ij,ij->i", matrix, matrix) - 2.0 * (matrix @ query) + float(query @ query),
                0.0,
            ))
        else:
            scores = matrix @ query
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
            limit = min(limit, selected)

        count = scores.shape[0]
        if limit < count:
            top = np.argpartition(-scores, limit - 1)[:limit]
            top = top[np.argsort(-scores[top], kind="stable")]
        else:
            top = np.argsort(-scores, kind="stable")

        positions = rows[top] if rows is not None else top
        sign = -1.0 if spec.distance == "euclid" else 1.0
        return [(snapshot.ids[int(pos)], sign * float(scores[int(idx)])) for pos, idx in zip(positions, top)]

    def text_search(
        self,
        collection: str,
        query: str,
        limit: int = 10,
        flt: SearchFilter | None = None,
        with_payload: bool | Sequence[str] = True,
    ) -> list[SearchPoint]:
        """
        BM25 full-text search over a fulltext-enabled collection's text field.

        Scores are positive; higher is better.
        """
        real = self.resolve(collection)
        info = self.get_collection(real, with_count=False)
        match = _fts_query(query)
        if info is None or not info.fulltext or match is None or limit <= 0:
            return []
        where, params = self._filter_sql(flt)
        try:
            rows = self._read(
                "SELECT points.id, bm25(points_fts) AS rank FROM points_fts "
                "JOIN points ON points.rowid = points_fts.rowid "
                f"WHERE points_fts MATCH ? AND points.collection = ?{' AND ' + where if where else ''} "
                "ORDER BY rank LIMIT ?",
                (match, real, *params, int(limit)),
            )
        except sqlite3.OperationalError:
            return []
        payloads = self._fetch_payloads(real, info, [row[0] for row in rows], with_payload)
        return [
            SearchPoint(id=point_id, score=-float(rank), payload=payloads.get(point_id, {}))
            for point_id, rank in rows
        ]

    def size_bytes(self) -> int:
        if self._path == ":memory:":
            return 0
        total = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(self._path + suffix)
            if candidate.exists():
                total += candidate.stat().st_size
        return total

    def optimize(self, vacuum: bool = False) -> dict:
        """
        Merge full-text segments, refresh planner stats, checkpoint the WAL, and optionally VACUUM.
        """
        before = self.size_bytes()
        with self._lock:
            conn = self._connection()
            conn.execute("INSERT INTO points_fts (points_fts) VALUES ('optimize')")
            conn.execute("PRAGMA optimize")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            if vacuum:
                conn.execute("VACUUM")
        return {"size_before": before, "size_after": self.size_bytes()}

    def integrity_check(self) -> list[str]:
        """
        Return SQLite integrity problems; an empty list means healthy.
        """
        rows = self._read("PRAGMA quick_check")
        problems = [row[0] for row in rows if row[0] != "ok"]
        return problems


def open_store(config: StoreConfig | None = None) -> VectorStore:
    """
    Open a VectorStore, creating the database file on first use.
    """
    return VectorStore(config).open()


__all__ = [
    "CollectionInfo",
    "CollectionNotFoundError",
    "PointRecord",
    "SearchFilter",
    "SearchPoint",
    "SearchRecord",
    "SearchRequest",
    "VectorStore",
    "normalize_path_key",
    "normalize_path_prefix",
    "open_store",
]
