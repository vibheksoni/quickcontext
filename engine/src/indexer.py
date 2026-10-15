from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time
import uuid

from engine.src.chunker import CodeChunk
from engine.src.embedder import EmbeddedChunk
from engine.src.vector_store import PointRecord, VectorStore


@dataclass(frozen=True, slots=True)
class IndexStats:
    """
    Statistics for indexing operation.

    Args:
        total_chunks: Total chunks processed
        upserted_points: Points successfully written to the vector store
        failed_points: Points that failed to upsert
        total_tokens: Total tokens used for descriptions
        duration_seconds: Total indexing duration
        llm_cost_usd: Cost in USD for LLM description generation
        embedding_cost_usd: Cost in USD for embedding generation
        skipped_small_chunks: Chunks skipped by minimum chunk size filter
        skipped_minified_chunks: Chunks skipped by minified/noise filter
        skipped_capped_chunks: Chunks dropped by max-per-file cap
        files_capped: Number of files affected by max-per-file cap
        descriptions_enabled: Whether LLM description generation was enabled
        embedding_requests: Number of embedding requests issued
        embedding_retries: Number of embedding retries performed
        embedding_failed_requests: Number of permanently failed embedding requests
        embedding_input_count: Number of texts embedded across both embedding spaces
        embedding_stage_duration_seconds: Time spent inside embedder execution
        embedding_final_batch_size: Final adaptive embedding batch size used
        embedding_batch_shrink_events: Number of adaptive batch-size decreases
        embedding_batch_grow_events: Number of adaptive batch-size increases
        scan_stage_duration_seconds: Time spent discovering supported files
        artifact_profile_stage_duration_seconds: Time spent profiling large artifact candidates
        extract_stage_duration_seconds: Time spent in Rust extraction
        chunk_build_stage_duration_seconds: Time spent converting extraction results into chunks
        filter_stage_duration_seconds: Time spent filtering and capping chunks
        dedup_stage_duration_seconds: Time spent deduplicating chunks before embedding
        description_stage_duration_seconds: Time spent generating descriptions
        point_build_stage_duration_seconds: Time spent converting embedded chunks into store points
        upsert_stage_duration_seconds: Time spent writing point batches to the vector store
    """
    total_chunks: int
    upserted_points: int
    failed_points: int
    total_tokens: int
    duration_seconds: float
    llm_cost_usd: float = 0.0
    embedding_cost_usd: float = 0.0
    skipped_small_chunks: int = 0
    skipped_minified_chunks: int = 0
    skipped_capped_chunks: int = 0
    files_capped: int = 0
    descriptions_enabled: bool = True
    embedding_requests: int = 0
    embedding_retries: int = 0
    embedding_failed_requests: int = 0
    embedding_input_count: int = 0
    embedding_stage_duration_seconds: float = 0.0
    embedding_final_batch_size: int = 0
    embedding_batch_shrink_events: int = 0
    embedding_batch_grow_events: int = 0
    scan_stage_duration_seconds: float = 0.0
    artifact_profile_stage_duration_seconds: float = 0.0
    extract_stage_duration_seconds: float = 0.0
    chunk_build_stage_duration_seconds: float = 0.0
    filter_stage_duration_seconds: float = 0.0
    dedup_stage_duration_seconds: float = 0.0
    description_stage_duration_seconds: float = 0.0
    point_build_stage_duration_seconds: float = 0.0
    upsert_stage_duration_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class IndexedFileState:
    file_hash: Optional[str]
    point_count: int


def chunk_point_id(chunk_id: str) -> str:
    """
    Deterministic store point ID for a chunk ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class ChunkIndexer:
    """
    Batched writer of embedded code chunks into one store collection.

    Each batch is a single SQLite transaction carrying both vector spaces
    (code + description) and the structured chunk payload.
    """

    def __init__(
        self,
        store: VectorStore,
        collection_name: str,
        batch_size: int = 512,
    ):
        """
        store: VectorStore — Open vector store.
        collection_name: str — Target collection or project alias.
        batch_size: int — Points per write transaction.
        """
        self._store = store
        self._collection_name = collection_name
        self._batch_size = max(1, int(batch_size))

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def upsert_chunks(self, embedded_chunks: list[EmbeddedChunk], progress_callback=None) -> IndexStats:
        """
        Write embedded chunks in batches.

        embedded_chunks: list[EmbeddedChunk] — Chunks with code and description vectors.
        progress_callback: Callable[[int, int, int], None] | None — (written, total, failed).
        Returns: IndexStats — Operation metrics.
        """
        start_time = time.time()
        total_chunks = len(embedded_chunks)
        total_tokens = sum(chunk.description.token_count for chunk in embedded_chunks)

        point_build_started = time.time()
        points = [self._chunk_to_point(chunk) for chunk in embedded_chunks]
        point_build_duration = time.time() - point_build_started

        upserted = 0
        failed = 0
        upsert_started = time.time()
        for offset in range(0, total_chunks, self._batch_size):
            batch = points[offset:offset + self._batch_size]
            try:
                upserted += self._store.upsert(self._collection_name, batch)
            except Exception as exc:
                failed += len(batch)
                print(f"Batch upsert failed: {exc}")
            if progress_callback is not None:
                progress_callback(upserted, total_chunks, failed)

        finished = time.time()
        return IndexStats(
            total_chunks=total_chunks,
            upserted_points=upserted,
            failed_points=failed,
            total_tokens=total_tokens,
            duration_seconds=finished - start_time,
            point_build_stage_duration_seconds=point_build_duration,
            upsert_stage_duration_seconds=finished - upsert_started,
        )

    def _chunk_to_point(self, chunk: EmbeddedChunk) -> PointRecord:
        """
        Convert an embedded chunk to a store point.
        """
        code_chunk = chunk.chunk
        payload = {
            "chunk_id": code_chunk.chunk_id,
            "language": code_chunk.language,
            "file_path": code_chunk.file_path,
            "symbol_name": code_chunk.symbol_name,
            "symbol_kind": code_chunk.symbol_kind,
            "symbol_type": code_chunk.symbol_kind,
            "line_start": code_chunk.line_start,
            "line_end": code_chunk.line_end,
            "byte_start": code_chunk.byte_start,
            "byte_end": code_chunk.byte_end,
            "file_hash": code_chunk.file_hash,
            "description": chunk.description.description,
            "keywords": chunk.description.keywords,
            "source": code_chunk.source,
            "path_context": self._build_path_context(code_chunk.file_path),
        }
        for key in ("signature", "docstring", "parent", "visibility", "role"):
            value = getattr(code_chunk, key, None)
            if value:
                payload[key] = value

        return PointRecord(
            id=chunk_point_id(chunk.chunk_id),
            payload=payload,
            vectors={
                "code": chunk.code_vector,
                "description": chunk.desc_vector,
            },
        )

    def _build_path_context(self, file_path: str) -> str:
        """
        Build lightweight path context string for retrieval-time boosting.

        file_path: str — Absolute or relative file path.
        Returns: str — Slash-joined trailing path segments.
        """
        parts = [segment for segment in Path(file_path).parts if segment not in ("", "/", "\\")]
        if not parts:
            return ""
        return " / ".join(parts[-5:])

    def delete_by_file(self, file_path: str) -> None:
        """
        Delete all chunks for one file.
        """
        self._store.delete_by_files(self._collection_name, [file_path])

    def delete_by_files(self, file_paths: list[str]) -> None:
        """
        Delete all chunks for a batch of files.
        """
        if file_paths:
            self._store.delete_by_files(self._collection_name, file_paths)

    def delete_by_chunk_ids(self, point_ids: list[str]) -> None:
        """
        Delete specific points by store point ID.
        """
        if point_ids:
            self._store.delete_points(self._collection_name, point_ids)

    def delete_by_identities(self, identities: list) -> None:
        """
        Delete chunks by their ChunkIdentity objects.
        """
        self.delete_by_chunk_ids([chunk_point_id(identity.chunk_id) for identity in identities])

    def get_file_hashes(self, file_paths: list[str]) -> dict[str, Optional[str]]:
        """
        Return {file_path: indexed file hash or None} for change detection.
        """
        hashes: dict[str, Optional[str]] = {path: None for path in file_paths}
        if not file_paths:
            return hashes
        try:
            states = self._store.file_states(self._collection_name, file_paths)
        except Exception:
            return hashes
        for path, (file_hash, _) in states.items():
            hashes[path] = file_hash
        return hashes

    def get_file_index_state(self, file_paths: list[str]) -> dict[str, IndexedFileState]:
        """
        Return indexed file hash and point count per file.
        """
        if not file_paths:
            return {}
        try:
            states = self._store.file_states(self._collection_name, file_paths)
        except Exception:
            states = {}
        return {
            path: IndexedFileState(
                file_hash=states.get(path, (None, 0))[0],
                point_count=states.get(path, (None, 0))[1],
            )
            for path in file_paths
        }

    def get_chunks_by_file(self, file_path: str) -> list[CodeChunk]:
        """
        Fetch all stored chunks for one file.
        """
        return self.get_chunks_by_files([file_path]).get(file_path, [])

    def get_chunks_by_files(self, file_paths: list[str]) -> dict[str, list[CodeChunk]]:
        """
        Fetch all stored chunks for a batch of files, grouped by file path.
        """
        grouped: dict[str, list[CodeChunk]] = {file_path: [] for file_path in file_paths}
        if not file_paths:
            return grouped
        try:
            records = self._store.records_by_files(self._collection_name, file_paths)
        except Exception as exc:
            print(f"Failed to fetch chunks: {exc}")
            return grouped

        for record in records:
            payload = record.payload
            file_path = payload.get("file_path")
            if file_path not in grouped:
                continue
            grouped[file_path].append(CodeChunk(
                chunk_id=payload.get("chunk_id") or self._legacy_chunk_id_from_payload(payload),
                language=payload.get("language", "unknown"),
                file_path=file_path,
                symbol_name=payload["symbol_name"],
                symbol_kind=payload["symbol_kind"],
                line_start=payload["line_start"],
                line_end=payload["line_end"],
                byte_start=payload["byte_start"],
                byte_end=payload["byte_end"],
                file_hash=payload["file_hash"],
                source=payload.get("source", ""),
                signature=payload.get("signature"),
                docstring=payload.get("docstring"),
                parent=payload.get("parent"),
                visibility=payload.get("visibility"),
                role=payload.get("role"),
            ))
        return grouped

    def _legacy_chunk_id_from_payload(self, payload: dict) -> str:
        parent_str = f":{payload.get('parent')}" if payload.get("parent") else ""
        return f"{payload['file_path']}:{payload['symbol_name']}:{payload['symbol_kind']}{parent_str}"
