from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time
import uuid

from qdrant_client import QdrantClient, models

from engine.src.embedder import EmbeddedChunk
from engine.src.chunker import CodeChunk


@dataclass(frozen=True, slots=True)
class IndexStats:
    """
    Statistics for indexing operation.

    Args:
        total_chunks: Total chunks processed
        upserted_points: Points successfully upserted to Qdrant
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


@dataclass(frozen=True, slots=True)
class IndexedFileState:
    file_hash: Optional[str]
    point_count: int


class QdrantIndexer:
    """
    Batched upserter for Qdrant with dual-vector support.

    Handles:
    - Batch upsert with configurable batch size
    - Optional concurrent upsert workers
    - Dual vector spaces (code + description)
    - Structured payloads with metadata
    - Error handling and retry logic
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        batch_size: int = 100,
        wait: bool = True,
        upsert_concurrency: int = 1,
    ):
        """
        Args:
            client: Qdrant client instance
            collection_name: Target collection name
            batch_size: Number of points per batch
            wait: Wait for indexing to complete before returning
            upsert_concurrency: Number of concurrent upsert workers
        """
        self._client = client
        self._collection_name = collection_name
        self._batch_size = max(1, int(batch_size))
        self._wait = wait
        self._upsert_concurrency = max(1, int(upsert_concurrency))

    def upsert_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> IndexStats:
        """
        Upsert embedded chunks to Qdrant in batches.

        Args:
            embedded_chunks: List of embedded chunks with dual vectors

        Returns:
            IndexStats with operation metrics
        """
        start_time = time.time()

        total_chunks = len(embedded_chunks)
        upserted = 0
        failed = 0
        total_tokens = sum(chunk.description.token_count for chunk in embedded_chunks)

        batches = [
            embedded_chunks[i:i + self._batch_size]
            for i in range(0, total_chunks, self._batch_size)
        ]

        if self._upsert_concurrency <= 1 or len(batches) <= 1:
            for batch in batches:
                points = [self._chunk_to_point(chunk) for chunk in batch]
                try:
                    self._client.upsert(
                        collection_name=self._collection_name,
                        points=points,
                        wait=self._wait,
                    )
                    upserted += len(points)
                except Exception as e:
                    failed += len(points)
                    print(f"Batch upsert failed: {e}")
        else:
            with ThreadPoolExecutor(max_workers=min(self._upsert_concurrency, len(batches))) as executor:
                futures = {
                    executor.submit(self._upsert_points, [self._chunk_to_point(chunk) for chunk in batch]): len(batch)
                    for batch in batches
                }
                for future in as_completed(futures):
                    batch_size = futures[future]
                    try:
                        future.result()
                        upserted += batch_size
                    except Exception as e:
                        failed += batch_size
                        print(f"Batch upsert failed: {e}")

        duration = time.time() - start_time

        return IndexStats(
            total_chunks=total_chunks,
            upserted_points=upserted,
            failed_points=failed,
            total_tokens=total_tokens,
            duration_seconds=duration,
        )

    def _upsert_points(self, points: list[models.PointStruct]) -> None:
        """
        Upsert one point batch to Qdrant.

        points: list[models.PointStruct] — Prepared Qdrant points.
        """
        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=self._wait,
        )

    def _chunk_to_point(self, chunk: EmbeddedChunk) -> models.PointStruct:
        """
        Convert embedded chunk to Qdrant point.

        Args:
            chunk: Embedded chunk with dual vectors

        Returns:
            PointStruct ready for upsert
        """
        payload = {
            "chunk_id": chunk.chunk.chunk_id,
            "language": chunk.chunk.language,
            "file_path": chunk.chunk.file_path,
            "symbol_name": chunk.chunk.symbol_name,
            "symbol_kind": chunk.chunk.symbol_kind,
            "symbol_type": chunk.chunk.symbol_kind,
            "line_start": chunk.chunk.line_start,
            "line_end": chunk.chunk.line_end,
            "byte_start": chunk.chunk.byte_start,
            "byte_end": chunk.chunk.byte_end,
            "file_hash": chunk.chunk.file_hash,
            "description": chunk.description.description,
            "keywords": chunk.description.keywords,
            "source": chunk.chunk.source,
            "path_context": self._build_path_context(chunk.chunk.file_path),
        }

        if chunk.chunk.signature:
            payload["signature"] = chunk.chunk.signature

        if chunk.chunk.docstring:
            payload["docstring"] = chunk.chunk.docstring

        if chunk.chunk.parent:
            payload["parent"] = chunk.chunk.parent

        if chunk.chunk.visibility:
            payload["visibility"] = chunk.chunk.visibility

        if chunk.chunk.role:
            payload["role"] = chunk.chunk.role

        path_prefixes = self._build_path_prefixes(chunk.chunk.file_path)
        if path_prefixes:
            payload["path_prefixes"] = path_prefixes

        point_id = uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)

        return models.PointStruct(
            id=str(point_id),
            vector={
                "code": chunk.code_vector,
                "description": chunk.desc_vector,
            },
            payload=payload,
        )

    def _build_path_context(self, file_path: str) -> str:
        """
        Build lightweight path context string for retrieval-time boosting.

        file_path: str — Absolute or relative file path.
        Returns: str — Slash-joined parent path segments.
        """
        parts = [segment for segment in Path(file_path).parts if segment not in ("", "/", "\\")]
        if not parts:
            return ""

        tail = parts[-5:]
        return " / ".join(tail)

    def _build_path_prefixes(self, file_path: str) -> list[str]:
        """
        Build normalized path prefixes for server-side path-scoped retrieval.

        file_path: str — Absolute or relative file path.
        Returns: list[str] — Incremental lowercase path prefixes.
        """
        normalized = file_path.replace("\\", "/").strip("/")
        if not normalized:
            return []

        parts = [segment for segment in normalized.split("/") if segment]
        prefixes: list[str] = []
        current: list[str] = []
        for part in parts[:-1]:
            current.append(part)
            prefixes.append("/".join(current).lower())
        return prefixes

    def delete_by_file(self, file_path: str) -> None:
        """
        Delete all chunks for a specific file.

        Args:
            file_path: Path to file whose chunks should be deleted
        """
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=file_path),
                        )
                    ]
                )
            ),
        )

    def delete_by_files(self, file_paths: list[str]) -> None:
        """
        Delete all chunks for a batch of files.

        file_paths: list[str] — File paths whose chunks should be deleted.
        """
        if not file_paths:
            return

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchAny(any=file_paths),
                        )
                    ]
                )
            ),
        )

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """
        Delete specific chunks by ID.

        chunk_ids: list[str] — List of chunk IDs to delete.
        """
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(
                points=chunk_ids,
            ),
        )

    def get_file_hashes(self, file_paths: list[str]) -> dict[str, Optional[str]]:
        """
        Get file hashes for change detection.

        Args:
            file_paths: List of file paths to check

        Returns:
            Dict mapping file_path to file_hash (None if not indexed)
        """
        hashes: dict[str, Optional[str]] = {path: None for path in file_paths}

        if not file_paths:
            return hashes

        batch_size = 256
        for offset in range(0, len(file_paths), batch_size):
            batch = file_paths[offset:offset + batch_size]
            try:
                results, _ = self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_path",
                                match=models.MatchAny(any=batch),
                            )
                        ]
                    ),
                    limit=len(batch),
                    with_payload=["file_path", "file_hash"],
                    with_vectors=False,
                )
            except Exception:
                continue

            for point in results:
                point_file_path = point.payload.get("file_path")
                if point_file_path in hashes and hashes[point_file_path] is None:
                    hashes[point_file_path] = point.payload.get("file_hash")

        return hashes

    def get_file_index_state(self, file_paths: list[str]) -> dict[str, IndexedFileState]:
        """
        Get lightweight indexing state for a set of files from the target collection.

        file_paths: list[str] — File paths to inspect.
        Returns: dict[str, IndexedFileState] — Indexed file hash and point count per file.
        """
        state: dict[str, dict[str, Optional[str] | int]] = {
            path: {"file_hash": None, "point_count": 0}
            for path in file_paths
        }
        if not file_paths:
            return {}

        batch_size = 128
        scroll_limit = 512
        for offset in range(0, len(file_paths), batch_size):
            batch = file_paths[offset:offset + batch_size]
            next_offset = None

            while True:
                try:
                    records, next_offset = self._client.scroll(
                        collection_name=self._collection_name,
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="file_path",
                                    match=models.MatchAny(any=batch),
                                )
                            ]
                        ),
                        limit=scroll_limit,
                        offset=next_offset,
                        with_payload=["file_path", "file_hash"],
                        with_vectors=False,
                    )
                except Exception:
                    break

                if not records:
                    break

                for point in records:
                    point_file_path = point.payload.get("file_path")
                    if point_file_path not in state:
                        continue
                    state[point_file_path]["point_count"] = int(state[point_file_path]["point_count"]) + 1
                    if state[point_file_path]["file_hash"] is None:
                        state[point_file_path]["file_hash"] = point.payload.get("file_hash")

                if next_offset is None:
                    break

        return {
            path: IndexedFileState(
                file_hash=item["file_hash"],
                point_count=int(item["point_count"]),
            )
            for path, item in state.items()
        }

    def get_chunks_by_file(self, file_path: str) -> list[CodeChunk]:
        """
        Fetch all existing chunks for a file from Qdrant.

        file_path: str — Path to file whose chunks to fetch.
        Returns: list[CodeChunk] — Existing chunks for the file.
        """
        chunks: list[CodeChunk] = []

        try:
            results = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path",
                            match=models.MatchValue(value=file_path),
                        )
                    ]
                ),
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )

            for point in results[0]:
                payload = point.payload
                chunk_id = payload.get("chunk_id") or self._legacy_chunk_id_from_payload(payload)
                chunk = CodeChunk(
                    chunk_id=chunk_id,
                    language=payload.get("language", "unknown"),
                    file_path=payload["file_path"],
                    symbol_name=payload["symbol_name"],
                    symbol_kind=payload["symbol_kind"],
                    line_start=payload["line_start"],
                    line_end=payload["line_end"],
                    byte_start=payload["byte_start"],
                    byte_end=payload["byte_end"],
                    file_hash=payload["file_hash"],
                    source=payload["source"],
                    signature=payload.get("signature"),
                    docstring=payload.get("docstring"),
                    parent=payload.get("parent"),
                    visibility=payload.get("visibility"),
                )
                chunks.append(chunk)

        except Exception as e:
            print(f"Failed to fetch chunks for {file_path}: {e}")

        return chunks

    def get_chunks_by_files(self, file_paths: list[str]) -> dict[str, list[CodeChunk]]:
        """
        Fetch all existing chunks for a batch of files from Qdrant.

        file_paths: list[str] — File paths whose chunks to fetch.
        Returns: dict[str, list[CodeChunk]] — Existing chunks grouped by file path.
        """
        grouped: dict[str, list[CodeChunk]] = {file_path: [] for file_path in file_paths}
        if not file_paths:
            return grouped

        batch_size = 128
        for offset in range(0, len(file_paths), batch_size):
            batch = file_paths[offset:offset + batch_size]
            try:
                results, _ = self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="file_path",
                                match=models.MatchAny(any=batch),
                            )
                        ]
                    ),
                    limit=10000,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                print(f"Failed to fetch chunks for batch: {e}")
                continue

            for point in results:
                payload = point.payload
                file_path = payload.get("file_path")
                if file_path not in grouped:
                    continue
                chunk_id = payload.get("chunk_id") or self._legacy_chunk_id_from_payload(payload)
                grouped[file_path].append(CodeChunk(
                    chunk_id=chunk_id,
                    language=payload.get("language", "unknown"),
                    file_path=file_path,
                    symbol_name=payload["symbol_name"],
                    symbol_kind=payload["symbol_kind"],
                    line_start=payload["line_start"],
                    line_end=payload["line_end"],
                    byte_start=payload["byte_start"],
                    byte_end=payload["byte_end"],
                    file_hash=payload["file_hash"],
                    source=payload["source"],
                    signature=payload.get("signature"),
                    docstring=payload.get("docstring"),
                    parent=payload.get("parent"),
                    visibility=payload.get("visibility"),
                    role=payload.get("role"),
                ))

        return grouped

    def delete_by_identities(self, identities: list) -> None:
        """
        Delete chunks by their ChunkIdentity objects.

        identities: list[ChunkIdentity] — List of chunk identities to delete.
        """
        if not identities:
            return

        chunk_ids = []
        for identity in identities:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, identity.chunk_id))
            chunk_ids.append(point_id)

        if chunk_ids:
            self.delete_by_chunk_ids(chunk_ids)

    def _legacy_chunk_id_from_payload(self, payload: dict) -> str:
        """
        Reconstruct the old chunk ID format for points indexed before chunk_id
        was stored in payload.

        payload: dict — Stored Qdrant payload.
        Returns: str — Legacy chunk ID string.
        """
        parent_str = f":{payload.get('parent')}" if payload.get("parent") else ""
        return f"{payload['file_path']}:{payload['symbol_name']}:{payload['symbol_kind']}{parent_str}"
