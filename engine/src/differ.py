from dataclasses import dataclass
from typing import Optional

from engine.src.chunker import CodeChunk


@dataclass(frozen=True, slots=True)
class ChunkIdentity:
    """
    Unique identifier for a code chunk based on symbol identity.
    Includes chunk_id so deletions map to the exact stored point, and uses
    signature/span fields to avoid collisions for overloaded symbols.

    chunk_id: str — Deterministic chunk ID used for point ID derivation.
    file_path: str — Absolute file path.
    symbol_name: str — Symbol name (function/class/etc).
    symbol_kind: str — Symbol kind (function, class, method, etc).
    parent: Optional[str] — Parent symbol name (class for methods, None for top-level).
    signature: Optional[str] — Signature text when available.
    byte_start: int — Byte start for disambiguation.
    byte_end: int — Byte end for disambiguation.
    """
    chunk_id: str
    file_path: str
    symbol_name: str
    symbol_kind: str
    parent: Optional[str]
    signature: Optional[str]
    byte_start: int
    byte_end: int

    @staticmethod
    def from_chunk(chunk: CodeChunk) -> "ChunkIdentity":
        """
        Create ChunkIdentity from a CodeChunk.

        chunk: CodeChunk — Code chunk to extract identity from.
        Returns: ChunkIdentity — Unique identifier for the chunk.
        """
        return ChunkIdentity(
            chunk_id=chunk.chunk_id,
            file_path=chunk.file_path,
            symbol_name=chunk.symbol_name,
            symbol_kind=chunk.symbol_kind,
            parent=chunk.parent,
            signature=chunk.signature,
            byte_start=chunk.byte_start,
            byte_end=chunk.byte_end,
        )


@dataclass(frozen=True, slots=True)
class ChunkDiff:
    """
    Result of comparing old and new chunks for a file.

    added: list[CodeChunk] — New chunks not in old set.
    modified: list[CodeChunk] — Chunks with same identity but different content.
    deleted: list[ChunkIdentity] — Old chunk identities not in new set.
    unchanged: list[ChunkIdentity] — Chunks with same identity and content.
    """
    added: list[CodeChunk]
    modified: list[CodeChunk]
    deleted: list[ChunkIdentity]
    unchanged: list[ChunkIdentity]

    @property
    def needs_reindex(self) -> list[CodeChunk]:
        """
        Returns: list[CodeChunk] — All chunks that need re-indexing (added + modified).
        """
        return self.added + self.modified

    @property
    def needs_deletion(self) -> list[ChunkIdentity]:
        """
        Returns: list[ChunkIdentity] — All chunk identities that need deletion.
        """
        return self.deleted


class ChunkDiffer:
    """
    Compares old and new chunks to detect changes at chunk granularity.

    Strategy:
    - Identity: chunk identity plus signature/span disambiguation
    - Added: New chunks not in old set
    - Modified: Same identity but different file_hash or source
    - Deleted: Old chunks not in new set
    - Unchanged: Same identity and same file_hash
    """

    @staticmethod
    def diff(old_chunks: list[CodeChunk], new_chunks: list[CodeChunk]) -> ChunkDiff:
        """
        Compare old and new chunks to detect changes.

        old_chunks: list[CodeChunk] — Existing chunks from Qdrant.
        new_chunks: list[CodeChunk] — Newly extracted chunks from file.
        Returns: ChunkDiff — Categorized changes.
        """
        old_map: dict[ChunkIdentity, CodeChunk] = {
            ChunkIdentity.from_chunk(chunk): chunk for chunk in old_chunks
        }
        new_map: dict[ChunkIdentity, CodeChunk] = {
            ChunkIdentity.from_chunk(chunk): chunk for chunk in new_chunks
        }

        old_ids = set(old_map.keys())
        new_ids = set(new_map.keys())

        added_ids = new_ids - old_ids
        deleted_ids = old_ids - new_ids
        common_ids = old_ids & new_ids

        added: list[CodeChunk] = [new_map[id_] for id_ in added_ids]
        deleted: list[ChunkIdentity] = list(deleted_ids)
        modified: list[CodeChunk] = []
        unchanged: list[ChunkIdentity] = []

        for id_ in common_ids:
            old_chunk = old_map[id_]
            new_chunk = new_map[id_]

            if old_chunk.file_hash != new_chunk.file_hash or old_chunk.source != new_chunk.source:
                modified.append(new_chunk)
            else:
                unchanged.append(id_)

        return ChunkDiff(
            added=added,
            modified=modified,
            deleted=deleted,
            unchanged=unchanged,
        )
