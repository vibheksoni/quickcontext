import hashlib
from dataclasses import dataclass

from engine.src.chunker import CodeChunk
from engine.src.describer import ChunkDescription
from engine.src.embedder import EmbeddedChunk


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    """
    Result of content-based chunk deduplication.

    unique_chunks: list[CodeChunk] — Representative chunks (one per unique source hash).
    chunk_groups: dict[str, list[CodeChunk]] — Maps content hash to all chunks with that source.
    total_chunks: int — Total input chunk count.
    unique_count: int — Number of unique source contents.
    duplicate_count: int — Number of duplicates skipped.
    """
    unique_chunks: list[CodeChunk]
    chunk_groups: dict[str, list[CodeChunk]]
    total_chunks: int
    unique_count: int
    duplicate_count: int


def content_hash(source: str) -> str:
    """
    Compute SHA256 hash of source code content for dedup grouping.

    source: str — Raw source code text.
    Returns: str — Hex digest of SHA256 hash.
    """
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def dedup_hash(chunk: CodeChunk) -> str:
    """
    Compute a context-aware dedup hash for description and embedding reuse.

    Identical source in materially different symbol contexts can require
    different descriptions, so the dedup key includes lightweight semantic
    context in addition to raw source.

    chunk: CodeChunk — Chunk to hash.
    Returns: str — Hex digest used for dedup grouping.
    """
    parts = [
        chunk.source,
        chunk.symbol_name,
        chunk.symbol_kind,
        chunk.parent or "",
        chunk.signature or "",
        chunk.docstring or "",
    ]
    return hashlib.sha256("\n\0".join(parts).encode("utf-8")).hexdigest()


def deduplicate_chunks(chunks: list[CodeChunk]) -> DeduplicationResult:
    """
    Group chunks by context-aware dedup hash, picking one representative per group.

    Chunks only dedup when both the raw source and lightweight semantic context
    align closely enough to safely share generated descriptions and embeddings.

    chunks: list[CodeChunk] — All chunks to deduplicate.
    Returns: DeduplicationResult — Unique representatives and grouping info.
    """
    groups: dict[str, list[CodeChunk]] = {}

    for chunk in chunks:
        h = dedup_hash(chunk)
        if h not in groups:
            groups[h] = []
        groups[h].append(chunk)

    unique = [group[0] for group in groups.values()]

    return DeduplicationResult(
        unique_chunks=unique,
        chunk_groups=groups,
        total_chunks=len(chunks),
        unique_count=len(unique),
        duplicate_count=len(chunks) - len(unique),
    )


def expand_descriptions(
    unique_descriptions: list[ChunkDescription],
    dedup_result: DeduplicationResult,
) -> list[ChunkDescription]:
    """
    Expand descriptions from unique representatives to all chunks in each group.

    For each unique chunk that was described, clone the description for every
    duplicate chunk in the same content-hash group (with the correct chunk_id).

    unique_descriptions: list[ChunkDescription] — Descriptions for unique chunks only.
    dedup_result: DeduplicationResult — Grouping info from deduplicate_chunks().
    Returns: list[ChunkDescription] — Descriptions for ALL original chunks.
    """
    unique_id_to_desc: dict[str, ChunkDescription] = {
        d.chunk_id: d for d in unique_descriptions
    }

    all_descriptions: list[ChunkDescription] = []

    for group in dedup_result.chunk_groups.values():
        representative = group[0]
        desc = unique_id_to_desc.get(representative.chunk_id)
        if desc is None:
            continue

        for chunk in group:
            if chunk.chunk_id == representative.chunk_id:
                all_descriptions.append(desc)
            else:
                all_descriptions.append(ChunkDescription(
                    chunk_id=chunk.chunk_id,
                    description=desc.description,
                    keywords=desc.keywords,
                    token_count=0,
                    cost_usd=0.0,
                ))

    return all_descriptions


def expand_embeddings(
    unique_embedded: list[EmbeddedChunk],
    all_chunks: list[CodeChunk],
    all_descriptions: list[ChunkDescription],
    dedup_result: DeduplicationResult,
) -> list[EmbeddedChunk]:
    """
    Expand embeddings from unique representatives to all chunks in each group.

    Duplicate chunks reuse the same code_vector and desc_vector from their
    representative, but get their own chunk metadata (file_path, symbol_name, etc).

    unique_embedded: list[EmbeddedChunk] — Embedded unique chunks only.
    all_chunks: list[CodeChunk] — All original chunks (for metadata).
    all_descriptions: list[ChunkDescription] — All expanded descriptions.
    dedup_result: DeduplicationResult — Grouping info.
    Returns: list[EmbeddedChunk] — Embedded chunks for ALL original chunks.
    """
    unique_id_to_embedded: dict[str, EmbeddedChunk] = {
        e.chunk_id: e for e in unique_embedded
    }

    chunk_map: dict[str, CodeChunk] = {c.chunk_id: c for c in all_chunks}
    desc_map: dict[str, ChunkDescription] = {d.chunk_id: d for d in all_descriptions}

    all_embedded: list[EmbeddedChunk] = []

    for group in dedup_result.chunk_groups.values():
        representative = group[0]
        source_embedded = unique_id_to_embedded.get(representative.chunk_id)
        if source_embedded is None:
            continue

        for chunk in group:
            if chunk.chunk_id == representative.chunk_id:
                all_embedded.append(source_embedded)
            else:
                desc = desc_map.get(chunk.chunk_id)
                if desc is None:
                    continue
                all_embedded.append(EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    code_vector=source_embedded.code_vector,
                    desc_vector=source_embedded.desc_vector,
                    chunk=chunk,
                    description=desc,
                    embedding_cost_usd=0.0,
                ))

    return all_embedded
