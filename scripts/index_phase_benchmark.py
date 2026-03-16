import argparse
import time
from collections import Counter
from pathlib import Path
import sys
import uuid

from qdrant_client import models

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine import QuickContext
from engine.src.artifact_index import (
    ARTIFACT_FALLBACK_ERROR,
    analyze_artifact_candidate,
    should_downgrade_artifact_profile,
)
from engine.src.chunk_filter import ChunkFilterConfig, filter_chunks
from engine.src.chunker import ChunkBuilder
from engine.src.config import EngineConfig
from engine.src.dedup import deduplicate_chunks
from engine.src.describer import build_fallback_descriptions
from engine.src.indexer import QdrantIndexer
from engine.src.parsing import ExtractionResult, RustParserService


def _human_mb(num_bytes: int) -> float:
    return num_bytes / 1024.0 / 1024.0


def _raw_file_line_stats(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {
            "line_count": 0,
            "max_line_length": 0,
            "avg_line_length": 0.0,
            "raw_minified_like": False,
        }

    lines = text.splitlines()
    if not lines:
        return {
            "line_count": 0,
            "max_line_length": 0,
            "avg_line_length": 0.0,
            "raw_minified_like": False,
        }

    max_line_length = max(len(line) for line in lines)
    avg_line_length = sum(len(line) for line in lines) / len(lines)
    whitespace_ratio = sum(1 for ch in text if ch.isspace()) / max(1, len(text))
    raw_minified_like = (
        max_line_length >= 1200
        or (len(lines) <= 6 and avg_line_length >= 220)
        or (len(lines) <= 3 and whitespace_ratio < 0.04 and avg_line_length > 120)
    )
    return {
        "line_count": len(lines),
        "max_line_length": max_line_length,
        "avg_line_length": avg_line_length,
        "raw_minified_like": raw_minified_like,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile indexing phases on a large directory without running a full index.")
    parser.add_argument("--config", default="quickcontext.json", help="Path to config JSON.")
    parser.add_argument("--path", required=True, help="Target directory to profile.")
    parser.add_argument("--largest-files", type=int, default=12, help="Number of largest supported files to extract/profile.")
    parser.add_argument("--embed-sample-chunks", type=int, default=64, help="Number of filtered chunks to embed as a sample. Use 0 to skip embeddings.")
    parser.add_argument("--upsert-sample-chunks", type=int, default=64, help="Number of embedded chunks to upsert into a temp collection. Use 0 to skip Qdrant upsert.")
    parser.add_argument("--min-chunk-bytes", type=int, default=200, help="Chunk min-byte filter used in fast indexing mode.")
    parser.add_argument("--max-chunks-per-file", type=int, default=180, help="Per-file chunk cap used in fast indexing mode.")
    parser.add_argument("--artifact-aware-fast", action="store_true", help="Downgrade obvious bundle artifacts to coarse file chunks before extraction.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(args.path).resolve()
    config = EngineConfig.from_json(args.config)
    parser_service = RustParserService()

    overall_start = time.perf_counter()
    scan_start = time.perf_counter()
    scan_entries = parser_service.scan_files(root)
    scan_ms = (time.perf_counter() - scan_start) * 1000

    total_supported_bytes = sum(entry.file_size for entry in scan_entries)
    ext_counts = Counter(Path(entry.file_path).suffix.lower() or "<noext>" for entry in scan_entries)
    sample_entries = sorted(scan_entries, key=lambda entry: entry.file_size, reverse=True)[: max(1, int(args.largest_files))]

    per_file: list[dict] = []
    extraction_results = []
    extract_total_ms = 0.0

    for entry in sample_entries:
        file_path = Path(entry.file_path)
        raw_stats = _raw_file_line_stats(file_path)
        downgraded = False
        started = time.perf_counter()
        if args.artifact_aware_fast and entry.file_size >= 128 * 1024 and entry.language.lower() in {"javascript", "typescript", "tsx", "jsx"}:
            profile = analyze_artifact_candidate(
                file_path=file_path,
                language=entry.language,
                file_size=entry.file_size,
                file_mtime=entry.file_mtime,
            )
            if should_downgrade_artifact_profile(profile):
                extracted = [
                    ExtractionResult(
                        file_path=str(file_path),
                        language=entry.language,
                        symbols=[],
                        errors=[ARTIFACT_FALLBACK_ERROR],
                        file_hash=profile.file_hash,
                        file_size=entry.file_size,
                        file_mtime=entry.file_mtime,
                    )
                ]
                downgraded = True
            else:
                extracted = parser_service.extract(file_path)
        else:
            extracted = parser_service.extract(file_path)
        extract_ms = (time.perf_counter() - started) * 1000
        extract_total_ms += extract_ms
        extraction_results.extend(extracted)
        symbol_count = sum(len(item.symbols) for item in extracted)
        per_file.append(
            {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": entry.file_size,
                "extract_ms": extract_ms,
                "symbol_count": symbol_count,
                "downgraded": downgraded,
                **raw_stats,
            }
        )

    parser_service.close()

    build_start = time.perf_counter()
    chunks = ChunkBuilder().build_chunks(extraction_results)
    chunk_build_ms = (time.perf_counter() - build_start) * 1000

    filter_start = time.perf_counter()
    filtered_chunks, filter_stats = filter_chunks(
        chunks,
        ChunkFilterConfig(
            min_chunk_bytes=max(1, int(args.min_chunk_bytes)),
            max_chunks_per_file=max(1, int(args.max_chunks_per_file)),
            skip_minified=True,
        ),
    )
    filter_ms = (time.perf_counter() - filter_start) * 1000

    dedup_start = time.perf_counter()
    dedup_result = deduplicate_chunks(filtered_chunks)
    dedup_ms = (time.perf_counter() - dedup_start) * 1000

    chunk_counts_before = Counter(chunk.file_path for chunk in chunks)
    chunk_counts_after = Counter(chunk.file_path for chunk in filtered_chunks)
    for item in per_file:
        item["chunks_before"] = chunk_counts_before.get(item["file_path"], 0)
        item["chunks_after"] = chunk_counts_after.get(item["file_path"], 0)

    embed_ms = 0.0
    embed_requests = 0
    embed_input_count = 0
    embed_sample_count = 0
    upsert_ms = 0.0
    point_build_ms = 0.0
    upserted_points = 0

    if args.embed_sample_chunks > 0 and filtered_chunks:
        embed_sample = filtered_chunks[: min(len(filtered_chunks), int(args.embed_sample_chunks))]
        embed_sample_count = len(embed_sample)
        descriptions = build_fallback_descriptions(embed_sample)

        with QuickContext(config) as qc:
            started = time.perf_counter()
            embedded_chunks, _embedding_cost = qc.embedder.embed_batch(
                embed_sample,
                descriptions,
                batch_size_override=16,
                adaptive_batching_override=False,
            )
            embed_ms = (time.perf_counter() - started) * 1000
            embed_stats = qc.embedder.last_run_stats
            embed_requests = embed_stats.request_count
            embed_input_count = embed_stats.input_count

            if args.upsert_sample_chunks > 0 and embedded_chunks:
                upsert_sample = embedded_chunks[: min(len(embedded_chunks), int(args.upsert_sample_chunks))]
                client = qc.connection.client
                collection_name = f"qc-index-phase-benchmark-{uuid.uuid4().hex[:8]}"
                try:
                    client.delete_collection(collection_name)
                except Exception:
                    pass
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        config.vectors[0].name: models.VectorParams(
                            size=config.vectors[0].dimension,
                            distance=models.Distance.COSINE,
                        ),
                        config.vectors[1].name: models.VectorParams(
                            size=config.vectors[1].dimension,
                            distance=models.Distance.COSINE,
                        ),
                    },
                )

                indexer = QdrantIndexer(
                    client=client,
                    collection_name=collection_name,
                    batch_size=32,
                    upsert_concurrency=1,
                )
                started = time.perf_counter()
                points = [indexer._chunk_to_point(chunk) for chunk in upsert_sample]
                point_build_ms = (time.perf_counter() - started) * 1000

                started = time.perf_counter()
                client.upsert(collection_name=collection_name, points=points, wait=True)
                upsert_ms = (time.perf_counter() - started) * 1000
                upserted_points = len(points)
                client.delete_collection(collection_name)

    total_ms = (time.perf_counter() - overall_start) * 1000

    print("Index Phase Benchmark")
    print(f"  Target: {root}")
    print(f"  Supported files: {len(scan_entries)}")
    print(f"  Supported bytes: {_human_mb(total_supported_bytes):.2f} MB")
    print(f"  Scan time: {scan_ms:.2f} ms")
    print("  Top extensions:")
    for ext, count in ext_counts.most_common(8):
        print(f"    {ext}: {count}")

    sample_bytes = sum(item["file_size"] for item in per_file)
    print()
    print("Sample Slice")
    print(f"  Largest files sampled: {len(per_file)}")
    print(f"  Sample bytes: {_human_mb(sample_bytes):.2f} MB")
    print(f"  Extraction time: {extract_total_ms:.2f} ms")
    print(f"  Chunk build time: {chunk_build_ms:.2f} ms")
    print(f"  Filter time: {filter_ms:.2f} ms")
    print(f"  Dedup time: {dedup_ms:.2f} ms")
    print(f"  Chunks before filter: {len(chunks)}")
    print(f"  Chunks after filter: {len(filtered_chunks)}")
    print(f"  Dedup unique chunks: {dedup_result.unique_count}")
    print(f"  Dedup duplicates: {dedup_result.duplicate_count}")
    print(f"  Skipped small chunks: {filter_stats.skipped_small}")
    print(f"  Skipped minified chunks: {filter_stats.skipped_minified}")
    print(f"  Skipped per-file cap: {filter_stats.skipped_per_file_cap}")
    print(f"  Files capped: {filter_stats.files_capped}")

    if embed_sample_count:
        print()
        print("Embedding Sample")
        print(f"  Sample chunks embedded: {embed_sample_count}")
        print(f"  Embed time: {embed_ms:.2f} ms")
        print(f"  Embed requests: {embed_requests}")
        print(f"  Embed inputs: {embed_input_count}")
        if embed_input_count:
            print(f"  Mean ms per embed input: {embed_ms / embed_input_count:.2f}")
        if upserted_points:
            print(f"  Point build time: {point_build_ms:.2f} ms")
            print(f"  Qdrant upsert time: {upsert_ms:.2f} ms")
            print(f"  Upserted points: {upserted_points}")

    print()
    print("Heaviest Sample Files")
    for item in sorted(per_file, key=lambda row: row["extract_ms"], reverse=True):
        print(
            "  "
            f"{_human_mb(item['file_size']):6.2f} MB | "
            f"{item['extract_ms']:9.2f} ms | "
            f"symbols={item['symbol_count']:6d} | "
            f"chunks={item['chunks_before']:6d}->{item['chunks_after']:4d} | "
            f"lines={item['line_count']:6d} | "
            f"max_line={item['max_line_length']:7d} | "
            f"avg_line={item['avg_line_length']:8.2f} | "
            f"downgraded={str(item['downgraded']).lower():5s} | "
            f"raw_minified={str(item['raw_minified_like']).lower():5s} | "
            f"{item['file_name']}"
        )

    print()
    print(f"Total script time: {total_ms:.2f} ms")


if __name__ == "__main__":
    main()
