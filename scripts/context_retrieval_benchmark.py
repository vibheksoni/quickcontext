from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.sdk import QuickContext
from engine.src.cli import _optimize_search_config
from engine.src.config import EngineConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark mixed AI retrieval quality and latency.")
    parser.add_argument("--config", default=None, help="Path to quickcontext config JSON.")
    parser.add_argument("--project", default="quickcontext", help="Indexed project name.")
    parser.add_argument("--cases-file", required=True, help="JSON file containing benchmark cases.")
    parser.add_argument(
        "--strategy",
        choices=("semantic-auto", "context-auto"),
        default="context-auto",
        help="Retrieval strategy to benchmark.",
    )
    parser.add_argument("--limit", type=int, default=3, help="Primary result limit.")
    parser.add_argument("--related-seed-files", type=int, default=1, help="Seed files for related context expansion.")
    parser.add_argument("--related-file-limit", type=int, default=8, help="Related files to return.")
    parser.add_argument("--show-top", type=int, default=2, help="Show the top N results per query.")
    return parser.parse_args()


def _load_config(config_path: str | None) -> EngineConfig:
    if config_path:
        return EngineConfig.from_json(config_path)
    return EngineConfig.auto()


def _normalize_path(path: str) -> str:
    return path.replace("\\\\?\\", "").replace("\\", "/").lower()


def _relative_path(path: str, root: Path) -> str:
    candidate = Path(path.replace("\\\\?\\", ""))
    try:
        return str(candidate.resolve().relative_to(root)).replace("\\", "/")
    except Exception:
        return str(candidate).replace("\\", "/")


def _match_rank(paths: list[str], expected_paths: list[str]) -> int | None:
    normalized_expected = tuple(fragment.lower() for fragment in expected_paths)
    for idx, path in enumerate(paths, 1):
        normalized_path = _normalize_path(path)
        if any(fragment in normalized_path for fragment in normalized_expected):
            return idx
    return None


def _coverage(paths: list[str], expected_paths: list[str], limit: int) -> float:
    if not expected_paths:
        return 0.0

    matched = 0
    for fragment in expected_paths:
        fragment_lower = fragment.lower()
        if any(fragment_lower in _normalize_path(path) for path in paths[:limit]):
            matched += 1
    return matched / len(expected_paths)


def _run_strategy(
    qc: QuickContext,
    strategy: str,
    query: str,
    project_name: str,
    limit: int,
    related_seed_files: int,
    related_file_limit: int,
) -> dict:
    if strategy == "semantic-auto":
        return qc.semantic_search_auto(
            query=query,
            project_name=project_name,
            limit=limit,
            related_seed_files=related_seed_files,
            related_file_limit=related_file_limit,
        )

    return qc.retrieve_context_auto(
        query=query,
        project_name=project_name,
        limit=limit,
        related_seed_files=related_seed_files,
        related_file_limit=related_file_limit,
    )


def _warmup(qc: QuickContext, project_name: str) -> None:
    qc.symbol_lookup("CollectionManager", path=Path.cwd(), limit=1)
    qc.semantic_search("query embedding cache", project_name=project_name, limit=1)


def main() -> None:
    args = _parse_args()
    repo_root = Path.cwd().resolve()
    cases = json.loads(Path(args.cases_file).read_text(encoding="utf-8"))
    config = _optimize_search_config(_load_config(args.config))

    latencies: list[float] = []
    hit_ranks: list[int | None] = []
    anchor_coverages: list[float] = []
    context_coverages: list[float] = []
    mode_counts: dict[str, int] = {}
    case_rows: list[dict] = []

    with QuickContext(config) as qc:
        _warmup(qc, args.project)
        for case in cases:
            query = str(case["query"])
            expected_paths = [str(path) for path in case["expected_paths"]]
            started = time.perf_counter()
            payload = _run_strategy(
                qc=qc,
                strategy=args.strategy,
                query=query,
                project_name=args.project,
                limit=args.limit,
                related_seed_files=args.related_seed_files,
                related_file_limit=args.related_file_limit,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)

            result_paths = [_relative_path(item.file_path, repo_root) for item in payload["results"]]
            bundle_paths = result_paths + [
                _relative_path(item["file_path"], repo_root)
                for item in payload["related_files"]
            ]
            hit_rank = _match_rank(result_paths, expected_paths)
            hit_ranks.append(hit_rank)
            anchor_coverages.append(_coverage(result_paths, expected_paths, limit=max(args.limit, 1)))
            context_coverages.append(
                _coverage(
                    bundle_paths,
                    expected_paths,
                    limit=max(args.limit + args.related_file_limit, 1),
                )
            )

            mode = str(payload.get("mode", "unknown"))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            case_rows.append(
                {
                    "query": query,
                    "mode": mode,
                    "latency_ms": latency_ms,
                    "hit_rank": hit_rank,
                    "result_paths": result_paths,
                }
            )

    hit1 = sum(1 for rank in hit_ranks if rank == 1)
    hit3 = sum(1 for rank in hit_ranks if rank is not None and rank <= 3)
    mrr = sum(0.0 if rank is None else 1.0 / rank for rank in hit_ranks) / len(hit_ranks)

    print("Summary")
    print(f"  Cases: {len(hit_ranks)}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Hit@1: {hit1}/{len(hit_ranks)}")
    print(f"  Hit@3: {hit3}/{len(hit_ranks)}")
    print(f"  MRR: {mrr:.4f}")
    print(f"  Mean anchor top-{args.limit} coverage: {sum(anchor_coverages) / len(anchor_coverages):.4f}")
    print(f"  Mean context coverage: {sum(context_coverages) / len(context_coverages):.4f}")
    print(f"  Mean latency: {mean(latencies):.2f} ms")
    print(f"  Median latency: {median(latencies):.2f} ms")
    print("  Modes:")
    for mode, count in sorted(mode_counts.items()):
        print(f"    {mode}: {count}")
    print()

    for row in case_rows:
        rank_text = str(row["hit_rank"]) if row["hit_rank"] is not None else "miss"
        print(row["query"])
        print(f"  mode: {row['mode']}")
        print(f"  hit_rank: {rank_text}")
        print(f"  latency_ms: {row['latency_ms']:.2f}")
        for idx, path in enumerate(row["result_paths"][: max(args.show_top, 1)], 1):
            print(f"  {idx}. {path}")
        print()


if __name__ == "__main__":
    main()
