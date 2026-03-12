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
from engine.src.config import EngineConfig
from engine.src.cli import _optimize_search_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark semantic_search_bundle coverage.")
    parser.add_argument("--config", default=None, help="Path to quickcontext config JSON.")
    parser.add_argument("--project", default="quickcontext", help="Indexed project name.")
    parser.add_argument("--cases-file", required=True, help="JSON file containing benchmark cases.")
    parser.add_argument("--strategy", choices=("semantic", "bundle", "auto"), default="bundle", help="Retrieval strategy to benchmark.")
    parser.add_argument("--limit", type=int, default=3, help="Semantic result limit.")
    parser.add_argument("--related-seed-files", type=int, default=1, help="Seed files for graph expansion.")
    parser.add_argument("--related-file-limit", type=int, default=8, help="Related files to return.")
    return parser.parse_args()


def _load_config(config_path: str | None) -> EngineConfig:
    if config_path:
        return EngineConfig.from_json(config_path)
    return EngineConfig.auto()


def _normalize_path(path: str) -> str:
    return path.replace("\\\\?\\", "").replace("\\", "/").lower()


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


def main() -> None:
    args = _parse_args()
    config = _optimize_search_config(_load_config(args.config))
    cases = json.loads(Path(args.cases_file).read_text(encoding="utf-8"))

    latencies: list[float] = []
    hit_ranks: list[int | None] = []
    result_coverages: list[float] = []
    bundle_coverages: list[float] = []

    with QuickContext(config) as qc:
        for case in cases:
            query = case["query"]
            expected_paths = [str(path) for path in case["expected_paths"]]
            started = time.perf_counter()
            if args.strategy == "semantic":
                bundle = {
                    "results": qc.semantic_search(
                        query=query,
                        project_name=args.project,
                        limit=args.limit,
                    ),
                    "related_files": [],
                    "related_callers": [],
                }
            elif args.strategy == "auto":
                bundle = qc.semantic_search_auto(
                    query=query,
                    project_name=args.project,
                    limit=args.limit,
                    related_seed_files=args.related_seed_files,
                    related_file_limit=args.related_file_limit,
                )
            else:
                bundle = qc.semantic_search_bundle(
                    query=query,
                    project_name=args.project,
                    limit=args.limit,
                    related_seed_files=args.related_seed_files,
                    related_file_limit=args.related_file_limit,
                )
            latencies.append((time.perf_counter() - started) * 1000)

            result_paths = [item.file_path for item in bundle["results"]]
            bundle_paths = result_paths + [item["file_path"] for item in bundle["related_files"]]
            hit_ranks.append(_match_rank(result_paths, expected_paths))
            result_coverages.append(_coverage(result_paths, expected_paths, limit=3))
            bundle_coverages.append(_coverage(bundle_paths, expected_paths, limit=max(args.limit + args.related_file_limit, 1)))

    hit1 = sum(1 for rank in hit_ranks if rank == 1)
    hit3 = sum(1 for rank in hit_ranks if rank is not None and rank <= 3)
    mrr = sum(0.0 if rank is None else 1.0 / rank for rank in hit_ranks) / len(hit_ranks)

    print("Summary")
    print(f"  Cases: {len(hit_ranks)}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Hit@1: {hit1}/{len(hit_ranks)}")
    print(f"  Hit@3: {hit3}/{len(hit_ranks)}")
    print(f"  MRR: {mrr:.4f}")
    print(f"  Mean anchor top-3 coverage: {sum(result_coverages) / len(result_coverages):.4f}")
    print(f"  Mean context coverage: {sum(bundle_coverages) / len(bundle_coverages):.4f}")
    print(f"  Mean latency: {mean(latencies):.2f} ms")
    print(f"  Median latency: {median(latencies):.2f} ms")


if __name__ == "__main__":
    main()
