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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Rust text_search quality and latency.")
    parser.add_argument("--config", default=None, help="Path to quickcontext config JSON.")
    parser.add_argument("--project", default="quickcontext", help="Unused project label for parity with other scripts.")
    parser.add_argument("--cases-file", required=True, help="JSON file containing retrieval cases.")
    parser.add_argument("--limit", type=int, default=5, help="Text-search result limit.")
    parser.add_argument("--show-top", type=int, default=3, help="Show the top N file paths per query.")
    parser.add_argument("--intent-level", type=int, default=2, help="Intent expansion level for text_search.")
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


def main() -> None:
    args = _parse_args()
    repo_root = Path.cwd().resolve()
    cases = json.loads(Path(args.cases_file).read_text(encoding="utf-8"))
    config = _load_config(args.config)

    hit_ranks: list[int | None] = []
    latencies: list[float] = []
    rows: list[dict] = []

    with QuickContext(config) as qc:
        for case in cases:
            query = str(case["query"])
            expected_paths = [str(path) for path in case["expected_paths"]]
            started = time.perf_counter()
            result = qc.text_search(
                query=query,
                path=Path.cwd(),
                limit=args.limit,
                intent_mode=True,
                intent_level=args.intent_level,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)

            top_paths = [
                _relative_path(item.file_path, repo_root)
                for item in result.matches[: max(args.show_top, 1)]
            ]
            hit_rank = _match_rank(top_paths, expected_paths)
            hit_ranks.append(hit_rank)
            rows.append(
                {
                    "query": query,
                    "hit_rank": hit_rank,
                    "latency_ms": latency_ms,
                    "top_paths": top_paths,
                }
            )

    hit1 = sum(1 for rank in hit_ranks if rank == 1)
    hit3 = sum(1 for rank in hit_ranks if rank is not None and rank <= 3)
    mrr = sum(0.0 if rank is None else 1.0 / rank for rank in hit_ranks) / len(hit_ranks)

    print("Summary")
    print(f"  Cases: {len(rows)}")
    print(f"  Hit@1: {hit1}/{len(rows)}")
    print(f"  Hit@3: {hit3}/{len(rows)}")
    print(f"  MRR: {mrr:.4f}")
    print(f"  Mean latency: {mean(latencies):.2f} ms")
    print(f"  Median latency: {median(latencies):.2f} ms")
    print()

    for row in rows:
        rank_text = str(row["hit_rank"]) if row["hit_rank"] is not None else "miss"
        print(row["query"])
        print(f"  hit_rank: {rank_text}")
        print(f"  latency_ms: {row['latency_ms']:.2f}")
        for idx, path in enumerate(row["top_paths"], 1):
            print(f"  {idx}. {path}")
        print()


if __name__ == "__main__":
    main()
