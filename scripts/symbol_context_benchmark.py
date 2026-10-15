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
    parser = argparse.ArgumentParser(description="Benchmark helper-level symbol context retrieval quality.")
    parser.add_argument("--config", default=None, help="Path to quickcontext config JSON.")
    parser.add_argument("--project", default="quickcontext", help="Indexed project name.")
    parser.add_argument("--cases-file", required=True, help="JSON file containing helper-context benchmark cases.")
    parser.add_argument("--strategy", choices=("anchor", "context-auto"), default="context-auto", help="Retrieval strategy to benchmark.")
    parser.add_argument("--limit", type=int, default=4, help="Maximum returned results.")
    parser.add_argument("--show-top", type=int, default=4, help="Show the top N symbol names per query.")
    return parser.parse_args()


def _load_config(config_path: str | None) -> EngineConfig:
    if config_path:
        return EngineConfig.from_json(config_path)
    return EngineConfig.auto()


def _helper_coverage(result_names: list[str], expected_helpers: list[str]) -> float:
    if not expected_helpers:
        return 0.0
    lowered = {name.lower() for name in result_names}
    matched = sum(1 for name in expected_helpers if name.lower() in lowered)
    return matched / len(expected_helpers)


def _run_anchor_strategy(qc: QuickContext, query: str, limit: int) -> list:
    symbol_query = qc._extract_symbol_query_candidate(query)
    if not symbol_query:
        return []
    return qc._symbol_lookup_search_results(query=symbol_query, limit=limit)


def main() -> None:
    args = _parse_args()
    cases = json.loads(Path(args.cases_file).read_text(encoding="utf-8"))
    config = _load_config(args.config)

    latencies: list[float] = []
    coverages: list[float] = []
    mode_counts: dict[str, int] = {}
    rows: list[dict] = []

    with QuickContext(config) as qc:
        for case in cases:
            query = str(case["query"])
            expected_helpers = [str(item) for item in case["expected_helpers"]]
            started = time.perf_counter()
            if args.strategy == "anchor":
                results = _run_anchor_strategy(qc, query, args.limit)
                payload = {"mode": "anchor", "results": results}
            else:
                payload = qc.retrieve_context_auto(query=query, project_name=args.project, limit=args.limit)
            latency_ms = (time.perf_counter() - started) * 1000
            latencies.append(latency_ms)

            result_names = [item.symbol_name for item in payload["results"]]
            coverage = _helper_coverage(result_names, expected_helpers)
            coverages.append(coverage)

            mode = str(payload.get("mode", "unknown"))
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            rows.append(
                {
                    "query": query,
                    "mode": mode,
                    "latency_ms": latency_ms,
                    "coverage": coverage,
                    "result_names": result_names,
                    "expected_helpers": expected_helpers,
                }
            )

    full_hits = sum(1 for coverage in coverages if coverage >= 1.0)
    any_hits = sum(1 for coverage in coverages if coverage > 0.0)

    print("Summary")
    print(f"  Cases: {len(rows)}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Full helper coverage: {full_hits}/{len(rows)}")
    print(f"  Any helper coverage: {any_hits}/{len(rows)}")
    print(f"  Mean helper coverage: {sum(coverages) / len(coverages):.4f}")
    print(f"  Mean latency: {mean(latencies):.2f} ms")
    print(f"  Median latency: {median(latencies):.2f} ms")
    print("  Modes:")
    for mode, count in sorted(mode_counts.items()):
        print(f"    {mode}: {count}")
    print()

    for row in rows:
        print(row["query"])
        print(f"  mode: {row['mode']}")
        print(f"  helper_coverage: {row['coverage']:.4f}")
        print(f"  latency_ms: {row['latency_ms']:.2f}")
        print(f"  expected: {', '.join(row['expected_helpers'])}")
        for idx, name in enumerate(row["result_names"][: max(args.show_top, 1)], 1):
            print(f"  {idx}. {name}")
        print()


if __name__ == "__main__":
    main()
