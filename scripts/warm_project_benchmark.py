from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.sdk import QuickContext
from engine.src.cli import _optimize_search_config
from engine.src.config import EngineConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cold query latency with and without warm_project().")
    parser.add_argument("--config", default=None, help="Path to quickcontext config JSON.")
    parser.add_argument("--project", default="quickcontext", help="Project name for retrieval calls.")
    parser.add_argument("--path", default=".", help="Project path to warm.")
    parser.add_argument("--query", required=True, help="Query to benchmark.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of cold-vs-warm trials.")
    parser.add_argument("--limit", type=int, default=4, help="Result limit for retrieve_context_auto.")
    return parser.parse_args()


def _load_config(config_path: str | None) -> EngineConfig:
    if config_path:
        return EngineConfig.from_json(config_path)
    return EngineConfig.auto()


def _stop_service() -> None:
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Get-Process quickcontext-service -ErrorAction SilentlyContinue | Stop-Process -Force"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _measure_query(config: EngineConfig, query: str, project: str, limit: int, do_warm: bool, warm_path: str) -> tuple[float, dict | None]:
    _stop_service()
    with QuickContext(config) as qc:
        warm_stats = qc.warm_project(warm_path) if do_warm else None
        started = time.perf_counter()
        qc.retrieve_context_auto(query=query, project_name=project, limit=limit)
        latency_ms = (time.perf_counter() - started) * 1000
        return latency_ms, warm_stats


def main() -> None:
    args = _parse_args()
    config = _optimize_search_config(_load_config(args.config))

    cold_latencies: list[float] = []
    warm_latencies: list[float] = []
    last_warm_stats: dict | None = None

    for _ in range(max(1, args.repeats)):
        cold_ms, _ = _measure_query(config, args.query, args.project, args.limit, False, args.path)
        cold_latencies.append(cold_ms)
        warm_ms, warm_stats = _measure_query(config, args.query, args.project, args.limit, True, args.path)
        warm_latencies.append(warm_ms)
        last_warm_stats = warm_stats

    print("Summary")
    print(f"  Repeats: {len(cold_latencies)}")
    print(f"  Query: {args.query}")
    if last_warm_stats:
        print(f"  Warm symbol count: {last_warm_stats['symbol_count']}")
        print(f"  Warm text doc count: {last_warm_stats['text_doc_count']}")
    print(f"  Cold mean latency: {mean(cold_latencies):.2f} ms")
    print(f"  Cold median latency: {median(cold_latencies):.2f} ms")
    print(f"  Warm mean latency: {mean(warm_latencies):.2f} ms")
    print(f"  Warm median latency: {median(warm_latencies):.2f} ms")
    print()
    print(f"  Cold runs: {[round(item, 2) for item in cold_latencies]}")
    print(f"  Warm runs: {[round(item, 2) for item in warm_latencies]}")


if __name__ == "__main__":
    main()
