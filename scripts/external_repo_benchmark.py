"""
Benchmark retrieval quality and latency on any indexed repository.

Cases are JSON objects with "query", "expected_paths" (path fragments), and
optionally "expected_symbols" (any of these symbol names counts as a hit).

Example:
    python scripts/external_repo_benchmark.py --config cfg.json --project sbm \
        --path C:/repo --cases-file cases.json --strategies auto,semantic,text
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.sdk import QuickContext
from engine.src.config import EngineConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="File- and symbol-level retrieval benchmark for an indexed repo.")
    parser.add_argument("--config", required=True, help="quickcontext config JSON.")
    parser.add_argument("--project", required=True, help="Indexed project name.")
    parser.add_argument("--path", required=True, help="Repository root.")
    parser.add_argument("--cases-file", required=True, help="Benchmark cases JSON.")
    parser.add_argument("--strategies", default="auto,semantic,text", help="Comma list of auto, semantic, text.")
    parser.add_argument("--limit", type=int, default=5, help="Results per query.")
    parser.add_argument("--show-misses", action="store_true", help="Print cases missing at rank 1.")
    parser.add_argument("--json-out", default=None, help="Optional path for machine-readable results.")
    return parser.parse_args()


def _norm(path: str) -> str:
    return str(path or "").replace("\\\\?\\", "").replace("\\", "/").lower()


def _items(strategy: str, qc: QuickContext, query: str, args: argparse.Namespace) -> list[tuple[str, str | None]]:
    if strategy == "auto":
        payload = qc.retrieve_context_auto(query, project_name=args.project, path=args.path, limit=args.limit)
        results = payload.get("results", [])
    elif strategy == "semantic":
        results = qc.semantic_search(query, mode="hybrid", limit=args.limit, project_name=args.project, path=args.path)
    elif strategy == "text":
        results = qc.text_search(query=query, path=args.path, limit=args.limit, intent_mode=True, intent_level=2).matches
    else:
        raise ValueError(strategy)
    return [(getattr(item, "file_path", ""), getattr(item, "symbol_name", None) or getattr(item, "name", None)) for item in results]


def _rank(values: list, predicate) -> int | None:
    for idx, value in enumerate(values, 1):
        if predicate(value):
            return idx
    return None


def main() -> None:
    args = _parse_args()
    cases = json.loads(Path(args.cases_file).read_text(encoding="utf-8"))
    config = EngineConfig.from_json(args.config)
    report: dict[str, dict] = {}

    with QuickContext(config) as qc:
        qc.warm_project(args.path)
        for strategy in [item.strip() for item in args.strategies.split(",") if item.strip()]:
            rows = []
            cold_ms = None
            for case in cases:
                start = time.perf_counter()
                items = _items(strategy, qc, case["query"], args)
                latency = (time.perf_counter() - start) * 1000
                if cold_ms is None:
                    cold_ms = latency
                expected_paths = [fragment.lower() for fragment in case["expected_paths"]]
                expected_symbols = {name.lower() for name in case.get("expected_symbols", [])}
                file_rank = _rank(items, lambda item: any(fragment in _norm(item[0]) for fragment in expected_paths))
                symbol_rank = _rank(
                    items,
                    lambda item: bool(item[1])
                    and item[1].lower() in expected_symbols
                    and any(fragment in _norm(item[0]) for fragment in expected_paths),
                ) if expected_symbols else None
                rows.append({
                    "query": case["query"],
                    "kind": case.get("kind", ""),
                    "file_rank": file_rank,
                    "symbol_rank": symbol_rank,
                    "latency_ms": latency,
                    "top": [f"{Path(_norm(path)).name}:{name}" for path, name in items[:3]],
                })

            def mrr(key: str) -> float:
                return sum(1.0 / row[key] for row in rows if row[key]) / len(rows)

            latencies = sorted(row["latency_ms"] for row in rows[1:]) or [0.0]
            summary = {
                "cases": len(rows),
                "file_hit1": sum(1 for row in rows if row["file_rank"] == 1),
                "file_mrr": round(mrr("file_rank"), 4),
                "symbol_hit1": sum(1 for row in rows if row["symbol_rank"] == 1),
                "symbol_hit5": sum(1 for row in rows if row["symbol_rank"]),
                "symbol_mrr": round(mrr("symbol_rank"), 4),
                "cold_ms": round(cold_ms or 0.0, 1),
                "median_ms": round(median(latencies), 1),
                "p90_ms": round(latencies[int(0.9 * (len(latencies) - 1))], 1),
            }
            report[strategy] = {"summary": summary, "rows": rows}
            print(
                f"{strategy:9s} file Hit@1 {summary['file_hit1']}/{len(rows)}  file MRR {summary['file_mrr']:.3f}  "
                f"symbol Hit@1 {summary['symbol_hit1']}/{len(rows)}  Hit@5 {summary['symbol_hit5']}/{len(rows)}  "
                f"symbol MRR {summary['symbol_mrr']:.3f}  latency cold {summary['cold_ms']} ms, "
                f"median {summary['median_ms']} ms, p90 {summary['p90_ms']} ms"
            )
            if args.show_misses:
                for row in rows:
                    if row["file_rank"] != 1 or (row["symbol_rank"] or 99) > 1:
                        print(f"    file={row['file_rank']} sym={row['symbol_rank']}  {row['query'][:80]}  -> {row['top']}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
