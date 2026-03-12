from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine import QuickContext
from engine.src.cli import _optimize_search_config
from engine.src.config import EngineConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure real-config semantic search phases.")
    parser.add_argument("--config", default=None, help="Path to quickcontext config JSON.")
    parser.add_argument("--project", default="quickcontext", help="Target indexed project.")
    parser.add_argument("--query", required=True, help="Search query text.")
    parser.add_argument("--limit", type=int, default=5, help="Result limit.")
    parser.add_argument("--mode", choices=("code", "desc", "hybrid"), default="hybrid")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--force-connect", action="store_true", help="Include explicit Qdrant connect timing.")
    parser.add_argument("--include-source", action="store_true", help="Hydrate full source content in final results.")
    return parser.parse_args()


def _load_config(config_path: str | None) -> EngineConfig:
    if config_path:
        return EngineConfig.from_json(config_path)
    return EngineConfig.auto()


def _wrap_method(obj, name: str, timings: dict[str, float], counts: dict[str, int]):
    original = getattr(obj, name)

    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timings[name] += (time.perf_counter() - start) * 1000
            counts[name] += 1

    setattr(obj, name, wrapped)
    return original


def main() -> None:
    args = _parse_args()
    config = _optimize_search_config(_load_config(args.config))

    timings: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    t0 = time.perf_counter()
    qc = QuickContext(config)
    t1 = time.perf_counter()
    if args.force_connect:
        qc.connect()
    t2 = time.perf_counter()
    code_provider = qc.code_provider
    t3 = time.perf_counter()
    desc_provider = qc.desc_provider
    t4 = time.perf_counter()
    searcher = qc._get_searcher(args.project, rerank=args.rerank)
    t5 = time.perf_counter()

    originals = []
    for method_name in (
        "_embed_query_cached",
        "_embed_query_with_provider_cached",
        "_extract_keywords_cached",
        "_batch_search",
        "_search",
        "_hydrate_results",
        "_rrf_fuse",
        "_blend_with_rerank",
    ):
        if hasattr(searcher, method_name):
            originals.append((searcher, method_name, _wrap_method(searcher, method_name, timings, counts)))

    client = searcher._client
    for method_name in ("query_points", "query_batch_points", "retrieve"):
        if hasattr(client, method_name):
            originals.append((client, method_name, _wrap_method(client, method_name, timings, counts)))

    try:
        start = time.perf_counter()
        if args.mode == "code":
            results = searcher.search_code(args.query, limit=args.limit, rerank=args.rerank, include_source=args.include_source)
        elif args.mode == "desc":
            results = searcher.search_description(args.query, limit=args.limit, rerank=args.rerank, include_source=args.include_source)
        else:
            results = searcher.search_hybrid(args.query, limit=args.limit, rerank=args.rerank, include_source=args.include_source)
        t6 = time.perf_counter()
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)
        qc.close()

    print(f"query={args.query!r}")
    print(f"mode={args.mode}")
    print(f"limit={args.limit}")
    print(f"rerank={args.rerank}")
    print(f"force_connect={args.force_connect}")
    print(f"include_source={args.include_source}")
    print(f"startup_construct_ms={round((t1 - t0) * 1000, 2)}")
    print(f"startup_connect_ms={round((t2 - t1) * 1000, 2)}")
    print(f"startup_code_provider_ms={round((t3 - t2) * 1000, 2)}")
    print(f"startup_desc_provider_ms={round((t4 - t3) * 1000, 2)}")
    print(f"startup_searcher_ms={round((t5 - t4) * 1000, 2)}")
    print(f"search_total_ms={round((t6 - start) * 1000, 2)}")
    print(f"process_total_ms={round((t6 - t0) * 1000, 2)}")
    print("phases:")
    for name in sorted(timings):
        print(f"  {name}_ms={round(timings[name], 2)} calls={counts[name]}")
    response_meta = getattr(client, "last_response_meta", None)
    if response_meta is not None:
        if response_meta.server_time_ms is not None:
            print(f"query_server_time_ms={round(response_meta.server_time_ms, 2)}")
        print(f"query_response_bytes={response_meta.response_bytes}")
        print(f"query_status_code={response_meta.status_code}")
    if results:
        top = results[0]
        print(f"top_file={top.file_path}")
        print(f"top_symbol={top.symbol_name}")


if __name__ == "__main__":
    main()
