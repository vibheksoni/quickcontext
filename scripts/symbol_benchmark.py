from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean, median
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.src.parsing import RustParserService


@dataclass(frozen=True, slots=True)
class SymbolEvalCase:
    query: str
    expected_path: str
    expected_name: str
    expected_parent: str | None = None


@dataclass(frozen=True, slots=True)
class SymbolEvalResult:
    case: SymbolEvalCase
    latency_ms: float
    hit_rank: int | None
    top_results: tuple[tuple[str, str, str | None, str], ...]


DEFAULT_CASES = (
    SymbolEvalCase(
        query="CollectionManager",
        expected_path="engine/src/collection.py",
        expected_name="CollectionManager",
    ),
    SymbolEvalCase(
        query="CodeSearcher._embed_query_with_provider_cached",
        expected_path="engine/src/searcher.py",
        expected_name="_embed_query_with_provider_cached",
        expected_parent="CodeSearcher",
    ),
    SymbolEvalCase(
        query="QuickContext._get_searcher",
        expected_path="engine/sdk.py",
        expected_name="_get_searcher",
        expected_parent="QuickContext",
    ),
    SymbolEvalCase(
        query="CollectionManager.active_collection",
        expected_path="engine/src/collection.py",
        expected_name="active_collection",
        expected_parent="CollectionManager",
    ),
    SymbolEvalCase(
        query="QdrantConnection.connect",
        expected_path="engine/src/qdrant.py",
        expected_name="connect",
        expected_parent="QdrantConnection",
    ),
    SymbolEvalCase(
        query="RustParserService.scan_files",
        expected_path="engine/src/parsing.py",
        expected_name="scan_files",
        expected_parent="RustParserService",
    ),
    SymbolEvalCase(
        query="FileSignatureCache.is_unchanged_from_metadata",
        expected_path="engine/src/filecache.py",
        expected_name="is_unchanged_from_metadata",
        expected_parent="FileSignatureCache",
    ),
    SymbolEvalCase(
        query="ChunkBuilder.build_chunks",
        expected_path="engine/src/chunker.py",
        expected_name="build_chunks",
        expected_parent="ChunkBuilder",
    ),
    SymbolEvalCase(
        query="QdrantIndexer.delete_by_file",
        expected_path="engine/src/indexer.py",
        expected_name="delete_by_file",
        expected_parent="QdrantIndexer",
    ),
    SymbolEvalCase(
        query="CodeSearcher.search_hybrid",
        expected_path="engine/src/searcher.py",
        expected_name="search_hybrid",
        expected_parent="CodeSearcher",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Rust-backed symbol lookup quality and latency.")
    parser.add_argument("--cases-file", default=None, help="Optional JSON file containing symbol benchmark cases.")
    parser.add_argument("--path", default=".", help="Project root or search scope path.")
    parser.add_argument("--limit", type=int, default=5, help="Results per query.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat the full benchmark N times.")
    parser.add_argument("--intent", action="store_true", help="Enable intent expansion.")
    parser.add_argument("--intent-level", type=int, default=2, help="Intent expansion aggressiveness.")
    parser.add_argument("--show-top", type=int, default=3, help="Show the top N results per query.")
    return parser.parse_args()


def _load_cases(cases_file: str | None) -> tuple[SymbolEvalCase, ...]:
    if not cases_file:
        return DEFAULT_CASES

    payload = json.loads(Path(cases_file).read_text(encoding="utf-8"))
    cases: list[SymbolEvalCase] = []
    for item in payload:
        query = str(item["query"]).strip()
        expected_path = str(item["expected_path"]).strip()
        expected_name = str(item["expected_name"]).strip()
        expected_parent = item.get("expected_parent")
        if not query or not expected_path or not expected_name:
            continue
        cases.append(
            SymbolEvalCase(
                query=query,
                expected_path=expected_path,
                expected_name=expected_name,
                expected_parent=str(expected_parent).strip() if expected_parent else None,
            )
        )
    return tuple(cases)


def _normalize_path(path: str) -> str:
    return path.replace("\\\\?\\", "").replace("\\", "/").lower()


def _relative_path(path: str, root: Path) -> str:
    candidate = Path(path.replace("\\\\?\\", ""))
    try:
        return str(candidate.resolve().relative_to(root)).replace("\\", "/")
    except Exception:
        return str(candidate).replace("\\", "/")


def _match_rank(
    results: list[tuple[str, str, str | None, str]],
    case: SymbolEvalCase,
) -> int | None:
    expected_path = case.expected_path.lower()
    expected_name = case.expected_name.lower()
    expected_parent = case.expected_parent.lower() if case.expected_parent else None

    for idx, (kind, name, parent, path) in enumerate(results, 1):
        del kind
        if expected_path not in path.lower():
            continue
        if name.lower() != expected_name:
            continue
        if expected_parent is not None and (parent or "").lower() != expected_parent:
            continue
        return idx
    return None


def _evaluate_case(
    svc: RustParserService,
    case: SymbolEvalCase,
    target_path: str,
    limit: int,
    intent_mode: bool,
    intent_level: int,
    repo_root: Path,
) -> SymbolEvalResult:
    started = time.perf_counter()
    result = svc.symbol_lookup(
        query=case.query,
        path=target_path,
        limit=limit,
        intent_mode=intent_mode,
        intent_level=intent_level,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    top_results = tuple(
        (
            item.kind,
            item.name,
            item.parent,
            _relative_path(item.file_path, repo_root),
        )
        for item in result.results
    )
    return SymbolEvalResult(
        case=case,
        latency_ms=latency_ms,
        hit_rank=_match_rank(list(top_results), case),
        top_results=top_results,
    )


def _print_results(results: list[SymbolEvalResult], show_top: int) -> None:
    hit1 = sum(1 for result in results if result.hit_rank == 1)
    hit3 = sum(1 for result in results if result.hit_rank is not None and result.hit_rank <= 3)
    mrr = sum(0.0 if result.hit_rank is None else 1.0 / result.hit_rank for result in results) / len(results)
    latencies = [result.latency_ms for result in results]

    print("Summary")
    print(f"  Cases: {len(results)}")
    print(f"  Hit@1: {hit1}/{len(results)}")
    print(f"  Hit@3: {hit3}/{len(results)}")
    print(f"  MRR: {mrr:.4f}")
    print(f"  Mean latency: {mean(latencies):.2f} ms")
    print(f"  Median latency: {median(latencies):.2f} ms")
    print()

    for result in results:
        rank_text = str(result.hit_rank) if result.hit_rank is not None else "miss"
        print(result.case.query)
        print(f"  hit_rank: {rank_text}")
        print(f"  latency_ms: {result.latency_ms:.2f}")
        for idx, (kind, name, parent, path) in enumerate(result.top_results[: max(1, show_top)], 1):
            parent_text = f" ({parent})" if parent else ""
            print(f"  {idx}. {kind} {name}{parent_text} [{path}]")
        print()


def main() -> None:
    args = _parse_args()
    repo_root = Path.cwd().resolve()
    cases = _load_cases(args.cases_file)
    target_path = str(Path(args.path).resolve())

    all_results: list[SymbolEvalResult] = []
    svc = RustParserService()
    try:
        svc._client.ensure_server(timeout_ms=10000)
        for _ in range(max(1, args.repeats)):
            for case in cases:
                all_results.append(
                    _evaluate_case(
                        svc=svc,
                        case=case,
                        target_path=target_path,
                        limit=args.limit,
                        intent_mode=args.intent,
                        intent_level=args.intent_level,
                        repo_root=repo_root,
                    )
                )
    finally:
        svc.close()

    _print_results(all_results, show_top=args.show_top)


if __name__ == "__main__":
    main()
