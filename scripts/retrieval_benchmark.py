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

from engine import QuickContext
from engine.src.config import EngineConfig
from engine.src.project import detect_project_name


@dataclass(frozen=True, slots=True)
class EvalCase:
    query: str
    expected_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalResult:
    case: EvalCase
    latency_ms: float
    hit_rank: int | None
    top_paths: tuple[str, ...]
    top3_coverage: float


DEFAULT_CASES = (
    EvalCase(
        query="How does the Python layer decide how to connect to the Rust service on Windows versus Linux?",
        expected_paths=("engine/src/parsing.py", "engine/src/pipe.py"),
    ),
    EvalCase(
        query="How does full indexing avoid replacing unchanged data when it builds a shadow collection?",
        expected_paths=("engine/__init__.py", "engine/src/collection.py"),
    ),
    EvalCase(
        query="How are query embeddings cached or reused in the search layer for repeated searches?",
        expected_paths=("engine/src/searcher.py",),
    ),
    EvalCase(
        query="How does this codebase make parser-only Python commands work without importing the vector store at startup?",
        expected_paths=("engine/src/parsing.py", "engine/__init__.py"),
    ),
    EvalCase(
        query="How are stale chunks removed when a file changes and some chunks are filtered out?",
        expected_paths=("engine/__init__.py", "engine/src/differ.py", "engine/src/indexer.py"),
    ),
    EvalCase(
        query="How does the code detect unchanged files before indexing?",
        expected_paths=("engine/__init__.py", "engine/src/filecache.py"),
    ),
    EvalCase(
        query="Where are the text and symbol indexes stored on disk?",
        expected_paths=("service/src/text_index.rs", "service/src/symbol_index.rs"),
    ),
    EvalCase(
        query="How are provider dependencies loaded so one-shot CLI commands start faster?",
        expected_paths=(
            "engine/src/providers/litellm_provider.py",
            "engine/src/providers/fastembed_provider.py",
            "engine/src/providers/factory.py",
        ),
    ),
    EvalCase(
        query="How does path-scoped semantic search work in this codebase?",
        expected_paths=("engine/src/searcher.py", "engine/src/indexer.py", "engine/__init__.py"),
    ),
    EvalCase(
        query="How are call edges built for symbol caller tracing?",
        expected_paths=("service/src/symbol_index.rs",),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark broader architecture-question retrieval quality.")
    parser.add_argument("--config", default=None, help="Path to quickcontext config JSON. Defaults to auto-discovery.")
    parser.add_argument("--project", default=None, help="Indexed project name. Defaults to auto-detect from cwd.")
    parser.add_argument("--cases-file", default=None, help="Optional JSON file containing benchmark cases.")
    parser.add_argument("--mode", default="hybrid", choices=("code", "desc", "hybrid"), help="Semantic search mode.")
    parser.add_argument("--limit", type=int, default=5, help="Results per query.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat the full benchmark N times.")
    parser.add_argument("--use-keywords", action="store_true", help="Enable payload keyword blending.")
    parser.add_argument("--rerank", action="store_true", help="Enable reranking.")
    parser.add_argument("--show-top", type=int, default=3, help="Show the top N result paths per query.")
    return parser.parse_args()


def _load_config(config_path: str | None) -> EngineConfig:
    if config_path:
        return EngineConfig.from_json(config_path)
    return EngineConfig.auto()


def _load_cases(cases_file: str | None) -> tuple[EvalCase, ...]:
    if not cases_file:
        return DEFAULT_CASES

    payload = json.loads(Path(cases_file).read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for item in payload:
        query = str(item["query"]).strip()
        expected_paths = tuple(str(path) for path in item["expected_paths"])
        if not query or not expected_paths:
            continue
        cases.append(EvalCase(query=query, expected_paths=expected_paths))
    return tuple(cases)


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lower()


def _relative_path(path: str, root: Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(root)).replace("\\", "/")
    except Exception:
        return str(candidate).replace("\\", "/")


def _match_rank(paths: list[str], expected_paths: tuple[str, ...]) -> int | None:
    normalized_expected = tuple(fragment.lower() for fragment in expected_paths)
    for idx, path in enumerate(paths, 1):
        normalized_path = _normalize_path(path)
        if any(fragment in normalized_path for fragment in normalized_expected):
            return idx
    return None


def _top_path_coverage(paths: list[str], expected_paths: tuple[str, ...], limit: int = 3) -> float:
    if not expected_paths:
        return 0.0

    normalized_expected = tuple(fragment.lower() for fragment in expected_paths)
    matched = 0
    for fragment in normalized_expected:
        if any(fragment in _normalize_path(path) for path in paths[:limit]):
            matched += 1
    return matched / len(normalized_expected)


def _evaluate_case(
    qc: QuickContext,
    case: EvalCase,
    project_name: str,
    mode: str,
    limit: int,
    use_keywords: bool,
    rerank: bool,
    repo_root: Path,
) -> EvalResult:
    started = time.perf_counter()
    results = qc.semantic_search(
        query=case.query,
        mode=mode,
        limit=limit,
        project_name=project_name,
        use_keywords=use_keywords,
        rerank=rerank,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    top_paths = tuple(_relative_path(item.file_path, repo_root) for item in results)
    return EvalResult(
        case=case,
        latency_ms=latency_ms,
        hit_rank=_match_rank(list(top_paths), case.expected_paths),
        top_paths=top_paths,
        top3_coverage=_top_path_coverage(list(top_paths), case.expected_paths, limit=3),
    )


def _print_results(results: list[EvalResult], show_top: int) -> None:
    hit1 = sum(1 for result in results if result.hit_rank == 1)
    hit3 = sum(1 for result in results if result.hit_rank is not None and result.hit_rank <= 3)
    mrr = sum(0.0 if result.hit_rank is None else 1.0 / result.hit_rank for result in results) / len(results)
    latencies = [result.latency_ms for result in results]
    mean_coverage = sum(result.top3_coverage for result in results) / len(results)

    print("Summary")
    print(f"  Cases: {len(results)}")
    print(f"  Hit@1: {hit1}/{len(results)}")
    print(f"  Hit@3: {hit3}/{len(results)}")
    print(f"  MRR: {mrr:.4f}")
    print(f"  Mean top-3 expected-path coverage: {mean_coverage:.4f}")
    print(f"  Mean latency: {mean(latencies):.2f} ms")
    print(f"  Median latency: {median(latencies):.2f} ms")
    print()

    for result in results:
        rank_text = str(result.hit_rank) if result.hit_rank is not None else "miss"
        print(result.case.query)
        print(f"  hit_rank: {rank_text}")
        print(f"  top3_coverage: {result.top3_coverage:.2f}")
        print(f"  latency_ms: {result.latency_ms:.2f}")
        for idx, path in enumerate(result.top_paths[: max(1, show_top)], 1):
            print(f"  {idx}. {path}")
        print()


def main() -> None:
    args = _parse_args()
    repo_root = Path.cwd().resolve()
    config = _load_config(args.config)
    project_name = args.project or detect_project_name(repo_root, manual_override=None)
    cases = _load_cases(args.cases_file)

    all_results: list[EvalResult] = []
    with QuickContext(config) as qc:
        for _ in range(max(1, args.repeats)):
            for case in cases:
                all_results.append(
                    _evaluate_case(
                        qc=qc,
                        case=case,
                        project_name=project_name,
                        mode=args.mode,
                        limit=args.limit,
                        use_keywords=args.use_keywords,
                        rerank=args.rerank,
                        repo_root=repo_root,
                    )
                )

    _print_results(all_results, show_top=args.show_top)


if __name__ == "__main__":
    main()
