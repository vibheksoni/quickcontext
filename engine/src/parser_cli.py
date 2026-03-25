import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from engine.src.config import EngineConfig
from engine.src.parsing import RustParserService


def _load_config(config_path: str | None) -> EngineConfig:
    if config_path:
        return EngineConfig.from_json(config_path)
    return EngineConfig.auto()


def _service(config: EngineConfig) -> RustParserService:
    return RustParserService(config=config)


def _print_json(data) -> None:
    sys.stdout.write(json.dumps(data, ensure_ascii=False))
    sys.stdout.write("\n")


def _to_data(value):
    return asdict(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m engine")
    parser.add_argument("--config", dest="config_path", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_cmd = subparsers.add_parser("parse")
    parse_cmd.add_argument("path")
    parse_cmd.add_argument("--json-output", action="store_true")

    grep_cmd = subparsers.add_parser("grep")
    grep_cmd.add_argument("query")
    grep_cmd.add_argument("--path", dest="target_path", default=None)
    grep_cmd.add_argument("--no-gitignore", action="store_true")
    grep_cmd.add_argument("--limit", type=int, default=200)
    grep_cmd.add_argument("--context", type=int, default=0)
    grep_cmd.add_argument("--before-context", type=int, default=None)
    grep_cmd.add_argument("--after-context", type=int, default=None)

    symbol_cmd = subparsers.add_parser("symbol-lookup")
    symbol_cmd.add_argument("query")
    symbol_cmd.add_argument("--path", dest="target_path", default=None)
    symbol_cmd.add_argument("--no-gitignore", action="store_true")
    symbol_cmd.add_argument("--limit", type=int, default=50)
    symbol_cmd.add_argument("--intent", dest="intent_mode", action="store_true")
    symbol_cmd.add_argument("--intent-level", type=int, default=2)

    callers_cmd = subparsers.add_parser("find-callers")
    callers_cmd.add_argument("symbol")
    callers_cmd.add_argument("--path", dest="target_path", default=None)
    callers_cmd.add_argument("--no-gitignore", action="store_true")
    callers_cmd.add_argument("--limit", type=int, default=100)

    trace_cmd = subparsers.add_parser("trace-call-graph")
    trace_cmd.add_argument("symbol")
    trace_cmd.add_argument("--path", dest="target_path", default=None)
    trace_cmd.add_argument("--no-gitignore", action="store_true")
    trace_cmd.add_argument("--direction", choices=["upstream", "downstream", "both"], default="both")
    trace_cmd.add_argument("--depth", type=int, default=5)

    skeleton_cmd = subparsers.add_parser("skeleton")
    skeleton_cmd.add_argument("path")
    skeleton_cmd.add_argument("--depth", type=int, default=None)
    skeleton_cmd.add_argument("--no-signatures", action="store_true")
    skeleton_cmd.add_argument("--lines", action="store_true")
    skeleton_cmd.add_argument("--collapse", type=int, default=0)
    skeleton_cmd.add_argument("--no-gitignore", action="store_true")
    skeleton_cmd.add_argument("--markdown", action="store_true")

    import_graph_cmd = subparsers.add_parser("import-graph")
    import_graph_cmd.add_argument("file")
    import_graph_cmd.add_argument("--path", dest="project_path", default=None)
    import_graph_cmd.add_argument("--no-gitignore", action="store_true")

    find_importers_cmd = subparsers.add_parser("find-importers")
    find_importers_cmd.add_argument("file")
    find_importers_cmd.add_argument("--path", dest="project_path", default=None)
    find_importers_cmd.add_argument("--no-gitignore", action="store_true")

    return parser


def _cmd_parse(args, config: EngineConfig) -> int:
    svc = _service(config)
    try:
        result = svc.extract(Path(args.path))
        if args.json_output:
            _print_json([_to_data(item) for item in result])
        else:
            _print_json([_to_data(item) for item in result])
        return 0
    finally:
        svc.close()


def _cmd_grep(args, config: EngineConfig) -> int:
    svc = _service(config)
    try:
        before_context = max(0, args.context if args.before_context is None else args.before_context)
        after_context = max(0, args.context if args.after_context is None else args.after_context)
        result = svc.grep(
            query=args.query,
            path=args.target_path,
            respect_gitignore=not args.no_gitignore,
            limit=args.limit,
            before_context=before_context,
            after_context=after_context,
        )
        _print_json(_to_data(result))
        return 0
    finally:
        svc.close()


def _cmd_symbol_lookup(args, config: EngineConfig) -> int:
    svc = _service(config)
    try:
        result = svc.symbol_lookup(
            query=args.query,
            path=args.target_path,
            respect_gitignore=not args.no_gitignore,
            limit=args.limit,
            intent_mode=args.intent_mode,
            intent_level=args.intent_level,
        )
        _print_json(_to_data(result))
        return 0
    finally:
        svc.close()


def _cmd_find_callers(args, config: EngineConfig) -> int:
    svc = _service(config)
    try:
        result = svc.find_callers(
            symbol=args.symbol,
            path=args.target_path,
            respect_gitignore=not args.no_gitignore,
            limit=args.limit,
        )
        _print_json(_to_data(result))
        return 0
    finally:
        svc.close()


def _cmd_trace_call_graph(args, config: EngineConfig) -> int:
    svc = _service(config)
    try:
        result = svc.trace_call_graph(
            symbol=args.symbol,
            path=args.target_path,
            respect_gitignore=not args.no_gitignore,
            direction=args.direction,
            max_depth=args.depth,
        )
        _print_json(_to_data(result))
        return 0
    finally:
        svc.close()


def _cmd_skeleton(args, config: EngineConfig) -> int:
    svc = _service(config)
    try:
        fmt = "markdown" if args.markdown else "json"
        result = svc.skeleton(
            path=Path(args.path),
            max_depth=args.depth,
            include_signatures=not args.no_signatures,
            include_line_numbers=args.lines,
            collapse_threshold=args.collapse,
            respect_gitignore=not args.no_gitignore,
            format=fmt,
        )
        if args.markdown and result.markdown:
            sys.stdout.write(result.markdown)
            if not result.markdown.endswith("\n"):
                sys.stdout.write("\n")
        else:
            _print_json(_to_data(result))
        return 0
    finally:
        svc.close()


def _cmd_import_graph(args, config: EngineConfig) -> int:
    svc = _service(config)
    try:
        result = svc.import_graph(
            file=Path(args.file),
            path=args.project_path,
            respect_gitignore=not args.no_gitignore,
        )
        _print_json(_to_data(result))
        return 0
    finally:
        svc.close()


def _cmd_find_importers(args, config: EngineConfig) -> int:
    svc = _service(config)
    try:
        result = svc.find_importers(
            file=Path(args.file),
            path=args.project_path,
            respect_gitignore=not args.no_gitignore,
        )
        _print_json(_to_data(result))
        return 0
    finally:
        svc.close()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _load_config(args.config_path)

    dispatch = {
        "parse": _cmd_parse,
        "grep": _cmd_grep,
        "symbol-lookup": _cmd_symbol_lookup,
        "find-callers": _cmd_find_callers,
        "trace-call-graph": _cmd_trace_call_graph,
        "skeleton": _cmd_skeleton,
        "import-graph": _cmd_import_graph,
        "find-importers": _cmd_find_importers,
    }
    return dispatch[args.command](args, config)
