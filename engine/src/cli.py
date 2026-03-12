import sys
from dataclasses import replace
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from engine import QuickContext
from engine.src.config import EngineConfig
from engine.src.project import detect_project_name


console = Console(legacy_windows=False)


def _load_config(config_path: str | None) -> EngineConfig:
    """
    Load engine config from JSON file, auto-discovery, or defaults.

    config_path: str | None — Path to JSON config file, or None for auto-discovery.
    Returns: EngineConfig — Resolved configuration.
    """
    if config_path:
        return EngineConfig.from_json(config_path)
    return EngineConfig.auto()


def _optimize_search_config(config: EngineConfig) -> EngineConfig:
    """
    Prefer REST for local one-shot search traffic where it is measurably faster.
    """
    if config.qdrant is None:
        return config

    host = (config.qdrant.host or "").lower()
    if config.qdrant.prefer_grpc and host in {"localhost", "127.0.0.1"}:
        return replace(config, qdrant=replace(config.qdrant, prefer_grpc=False))
    return config


@click.group()
@click.option("--config", "config_path", default=None, help="Path to JSON config file.")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None) -> None:
    """
    quickcontext engine — local code context engine CLI.
    """
    ctx.ensure_object(dict)
    ctx.obj["config"] = _load_config(config_path)


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """
    Show engine status: Qdrant connection, collection info, embedding providers.
    """
    config = _optimize_search_config(ctx.obj["config"])
    qc = QuickContext(config)

    table = Table(title="quickcontext status")
    table.add_column("Component", style="cyan")
    table.add_column("Detail", style="white")
    table.add_column("Status", style="green")

    if config.qdrant is not None:
        alive = False
        try:
            qc.connect(verify=True)
            alive = True
        except ConnectionError:
            pass

        table.add_row(
            "Qdrant",
            config.qdrant.url,
            "[green]connected[/green]" if alive else "[red]unreachable[/red]",
        )

        if alive:
            try:
                info = qc.collection.info()
                table.add_row(
                    "Collection",
                    info["name"],
                    f'[green]{info["status"]}[/green] ({info["points_count"]} points)',
                )
                for vec_name, vec_info in info["vectors"].items():
                    table.add_row(
                        f"  Vector: {vec_name}",
                        f'dim={vec_info["size"]} dist={vec_info["distance"]}',
                        "[green]ok[/green]",
                    )
            except Exception as exc:
                table.add_row("Collection", str(config.qdrant.collection), f"[red]{exc}[/red]")
    else:
        table.add_row("Qdrant", "disabled", "[dim]not configured[/dim]")

    if config.code_embedding is not None:
        table.add_row(
            "Code Embedding",
            f"{config.code_embedding.provider} / {config.code_embedding.model}",
            f"dim={config.code_embedding.dimension}",
        )
    else:
        table.add_row("Code Embedding", "disabled", "[dim]not configured[/dim]")

    if config.desc_embedding is not None:
        table.add_row(
            "Desc Embedding",
            f"{config.desc_embedding.provider} / {config.desc_embedding.model}",
            f"dim={config.desc_embedding.dimension}",
        )
    else:
        table.add_row("Desc Embedding", "disabled", "[dim]not configured[/dim]")

    if config.llm is not None:
        table.add_row(
            "LLM",
            f"{config.llm.provider} / {config.llm.model}",
            f"max_tokens={config.llm.max_tokens}",
        )
    else:
        table.add_row("LLM", "disabled", "[dim]not configured[/dim]")

    console.print(table)
    qc.close()


@cli.command("mcp-hints")
@click.option("--project", default=None, help="Project name (collection) to inspect. Auto-detected from cwd if not specified.")
@click.pass_context
def mcp_hints(ctx: click.Context, project: str | None) -> None:
    """
    Print dynamic search instructions suitable for MCP/system prompts.
    """
    config = _optimize_search_config(ctx.obj["config"])

    try:
        with QuickContext(config) as qc:
            project_name = project if project else detect_project_name(Path.cwd(), manual_override=None)
            hints = qc.build_dynamic_search_instructions(project_name=project_name)
            console.print(hints)
    except Exception as exc:
        console.print(f"[red]Failed to build MCP hints:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """
    Initialize: connect to Qdrant, create collection if needed.
    """
    config = ctx.obj["config"]

    if config.qdrant is None:
        console.print("[red]Qdrant is disabled in config. Cannot initialize.[/red]")
        sys.exit(1)

    try:
        with QuickContext(config) as qc:
            info = qc.collection.info()
            console.print(f"[green]Collection '{info['name']}' ready[/green] — {info['points_count']} points")
    except ConnectionError as exc:
        console.print(f"[red]Connection failed:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt.")
@click.option("--project", default=None, help="Project name (collection) to reset. If not specified, resets default 'codebase' collection.")
@click.pass_context
def reset(ctx: click.Context, confirm: bool, project: str | None) -> None:
    """
    Delete and recreate the collection. Destroys all indexed data.
    """
    config = ctx.obj["config"]

    if config.qdrant is None:
        console.print("[red]Qdrant is disabled in config. Cannot reset.[/red]")
        sys.exit(1)

    collection_name = project if project else (config.qdrant.collection or "codebase")

    if not confirm:
        click.confirm(
            f"Delete collection '{collection_name}' and all data?",
            abort=True,
        )

    try:
        with QuickContext(config) as qc:
            coll = qc._get_collection(collection_name)
            coll.delete()
            coll.create()
            console.print(f"[green]Collection '{collection_name}' reset[/green]")
    except ConnectionError as exc:
        console.print(f"[red]Connection failed:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--json-output", is_flag=True, help="Print raw extraction JSON.")
@click.pass_context
def parse(ctx: click.Context, path: Path, json_output: bool) -> None:
    """
    Parse symbols via Rust pipe service for a file or directory.
    """
    config = ctx.obj["config"]

    try:
        qc = QuickContext(config)
        results = qc.extract_symbols(path)
    except Exception as exc:
        console.print(f"[red]Parse failed:[/red] {exc}")
        sys.exit(1)

    if json_output:
        import json

        payload = [
            {
                "file_path": item.file_path,
                "language": item.language,
                "errors": item.errors,
                "symbols": [
                    {
                        "name": symbol.name,
                        "kind": symbol.kind,
                        "language": symbol.language,
                        "file_path": symbol.file_path,
                        "line_start": symbol.line_start,
                        "line_end": symbol.line_end,
                        "byte_start": symbol.byte_start,
                        "byte_end": symbol.byte_end,
                        "source": symbol.source,
                        "signature": symbol.signature,
                        "docstring": symbol.docstring,
                        "params": symbol.params,
                        "return_type": symbol.return_type,
                        "parent": symbol.parent,
                        "visibility": symbol.visibility,
                        "role": symbol.role,
                    }
                    for symbol in item.symbols
                ],
            }
            for item in results
        ]
        console.print_json(json.dumps(payload))
        return

    file_count = len(results)
    symbol_count = sum(len(item.symbols) for item in results)
    error_count = sum(len(item.errors) for item in results)

    table = Table(title="parse results")
    table.add_column("File", style="cyan")
    table.add_column("Language", style="white")
    table.add_column("Symbols", style="green")
    table.add_column("Roles", style="magenta")
    table.add_column("Errors", style="red")

    for item in results:
        role_counts: dict[str, int] = {}
        for sym in item.symbols:
            r = sym.role or "unknown"
            role_counts[r] = role_counts.get(r, 0) + 1
        role_summary = " ".join(f"{k}:{v}" for k, v in sorted(role_counts.items()))
        table.add_row(
            item.file_path,
            item.language,
            str(len(item.symbols)),
            role_summary,
            str(len(item.errors)),
        )

    console.print(table)
    console.print(
        f"[green]files={file_count}[/green]  "
        f"[green]symbols={symbol_count}[/green]  "
        f"[red]errors={error_count}[/red]"
    )


@cli.command("extract-symbol")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("symbol")
@click.option("--json-output", is_flag=True, help="Print raw JSON response.")
@click.pass_context
def extract_symbol(ctx: click.Context, file: Path, symbol: str, json_output: bool) -> None:
    """
    Extract a specific symbol by name from FILE.

    SYMBOL can be a plain name like "extract_file" or "Parent.method" for disambiguation.
    """
    config = ctx.obj["config"]

    try:
        qc = QuickContext(config)
        result = qc.extract_symbol(file, symbol)
    except Exception as exc:
        console.print(f"[red]Extract symbol failed:[/red] {exc}")
        sys.exit(1)

    if json_output:
        import json

        payload = {
            "file_path": result.file_path,
            "language": result.language,
            "query": result.query,
            "total_matches": result.total_matches,
            "symbols": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "language": s.language,
                    "file_path": s.file_path,
                    "line_start": s.line_start,
                    "line_end": s.line_end,
                    "byte_start": s.byte_start,
                    "byte_end": s.byte_end,
                    "source": s.source,
                    "signature": s.signature,
                    "docstring": s.docstring,
                    "params": s.params,
                    "return_type": s.return_type,
                    "parent": s.parent,
                    "visibility": s.visibility,
                    "role": s.role,
                }
                for s in result.symbols
            ],
        }
        console.print_json(json.dumps(payload))
        return

    console.print(
        f"[cyan]{result.file_path}[/cyan]  "
        f"[white]{result.language}[/white]  "
        f"query=[yellow]{result.query}[/yellow]  "
        f"matches=[green]{result.total_matches}[/green]"
    )

    table = Table(title="matched symbols")
    table.add_column("Name", style="cyan")
    table.add_column("Kind", style="white")
    table.add_column("Lines", style="green")
    table.add_column("Parent", style="magenta")
    table.add_column("Role", style="yellow")
    table.add_column("Signature", style="dim")

    for s in result.symbols:
        table.add_row(
            s.name,
            s.kind,
            f"{s.line_start}-{s.line_end}",
            s.parent or "",
            s.role or "",
            s.signature or "",
        )

    console.print(table)


@cli.command()
@click.argument("text")
@click.option("--provider", "which", default="code", type=click.Choice(["code", "desc"]))
@click.pass_context
def embed_test(ctx: click.Context, text: str, which: str) -> None:
    """
    Test embedding generation. Embeds TEXT and prints vector stats.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    provider = qc.code_provider if which == "code" else qc.desc_provider

    console.print(f"[cyan]Provider:[/cyan] {provider._config.provider} / {provider.model}")
    console.print(f"[cyan]Embedding:[/cyan] {text!r}")

    vec = provider.embed_single(text)
    console.print(f"[green]Dimension:[/green] {vec.shape[0]}")
    console.print(f"[green]Min:[/green] {vec.min():.6f}  [green]Max:[/green] {vec.max():.6f}  [green]Mean:[/green] {vec.mean():.6f}")


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--force", is_flag=True, help="Force re-indexing even if files unchanged.")
@click.option("--quiet", is_flag=True, help="Suppress progress messages.")
@click.option("--project", default=None, help="Project name override. Auto-detected from directory if not specified.")
@click.option("--fast", is_flag=True, help="Fast indexing mode: skip descriptions and use stricter chunk filtering.")
@click.option("--target-seconds", default=None, type=int, help="SLA mode target duration in seconds. Applies aggressive speed defaults when <= 60 unless explicitly overridden.")
@click.option("--no-descriptions", is_flag=True, help="Skip LLM description generation and embed with lightweight fallback descriptions.")
@click.option("--min-chunk-bytes", default=120, type=int, help="Minimum chunk size in bytes to keep.")
@click.option("--max-chunks-per-file", default=400, type=int, help="Maximum chunks kept per file after quality ranking.")
@click.option("--max-total-chunks", default=None, type=int, help="Maximum total chunks kept for the indexing run.")
@click.option("--compress-for-embedding", default=None, type=click.Choice(["light", "medium", "aggressive"]), help="Compress source before embeddings to reduce comments and boilerplate.")
@click.option("--incremental-resume", is_flag=True, help="Continue indexing changed files directly on active collection (no shadow swap).")
@click.option("--resume-batch-files", default=200, type=int, help="When --incremental-resume is set, files processed per batch.")
@click.option("--no-skip-minified", is_flag=True, help="Disable minified/noise chunk skipping.")
@click.option("--embedding-concurrency", default=None, type=int, help="Override embedding request concurrency for this run.")
@click.option("--embedding-max-retries", default=None, type=int, help="Override embedding retry count for this run.")
@click.option("--embedding-batch-size", default=None, type=int, help="Override embedding batch size for this run.")
@click.option("--embedding-fixed-batch", is_flag=True, help="Disable adaptive embedding batch resizing for this run.")
@click.option("--upsert-batch-size", default=None, type=int, help="Override Qdrant upsert batch size for this run.")
@click.option("--upsert-concurrency", default=None, type=int, help="Override Qdrant upsert worker concurrency for this run.")
@click.pass_context
def index(
    ctx: click.Context,
    directory: Path,
    force: bool,
    quiet: bool,
    project: str | None,
    fast: bool,
    target_seconds: int | None,
    no_descriptions: bool,
    min_chunk_bytes: int,
    max_chunks_per_file: int,
    max_total_chunks: int | None,
    compress_for_embedding: str | None,
    incremental_resume: bool,
    resume_batch_files: int,
    no_skip_minified: bool,
    embedding_concurrency: int | None,
    embedding_max_retries: int | None,
    embedding_batch_size: int | None,
    embedding_fixed_batch: bool,
    upsert_batch_size: int | None,
    upsert_concurrency: int | None,
) -> None:
    """
    Index a directory: extract, chunk, optionally describe, embed, and upsert to Qdrant.
    """
    config = ctx.obj["config"]

    try:
        effective_config = config
        if upsert_batch_size is not None or upsert_concurrency is not None:
            if effective_config.qdrant is None:
                raise RuntimeError("Cannot override upsert settings when qdrant is disabled")
            qdrant_cfg = effective_config.qdrant
            if upsert_batch_size is not None:
                qdrant_cfg = replace(qdrant_cfg, upsert_batch_size=max(1, int(upsert_batch_size)))
            if upsert_concurrency is not None:
                qdrant_cfg = replace(qdrant_cfg, upsert_concurrency=max(1, int(upsert_concurrency)))
            effective_config = replace(effective_config, qdrant=qdrant_cfg)

        with QuickContext(effective_config) as qc:
            effective_fast = fast
            effective_no_descriptions = no_descriptions
            effective_min_chunk_bytes = max(1, int(min_chunk_bytes))
            effective_max_chunks_per_file = max(1, int(max_chunks_per_file))
            effective_skip_minified = not no_skip_minified
            effective_embedding_concurrency = embedding_concurrency
            effective_embedding_max_retries = embedding_max_retries
            effective_embedding_batch_size = embedding_batch_size
            effective_embedding_adaptive_batching = None if not embedding_fixed_batch else False

            if target_seconds is not None and target_seconds <= 60:
                effective_fast = True
                effective_no_descriptions = True
                if effective_embedding_concurrency is None:
                    effective_embedding_concurrency = 16
                if effective_embedding_max_retries is None:
                    effective_embedding_max_retries = 2
                if effective_embedding_batch_size is None:
                    effective_embedding_batch_size = 64
                effective_embedding_adaptive_batching = True
                effective_min_chunk_bytes = max(effective_min_chunk_bytes, 240)
                effective_max_chunks_per_file = min(effective_max_chunks_per_file, 140)
                effective_skip_minified = True

            stats = qc.index_directory(
                directory=directory,
                force_refresh=force,
                show_progress=not quiet,
                project_name=project,
                fast=effective_fast,
                generate_descriptions=not effective_no_descriptions,
                min_chunk_bytes=effective_min_chunk_bytes,
                max_chunks_per_file=effective_max_chunks_per_file,
                max_total_chunks=max_total_chunks,
                compress_for_embedding=compress_for_embedding,
                incremental_resume=incremental_resume,
                resume_batch_files=max(1, int(resume_batch_files)),
                skip_minified=effective_skip_minified,
                embedding_concurrency=effective_embedding_concurrency,
                embedding_max_retries=effective_embedding_max_retries,
                embedding_batch_size=effective_embedding_batch_size,
                embedding_adaptive_batching=effective_embedding_adaptive_batching,
            )

            if not quiet:
                console.print("\n[green]Indexing complete[/green]")
                console.print(f"  Chunks: {stats.total_chunks}")
                console.print(f"  Upserted: {stats.upserted_points}")
                console.print(f"  Failed: {stats.failed_points}")
                console.print(f"  Tokens: {stats.total_tokens}")
                console.print(f"  Duration: {stats.duration_seconds:.2f}s")
                console.print(
                    "  Filtering: "
                    f"small={stats.skipped_small_chunks}, "
                    f"minified={stats.skipped_minified_chunks}, "
                    f"capped={stats.skipped_capped_chunks}, "
                    f"files_capped={stats.files_capped}"
                )
                console.print(f"  Descriptions enabled: {stats.descriptions_enabled}")
                console.print(
                    "  Embedding telemetry: "
                    f"requests={stats.embedding_requests}, "
                    f"retries={stats.embedding_retries}, "
                    f"failed_requests={stats.embedding_failed_requests}, "
                    f"inputs={stats.embedding_input_count}, "
                    f"duration={stats.embedding_stage_duration_seconds:.2f}s, "
                    f"final_batch={stats.embedding_final_batch_size}, "
                    f"shrink={stats.embedding_batch_shrink_events}, "
                    f"grow={stats.embedding_batch_grow_events}"
                )
                total_cost = stats.llm_cost_usd + stats.embedding_cost_usd
                console.print(
                    f"  Cost: ${total_cost:.4f} "
                    f"(LLM: ${stats.llm_cost_usd:.4f}, Embeddings: ${stats.embedding_cost_usd:.4f})"
                )

    except Exception as exc:
        console.print(f"[red]Indexing failed:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--quiet", is_flag=True, help="Suppress progress messages.")
@click.option("--fast", is_flag=True, help="Fast refresh mode: skip descriptions and use stricter chunk filtering.")
@click.option("--target-seconds", default=None, type=int, help="SLA mode target duration in seconds. Applies aggressive speed defaults when <= 60 unless explicitly overridden.")
@click.option("--no-descriptions", is_flag=True, help="Skip LLM description generation and use lightweight fallback descriptions.")
@click.option("--min-chunk-bytes", default=120, type=int, help="Minimum chunk size in bytes to keep.")
@click.option("--max-chunks-per-file", default=400, type=int, help="Maximum chunks kept per file after quality ranking.")
@click.option("--max-total-chunks", default=None, type=int, help="Maximum total chunks kept for the refresh run.")
@click.option("--compress-for-embedding", default=None, type=click.Choice(["light", "medium", "aggressive"]), help="Compress source before embeddings to reduce comments and boilerplate.")
@click.option("--no-skip-minified", is_flag=True, help="Disable minified/noise chunk skipping.")
@click.option("--embedding-concurrency", default=None, type=int, help="Override embedding request concurrency for this run.")
@click.option("--embedding-max-retries", default=None, type=int, help="Override embedding retry count for this run.")
@click.option("--embedding-batch-size", default=None, type=int, help="Override embedding batch size for this run.")
@click.option("--embedding-fixed-batch", is_flag=True, help="Disable adaptive embedding batch resizing for this run.")
@click.option("--upsert-batch-size", default=None, type=int, help="Override Qdrant upsert batch size for this run.")
@click.option("--upsert-concurrency", default=None, type=int, help="Override Qdrant upsert worker concurrency for this run.")
@click.pass_context
def refresh(
    ctx: click.Context,
    files: tuple[Path, ...],
    quiet: bool,
    fast: bool,
    target_seconds: int | None,
    no_descriptions: bool,
    min_chunk_bytes: int,
    max_chunks_per_file: int,
    max_total_chunks: int | None,
    compress_for_embedding: str | None,
    no_skip_minified: bool,
    embedding_concurrency: int | None,
    embedding_max_retries: int | None,
    embedding_batch_size: int | None,
    embedding_fixed_batch: bool,
    upsert_batch_size: int | None,
    upsert_concurrency: int | None,
) -> None:
    """
    Refresh specific files: delete old chunks, re-extract, re-embed, re-index.
    """
    config = ctx.obj["config"]

    if not files:
        console.print("[red]No files specified[/red]")
        sys.exit(1)

    try:
        effective_config = config
        if upsert_batch_size is not None or upsert_concurrency is not None:
            if effective_config.qdrant is None:
                raise RuntimeError("Cannot override upsert settings when qdrant is disabled")
            qdrant_cfg = effective_config.qdrant
            if upsert_batch_size is not None:
                qdrant_cfg = replace(qdrant_cfg, upsert_batch_size=max(1, int(upsert_batch_size)))
            if upsert_concurrency is not None:
                qdrant_cfg = replace(qdrant_cfg, upsert_concurrency=max(1, int(upsert_concurrency)))
            effective_config = replace(effective_config, qdrant=qdrant_cfg)

        with QuickContext(effective_config) as qc:
            effective_fast = fast
            effective_no_descriptions = no_descriptions
            effective_min_chunk_bytes = max(1, int(min_chunk_bytes))
            effective_max_chunks_per_file = max(1, int(max_chunks_per_file))
            effective_skip_minified = not no_skip_minified
            effective_embedding_concurrency = embedding_concurrency
            effective_embedding_max_retries = embedding_max_retries
            effective_embedding_batch_size = embedding_batch_size
            effective_embedding_adaptive_batching = None if not embedding_fixed_batch else False

            if target_seconds is not None and target_seconds <= 60:
                effective_fast = True
                effective_no_descriptions = True
                if effective_embedding_concurrency is None:
                    effective_embedding_concurrency = 16
                if effective_embedding_max_retries is None:
                    effective_embedding_max_retries = 2
                if effective_embedding_batch_size is None:
                    effective_embedding_batch_size = 64
                effective_embedding_adaptive_batching = True
                effective_min_chunk_bytes = max(effective_min_chunk_bytes, 240)
                effective_max_chunks_per_file = min(effective_max_chunks_per_file, 140)
                effective_skip_minified = True

            stats = qc.refresh_files(
                file_paths=list(files),
                show_progress=not quiet,
                fast=effective_fast,
                generate_descriptions=not effective_no_descriptions,
                min_chunk_bytes=effective_min_chunk_bytes,
                max_chunks_per_file=effective_max_chunks_per_file,
                max_total_chunks=max_total_chunks,
                compress_for_embedding=compress_for_embedding,
                skip_minified=effective_skip_minified,
                embedding_concurrency=effective_embedding_concurrency,
                embedding_max_retries=effective_embedding_max_retries,
                embedding_batch_size=effective_embedding_batch_size,
                embedding_adaptive_batching=effective_embedding_adaptive_batching,
            )

            if not quiet:
                console.print(f"\n[green]Refresh complete[/green]")
                console.print(f"  Chunks: {stats.total_chunks}")
                console.print(f"  Upserted: {stats.upserted_points}")
                console.print(f"  Failed: {stats.failed_points}")
                console.print(f"  Tokens: {stats.total_tokens}")
                console.print(f"  Duration: {stats.duration_seconds:.2f}s")
                console.print(
                    "  Filtering: "
                    f"small={stats.skipped_small_chunks}, "
                    f"minified={stats.skipped_minified_chunks}, "
                    f"capped={stats.skipped_capped_chunks}, "
                    f"files_capped={stats.files_capped}"
                )
                console.print(f"  Descriptions enabled: {stats.descriptions_enabled}")
                console.print(
                    "  Embedding telemetry: "
                    f"requests={stats.embedding_requests}, "
                    f"retries={stats.embedding_retries}, "
                    f"failed_requests={stats.embedding_failed_requests}, "
                    f"inputs={stats.embedding_input_count}, "
                    f"duration={stats.embedding_stage_duration_seconds:.2f}s, "
                    f"final_batch={stats.embedding_final_batch_size}, "
                    f"shrink={stats.embedding_batch_shrink_events}, "
                    f"grow={stats.embedding_batch_grow_events}"
                )
                total_cost = stats.llm_cost_usd + stats.embedding_cost_usd
                console.print(f"  Cost: ${total_cost:.4f} (LLM: ${stats.llm_cost_usd:.4f}, Embeddings: ${stats.embedding_cost_usd:.4f})")

    except Exception as exc:
        console.print(f"[red]Refresh failed:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("--mode", default="hybrid", type=click.Choice(["code", "desc", "hybrid"]), help="Search mode: code (exact), desc (conceptual), hybrid (both)")
@click.option("--limit", default=5, type=int, help="Maximum number of results")
@click.option("--language", default=None, help="Filter by programming language")
@click.option("--path", default=None, help="Filter by file path prefix (e.g., 'src/api' to search only in that directory)")
@click.option("--symbol-kind", default=None, help="Filter by symbol kind (e.g., function, class, method, variable)")
@click.option("--project", default=None, help="Project name (collection) to search. Auto-detected from cwd if not specified.")
@click.option("--show-source", is_flag=True, help="Show full source code in results")
@click.option("--use-keywords", is_flag=True, help="Enable keyword-based boosting for better relevance")
@click.option("--keyword-weight", default=0.3, type=float, help="Weight for keyword overlap score (0.0-1.0, default: 0.3)")
@click.option("--max-tokens", default=None, type=int, help="Token budget: pack results to fit within N tokens. Truncates source and drops low-ranked results as needed.")
@click.option("--compress", default=None, type=click.Choice(["light", "medium", "aggressive"]), help="Compress source code before packing: light (blanks), medium (+comments), aggressive (+imports+boilerplate). Requires --max-tokens.")
@click.option("--rerank", is_flag=True, help="Enable ColBERT reranking for improved relevance (downloads model on first use)")
@click.option("--bias", default=None, type=click.Choice(["precise", "discovery", "implementation", "debug", "planning"]), help="Search bias preset: precise (exact match), discovery (broad explore), implementation (code-heavy), debug (balanced+rerank), planning (description-only)")
@click.option("--structured", is_flag=True, help="Interpret query as typed lines (lex:/vec:/hyde:) and use RRF fusion")
@click.option("--rrf-k", default=60, type=int, help="RRF constant (higher flattens rank influence)")
@click.option("--rrf-bonus-top1", default=0.05, type=float, help="RRF additive bonus for rank #1 in any list")
@click.option("--rrf-bonus-top3", default=0.02, type=float, help="RRF additive bonus for ranks #2-3 in any list")
@click.option("--first-query-weight", default=2.0, type=float, help="Extra weight for first structured sub-query")
@click.option("--rerank-top3-weight", default=0.75, type=float, help="Retrieval weight for ranks 1-3 during rerank blending")
@click.option("--rerank-top10-weight", default=0.60, type=float, help="Retrieval weight for ranks 4-10 during rerank blending")
@click.option("--rerank-tail-weight", default=0.40, type=float, help="Retrieval weight for ranks 11+ during rerank blending")
@click.option("--rerank-candidate-multiplier", default=4, type=int, help="How many candidates to fetch before rerank relative to limit")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    mode: str,
    limit: int,
    language: str | None,
    path: str | None,
    symbol_kind: str | None,
    project: str | None,
    show_source: bool,
    use_keywords: bool,
    keyword_weight: float,
    max_tokens: int | None,
    compress: str | None,
    rerank: bool,
    bias: str | None,
    structured: bool,
    rrf_k: int,
    rrf_bonus_top1: float,
    rrf_bonus_top3: float,
    first_query_weight: float,
    rerank_top3_weight: float,
    rerank_top10_weight: float,
    rerank_tail_weight: float,
    rerank_candidate_multiplier: int,
) -> None:
    """
    Search indexed code using semantic similarity.
    """
    config = ctx.obj["config"]

    search_client = None
    qc = None
    try:
        from engine.src.qdrant_search import RestQdrantSearchClient
        from engine.src.searcher import CodeSearcher

        qc = QuickContext(config)
        project_name = project if project else detect_project_name(Path.cwd(), manual_override=None)

        if bias is not None:
            from engine.src.search_modes import get_bias, apply_bias
            bias_preset = get_bias(bias)
            explicit_mode = mode if ctx.get_parameter_source("mode") == click.core.ParameterSource.COMMANDLINE else None
            explicit_kw = use_keywords if use_keywords else None
            explicit_kw_weight = keyword_weight if ctx.get_parameter_source("keyword_weight") == click.core.ParameterSource.COMMANDLINE else None
            explicit_rerank = rerank if rerank else None
            resolved = apply_bias(bias_preset, limit, mode=explicit_mode, use_keywords=explicit_kw, keyword_weight=explicit_kw_weight, rerank=explicit_rerank)
            mode = resolved["vector_mode"]
            limit = resolved["limit"]
            use_keywords = resolved["use_keywords"]
            keyword_weight = resolved["keyword_weight"]
            rerank = resolved["rerank"]
            bias_code_weight = resolved["code_weight"]
            bias_desc_weight = resolved["desc_weight"]

        if config.qdrant is None:
            raise RuntimeError("Qdrant is disabled in config. Search is unavailable.")

        search_client = RestQdrantSearchClient(config.qdrant)
        reranker = qc._get_reranker() if rerank else None
        searcher = CodeSearcher(
            client=search_client,
            collection_name=project_name,
            code_provider=qc.code_provider,
            desc_provider=qc.desc_provider,
            reranker=reranker,
        )

        from engine.src.query_dsl import looks_like_structured_query, parse_structured_query

        use_structured = structured or looks_like_structured_query(query)
        if use_structured:
            sub_queries = parse_structured_query(query)
            results = searcher.search_structured(
                sub_queries=sub_queries,
                limit=limit,
                language=language,
                path_prefix=path,
                symbol_kind=symbol_kind,
                use_keywords=True,
                keyword_weight=keyword_weight,
                rerank=rerank,
                first_query_weight=first_query_weight,
                rrf_k=rrf_k,
                top_rank_bonus_1=rrf_bonus_top1,
                top_rank_bonus_2_3=rrf_bonus_top3,
                rerank_top3_retrieval_weight=rerank_top3_weight,
                rerank_top10_retrieval_weight=rerank_top10_weight,
                rerank_tail_retrieval_weight=rerank_tail_weight,
                rerank_candidate_multiplier=rerank_candidate_multiplier,
            )
            mode = "structured"
        elif mode == "code":
            results = searcher.search_code(query=query, limit=limit, language=language, path_prefix=path, symbol_kind=symbol_kind, use_keywords=use_keywords, keyword_weight=keyword_weight, rerank=rerank)
        elif mode == "desc":
            results = searcher.search_description(query=query, limit=limit, language=language, path_prefix=path, symbol_kind=symbol_kind, use_keywords=use_keywords, keyword_weight=keyword_weight, rerank=rerank)
        elif bias is not None:
            results = searcher.search_hybrid(
                query=query,
                limit=limit,
                code_weight=bias_code_weight,
                desc_weight=bias_desc_weight,
                language=language,
                path_prefix=path,
                symbol_kind=symbol_kind,
                use_keywords=use_keywords,
                keyword_weight=keyword_weight,
                rerank=rerank,
                rrf_k=rrf_k,
                top_rank_bonus_1=rrf_bonus_top1,
                top_rank_bonus_2_3=rrf_bonus_top3,
                rerank_top3_retrieval_weight=rerank_top3_weight,
                rerank_top10_retrieval_weight=rerank_top10_weight,
                rerank_tail_retrieval_weight=rerank_tail_weight,
                rerank_candidate_multiplier=rerank_candidate_multiplier,
            )
        else:
            results = searcher.search_hybrid(
                query=query,
                limit=limit,
                language=language,
                path_prefix=path,
                symbol_kind=symbol_kind,
                use_keywords=use_keywords,
                keyword_weight=keyword_weight,
                rerank=rerank,
                rrf_k=rrf_k,
                top_rank_bonus_1=rrf_bonus_top1,
                top_rank_bonus_2_3=rrf_bonus_top3,
                rerank_top3_retrieval_weight=rerank_top3_weight,
                rerank_top10_retrieval_weight=rerank_top10_weight,
                rerank_tail_retrieval_weight=rerank_tail_weight,
                rerank_candidate_multiplier=rerank_candidate_multiplier,
            )

            if not results:
                console.print("[yellow]No results found[/yellow]")
                return

            if max_tokens is not None:
                from engine.src.packer import pack_search_results
                packed = pack_search_results(results, max_tokens=max_tokens, include_source=True, compress=compress)

                console.print(f"\n[cyan]Search mode:[/cyan] {mode}")
                if bias is not None:
                    console.print(f"[cyan]Bias:[/cyan] {bias} -- {bias_preset.description}")
                console.print(f"[cyan]Query:[/cyan] {query!r}")
                if use_keywords:
                    console.print(f"[cyan]Keyword boosting:[/cyan] enabled (weight={keyword_weight})")
                if rerank:
                    console.print(f"[cyan]Reranking:[/cyan] ColBERT MaxSim")
                if path:
                    console.print(f"[cyan]Path filter:[/cyan] {path}")
                if symbol_kind:
                    console.print(f"[cyan]Symbol kind filter:[/cyan] {symbol_kind}")
                if compress:
                    console.print(f"[cyan]Compression:[/cyan] {compress}")
                console.print(f"[cyan]Token budget:[/cyan] {packed.total_tokens}/{packed.max_tokens} tokens")
                console.print(f"[cyan]Results:[/cyan] {packed.results_included} included, {packed.results_truncated} truncated, {packed.results_dropped} dropped\n")

                for i, result in enumerate(packed.results, 1):
                    trunc_tag = " [yellow](truncated)[/yellow]" if result.truncated else ""
                    console.print(f"[bold cyan]{i}. {result.symbol_name}[/bold cyan] [dim]({result.symbol_kind})[/dim]{trunc_tag}")
                    console.print(f"   [green]Score:[/green] {result.score:.4f}")
                    console.print(f"   [blue]File:[/blue] {result.file_path}:{result.line_start}-{result.line_end}")

                    if result.language:
                        console.print(f"   [magenta]Language:[/magenta] {result.language}")
                    if result.parent:
                        console.print(f"   [yellow]Parent:[/yellow] {result.parent}")
                    if result.path_context:
                        console.print(f"   [yellow]Context:[/yellow] {result.path_context}")
                    if result.signature:
                        console.print(f"   [white]Signature:[/white] {result.signature}")

                    desc_text = result.description[:150].encode('ascii', 'replace').decode('ascii')
                    console.print(f"   [dim]Description:[/dim] {desc_text}{'...' if len(result.description) > 150 else ''}")

                    if show_source and result.source:
                        console.print(f"   [dim]Source:[/dim]")
                        for line in result.source.split('\n')[:20]:
                            safe_line = line.encode('ascii', 'replace').decode('ascii')
                            console.print(f"     {safe_line}")

                    console.print()
                return

            console.print(f"\n[cyan]Search mode:[/cyan] {mode}")
            if bias is not None:
                console.print(f"[cyan]Bias:[/cyan] {bias} -- {bias_preset.description}")
            console.print(f"[cyan]Query:[/cyan] {query!r}")
            if use_keywords:
                console.print(f"[cyan]Keyword boosting:[/cyan] enabled (weight={keyword_weight})")
            if rerank:
                console.print(f"[cyan]Reranking:[/cyan] ColBERT MaxSim")
            if path:
                console.print(f"[cyan]Path filter:[/cyan] {path}")
            console.print(f"[cyan]Results:[/cyan] {len(results)}\n")

            for i, result in enumerate(results, 1):
                console.print(f"[bold cyan]{i}. {result.symbol_name}[/bold cyan] [dim]({result.symbol_kind})[/dim]")
                console.print(f"   [green]Score:[/green] {result.score:.4f}")
                console.print(f"   [blue]File:[/blue] {result.file_path}:{result.line_start}-{result.line_end}")

                if result.language:
                    console.print(f"   [magenta]Language:[/magenta] {result.language}")

                if result.parent:
                    console.print(f"   [yellow]Parent:[/yellow] {result.parent}")

                if result.path_context:
                    console.print(f"   [yellow]Context:[/yellow] {result.path_context}")

                if result.signature:
                    console.print(f"   [white]Signature:[/white] {result.signature}")

                desc_text = result.description[:150].encode('ascii', 'replace').decode('ascii')
                console.print(f"   [dim]Description:[/dim] {desc_text}{'...' if len(result.description) > 150 else ''}")

                if show_source:
                    console.print(f"   [dim]Source:[/dim]")
                    source_lines = result.source.split('\n')[:10]
                    for line in source_lines:
                        safe_line = line.encode('ascii', 'replace').decode('ascii')
                        console.print(f"     {safe_line}")
                    if len(result.source.split('\n')) > 10:
                        console.print("     ...")

                console.print()

    except Exception as exc:
        console.print(f"[red]Search failed:[/red] {exc}")
        sys.exit(1)
    finally:
        if search_client is not None:
            search_client.close()
        if qc is not None:
            qc.close()


@cli.command()
@click.pass_context
def list_projects(ctx: click.Context) -> None:
    """
    List all indexed projects (collections) in Qdrant.

    Uses alias-based discovery to show logical project names,
    filtering out internal versioned backing collections.
    """
    config = ctx.obj["config"]

    if config.qdrant is None:
        console.print("[red]Qdrant is disabled in config. Cannot list projects.[/red]")
        sys.exit(1)

    try:
        with QuickContext(config) as qc:
            from engine.src.collection import CollectionManager
            mgr = CollectionManager(
                client=qc.connection.client,
                config=config,
                collection_name="_dummy",
            )
            projects = mgr.list_all_projects()

            if not projects:
                console.print("[yellow]No projects indexed[/yellow]")
                return

            table = Table(title="Indexed Projects")
            table.add_column("Project", style="cyan")
            table.add_column("Collection", style="dim")
            table.add_column("Chunks", style="green")
            table.add_column("Status", style="white")
            table.add_column("Vectors", style="magenta")

            for proj in projects:
                vector_info = ", ".join(
                    f"{name}({v['size']}d)" for name, v in proj["vectors"].items()
                )
                table.add_row(
                    proj["name"],
                    proj["real_collection"],
                    str(proj["points_count"]),
                    proj["status"],
                    vector_info,
                )

            console.print(table)
            console.print(f"\n[green]Total projects:[/green] {len(projects)}")

    except Exception as exc:
        console.print(f"[red]Failed to list projects:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("--path", "target_path", default=None, help="File or directory path to search in. Defaults to current directory.")
@click.option("--no-gitignore", is_flag=True, help="Disable .gitignore/.ignore filtering for this grep run.")
@click.option("--limit", default=200, type=int, help="Maximum number of matches to return.")
@click.option("--context", default=0, type=int, help="Include NUM lines of context before and after each match.")
@click.option("--before-context", default=None, type=int, help="Include NUM lines of leading context before each match.")
@click.option("--after-context", default=None, type=int, help="Include NUM lines of trailing context after each match.")
@click.option("--max-tokens", default=None, type=int, help="Token budget: pack grep matches to fit within N tokens.")
@click.option("--compress", is_flag=True, help="Compress matched lines (collapse whitespace) before packing. Requires --max-tokens.")
@click.pass_context
def grep(
    ctx: click.Context,
    query: str,
    target_path: str | None,
    no_gitignore: bool,
    limit: int,
    context: int,
    before_context: int | None,
    after_context: int | None,
    max_tokens: int | None,
    compress: bool,
) -> None:
    """
    Fast literal grep via Rust service with optional path scope and gitignore toggle.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    before_ctx = max(0, context if before_context is None else before_context)
    after_ctx = max(0, context if after_context is None else after_context)

    try:
        result = qc.grep_text(
            query=query,
            path=target_path,
            respect_gitignore=not no_gitignore,
            limit=limit,
            before_context=before_ctx,
            after_context=after_ctx,
        )
    except Exception as exc:
        console.print(f"[red]Grep failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    matches = result.matches

    if max_tokens is not None:
        if compress:
            from engine.src.compressor import compress_grep_line
            for m in matches:
                m.line = compress_grep_line(m.line)
        from engine.src.packer import pack_grep_results
        matches, total_tok, dropped = pack_grep_results(matches, max_tokens=max_tokens)
        console.print(f"\n[cyan]Query:[/cyan] {query!r}")
        console.print(f"[cyan]Scope:[/cyan] {target_path if target_path else str(Path.cwd())}")
        console.print(f"[cyan]Context:[/cyan] before={before_ctx}, after={after_ctx}")
        console.print(f"[cyan]Token budget:[/cyan] {total_tok}/{max_tokens} tokens")
        console.print(f"[cyan]Matches:[/cyan] {len(matches)} included, {dropped} dropped\n")
    else:
        console.print(f"\n[cyan]Query:[/cyan] {query!r}")
        console.print(f"[cyan]Scope:[/cyan] {target_path if target_path else str(Path.cwd())}")
        console.print(f"[cyan]Gitignore:[/cyan] {'enabled' if not no_gitignore else 'disabled'}")
        console.print(f"[cyan]Context:[/cyan] before={before_ctx}, after={after_ctx}")
        console.print(f"[cyan]Searched files:[/cyan] {result.searched_files}")
        console.print(f"[cyan]Duration:[/cyan] {result.duration_ms} ms")
        console.print(f"[cyan]Matches:[/cyan] {len(matches)}{' (truncated)' if result.truncated else ''}\n")

    for i, match in enumerate(matches, 1):
        safe_line = match.line.encode('ascii', 'replace').decode('ascii')
        console.print(
            f"[bold cyan]{i}.[/bold cyan] [blue]{match.file_path}[/blue]:"
            f"[green]{match.line_number}[/green]:"
            f"[yellow]{match.column_start}-{match.column_end}[/yellow]"
        )
        if match.context_before:
            start_line = match.line_number - len(match.context_before)
            for idx, ctx_line in enumerate(match.context_before):
                ctx_line_no = start_line + idx
                safe_ctx = ctx_line.encode('ascii', 'replace').decode('ascii')
                console.print(f"   [dim]{ctx_line_no:>6}- {safe_ctx}[/dim]")
        console.print(f"   {safe_line}")
        if match.context_after:
            for idx, ctx_line in enumerate(match.context_after, 1):
                ctx_line_no = match.line_number + idx
                safe_ctx = ctx_line.encode('ascii', 'replace').decode('ascii')
                console.print(f"   [dim]{ctx_line_no:>6}+ {safe_ctx}[/dim]")

@cli.command()
@click.argument("query")
@click.option("--path", "target_path", default=None, help="Project root or search scope path. Defaults to current directory.")
@click.option("--no-gitignore", is_flag=True, help="Disable .gitignore/.ignore filtering.")
@click.option("--limit", default=50, type=int, help="Maximum number of symbols to return.")
@click.option("--intent", "intent_mode", is_flag=True, help="Enable non-embedding intent expansion.")
@click.option("--intent-level", default=2, type=click.IntRange(1, 3), help="Intent expansion aggressiveness from 1 to 3.")
@click.pass_context
def symbol_lookup(
    ctx: click.Context,
    query: str,
    target_path: str | None,
    no_gitignore: bool,
    limit: int,
    intent_mode: bool,
    intent_level: int,
) -> None:
    """
    Fast symbol lookup by keyword via Rust in-memory index.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.symbol_lookup(
            query=query,
            path=target_path,
            respect_gitignore=not no_gitignore,
            limit=limit,
            intent_mode=intent_mode,
            intent_level=intent_level,
        )
    except Exception as exc:
        console.print(f"[red]Symbol lookup failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    console.print(f"\n[cyan]Query:[/cyan] {query!r}")
    console.print(f"[cyan]Project root:[/cyan] {result.project_root}")
    console.print(f"[cyan]Index:[/cyan] {result.indexed_files} files, {result.indexed_symbols} symbols")
    console.print(f"[cyan]Intent mode:[/cyan] {'enabled' if intent_mode else 'disabled'} (level={intent_level})")
    console.print(f"[cyan]Cache hit:[/cyan] {'yes' if result.from_cache else 'no'}")
    console.print(f"[cyan]Results:[/cyan] {len(result.results)}\n")

    for i, item in enumerate(result.results, 1):
        parent_str = f" [yellow]({item.parent})[/yellow]" if item.parent else ""
        console.print(
            f"[bold cyan]{i}.[/bold cyan] [green]{item.name}[/green] "
            f"[dim]({item.kind})[/dim]{parent_str}"
        )
        console.print(f"   [blue]{item.file_path}[/blue]:[green]{item.line_start}-{item.line_end}[/green]")
        console.print(f"   [magenta]{item.language}[/magenta]")
        if item.signature:
            sig = item.signature[:120].encode('ascii', 'replace').decode('ascii')
            console.print(f"   [dim]{sig}[/dim]")
        console.print()

@cli.command()
@click.argument("symbol")
@click.option("--path", "target_path", default=None, help="Project root or search scope path. Defaults to current directory.")
@click.option("--no-gitignore", is_flag=True, help="Disable .gitignore/.ignore filtering.")
@click.option("--limit", default=100, type=int, help="Maximum number of callers to return.")
@click.pass_context
def find_callers(ctx: click.Context, symbol: str, target_path: str | None, no_gitignore: bool, limit: int) -> None:
    """
    Find all call sites for a symbol via Rust call index.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.find_callers(
            symbol=symbol,
            path=target_path,
            respect_gitignore=not no_gitignore,
            limit=limit,
        )
    except Exception as exc:
        console.print(f"[red]Find callers failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    console.print(f"\n[cyan]Symbol:[/cyan] {symbol!r}")
    console.print(f"[cyan]Project root:[/cyan] {result.project_root}")
    console.print(f"[cyan]Index:[/cyan] {result.indexed_files} files, {result.indexed_symbols} symbols")
    console.print(f"[cyan]Cache hit:[/cyan] {'yes' if result.from_cache else 'no'}")
    console.print(f"[cyan]Callers:[/cyan] {len(result.callers)}\n")

    for i, caller in enumerate(result.callers, 1):
        console.print(
            f"[bold cyan]{i}.[/bold cyan] [green]{caller.caller_name}[/green] "
            f"[dim]({caller.caller_kind})[/dim] calls [yellow]{caller.callee_name}[/yellow]"
        )
        console.print(f"   [blue]{caller.caller_file_path}[/blue]:[green]{caller.caller_line}[/green]")
        console.print(f"   [magenta]{caller.caller_language}[/magenta]")
        console.print()


@cli.command("trace-call-graph")
@click.argument("symbol")
@click.option("--path", "target_path", default=None, help="Project root or search scope path. Defaults to current directory.")
@click.option("--no-gitignore", is_flag=True, help="Disable .gitignore/.ignore filtering.")
@click.option("--direction", default="both", type=click.Choice(["upstream", "downstream", "both"]), help="Trace direction: upstream (callers), downstream (callees), or both.")
@click.option("--depth", default=5, type=int, help="Maximum BFS traversal depth.")
@click.pass_context
def trace_call_graph(ctx: click.Context, symbol: str, target_path: str | None, no_gitignore: bool, direction: str, depth: int) -> None:
    """
    Trace multi-hop call graph from a symbol via BFS.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.trace_call_graph(
            symbol=symbol,
            path=target_path,
            respect_gitignore=not no_gitignore,
            direction=direction,
            max_depth=depth,
        )
    except Exception as exc:
        console.print(f"[red]Trace call graph failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    console.print(f"\n[cyan]Root symbol:[/cyan] {result.root_symbol!r}")
    console.print(f"[cyan]Direction:[/cyan] {result.direction}")
    console.print(f"[cyan]Max depth:[/cyan] {result.max_depth}")
    console.print(f"[cyan]Index:[/cyan] {result.indexed_files} files, {result.indexed_symbols} symbols")
    console.print(f"[cyan]Cache hit:[/cyan] {'yes' if result.from_cache else 'no'}")
    console.print(f"[cyan]Nodes:[/cyan] {len(result.nodes)}  [cyan]Edges:[/cyan] {len(result.edges)}")

    if result.cycles_detected:
        console.print(f"[yellow]Cycles:[/yellow] {len(result.cycles_detected)}")
        for cyc in result.cycles_detected:
            console.print(f"  [yellow]{cyc}[/yellow]")

    if result.nodes:
        console.print()
        table = Table(title="Call Graph Nodes")
        table.add_column("#", style="dim", width=4)
        table.add_column("Depth", style="cyan", width=5)
        table.add_column("Symbol", style="green")
        table.add_column("Kind", style="dim")
        table.add_column("File", style="blue")
        table.add_column("Lines", style="dim")

        for i, node in enumerate(result.nodes, 1):
            table.add_row(
                str(i),
                str(node.depth),
                node.name,
                node.kind,
                node.file_path,
                f"{node.line_start}-{node.line_end}",
            )
        console.print(table)

    if result.edges:
        console.print()
        table = Table(title="Call Graph Edges")
        table.add_column("#", style="dim", width=4)
        table.add_column("From", style="green")
        table.add_column("To", style="yellow")
        table.add_column("File", style="blue")
        table.add_column("Line", style="dim")

        for i, edge in enumerate(result.edges, 1):
            table.add_row(
                str(i),
                edge.from_name,
                edge.to_name,
                edge.file_path,
                str(edge.call_line),
            )
        console.print(table)

    console.print()


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--depth", default=None, type=int, help="Max directory recursion depth.")
@click.option("--no-signatures", is_flag=True, help="Omit function/class signatures.")
@click.option("--lines", is_flag=True, help="Include line number ranges.")
@click.option("--collapse", default=0, type=int, help="Collapse dirs with fewer files than this threshold.")
@click.option("--no-gitignore", is_flag=True, help="Disable .gitignore/.ignore filtering.")
@click.option("--markdown", is_flag=True, help="Output compact markdown tree instead of JSON.")
@click.option("--max-tokens", default=None, type=int, help="Token budget: truncate skeleton output to fit within N tokens.")
@click.option("--compress", is_flag=True, help="Strip blank lines from skeleton output before packing. Requires --max-tokens and --markdown.")
@click.pass_context
def skeleton(ctx: click.Context, path: Path, depth: int | None, no_signatures: bool, lines: bool, collapse: int, no_gitignore: bool, markdown: bool, max_tokens: int | None, compress: bool) -> None:
    """
    Generate a repo skeleton showing file tree + symbol signatures.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    fmt = "markdown" if markdown else "json"

    try:
        result = qc.skeleton(
            path=path,
            max_depth=depth,
            include_signatures=not no_signatures,
            include_line_numbers=lines,
            collapse_threshold=collapse,
            respect_gitignore=not no_gitignore,
            format=fmt,
        )
    except Exception as exc:
        console.print(f"[red]Skeleton failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    if max_tokens is not None and markdown and result.markdown:
        md_text = result.markdown
        if compress:
            from engine.src.compressor import compress_source
            md_text, _ = compress_source(md_text, level="light")
        from engine.src.packer import pack_skeleton
        output, tok_used, was_truncated = pack_skeleton(md_text, max_tokens=max_tokens)
        sys.stdout.buffer.write(output.encode("utf-8"))
        sys.stdout.buffer.flush()
        trunc_tag = " (truncated)" if was_truncated else ""
        console.print(
            f"\n[dim]{result.total_files} files, {result.total_symbols} symbols, "
            f"{result.total_directories} dirs in {result.duration_ms}ms[/dim]"
        )
        console.print(f"[cyan]Token budget:[/cyan] {tok_used}/{max_tokens} tokens{trunc_tag}")
        return

    if markdown and result.markdown:
        sys.stdout.buffer.write(result.markdown.encode("utf-8"))
        sys.stdout.buffer.flush()
    else:
        import json
        console.print_json(json.dumps(result.root, indent=2))

    console.print(
        f"\n[dim]{result.total_files} files, {result.total_symbols} symbols, "
        f"{result.total_directories} dirs in {result.duration_ms}ms[/dim]"
    )


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.pass_context
def cache_status(ctx: click.Context, directory: Path) -> None:
    """
    Show mtime file cache statistics for a project directory.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        stats = qc.cache_status(directory)
    except Exception as exc:
        console.print(f"[red]Cache status failed:[/red] {exc}")
        sys.exit(1)

    table = Table(title="File Cache Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Project root", stats["project_root"])
    table.add_row("Cache path", stats["cache_path"])
    table.add_row("Cached files", str(stats["entries"]))
    table.add_row("Disk size", f"{stats['disk_bytes']:,} bytes")

    console.print(table)


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt.")
@click.pass_context
def cache_clear(ctx: click.Context, directory: Path, confirm: bool) -> None:
    """
    Clear the mtime file cache for a project directory.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    if not confirm:
        click.confirm(
            f"Clear file cache for '{directory}'?",
            abort=True,
        )

    try:
        count = qc.cache_clear(directory)
        console.print(f"[green]Cleared {count} cached file entries[/green]")
    except Exception as exc:
        console.print(f"[red]Cache clear failed:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--project", default=None, help="Project name override. Auto-detected from directory if not specified.")
@click.option("--debounce", default=2.0, type=float, help="Seconds to wait after last change before refreshing (default: 2.0).")
@click.option("--quiet", is_flag=True, help="Suppress progress messages.")
@click.pass_context
def watch(ctx: click.Context, directory: Path, project: str | None, debounce: float, quiet: bool) -> None:
    """
    Watch a directory for file changes and auto-refresh the index.
    """
    config = ctx.obj["config"]

    try:
        with QuickContext(config) as qc:
            qc.watch(
                directory=directory,
                project_name=project,
                debounce_seconds=debounce,
                show_progress=not quiet,
            )
    except Exception as exc:
        console.print(f"[red]Watch failed:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--path", "project_path", default=None, help="Project root directory. Defaults to current directory.")
@click.option("--no-gitignore", is_flag=True, help="Disable .gitignore/.ignore filtering.")
@click.pass_context
def import_graph(ctx: click.Context, file: Path, project_path: str | None, no_gitignore: bool) -> None:
    """
    Show import dependencies for a file (what does this file import?).
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.import_graph(
            file=file,
            path=project_path,
            respect_gitignore=not no_gitignore,
        )
    except Exception as exc:
        console.print(f"[red]Import graph failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    console.print(f"\n[cyan]File:[/cyan] {result.file_path}")
    console.print(f"[cyan]Project root:[/cyan] {result.project_root}")
    console.print(f"[cyan]Scanned:[/cyan] {result.total_files} files, {result.total_imports} total imports")
    console.print(f"[cyan]Duration:[/cyan] {result.duration_ms} ms")
    console.print(f"[cyan]Imports:[/cyan] {len(result.edges)}\n")

    for i, edge in enumerate(result.edges, 1):
        console.print(
            f"[bold cyan]{i}.[/bold cyan] [green]{edge.module_path}[/green] "
            f"[dim]({edge.language})[/dim]"
        )
        console.print(f"   [blue]{edge.target_file}[/blue]")
        stmt = edge.import_statement[:120].encode('ascii', 'replace').decode('ascii')
        console.print(f"   [dim]L{edge.line}: {stmt}[/dim]")
        console.print()


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--path", "project_path", default=None, help="Project root directory. Defaults to current directory.")
@click.option("--no-gitignore", is_flag=True, help="Disable .gitignore/.ignore filtering.")
@click.pass_context
def find_importers(ctx: click.Context, file: Path, project_path: str | None, no_gitignore: bool) -> None:
    """
    Find files that import a given file (who imports this file?).
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.find_importers(
            file=file,
            path=project_path,
            respect_gitignore=not no_gitignore,
        )
    except Exception as exc:
        console.print(f"[red]Find importers failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    console.print(f"\n[cyan]File:[/cyan] {result.file_path}")
    console.print(f"[cyan]Project root:[/cyan] {result.project_root}")
    console.print(f"[cyan]Scanned:[/cyan] {result.total_files} files, {result.total_imports} total imports")
    console.print(f"[cyan]Duration:[/cyan] {result.duration_ms} ms")
    console.print(f"[cyan]Imported by:[/cyan] {len(result.edges)}\n")

    for i, edge in enumerate(result.edges, 1):
        console.print(
            f"[bold cyan]{i}.[/bold cyan] [green]{edge.source_file}[/green]"
        )
        stmt = edge.import_statement[:120].encode('ascii', 'replace').decode('ascii')
        console.print(f"   [dim]L{edge.line}: {stmt}[/dim]")
        console.print()


@cli.command()
@click.argument("query")
@click.option("--path", "target_path", default=None, help="Directory to search in. Defaults to current directory.")
@click.option("--no-gitignore", is_flag=True, help="Disable .gitignore/.ignore filtering.")
@click.option("--limit", default=20, type=int, help="Maximum number of results to return.")
@click.option("--intent", "intent_mode", is_flag=True, help="Enable non-embedding intent expansion.")
@click.option("--intent-level", default=2, type=click.IntRange(1, 3), help="Intent expansion aggressiveness from 1 to 3.")
@click.pass_context
def text_search(
    ctx: click.Context,
    query: str,
    target_path: str | None,
    no_gitignore: bool,
    limit: int,
    intent_mode: bool,
    intent_level: int,
) -> None:
    """
    BM25 full-text search with query operators (AND/OR/+/-/"exact"/file:/ext:/lang:).
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.text_search(
            query=query,
            path=target_path,
            respect_gitignore=not no_gitignore,
            limit=limit,
            intent_mode=intent_mode,
            intent_level=intent_level,
        )
    except Exception as exc:
        console.print(f"[red]Text search failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    console.print(f"\n[cyan]Query:[/cyan] {query!r}")
    console.print(f"[cyan]Scope:[/cyan] {target_path if target_path else str(Path.cwd())}")
    console.print(f"[cyan]Gitignore:[/cyan] {'enabled' if not no_gitignore else 'disabled'}")
    console.print(f"[cyan]Intent mode:[/cyan] {'enabled' if intent_mode else 'disabled'} (level={intent_level})")
    console.print(f"[cyan]Searched files:[/cyan] {result.searched_files}")
    console.print(f"[cyan]Duration:[/cyan] {result.duration_ms} ms")
    console.print(f"[cyan]Query expr:[/cyan] {result.query_expr}")
    console.print(f"[cyan]Results:[/cyan] {len(result.matches)}{' (truncated)' if result.truncated else ''}\n")

    for i, match in enumerate(result.matches, 1):
        console.print(
            f"[bold cyan]{i}.[/bold cyan] [green]{match.score:.4f}[/green] "
            f"[blue]{match.file_path}[/blue]"
        )
        if match.language:
            console.print(f"   [magenta]{match.language}[/magenta]")
        console.print(f"   [dim]Terms: {', '.join(match.matched_terms)}[/dim]")
        console.print(f"   [dim]Lines {match.snippet_line_start}-{match.snippet_line_end}:[/dim]")
        for line in match.snippet.split('\n')[:10]:
            safe_line = line.encode('ascii', 'replace').decode('ascii')
            console.print(f"     {safe_line}")
        console.print()


@cli.command("protocol-search")
@click.argument("query")
@click.option("--path", "target_path", default=None, help="Directory or file to search in. Defaults to current directory.")
@click.option("--no-gitignore", is_flag=True, help="Disable .gitignore/.ignore filtering.")
@click.option("--limit", default=20, type=int, help="Maximum number of contracts to return.")
@click.option("--context-radius", default=None, type=int, help="Context line radius around protocol markers.")
@click.option("--min-score", default=None, type=float, help="Minimum score threshold for returned contracts.")
@click.option("--include-marker", "include_markers", multiple=True, help="Extra marker treated as protocol signal (repeatable).")
@click.option("--exclude-marker", "exclude_markers", multiple=True, help="Marker that excludes candidate files (repeatable).")
@click.option("--max-input-fields", default=None, type=int, help="Maximum inferred input fields per contract.")
@click.option("--max-output-fields", default=None, type=int, help="Maximum inferred output fields per contract.")
@click.pass_context
def protocol_search(
    ctx: click.Context,
    query: str,
    target_path: str | None,
    no_gitignore: bool,
    limit: int,
    context_radius: int | None,
    min_score: float | None,
    include_markers: tuple[str, ...],
    exclude_markers: tuple[str, ...],
    max_input_fields: int | None,
    max_output_fields: int | None,
) -> None:
    """
    Extract protocol request/response contracts from code.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.protocol_search(
            query=query,
            path=target_path,
            respect_gitignore=not no_gitignore,
            limit=limit,
            context_radius=context_radius,
            min_score=min_score,
            include_markers=list(include_markers) if include_markers else None,
            exclude_markers=list(exclude_markers) if exclude_markers else None,
            max_input_fields=max_input_fields,
            max_output_fields=max_output_fields,
        )
    except Exception as exc:
        console.print(f"[red]Protocol search failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    console.print(f"\n[cyan]Query:[/cyan] {query!r}")
    console.print(f"[cyan]Scope:[/cyan] {target_path if target_path else str(Path.cwd())}")
    console.print(f"[cyan]Gitignore:[/cyan] {'enabled' if not no_gitignore else 'disabled'}")
    console.print(f"[cyan]Searched files:[/cyan] {result.searched_files}")
    console.print(f"[cyan]Duration:[/cyan] {result.duration_ms} ms")
    console.print(f"[cyan]Contracts:[/cyan] {len(result.contracts)}{' (truncated)' if result.truncated else ''}\n")

    for i, contract in enumerate(result.contracts, 1):
        endpoint = contract.endpoint if contract.endpoint else "-"
        op_name = contract.operation_name if contract.operation_name else "-"
        console.print(
            f"[bold cyan]{i}.[/bold cyan] [green]{contract.score:.3f}[/green] "
            f"[blue]{contract.file_path}[/blue]"
        )
        console.print(f"   [magenta]{contract.transport}[/magenta] op={contract.operation} endpoint={endpoint} opName={op_name}")
        console.print(f"   [dim]confidence={contract.confidence:.2f} matched={', '.join(contract.matched_terms)}[/dim]")

        if contract.input_fields:
            inputs = ", ".join(f"{f.name}{'*' if f.required else ''}" for f in contract.input_fields)
            inputs = inputs.encode('ascii', 'replace').decode('ascii')
            console.print(f"   [yellow]Inputs:[/yellow] {inputs}")

        if contract.output_fields:
            outputs = ", ".join(f.path for f in contract.output_fields)
            outputs = outputs.encode('ascii', 'replace').decode('ascii')
            console.print(f"   [yellow]Outputs:[/yellow] {outputs}")

        for ev in contract.evidence[:3]:
            text = ev.text.encode('ascii', 'replace').decode('ascii')
            console.print(f"   [dim]{ev.kind} L{ev.line}: {text}[/dim]")

        console.print()


@cli.command("pattern-search")
@click.argument("pattern", required=False, default=None)
@click.option("--lang", "language", required=True, help="Target language (python, rust, javascript, etc.).")
@click.option("--path", "target_path", default=None, help="Directory or file to search in. Defaults to current directory.")
@click.option("--no-gitignore", is_flag=True, help="Disable .gitignore/.ignore filtering.")
@click.option("--limit", default=50, type=int, help="Maximum number of matches to return.")
@click.option("--stdin", "use_stdin", is_flag=True, help="Read pattern from stdin (avoids shell $ expansion).")
@click.pass_context
def pattern_search(ctx: click.Context, pattern: str | None, language: str, target_path: str | None, no_gitignore: bool, limit: int, use_stdin: bool) -> None:
    """
    AST pattern search with metavariables ($NAME, $$$, $_).

    Pass pattern as argument or use --stdin to read from stdin
    (recommended on Git Bash/Windows to avoid $ expansion).

    pattern: str | None — Code pattern with metavariable placeholders.
    language: str — Target language name.
    target_path: str | None — Search scope path.
    no_gitignore: bool — Disable gitignore filtering.
    limit: int — Max matches to return.
    use_stdin: bool — Read pattern from stdin instead of argument.
    """
    if use_stdin:
        pattern = sys.stdin.readline().strip()
    if not pattern:
        console.print("[red]Pattern required:[/red] pass as argument or use --stdin")
        sys.exit(1)

    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.pattern_search(
            pattern=pattern,
            language=language,
            path=target_path,
            respect_gitignore=not no_gitignore,
            limit=limit,
        )
    except Exception as exc:
        console.print(f"[red]Pattern search failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    console.print(f"\n[cyan]Pattern:[/cyan] {pattern!r}")
    console.print(f"[cyan]Language:[/cyan] {language}")
    console.print(f"[cyan]Scope:[/cyan] {target_path if target_path else str(Path.cwd())}")
    console.print(f"[cyan]Gitignore:[/cyan] {'enabled' if not no_gitignore else 'disabled'}")
    console.print(f"[cyan]Searched files:[/cyan] {result.searched_files}")
    console.print(f"[cyan]Duration:[/cyan] {result.duration_ms} ms")
    console.print(f"[cyan]Matches:[/cyan] {len(result.matches)}{' (truncated)' if result.truncated else ''}\n")

    for i, match in enumerate(result.matches, 1):
        console.print(
            f"[bold cyan]{i}.[/bold cyan] "
            f"[blue]{match.file_path}[/blue]:[green]{match.line_start}[/green]-[green]{match.line_end}[/green]"
        )
        console.print(f"   [magenta]{match.language}[/magenta]")
        if match.captures:
            caps = ", ".join(f"{c.name}={c.text!r}" for c in match.captures)
            console.print(f"   [yellow]Captures:[/yellow] {caps}")
        for line in match.matched_text.split('\n')[:10]:
            safe_line = line.encode('ascii', 'replace').decode('ascii')
            console.print(f"     {safe_line}")
        console.print()


@cli.command("pattern-rewrite")
@click.argument("pattern", required=False, default=None)
@click.option("--replacement", "-r", required=True, help="Replacement template with metavariable substitution ($NAME, $$$ARGS).")
@click.option("--lang", "language", required=True, help="Target language (python, rust, javascript, etc.).")
@click.option("--path", "target_path", default=None, help="Directory or file to search in. Defaults to current directory.")
@click.option("--no-gitignore", is_flag=True, help="Disable .gitignore/.ignore filtering.")
@click.option("--limit", default=50, type=int, help="Maximum number of files to rewrite.")
@click.option("--apply", is_flag=True, help="Actually write changes to disk (default is dry-run).")
@click.option("--stdin", "use_stdin", is_flag=True, help="Read pattern from stdin (avoids shell $ expansion).")
@click.pass_context
def pattern_rewrite(ctx: click.Context, pattern: str | None, replacement: str, language: str, target_path: str | None, no_gitignore: bool, limit: int, apply: bool, use_stdin: bool) -> None:
    """
    AST pattern rewrite with metavariable substitution.

    Pass pattern as argument or use --stdin to read from stdin
    (recommended on Git Bash/Windows to avoid $ expansion).

    pattern: str | None — Code pattern with metavariable placeholders.
    replacement: str — Replacement template with metavariable substitution.
    language: str — Target language name.
    target_path: str | None — Search scope path.
    no_gitignore: bool — Disable gitignore filtering.
    limit: int — Max files to rewrite.
    apply: bool — Write changes to disk when True.
    use_stdin: bool — Read pattern from stdin instead of argument.
    """
    if use_stdin:
        pattern = sys.stdin.readline().strip()
    if not pattern:
        console.print("[red]Pattern required:[/red] pass as argument or use --stdin")
        sys.exit(1)

    dry_run = not apply
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.pattern_rewrite(
            pattern=pattern,
            replacement=replacement,
            language=language,
            path=target_path,
            respect_gitignore=not no_gitignore,
            limit=limit,
            dry_run=dry_run,
        )
    except Exception as exc:
        console.print(f"[red]Pattern rewrite failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    mode_label = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]APPLIED[/green]"
    console.print(f"\n[cyan]Mode:[/cyan] {mode_label}")
    console.print(f"[cyan]Pattern:[/cyan] {pattern!r}")
    console.print(f"[cyan]Replacement:[/cyan] {replacement!r}")
    console.print(f"[cyan]Language:[/cyan] {language}")
    console.print(f"[cyan]Scope:[/cyan] {target_path if target_path else str(Path.cwd())}")
    console.print(f"[cyan]Searched files:[/cyan] {result.searched_files}")
    console.print(f"[cyan]Duration:[/cyan] {result.duration_ms} ms")
    console.print(f"[cyan]Total edits:[/cyan] {result.total_edits}")
    console.print(f"[cyan]Files affected:[/cyan] {len(result.files)}\n")

    for file_result in result.files:
        console.print(f"[blue]{file_result.file_path}[/blue] ({len(file_result.edits)} edits)")
        for edit in file_result.edits:
            console.print(
                f"  [dim]L{edit.line_start}:{edit.column_start}-L{edit.line_end}:{edit.column_end}[/dim]"
            )
            console.print(f"    [red]- {edit.original}[/red]")
            console.print(f"    [green]+ {edit.replacement}[/green]")
        console.print()


@cli.command("lsp-definition")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.argument("line", type=int)
@click.argument("character", type=int)
@click.pass_context
def lsp_definition(ctx: click.Context, file: Path, line: int, character: int) -> None:
    """
    Go to definition of symbol at FILE:LINE:CHARACTER via LSP.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.lsp_definition(file=file, line=line, character=character)
    except Exception as exc:
        console.print(f"[red]LSP definition failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    import json
    console.print_json(json.dumps(result, indent=2))


@cli.command("lsp-references")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.argument("line", type=int)
@click.argument("character", type=int)
@click.option("--no-declaration", is_flag=True, help="Exclude the declaration from results.")
@click.pass_context
def lsp_references(ctx: click.Context, file: Path, line: int, character: int, no_declaration: bool) -> None:
    """
    Find all references to symbol at FILE:LINE:CHARACTER via LSP.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.lsp_references(
            file=file,
            line=line,
            character=character,
            include_declaration=not no_declaration,
        )
    except Exception as exc:
        console.print(f"[red]LSP references failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    import json
    console.print_json(json.dumps(result, indent=2))


@cli.command("lsp-hover")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.argument("line", type=int)
@click.argument("character", type=int)
@click.pass_context
def lsp_hover(ctx: click.Context, file: Path, line: int, character: int) -> None:
    """
    Get hover info (type, docs) for symbol at FILE:LINE:CHARACTER via LSP.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.lsp_hover(file=file, line=line, character=character)
    except Exception as exc:
        console.print(f"[red]LSP hover failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    import json
    console.print_json(json.dumps(result, indent=2))


@cli.command("lsp-symbols")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def lsp_symbols(ctx: click.Context, file: Path) -> None:
    """
    Get document symbols (outline) for FILE via LSP.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.lsp_symbols(file=file)
    except Exception as exc:
        console.print(f"[red]LSP symbols failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    import json
    console.print_json(json.dumps(result, indent=2))


@cli.command("lsp-format")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("--tab-size", default=4, type=int, help="Spaces per tab (default: 4).")
@click.option("--tabs", is_flag=True, help="Use tabs instead of spaces.")
@click.pass_context
def lsp_format(ctx: click.Context, file: Path, tab_size: int, tabs: bool) -> None:
    """
    Format FILE via LSP and print text edits.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.lsp_format(file=file, tab_size=tab_size, insert_spaces=not tabs)
    except Exception as exc:
        console.print(f"[red]LSP format failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    import json
    console.print_json(json.dumps(result, indent=2))


@cli.command("lsp-diagnostics")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def lsp_diagnostics(ctx: click.Context, file: Path) -> None:
    """
    Request diagnostics for FILE via LSP.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.lsp_diagnostics(file=file)
    except Exception as exc:
        console.print(f"[red]LSP diagnostics failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    import json
    console.print_json(json.dumps(result, indent=2))


@cli.command("lsp-completion")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.argument("line", type=int)
@click.argument("character", type=int)
@click.pass_context
def lsp_completion(ctx: click.Context, file: Path, line: int, character: int) -> None:
    """
    Request completions at FILE:LINE:CHARACTER via LSP.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.lsp_completion(file=file, line=line, character=character)
    except Exception as exc:
        console.print(f"[red]LSP completion failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    import json
    console.print_json(json.dumps(result, indent=2))


@cli.command("lsp-rename")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.argument("line", type=int)
@click.argument("character", type=int)
@click.argument("new_name", type=str)
@click.pass_context
def lsp_rename(ctx: click.Context, file: Path, line: int, character: int, new_name: str) -> None:
    """
    Rename symbol at FILE:LINE:CHARACTER to NEW_NAME via LSP.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.lsp_rename(file=file, line=line, character=character, new_name=new_name)
    except Exception as exc:
        console.print(f"[red]LSP rename failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    import json
    console.print_json(json.dumps(result, indent=2))


@cli.command("lsp-prepare-rename")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.argument("line", type=int)
@click.argument("character", type=int)
@click.pass_context
def lsp_prepare_rename(ctx: click.Context, file: Path, line: int, character: int) -> None:
    """
    Validate rename at FILE:LINE:CHARACTER and get symbol range via LSP.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.lsp_prepare_rename(file=file, line=line, character=character)
    except Exception as exc:
        console.print(f"[red]LSP prepare-rename failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    import json
    console.print_json(json.dumps(result, indent=2))


@cli.command("lsp-code-actions")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.argument("start_line", type=int)
@click.argument("start_character", type=int)
@click.argument("end_line", type=int)
@click.argument("end_character", type=int)
@click.pass_context
def lsp_code_actions(
    ctx: click.Context,
    file: Path,
    start_line: int,
    start_character: int,
    end_line: int,
    end_character: int,
) -> None:
    """
    Get code actions for range START_LINE:START_CHARACTER to END_LINE:END_CHARACTER in FILE via LSP.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.lsp_code_actions(
            file=file,
            start_line=start_line,
            start_character=start_character,
            end_line=end_line,
            end_character=end_character,
        )
    except Exception as exc:
        console.print(f"[red]LSP code-actions failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    import json
    console.print_json(json.dumps(result, indent=2))


@cli.command("lsp-signature-help")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.argument("line", type=int)
@click.argument("character", type=int)
@click.pass_context
def lsp_signature_help(ctx: click.Context, file: Path, line: int, character: int) -> None:
    """
    Get signature help at FILE:LINE:CHARACTER via LSP.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.lsp_signature_help(file=file, line=line, character=character)
    except Exception as exc:
        console.print(f"[red]LSP signature-help failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    import json
    console.print_json(json.dumps(result, indent=2))


@cli.command("lsp-workspace-symbols")
@click.argument("query", type=str)
@click.option("--file", type=click.Path(exists=True, path_type=Path), default=None, help="File to determine project root for server selection.")
@click.pass_context
def lsp_workspace_symbols(ctx: click.Context, query: str, file: Path | None) -> None:
    """
    Search workspace symbols matching QUERY via LSP.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    try:
        result = qc.lsp_workspace_symbols(query=query, file=file)
    except Exception as exc:
        console.print(f"[red]LSP workspace-symbols failed:[/red] {exc}")
        sys.exit(1)
    finally:
        qc.close()

    import json
    console.print_json(json.dumps(result, indent=2))


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without actually deleting.")
@click.pass_context
def gc(ctx: click.Context, dry_run: bool) -> None:
    """
    Garbage collect orphaned versioned collections and stale file caches.

    Scans all Qdrant collections for orphaned {project}_v{N} versions
    that no alias points to, and removes stale .quickcontext/file_cache.json
    files for projects that no longer exist.

    ctx: click.Context — CLI context with config.
    dry_run: bool — If True, only report what would be deleted.
    """
    config = ctx.obj["config"]

    if config.qdrant is None:
        console.print("[red]Qdrant is disabled in config.[/red]")
        sys.exit(1)

    try:
        with QuickContext(config) as qc:
            from engine.src.collection import CollectionManager
            client = qc.connection.client

            all_aliases = client.get_aliases()
            alias_targets: set[str] = set()
            alias_names: set[str] = set()
            for alias in all_aliases.aliases:
                alias_targets.add(alias.collection_name)
                alias_names.add(alias.alias_name)

            collections = client.get_collections().collections
            orphaned = []
            for col in collections:
                if col.name in alias_targets:
                    continue
                if col.name in alias_names:
                    continue
                import re
                if re.match(r"^.+_v\d+$", col.name):
                    orphaned.append(col.name)

            if orphaned:
                table = Table(title="Orphaned Collections")
                table.add_column("Collection", style="red")
                table.add_column("Action", style="yellow")
                for name in sorted(orphaned):
                    action = "would delete" if dry_run else "deleted"
                    table.add_row(name, action)
                    if not dry_run:
                        client.delete_collection(name)
                console.print(table)
            else:
                console.print("[green]No orphaned collections found.[/green]")

            cache_dir_cleaned = 0
            if not dry_run:
                for project_name in list(alias_names):
                    mgr = CollectionManager(
                        client=client,
                        config=config,
                        collection_name=project_name,
                    )
                    deleted = mgr.cleanup_old_versions(keep_current=True)
                    if deleted:
                        for d in deleted:
                            console.print(f"  [yellow]deleted:[/yellow] {d} (stale version of {project_name})")
                        cache_dir_cleaned += len(deleted)

            console.print(f"\n[green]GC complete.[/green] Orphaned: {len(orphaned)}, Stale versions: {cache_dir_cleaned}")
            if dry_run:
                console.print("[yellow]Dry run -- nothing was actually deleted.[/yellow]")

    except Exception as exc:
        console.print(f"[red]GC failed:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.option("--project", default=None, help="Target a specific project. If omitted, compacts all projects.")
@click.pass_context
def compact(ctx: click.Context, project: str | None) -> None:
    """
    Trigger Qdrant optimizer to merge segments for a project or all projects.

    Lowers the indexing_threshold temporarily to force segment merging,
    then restores it. This reduces segment count and improves query performance.

    ctx: click.Context — CLI context with config.
    project: str | None — Specific project name, or None for all.
    """
    config = ctx.obj["config"]

    if config.qdrant is None:
        console.print("[red]Qdrant is disabled in config.[/red]")
        sys.exit(1)

    try:
        with QuickContext(config) as qc:
            from qdrant_client import models
            from engine.src.collection import CollectionManager
            client = qc.connection.client

            if project:
                targets = [project]
            else:
                mgr = CollectionManager(
                    client=client,
                    config=config,
                    collection_name="_dummy",
                )
                projects = mgr.list_all_projects()
                targets = [p["name"] for p in projects]

            if not targets:
                console.print("[yellow]No projects to compact.[/yellow]")
                return

            for name in targets:
                console.print(f"Compacting [cyan]{name}[/cyan]...")
                try:
                    info = client.get_collection(name)
                    segments_before = info.segments_count

                    client.update_collection(
                        collection_name=name,
                        optimizer_config=models.OptimizersConfigDiff(
                            indexing_threshold=0,
                        ),
                    )

                    import time
                    time.sleep(2)

                    client.update_collection(
                        collection_name=name,
                        optimizer_config=models.OptimizersConfigDiff(
                            indexing_threshold=20000,
                        ),
                    )

                    info_after = client.get_collection(name)
                    segments_after = info_after.segments_count
                    console.print(
                        f"  [green]Done.[/green] Segments: {segments_before} -> {segments_after}"
                    )
                except Exception as exc:
                    console.print(f"  [red]Failed:[/red] {exc}")

    except Exception as exc:
        console.print(f"[red]Compact failed:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.pass_context
def audit(ctx: click.Context) -> None:
    """
    Health check across all indexed projects.

    Reports collection status, optimizer state, segment count,
    indexed vs total vectors, and flags potential issues.

    ctx: click.Context — CLI context with config.
    """
    config = ctx.obj["config"]

    if config.qdrant is None:
        console.print("[red]Qdrant is disabled in config.[/red]")
        sys.exit(1)

    try:
        with QuickContext(config) as qc:
            from engine.src.collection import CollectionManager
            client = qc.connection.client

            mgr = CollectionManager(
                client=client,
                config=config,
                collection_name="_dummy",
            )
            projects = mgr.list_all_projects()

            if not projects:
                console.print("[yellow]No projects indexed.[/yellow]")
                return

            warnings = []

            table = Table(title="Project Health Audit")
            table.add_column("Project", style="cyan")
            table.add_column("Collection", style="dim")
            table.add_column("Status", style="white")
            table.add_column("Optimizer", style="white")
            table.add_column("Segments", style="green")
            table.add_column("Points", style="green")
            table.add_column("Indexed Vecs", style="green")
            table.add_column("Warnings", style="yellow")

            for proj in projects:
                name = proj["name"]
                row_warnings = []

                try:
                    info = client.get_collection(name)
                    status = info.status.value
                    opt_raw = info.optimizer_status
                    if hasattr(opt_raw, "value"):
                        optimizer_status = str(opt_raw.value)
                    elif hasattr(opt_raw, "status"):
                        optimizer_status = str(opt_raw.status.value)
                    else:
                        optimizer_status = str(opt_raw) if opt_raw else "unknown"
                    segments = info.segments_count or 0
                    points = info.points_count or 0
                    indexed_vecs = info.indexed_vectors_count or 0

                    if status != "green":
                        row_warnings.append(f"status={status}")
                    if optimizer_status != "ok":
                        row_warnings.append(f"optimizer={optimizer_status}")
                    if segments > 10:
                        row_warnings.append(f"high segments ({segments})")
                    if points > 0 and indexed_vecs == 0:
                        row_warnings.append("no indexed vectors")

                    status_style = "green" if status == "green" else "red"
                    opt_style = "green" if optimizer_status == "ok" else "yellow"

                    table.add_row(
                        name,
                        proj["real_collection"],
                        f"[{status_style}]{status}[/{status_style}]",
                        f"[{opt_style}]{optimizer_status}[/{opt_style}]",
                        str(segments),
                        str(points),
                        str(indexed_vecs),
                        ", ".join(row_warnings) if row_warnings else "-",
                    )

                    if row_warnings:
                        for w in row_warnings:
                            warnings.append(f"{name}: {w}")

                except Exception as exc:
                    table.add_row(
                        name,
                        proj["real_collection"],
                        "[red]error[/red]",
                        "-",
                        "-",
                        "-",
                        "-",
                        str(exc),
                    )
                    warnings.append(f"{name}: unreachable ({exc})")

            console.print(table)

            all_collections = client.get_collections().collections
            all_aliases = client.get_aliases()
            alias_targets = {a.collection_name for a in all_aliases.aliases}
            alias_names = {a.alias_name for a in all_aliases.aliases}

            import re
            orphaned = [
                c.name for c in all_collections
                if c.name not in alias_targets
                and c.name not in alias_names
                and re.match(r"^.+_v\d+$", c.name)
            ]
            if orphaned:
                warnings.append(f"orphaned collections: {', '.join(orphaned)}")

            console.print(f"\n[green]Total projects:[/green] {len(projects)}")
            console.print(f"[green]Total collections:[/green] {len(all_collections)}")

            if warnings:
                console.print(f"\n[yellow]Warnings ({len(warnings)}):[/yellow]")
                for w in warnings:
                    console.print(f"  [yellow]- {w}[/yellow]")
            else:
                console.print("\n[green]All projects healthy.[/green]")

    except Exception as exc:
        console.print(f"[red]Audit failed:[/red] {exc}")
        sys.exit(1)


@cli.command()
@click.option("--dry-run", is_flag=True, help="Show what would be repaired without making changes.")
@click.pass_context
def repair(ctx: click.Context, dry_run: bool) -> None:
    """
    Repair collection state: migrate legacy collections to alias scheme,
    clean up orphaned versions, and ensure all projects have valid aliases.

    ctx: click.Context — CLI context with config.
    dry_run: bool — If True, only report what would be repaired.
    """
    config = ctx.obj["config"]

    if config.qdrant is None:
        console.print("[red]Qdrant is disabled in config.[/red]")
        sys.exit(1)

    try:
        with QuickContext(config) as qc:
            from engine.src.collection import CollectionManager
            client = qc.connection.client

            import re
            collections = client.get_collections().collections
            all_aliases = client.get_aliases()
            alias_names = {a.alias_name for a in all_aliases.aliases}
            alias_targets = {a.collection_name for a in all_aliases.aliases}

            legacy = []
            for col in collections:
                if col.name in alias_names:
                    continue
                if col.name in alias_targets:
                    continue
                if re.match(r"^.+_v\d+$", col.name):
                    continue
                legacy.append(col.name)

            repaired = 0
            if legacy:
                console.print(f"[yellow]Found {len(legacy)} legacy collection(s) without aliases:[/yellow]")
                for name in sorted(legacy):
                    if dry_run:
                        console.print(f"  [yellow]would migrate:[/yellow] {name} -> {name}_v1 + alias")
                    else:
                        console.print(f"  [cyan]Migrating:[/cyan] {name}...")
                        mgr = CollectionManager(
                            client=client,
                            config=config,
                            collection_name=name,
                        )
                        mgr.ensure()
                        console.print(f"  [green]Migrated:[/green] {name} -> alias scheme")
                    repaired += 1
            else:
                console.print("[green]No legacy collections found.[/green]")

            orphaned_cleaned = 0
            for alias_name in sorted(alias_names):
                mgr = CollectionManager(
                    client=client,
                    config=config,
                    collection_name=alias_name,
                )
                if dry_run:
                    prefix = alias_name + "_v"
                    for col in collections:
                        if col.name.startswith(prefix) and col.name not in alias_targets:
                            suffix = col.name[len(prefix):]
                            if suffix.isdigit():
                                console.print(f"  [yellow]would delete orphan:[/yellow] {col.name}")
                                orphaned_cleaned += 1
                else:
                    deleted = mgr.cleanup_old_versions(keep_current=True)
                    orphaned_cleaned += len(deleted)
                    for d in deleted:
                        console.print(f"  [green]Deleted orphan:[/green] {d}")

            console.print(f"\n[green]Repair complete.[/green] Legacy migrated: {repaired}, Orphans cleaned: {orphaned_cleaned}")
            if dry_run:
                console.print("[yellow]Dry run -- nothing was actually changed.[/yellow]")

    except Exception as exc:
        console.print(f"[red]Repair failed:[/red] {exc}")
        sys.exit(1)


@cli.command("file-read")
@click.argument("file", type=click.Path(exists=True))
@click.option("--start-line", type=int, default=None, help="Start line number (1-indexed)")
@click.option("--end-line", type=int, default=None, help="End line number (1-indexed)")
@click.option("--max-bytes", type=int, default=None, help="Maximum bytes to read")
@click.pass_context
def file_read_cmd(ctx: click.Context, file: str, start_line: int | None, end_line: int | None, max_bytes: int | None) -> None:
    """
    Read file content with optional line slicing.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)
    result = qc.file_read(file, start_line, end_line, max_bytes)

    console.print(f"[cyan]File:[/cyan] {result.file_path}")
    console.print(f"[cyan]Lines:[/cyan] {result.line_start}-{result.line_end} of {result.total_lines}")
    if result.truncated:
        console.print("[yellow]Truncated[/yellow]")
    console.print(f"[dim]Duration: {result.duration_ms}ms[/dim]\n")

    for line in result.lines:
        console.print(f"{line.line_number:4d} | {line.text}")


@cli.command("file-edit")
@click.argument("file", type=click.Path())
@click.option("--mode", type=click.Choice(["append", "insert", "replace", "delete", "batch"]), required=True)
@click.option("--start-line", type=int, help="Start line for edit")
@click.option("--end-line", type=int, help="End line for edit")
@click.option("--text", type=str, help="Text content for edit")
@click.option("--dry-run", is_flag=True, help="Preview changes without applying")
@click.option("--expected-hash", type=str, help="Expected SHA256 hash for safety")
@click.option("--no-undo", is_flag=True, help="Skip undo record creation")
@click.pass_context
def file_edit_cmd(ctx: click.Context, file: str, mode: str, start_line: int | None, end_line: int | None, text: str | None, dry_run: bool, expected_hash: str | None, no_undo: bool) -> None:
    """
    Apply file edit operation with reversible undo support.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)

    edits = None
    if start_line is not None:
        edits = [{"start_line": start_line, "end_line": end_line, "text": text}]

    result = qc.file_edit(file, mode, edits, text, dry_run, expected_hash, not no_undo)

    console.print(f"[cyan]File:[/cyan] {result.file_path}")
    console.print(f"[cyan]Applied:[/cyan] {result.applied}")
    console.print(f"[cyan]Before hash:[/cyan] {result.before_hash}")
    console.print(f"[cyan]After hash:[/cyan] {result.after_hash}")
    if result.edit_id:
        console.print(f"[green]Edit ID:[/green] {result.edit_id}")
    console.print(f"[dim]Duration: {result.duration_ms}ms[/dim]")

    if dry_run and result.updated_text:
        console.print("\n[yellow]Preview:[/yellow]")
        console.print(result.updated_text[:500])


@cli.command("file-edit-revert")
@click.argument("edit-id", type=str)
@click.option("--dry-run", is_flag=True, help="Preview revert without applying")
@click.option("--expected-hash", type=str, help="Expected current SHA256 hash")
@click.pass_context
def file_edit_revert_cmd(ctx: click.Context, edit_id: str, dry_run: bool, expected_hash: str | None) -> None:
    """
    Revert a previous file edit by undo ID.
    """
    config = ctx.obj["config"]
    qc = QuickContext(config)
    result = qc.file_edit_revert(edit_id, dry_run, expected_hash)

    console.print(f"[cyan]File:[/cyan] {result.file_path}")
    console.print(f"[cyan]Reverted:[/cyan] {result.reverted}")
    console.print(f"[cyan]Before hash:[/cyan] {result.before_hash}")
    console.print(f"[cyan]After hash:[/cyan] {result.after_hash}")
    console.print(f"[dim]Duration: {result.duration_ms}ms[/dim]")


@cli.command("symbol-edit")
@click.argument("file")
@click.argument("symbol_name")
@click.argument("new_source")
@click.option("--dry-run", is_flag=True, help="Simulate edit without writing")
@click.option("--expected-hash", help="Optional pre-edit hash guard")
@click.option("--no-undo", is_flag=True, help="Skip undo record")
@click.pass_context
def symbol_edit_cmd(ctx: click.Context, file: str, symbol_name: str, new_source: str, dry_run: bool, expected_hash: str | None, no_undo: bool) -> None:
    """Edit a symbol by name using AST extraction."""
    config = ctx.obj["config"]
    qc = QuickContext(config)
    result = qc.symbol_edit(
        file=file,
        symbol_name=symbol_name,
        new_source=new_source,
        dry_run=dry_run,
        expected_hash=expected_hash,
        record_undo=not no_undo,
    )
    console.print(f"[green]Edited:[/green] {result.file_path}")
    console.print(f"[cyan]Symbol:[/cyan] {result.symbol_name} (lines {result.line_start}-{result.line_end})")
    console.print(f"[cyan]Before hash:[/cyan] {result.before_hash}")
    console.print(f"[cyan]After hash:[/cyan] {result.after_hash}")
    if result.edit_id:
        console.print(f"[yellow]Edit ID:[/yellow] {result.edit_id}")
    console.print(f"[dim]Duration: {result.duration_ms}ms[/dim]")


def main() -> None:
    """
    CLI entry point.
    """
    cli()


if __name__ == "__main__":
    main()
