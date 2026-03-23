# quickcontext

![quickcontext](assets/QuickContext-banner.png)

quickcontext is a local code context engine for code search, indexing, parsing, and retrieval.

It currently has two main parts:

- `service/`: a Rust binary for parsing, grep, skeleton generation, text search, protocol search, pattern search, symbol and caller lookup, import graph analysis, and local IPC
- `engine/`: a Python SDK and CLI for indexing, Qdrant collection management, chunking, deduplication, embeddings, retrieval, watch mode, and edit operations
- `quickcontext_mcp/`: a thin FastMCP server that wraps the Python SDK with agent-friendly indexing and retrieval tools

The SDK now includes AI-facing retrieval helpers:

- `QuickContext.retrieve_context_auto(...)`
  Default AI entrypoint. It routes exact symbol questions to the Rust symbol index first, expands behavior-oriented symbol questions with same-file helper symbols from indexed Rust metadata plus targeted source hydration, falls back to semantic or bundle retrieval for broader natural-language questions, enriches strong text-primary queries with import-aware and module-local related files, accepts explicit `path=` scoping when the target repo is outside the current working directory, and degrades to Rust text retrieval when an external repo has not been vector-indexed yet.
- `QuickContext.project_info(...)`
  Stable project discovery entrypoint for downstream wrappers. It reports the detected project name, cache state, parser connectivity, Qdrant availability, current collection metadata, and optional folder scopes.
- `QuickContext.list_projects(...)`
  Returns typed project collection summaries so wrappers no longer need to import `CollectionManager` directly just to discover indexed projects.
- `QuickContext.qdrant_available(...)`
  Explicit backend health probe for wrappers that need to distinguish “no indexed projects” from “vector store is currently unreachable”.
- `QuickContext.start_index_directory(...)`
  Starts SDK-owned background indexing and returns a pollable operation snapshot instead of forcing each wrapper to invent its own job registry.
- `QuickContext.start_refresh_files(...)`
  Starts SDK-owned background refresh work with the same pollable snapshot model.
- `QuickContext.get_operation_status(...)` and `QuickContext.list_operation_statuses(...)`
  Poll live indexing and refresh snapshots, including stage, files remaining, chunks kept, description progress, embedding progress, and points upserted.
- `QuickContext.warm_project(...)`
  Optional startup warmup for the long-lived Rust service. It preloads the persisted Rust symbol and text indices for a project so the first real query is cheaper.
- `QuickContext.start_background_warm(...)`
  Optional idle warmup for long-lived SDK sessions. It schedules the same project warmup to happen once the session goes idle instead of blocking startup.
- `QuickContext.semantic_search_auto(...)`
  Lets the SDK choose between fast direct semantic retrieval and the deeper graph-aware bundle path. It also accepts explicit `path=` scoping for external repos.
- `QuickContext.semantic_search_bundle(...)`
  Returns semantic anchors plus distinct semantic neighbor files, related import-graph files, and caller context for deeper codebase exploration. It also accepts explicit `path=` scoping for external repos.

Use `retrieve_context_auto(...)` as the default for AI workflows, `semantic_search(...)` for direct semantic retrieval, `semantic_search_auto(...)` when you specifically want semantic-only auto-routing, and `semantic_search_bundle(...)` when you explicitly want the deeper cross-file expansion path.
If you run the service as a long-lived process, call `warm_project('.')` once after startup to preload local indices before the first heavy query. If you do not want to pay that startup cost up front, call `start_background_warm('.')` and let the SDK defer the same warmup until the session goes idle.
Entering `with QuickContext(...)` keeps subsystems lazy: parser-only flows do not preconnect Qdrant, and vector features still connect on first use.
When the repo you want to inspect is outside the process cwd, pass that repo root through `path=` on the AI-facing retrieval helpers so Rust symbol/text routing, helper expansion, caller expansion, and benchmark harnesses stay scoped to the right codebase.
Full alias/shadow indexing now writes a local resume manifest under `.quickcontext/`, so if indexing is interrupted it can reuse the in-progress shadow collection instead of discarding files that were already upserted.
When `fast=True`, the indexer now downgrades obvious large minified JavaScript bundle artifacts to a small number of coarse file chunks before deep extraction. That keeps generated trees indexable without paying full parser cost on huge hashed bundles.
Those generated artifact chunks now also prepend a deterministic semantic projection of extracted services, methods, types, fields, and relevant strings so semantic retrieval has better signal on dist-heavy JavaScript bundles even when fast mode skips full symbol extraction.
Generated artifact chunk sampling now prefers structurally interesting hotspots from extracted service/type/method/field/string signals before filling the remaining windows with broad file coverage, so large generated bundles are more likely to expose the relevant region to semantic retrieval.
Large generated bundle files now also emit one distributed summary packet that aggregates signals from multiple structural hotspots across the file, giving semantic retrieval one cheaper file-level anchor in addition to the per-window artifact chunks.
Generated artifact fallback descriptions can optionally be upgraded with compact batched LLM metadata behind the `artifact_metadata_*` LLM config knobs, so you can experiment with richer generated-code interpretation without enabling full per-chunk descriptions.
AI-facing retrieval now tries to replace top generated-bundle summary hits with a tighter file-local grep-backed focus snippet before returning results, so agents see the most relevant local evidence inside a generated file instead of only a broad artifact packet.
If Rust extraction returns no symbols for a file and a ready language server exists for that language, the SDK now opportunistically enriches that file with LSP document symbols before chunking so indexing can still use AST-backed structure instead of falling back straight to whole-file chunks.
On Windows, npm-installed language servers are now resolved directly from the npm global bin directory for readiness checks and Rust-side LSP spawning, so installing them does not require restarting the shell just to make them discoverable.
The indexing stats now include real phase timings for scan, artifact profiling, extraction, chunk building, filtering, dedup, description generation, embedding, point building, and Qdrant upsert.

## Status

This repository is still a work in progress.

- The current focus is the Rust service and Python engine
- The design can still be improved in many places
- We want contributors
- A first-class MCP layer is planned but not part of the tracked repo yet

If you want to help build it, issues and pull requests are welcome.

## AI Agent Setup

This repository includes [AI_DOCS.md](AI_DOCS.md) as the clean agent-facing project guide.

If your coding assistant expects `AGENTS.md` or `CLAUDE.md`, copy `AI_DOCS.md` to the filename your tool expects in your local setup.

## Python SDK Examples

```python
from engine.sdk import QuickContext
from engine.src.config import EngineConfig

config = EngineConfig.from_json("quickcontext.json")

with QuickContext(config) as qc:
    qc.warm_project(".")

    payload = qc.retrieve_context_auto(
        "How does the Python layer decide how to connect to the Rust service on Windows versus Linux?",
        project_name="quickcontext",
        path=".",
        limit=3,
    )

    for item in payload["results"]:
        print(item.file_path, item.symbol_name, item.line_start, item.line_end)
```

```python
from engine.sdk import QuickContext
from engine.src.config import EngineConfig

config = EngineConfig.from_json("quickcontext.json")

with QuickContext(config) as qc:
    stats = qc.index_directory(
        ".",
        project_name="quickcontext",
        fast=True,
        show_progress=False,
    )
    print(stats.total_chunks, stats.upserted_points)
```

## Repository Layout

- `engine/`: Python package and CLI entrypoint
- `quickcontext_mcp/`: FastMCP server package
- `service/`: Rust service and command-line binary
- `docker-compose.yml`: local Qdrant container
- `requirements.txt`: Python dependencies
- `quickcontext.local.example.json`: local config using `fastembed`
- `quickcontext.example.json`: cloud config using `litellm`

## Platform Support

The transport layer supports both Windows and Linux.

- Windows transport: named pipe `\\.\pipe\quickcontext`
- Linux transport: Unix socket from `QC_SOCKET_PATH`, then `$XDG_RUNTIME_DIR/quickcontext.sock`, then `/tmp/quickcontext-<user>.sock`

The request protocol is the same on both platforms: 4-byte little-endian length prefix plus JSON payload.

The Rust text-search path now persists file-category and path-field metadata alongside content terms, so lexical retrieval can use both content and file-path signals while downweighting low-priority generated/test-like files.

## Requirements

- Python 3.10 or newer
- Rust toolchain with Cargo
- Native C/C++ build toolchain for the tree-sitter grammar crates
- Docker if you want local Qdrant with `docker compose`
- Windows or Linux

## Configuration

The Python CLI resolves configuration in this order:

1. `quickcontext.json` or `.quickcontext.json`
2. `QC_*` environment variables
3. built-in defaults from `engine/src/config.py`

### Local config

Windows:

```powershell
Copy-Item quickcontext.local.example.json quickcontext.json
```

Linux:

```bash
cp quickcontext.local.example.json quickcontext.json
```

This example uses local Qdrant, `fastembed`, and `llm: null`.

For that setup, index with `--fast` or `--no-descriptions`.

### Cloud config

Windows:

```powershell
Copy-Item quickcontext.example.json quickcontext.json
```

Linux:

```bash
cp quickcontext.example.json quickcontext.json
```

Then replace the placeholder API keys before indexing.

### Service path

The Python layer looks for the service binary at:

- `service/target/release/quickcontext-service.exe`
- `service/target/release/quickcontext-service`

If the binary lives somewhere else, set `QC_SERVICE_PATH`.

## Setup

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cargo build --release --manifest-path service/Cargo.toml
docker compose up -d qdrant

Copy-Item quickcontext.local.example.json quickcontext.json

python -m engine status
python -m engine init
python -m engine index . --project quickcontext --fast
```

### Linux

```bash
sudo apt-get install -y build-essential
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cargo build --release --manifest-path service/Cargo.toml
docker compose up -d qdrant

cp quickcontext.local.example.json quickcontext.json

python -m engine status
python -m engine init
python -m engine index . --project quickcontext --fast
```

## Running The Rust Service

The Python layer can auto-start the Rust service if the binary exists at the default path. Running it manually is still the clearest option.

Windows:

```powershell
.\service\target\release\quickcontext-service.exe serve
```

Linux:

```bash
./service/target/release/quickcontext-service serve
```

## MCP Server

The tracked repository now includes a thin FastMCP wrapper around the Python SDK. The MCP surface is intentionally narrow:

- `project_info`: inspect one path, detect the project name, check index state, and list useful folder scopes
- `list_projects`: list currently indexed projects
- `index`: index a path for semantic retrieval and suppress duplicate active runs for the same target
- `index_status`: inspect the latest or active indexing run
- `search`: main AI-facing retrieval tool with `auto`, `text`, `semantic`, and `bundle` modes
- `grep`: exact literal grep through the Rust service
- `symbol_lookup`: exact or near-exact identifier lookup through the Rust symbol index

For path-scoped MCP tools, always pass an explicit target path. Do not rely on the MCP server process working directory.

Run it locally over stdio:

```text
venv/Scripts/python.exe -m quickcontext_mcp
```

Run it as a long-lived HTTP server:

```text
set QC_MCP_TRANSPORT=http
set QC_MCP_HOST=127.0.0.1
set QC_MCP_PORT=8000
venv/Scripts/python.exe -m quickcontext_mcp
```

Useful environment variables:

- `QC_MCP_CONFIG`: explicit config file path. Defaults to `quickcontext.json`, then `.quickcontext.json`, then auto config resolution.
- `QC_MCP_TRANSPORT`: `stdio` or `http`
- `QC_MCP_HOST`: HTTP bind host
- `QC_MCP_PORT`: HTTP bind port
- `QC_MCP_HTTP_PATH`: HTTP MCP route path, default `/mcp/`
- `QC_MCP_STATELESS_HTTP`: `true` or `false` for stateless HTTP mode

## Common Python CLI Commands

```text
python -m engine parse .
python -m engine lsp-setup <path>
python -m engine lsp-sessions
python -m engine lsp-shutdown-all
python -m engine grep "CollectionManager"
python -m engine skeleton . --markdown
python -m engine text-search "auth token"
python -m engine protocol-search "request response"
python -m engine pattern-search "(function_definition name: (identifier) @name)" --lang python
python -m engine search "chunk filter" --project quickcontext
python -m engine warm .
python -m engine benchmark-context --project quickcontext --path . --cases-file scripts/context_retrieval_cases_template.json
python -m engine benchmark-compare --project quickcontext --path . --cases-file scripts/context_retrieval_cases_template.json
python -m engine refresh engine/src/config.py --project quickcontext
python -m engine watch .
python -m engine list-projects
python -m engine status
```

## LSP Setup

quickcontext can detect which language servers a target project likely needs and print install commands for missing ones.

Preview the plan:

```text
python -m engine lsp-setup "C:/path/to/project"
```

Run supported auto-installs:

```text
python -m engine lsp-setup "C:/path/to/project" --install
```

Check readiness after install:

```text
python -m engine lsp-check "C:/path/to/project"
```

One-step guided flow:

```text
python -m engine lsp-setup "C:/path/to/project" --install --check
```

Windows PowerShell wrapper:

```powershell
.\scripts\setup_project_lsps.ps1 -Path "C:\path\to\project"
.\scripts\setup_project_lsps.ps1 -Path "C:\path\to\project" -Install
.\scripts\setup_project_lsps.ps1 -Path "C:\path\to\project" -Install -Check
```

Notes:

- The setup command is project-scoped. Pass the actual target repo path.
- Some servers have automatic install commands; others are reported as manual-only with notes.
- The command checks whether the expected LSP binary is already on `PATH`.
- `lsp-check` reports `ready`, `missing`, `installed`, or `error` based on binary presence plus a lightweight probe where available.
- `lsp-sessions` lists the language servers the Rust service is actively tracking, including pid and project root.
- `lsp-shutdown-all` explicitly shuts down those tracked language-server sessions for cleanup without stopping the whole service.

## Validation

Useful validation commands during development:

```text
cargo check --manifest-path service/Cargo.toml
cargo test --manifest-path service/Cargo.toml
python -m py_compile engine/src/pipe.py engine/src/parsing.py engine/src/cli.py engine/__init__.py
venv/Scripts/python.exe -m unittest engine.tests.test_regressions
venv/Scripts/python.exe scripts/retrieval_benchmark.py --config quickcontext.json --project quickcontext
venv/Scripts/python.exe scripts/context_retrieval_benchmark.py --config quickcontext.json --project quickcontext --cases-file scripts/context_retrieval_cases.json --strategy context-auto
venv/Scripts/python.exe scripts/context_retrieval_benchmark.py --config quickcontext.json --project external-bench --path <target-root> --cases-file scripts/augmentintent_dist_cases.json --strategy context-auto
venv/Scripts/python.exe -m engine --config quickcontext.json benchmark-context --project external-bench --path <target-root> --cases-file scripts/context_retrieval_cases_template.json --strategy context-auto
venv/Scripts/python.exe -m engine --config quickcontext.json benchmark-compare --project external-bench --path <target-root> --cases-file scripts/context_retrieval_cases_template.json
venv/Scripts/python.exe scripts/symbol_context_benchmark.py --config quickcontext.json --project quickcontext --cases-file scripts/symbol_context_cases.json --strategy context-auto
venv/Scripts/python.exe scripts/context_retrieval_benchmark.py --config quickcontext.json --project quickcontext --cases-file scripts/graph_retrieval_cases.json --strategy context-auto
venv/Scripts/python.exe scripts/text_retrieval_benchmark.py --config quickcontext.json --cases-file scripts/context_retrieval_cases.json --show-top 3
venv/Scripts/python.exe scripts/warm_project_benchmark.py --config quickcontext.json --project quickcontext --path . --query "How does CodeSearcher.search_hybrid merge code and description vectors?"
```

For local performance work, keep benchmark notes in `BENCHMARK_LOCAL.md`. That file is intentionally gitignored.

## Contributing

Contributors are wanted.

Good areas to help with:

- indexing quality
- retrieval and ranking
- protocol extraction quality
- chunking and deduplication
- Linux and cross-platform polish
- future MCP support

If you want to contribute, start by reading [AI_DOCS.md](AI_DOCS.md).

## License

This project is licensed under Apache-2.0. See `LICENSE`.
