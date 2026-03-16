# quickcontext

![quickcontext](assets/QuickContext-banner.png)

quickcontext is a local code context engine for code search, indexing, parsing, and retrieval.

It currently has two main parts:

- `service/`: a Rust binary for parsing, grep, skeleton generation, text search, protocol search, pattern search, symbol and caller lookup, import graph analysis, and local IPC
- `engine/`: a Python SDK and CLI for indexing, Qdrant collection management, chunking, deduplication, embeddings, retrieval, watch mode, and edit operations

The SDK now includes AI-facing retrieval helpers:

- `QuickContext.retrieve_context_auto(...)`
  Default AI entrypoint. It routes exact symbol questions to the Rust symbol index first, expands behavior-oriented symbol questions with same-file helper symbols from indexed Rust metadata plus targeted source hydration, falls back to semantic or bundle retrieval for broader natural-language questions, enriches strong text-primary queries with import-aware and module-local related files, accepts explicit `path=` scoping when the target repo is outside the current working directory, and degrades to Rust text retrieval when an external repo has not been vector-indexed yet.
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

## Common Python CLI Commands

```text
python -m engine parse .
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
