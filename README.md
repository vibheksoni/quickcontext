# quickcontext

![quickcontext](assets/QuickContext-banner.png)

quickcontext is a local code context engine for code search, indexing, parsing, and retrieval.

It currently has two main parts:

- `service/`: a Rust binary for parsing, grep, skeleton generation, text search, protocol search, pattern search, symbol and caller lookup, import graph analysis, and local IPC
- `engine/`: a Python SDK and CLI for indexing, Qdrant collection management, chunking, deduplication, embeddings, retrieval, watch mode, and edit operations

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
python -m engine refresh engine/src/config.py
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
