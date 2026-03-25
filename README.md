# quickcontext

![quickcontext](assets/QuickContext-banner.png)

quickcontext is a local code context engine built around a long-lived Rust service and a Python SDK.

The repository has four main surfaces:

- `engine/`
  Python SDK for indexing, retrieval, embeddings, Qdrant orchestration, and wrapper-friendly APIs
- CLI
  Terminal surface for setup, indexing, maintenance, inspection, and benchmarking
- `qc_mcp/`
  Thin FastMCP wrapper over the SDK
- `service/`
  Rust backend for parsing, grep, symbol lookup, text search, import graphs, protocol extraction, pattern matching, file operations, and IPC

## Quick Start

Use the bootstrap flow for the fastest local setup.

Windows:

```powershell
.\scripts\setup_quickcontext.ps1
```

Cross-platform:

```text
python scripts/bootstrap_quickcontext.py
```

If `quickcontext.json` already exists, the bootstrap keeps the existing provider and API settings and only fills in missing runtime defaults.

After bootstrap:

```text
.venv/Scripts/python.exe -m engine status
.venv/Scripts/python.exe -m engine index . --project quickcontext --no-descriptions
.venv/Scripts/python.exe -m qc_mcp
```

## Documentation

Detailed docs are split by surface:

- [SDK](docs/sdk.md)
- [CLI](docs/cli.md)
- [MCP](docs/mcp.md)
- [Service](docs/service.md)

Agent-facing implementation notes live in [AI_DOCS.md](AI_DOCS.md).

## Configuration

quickcontext reads `quickcontext.json` or `.quickcontext.json`.

Start from one of these examples:

- `quickcontext.local.example.json`
  Local Qdrant, local embeddings, no API keys required
- `quickcontext.example.json`
  Cloud embeddings and LLM configuration

The JSON config model includes:

- `qdrant`
- `code_embedding`
- `desc_embedding`
- `llm`
- `service`
- `transport`
- `mcp`

## Validation

Useful validation commands:

```text
cargo check --manifest-path service/Cargo.toml
cargo test --manifest-path service/Cargo.toml
python -m py_compile engine/src/pipe.py engine/src/parsing.py engine/src/cli.py engine/__init__.py
venv/Scripts/python.exe -m unittest engine.tests.test_regressions engine.tests.test_mcp_server engine.tests.test_bootstrap_setup
```

## Requirements

- Python 3.10 or newer
- Rust toolchain with Cargo
- Native build tooling for the tree-sitter grammar crates
- Docker if you want local Qdrant with `docker compose`
- Windows or Linux

## License

This project is licensed under Apache-2.0. See `LICENSE`.
