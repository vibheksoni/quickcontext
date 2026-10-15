# quickcontext

![quickcontext](assets/QuickContext-banner.png)

quickcontext is a local context SDK for agents: fast code retrieval over a
long-lived Rust service, plus an embedded vector store that agents can use to
collect and recall anything (notes, findings, snippets, docs, tool output).

No database server or Docker is required. Vectors live in a single SQLite file and
are searched exactly in memory with numpy.

The repository has four main surfaces:

- `engine/`
  Python SDK for indexing, retrieval, embeddings, the embedded vector store, and agent document collections
- CLI
  Terminal surface for setup, indexing, maintenance, inspection, collect/recall, and benchmarking
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

If `quickcontext.json` already exists, the bootstrap keeps the existing provider and API settings and only fills in missing runtime defaults. A legacy `qdrant` section is converted to the new `store` section.

After bootstrap:

```text
.venv/Scripts/python.exe -m engine status
.venv/Scripts/python.exe -m engine index . --project quickcontext --no-descriptions
.venv/Scripts/python.exe -m qc_mcp
```

## Agent Memory In Three Calls

```python
from engine import QuickContext

with QuickContext() as qc:
    qc.collect("research", "Auth tokens are cached in Redis for 15 minutes", tags=["auth"])
    hits = qc.recall("research", "how long do auth tokens live?")
    qc.forget("research", tags="auth")
```

The same operations are available as CLI commands (`collect`, `recall`, `forget`, `collections`) and MCP tools.

## Documentation

Detailed docs are split by surface:

- [SDK](docs/sdk.md)
- [Vector store and agent documents](docs/store.md)
- [Embeddings: local, remote, GPU, profiles](docs/embeddings.md)
- [CLI](docs/cli.md)
- [MCP](docs/mcp.md)
- [Service](docs/service.md)
- [Low memory setup](docs/low-memory-setup.md)

Agent-facing implementation notes live in [AI_DOCS.md](AI_DOCS.md).

## Configuration

quickcontext reads `quickcontext.json` or `.quickcontext.json`.

Start from one of these examples:

- `quickcontext.local.example.json`
  Local embeddings, no API keys required
- `quickcontext.example.json`
  Cloud embeddings and LLM configuration

The JSON config model includes:

- `store`
- `code_embedding`
- `desc_embedding`
- `llm`
- `service`
- `transport`
- `mcp`

Named `profiles` switch between embedding setups (for example remote and local) with `--profile` or `QC_PROFILE`; see [docs/embeddings.md](docs/embeddings.md).

Vector dimensions for code collections are derived from `code_embedding.dimension` and `desc_embedding.dimension` unless a `vectors` list is given explicitly.

## Validation

Useful validation commands:

```text
cargo check --manifest-path service/Cargo.toml
cargo test --manifest-path service/Cargo.toml
python -m py_compile engine/src/pipe.py engine/src/parsing.py engine/src/cli.py engine/__init__.py
venv/Scripts/python.exe -m unittest engine.tests.test_vector_store engine.tests.test_regressions engine.tests.test_mcp_server engine.tests.test_sdk_surface engine.tests.test_bootstrap_setup engine.tests.test_lsp_setup engine.tests.test_retrieval_focus
```

## Requirements

- Python 3.10 or newer (SQLite with FTS5, which the standard CPython builds include)
- Rust toolchain with Cargo
- Native build tooling for the tree-sitter grammar crates
- Windows or Linux

## License

This project is licensed under Apache-2.0. See `LICENSE`.
