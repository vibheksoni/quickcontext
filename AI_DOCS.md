# AI_DOCS

## Purpose

quickcontext is a local code context engine.

The repository currently focuses on:

- a Rust service for parsing and search operations
- a Python engine for indexing, retrieval, embeddings, and orchestration

The project is still in progress. There is room to improve architecture, speed, ranking quality, and developer experience. MCP support is planned but not yet part of the tracked repo.

## Product Direction

- The primary product target is a long-lived Rust service plus a Python SDK wrapper.
- Rust should keep handling parsing, symbol and text search, graph operations, and other hot paths where performance matters.
- Python should keep handling embeddings, Qdrant orchestration, higher-level retrieval logic, and the SDK surface used by other tools.
- The Python CLI is mainly a development and validation surface. It is useful, but it is not the main product target.
- MCP is intentionally a later thin layer over the SDK. It is low priority until speed and retrieval quality are strong.
- When evaluating performance work, prioritize persistent service and SDK workflows over one-shot CLI startup optics.

## Architecture

- `service/`: Rust binary and IPC server
- `engine/`: Python SDK and CLI

Flow:

1. Keep the Rust service process running.
2. Python talks to Rust through local IPC.
3. Rust handles parser, search, and other heavy operations.
4. Python handles indexing, embeddings, Qdrant collections, retrieval composition, and SDK ergonomics.

`QuickContext` keeps that split lazy in practice:

- entering `with QuickContext(...)` does not force a Qdrant connection up front
- parser-only and text-first workflows can start without paying vector-store startup cost
- Qdrant still connects lazily on first vector or collection operation

## IPC Transport

The protocol is the same on all supported platforms:

- 4-byte little-endian length prefix
- JSON request payload
- JSON response payload

Endpoints:

- Windows: `\\.\pipe\quickcontext`
- Linux: `QC_SOCKET_PATH`, then `$XDG_RUNTIME_DIR/quickcontext.sock`, then `/tmp/quickcontext-<user>.sock`

## Configuration

Configuration resolution order:

1. `quickcontext.json` or `.quickcontext.json`
2. `QC_*` environment variables
3. built-in defaults from `engine/src/config.py`

Important config ideas:

- parser-only commands do not require Qdrant or embeddings
- indexing and semantic search require Qdrant plus embedding configuration
- local example config uses `fastembed`
- cloud example config uses `litellm`
- embedding dimensions must match the Qdrant collection vector dimensions
- the Rust symbol index snapshot lives in `.quickcontext/symbol_index.redb` and uses a compact binary payload format for faster cold loads
- Rust extraction also reuses compiled tree-sitter queries across parses to reduce repeated extractor overhead

## Important Python Modules

- `engine/src/cli.py`: CLI surface
- `engine/src/config.py`: config schema and defaults
- `engine/src/collection.py`: Qdrant collection lifecycle and aliases
- `engine/src/indexer.py`: upsert and payload handling
- `engine/src/searcher.py`: semantic and structured retrieval
- `engine/src/query_dsl.py`: typed query parser
- `engine/src/chunker.py`: symbol-to-chunk conversion
- `engine/src/differ.py`: chunk diffing
- `engine/src/embedder.py`: embedding pipeline
- `engine/src/describer.py`: description generation
- `engine/src/parsing.py`: typed wrappers over Rust service calls
- `engine/src/pipe.py`: IPC client
- `engine/src/watcher.py`: filesystem watch and refresh

Useful SDK retrieval primitives:

- `QuickContext.retrieve_context_auto(...)`: default AI entrypoint; routes exact symbol questions to symbol lookup, expands behavior-oriented exact-symbol questions with helper symbols from the same implementation file using source-bearing same-file extraction, can use Rust text search as the primary path for strong non-symbol technical queries, and attaches scoped graph-lexical companions for import/dependency-style graph questions
- `QuickContext.warm_project(...)`: preload persisted Rust symbol and text indices for a project root before the first real query
- `QuickContext.start_background_warm(...)`: schedule the same Rust warmup to run once the SDK session goes idle instead of blocking startup
- `QuickContext.semantic_search(...)`: main semantic retrieval path
- `QuickContext.semantic_search_auto(...)`: semantic-only auto-routing between fast direct retrieval and the deeper bundle path
- `QuickContext.structured_search(...)`: typed multi-query retrieval
- `QuickContext.semantic_search_bundle(...)`: semantic anchors plus distinct semantic neighbor files, related import-graph files, and caller context for deeper follow-up exploration

Recommendation:

- Use `retrieve_context_auto(...)` as the safest default for AI workflows.
- Use `semantic_search(...)` for fast single-hop semantic retrieval.
- Use `semantic_search_auto(...)` when you specifically want semantic-only auto-routing.
- Use `semantic_search_bundle(...)` directly when you already know the question is broad, architectural, or likely to require cross-file follow-up context.
- Use `warm_project(...)` once near startup when you expect a long-lived service session and want the first heavy query to be cheaper.
- Use `start_background_warm(...)` when you want that same project warmup to happen opportunistically after the session goes idle instead of paying the cost up front.

## Important Rust Modules

- `service/src/main.rs`: service CLI
- `service/src/server.rs`: IPC server and request dispatch
- `service/src/protocol.rs`: request and response schema
- `service/src/extract.rs`: AST extraction
- `service/src/grep.rs`: literal grep
- `service/src/text_search.rs`: BM25-style text search
- `service/src/protocol_search.rs`: request and response contract extraction
- `service/src/pattern_match.rs`: structural pattern search
- `service/src/pattern_rewrite.rs`: structural rewrite
- `service/src/symbol_index.rs`: symbol lookup and caller tracing
- `service/src/import_graph.rs`: dependency and importer graph
- `service/src/lsp/`: language server integration
- `service/src/lang/`: tree-sitter language registry

## Execution Basics

Build the Rust service:

- Windows: `cargo build --release --manifest-path service/Cargo.toml`
- Linux: `cargo build --release --manifest-path service/Cargo.toml`

Run the service:

- Windows: `./service/target/release/quickcontext-service.exe serve`
- Linux: `./service/target/release/quickcontext-service serve`

Common SDK and engine flow:

- `python -m engine status`
- `python -m engine init`
- `python -m engine warm .`
- `python -m engine index <path> [--project <name>]`
- `python -m engine search "<query>" [--project <name>]`
- `python -m engine refresh <files...>`
- `python -m engine watch <dir>`

Use the CLI to validate and benchmark the same underlying service and SDK behavior. Do not treat CLI startup as the main product surface.

## Validation And Benchmarking

Validation commands:

- `cargo check --manifest-path service/Cargo.toml`
- `cargo test --manifest-path service/Cargo.toml`
- `python -m py_compile engine/src/pipe.py engine/src/parsing.py engine/src/cli.py engine/__init__.py`
- `venv/Scripts/python.exe -m unittest engine.tests.test_regressions`
- `venv/Scripts/python.exe scripts/retrieval_benchmark.py --config quickcontext.json --project quickcontext`
- `venv/Scripts/python.exe scripts/context_retrieval_benchmark.py --config quickcontext.json --project quickcontext --cases-file scripts/context_retrieval_cases.json --strategy context-auto`
- `venv/Scripts/python.exe scripts/context_retrieval_benchmark.py --config quickcontext.json --project quickcontext --cases-file scripts/graph_retrieval_cases.json --strategy context-auto`
- `venv/Scripts/python.exe scripts/symbol_context_benchmark.py --config quickcontext.json --project quickcontext --cases-file scripts/symbol_context_cases.json --strategy context-auto`
- `venv/Scripts/python.exe scripts/text_retrieval_benchmark.py --config quickcontext.json --cases-file scripts/context_retrieval_cases.json --show-top 3`
- `venv/Scripts/python.exe scripts/warm_project_benchmark.py --config quickcontext.json --project quickcontext --path . --query "How does CodeSearcher.search_hybrid merge code and description vectors?"`

Benchmarking guidance:

- Prefer direct before/after timings for any speed-related change.
- Use the Rust release binary for service-only timings.
- Benchmark live SDK calls against a running Rust service, not only one-shot CLI invocations.
- Use bounded targets for end-to-end indexing when the active config uses remote embeddings.
- Keep local benchmark notes in `BENCHMARK_LOCAL.md`. That file is intentionally gitignored.

## Strict Coding Guidelines

- Keep code clean and modular.
- Keep imports minimal and organized.
- Avoid comments unless they explain non-obvious logic.
- Add useful docstrings for public or non-obvious Python functions.
- Preserve existing CLI behavior unless a change is intentional and documented.
- Preserve the optional and lazy-loaded subsystem design.
- Do not hardcode secrets, tokens, or machine-specific paths.
- Keep transport and process code cross-platform when touching runtime boundaries.
- Update `requirements.txt` when Python dependencies change.
- Update `service/Cargo.toml` and `service/Cargo.lock` when Rust dependencies change.
- Keep README and AI_DOCS in sync with architecture or setup changes.
- Keep changes focused. Do not mix unrelated refactors into feature work.
- Prefer simple solutions over clever ones when maintaining IPC, indexing, and search code.

## Contribution Priorities

Useful areas for contributors:

- search relevance and ranking
- indexing performance
- chunk quality
- protocol extraction quality
- cross-platform support
- LSP reliability
- future MCP support

## Notes For AI Agents

- Read the code before proposing structural changes.
- Treat the persistent Rust service plus Python SDK as the core product surface.
- When touching indexing or search, check dimension compatibility and collection behavior.
- When touching IPC, keep Windows and Linux behavior aligned at the protocol level.
- When touching docs, keep them plain, direct, and easy to navigate.
- Prefer evidence-based changes: benchmark, validate, then iterate.
