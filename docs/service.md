# Service

## Purpose

The Rust service is the low-level execution engine behind quickcontext.

It handles the hot paths where a long-lived native process is valuable:

- parsing and extraction
- literal grep
- symbol lookup and caller tracing
- text search
- import graph operations
- protocol extraction
- AST pattern search
- file operations and symbol editing
- LSP-backed navigation and editing
- local IPC serving

The service package lives in `service/`.

## Runtime Role

The service is not the main user-facing product by itself.

Its role is to provide the local backend used by:

- the Python SDK
- the CLI
- the MCP wrapper

The SDK orchestrates indexing and retrieval around it.

## Core Modules

The crate currently exposes these major modules:

- `extract`
- `grep`
- `symbol_index`
- `skeleton`
- `import_graph`
- `text_search`
- `text_index`
- `protocol_search`
- `pattern_match`
- `pattern_rewrite`
- `file_ops`
- `symbol_edit`
- `lsp`
- `server`

This is the parser-first and lexical-first part of the system.

## Binary Commands

The Rust binary currently supports these commands:

- `extract`
- `extract-symbol`
- `grep`
- `skeleton`
- `text-search`
- `protocol-search`
- `pattern-search`
- `serve`

Examples:

```text
quickcontext-service extract <file_or_dir>
quickcontext-service grep "CollectionManager" --path .
quickcontext-service text-search "shadow collection alias"
quickcontext-service serve
```

## IPC Server

The most important runtime mode is `serve`.

In that mode, the service starts a local IPC daemon:

- Windows: named pipe `\\.\pipe\quickcontext`
- Unix: `transport.unix_socket_path` when configured, then `$XDG_RUNTIME_DIR/quickcontext.sock`, then `/tmp/quickcontext-<user>.sock`

The IPC protocol is:

- 4-byte little-endian frame length
- JSON request payload
- JSON response payload

The Python layer uses that transport through `engine/src/pipe.py` and `engine/src/parsing.py`.

## What The Service Owns

The service owns:

- parser-backed symbol extraction
- lexical search and grep
- symbol index lookups
- caller tracing
- text search
- import graph traversal
- protocol contract extraction
- AST pattern search
- AST rewrite support
- file read and edit primitives
- LSP-backed document and workspace operations

These operations do not require vector search and often remain usable even when Qdrant is unavailable.

## Build And Run

Build:

```text
cargo build --release --manifest-path service/Cargo.toml
```

Run manually:

Windows:

```powershell
.\service\target\release\quickcontext-service.exe serve
```

Linux:

```bash
./service/target/release/quickcontext-service serve
```

The Python layer can auto-start the service when `service.path` is available in `quickcontext.json`.

## Relationship To The SDK

The service does not manage:

- Qdrant collections
- embeddings
- optional LLM descriptions
- higher-level semantic retrieval composition

Those responsibilities stay in Python.

Current split:

- Rust service
  parsing, lexical search, symbol and graph operations, protocol extraction, file operations, LSP, IPC
- Python SDK
  chunking, deduplication, embeddings, optional LLM descriptions, Qdrant indexing, retrieval composition, wrappers

That split keeps the performance-sensitive execution in Rust while preserving an easy integration surface in Python.
