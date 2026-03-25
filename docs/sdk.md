# SDK

## Purpose

The Python SDK is the main product surface for quickcontext.

It is responsible for:

- orchestrating the long-lived Rust service
- indexing repositories into Qdrant
- running embeddings and optional LLM description generation
- composing retrieval strategies for agents and downstream tools
- exposing a stable integration surface for wrappers such as the CLI and MCP server

The main entrypoint is `engine.sdk.QuickContext`.

## Architecture

The SDK sits between the Rust service and downstream clients:

1. Rust handles parsing, grep, symbol lookup, text search, import graphs, protocol extraction, pattern search, file operations, and LSP-backed navigation.
2. The SDK handles chunking, deduplication, description generation, embeddings, Qdrant collection management, semantic retrieval, and higher-level orchestration.
3. Wrappers call the SDK instead of reimplementing indexing or retrieval logic.

`QuickContext` is lazy by design:

- Qdrant is not connected until vector features are needed.
- The Rust parser service is not started until parser or text operations are used.
- Embedding providers and the LLM description generator are created only when required.

## Configuration

The SDK reads `quickcontext.json` or `.quickcontext.json`.

Important sections:

- `qdrant`
  Connection details, HTTP port, optional API key, upsert batch sizing, and concurrency.
- `code_embedding`
  Provider and model for code vectors.
- `desc_embedding`
  Provider and model for description vectors.
- `llm`
  Optional LLM for description generation and artifact metadata generation.
- `service`
  Optional explicit path to the Rust service binary.
- `transport`
  Windows named pipe or Unix socket overrides.
- `mcp`
  Runtime settings used by the MCP wrapper.

For local development, `quickcontext.local.example.json` is the simplest starting point.

## Main Retrieval APIs

Use these methods for most integrations:

- `retrieve_context_auto(...)`
  Default AI-facing retrieval path. It routes exact symbol questions to symbol lookup first, uses Rust text retrieval when that is stronger, and falls back to semantic or bundle retrieval for broader questions.
- `semantic_search(...)`
  Direct semantic search over indexed vectors.
- `semantic_search_auto(...)`
  Semantic-only auto-routing between plain semantic search and the broader bundle path.
- `semantic_search_bundle(...)`
  Semantic retrieval with related-file and caller expansion for broader codebase exploration.
- `structured_search(...)`
  Typed multi-query semantic search.

Use these parser or lexical helpers when you need exact structure or exact text:

- `extract_symbols(...)`
- `extract_symbol(...)`
- `grep_text(...)`
- `symbol_lookup(...)`
- `find_callers(...)`
- `trace_call_graph(...)`
- `skeleton(...)`
- `import_graph(...)`
- `find_importers(...)`
- `import_neighbors(...)`
- `text_search(...)`
- `protocol_search(...)`
- `pattern_search(...)`
- `pattern_rewrite(...)`

## Indexing APIs

The SDK exposes both foreground and background indexing flows.

Foreground indexing:

- `index_directory(...)`
- `refresh_files(...)`

Background indexing:

- `start_index_directory(...)`
- `start_refresh_files(...)`
- `get_operation_status(...)`
- `list_operation_statuses(...)`
- `get_operation_status_for_target(...)`

Project discovery:

- `project_info(...)`
- `list_projects(...)`
- `project_collection_info(...)`
- `qdrant_available(...)`

Warmup:

- `warm_project(...)`
- `start_background_warm(...)`

## Indexing Profiles

Recommended default for most repositories:

```python
with QuickContext(config) as qc:
    qc.index_directory(
        ".",
        project_name="my-project",
        generate_descriptions=False,
        show_progress=False,
    )
```

Why this is the default:

- it keeps normal extraction and chunk coverage
- it still builds local fallback descriptions
- it avoids the main indexing cost, which is usually LLM description generation

Useful profile choices:

- `generate_descriptions=False`
  Best default for most repos.
- `fast=True`
  Fastest indexing path. It disables descriptions and uses stricter filtering defaults.
- `skip_minified=False`
  Important when generated frontend assets or dist bundles matter.
- `incremental_resume=True`
  Useful when you want to continue directly on the active collection instead of doing a shadow-swap style rebuild.

## Generated And Minified Code Handling

The SDK has dedicated logic for generated bundles and minified JavaScript.

Key behaviors:

- obvious large generated bundles can be downgraded into coarse artifact packets
- artifact packets carry deterministic semantic projections of extracted services, methods, types, fields, and strings
- artifact hotspot sampling prefers structurally interesting regions
- large generated files can emit one distributed summary packet for file-level recall
- AI-facing retrieval can replace a top bundle packet with a tighter local grep-backed focus snippet before returning it
- optional compact LLM artifact metadata can be enabled through `llm.artifact_metadata_*`

This is designed to make dist-heavy repositories indexable without treating hashed bundles like normal source files.

## LSP Support

The SDK can use LSP-backed document symbols and editing/navigation flows.

Available LSP surfaces:

- definitions, references, hover, symbols, format, diagnostics, completion
- rename and prepare-rename
- code actions and signature help
- workspace symbol search
- session listing and session shutdown
- project-specific setup and readiness reports

Relevant methods:

- `lsp_setup_plan(...)`
- `lsp_check(...)`
- `lsp_sessions(...)`
- `lsp_shutdown_all(...)`

## Basic Example

```python
from engine.sdk import QuickContext
from engine.src.config import EngineConfig

config = EngineConfig.from_json("quickcontext.json")

with QuickContext(config) as qc:
    qc.warm_project(".")

    payload = qc.retrieve_context_auto(
        "How does the SDK decide between text-first retrieval and semantic retrieval?",
        project_name="quickcontext",
        path=".",
        limit=3,
    )

    for item in payload["results"]:
        print(item.file_path, item.symbol_name, item.line_start, item.line_end)
```

## When To Use The SDK Directly

Use the SDK directly when:

- you are building another tool or agent wrapper
- you want full control over indexing and retrieval
- you need a long-lived process with reusable parser and vector state
- you want to compose parser, text, semantic, and graph retrieval in one place

If you only want a terminal surface, use the CLI.
If you need an agent-friendly tool protocol, use the MCP wrapper.
