# MCP

## Purpose

The MCP server is the agent-facing wrapper for quickcontext.

It is intentionally thin:

- it loads the same `quickcontext.json` config used by the CLI and SDK
- it creates `QuickContext`
- it forwards indexing and retrieval into the SDK
- it exposes a stable tool surface for agents

The package lives in `qc_mcp/`.

Run it with:

```text
python -m qc_mcp
```

## Design

The MCP layer does not implement its own parser, indexer, or retrieval stack.

Instead, it wraps the SDK and relies on the Rust service plus Python orchestration underneath.

That means:

- indexing behavior stays aligned with the SDK
- retrieval behavior stays aligned with the SDK
- there is only one real implementation of indexing and search logic

## Tools

The server exposes seven tools:

- `project_info`
  Inspect one path, detect the project name, report index state, and list folder scopes.
- `list_projects`
  List indexed projects visible through the current Qdrant configuration.
- `index`
  Start semantic indexing for a path. Duplicate active runs are attached instead of starting another one.
- `index_status`
  Inspect the current or latest indexing run by run ID or by path/project.
- `search`
  Main AI-facing retrieval tool with `auto`, `text`, `semantic`, and `bundle` modes.
- `grep`
  Exact literal grep through the Rust service.
- `symbol_lookup`
  Exact or near-exact symbol lookup through the Rust symbol index.

## Resources And Prompt

The server also exposes:

Resources:

- `quickcontext://capabilities`
- `quickcontext://projects`
- `quickcontext://jobs`

Prompt:

- `quickcontext_search_playbook`

These help agent runtimes understand available capabilities and current server state.

## Search Modes

The `search` tool supports four modes:

- `auto`
  Default AI-facing retrieval route.
- `text`
  Rust lexical retrieval only.
- `semantic`
  Direct vector retrieval only.
- `bundle`
  Broader semantic retrieval with related-file and caller expansion.

Recommended default:

- start with `auto`
- use `grep` for exact literals
- use `symbol_lookup` for exact identifiers

## Indexing Behavior

The `index` tool calls `QuickContext.start_index_directory(...)`.

Important behavior:

- `path` is required
- repeated calls for the same active target attach to the running job
- `reindex=true` force-refreshes changed files
- `fast=true` selects the SDK fast indexing profile
- job status is inspectable through `index_status`

This lets agents start indexing and then poll progress without maintaining their own job registry.

## Configuration

The MCP server uses the `mcp` section of `quickcontext.json`:

```json
{
  "mcp": {
    "transport": "stdio",
    "host": "127.0.0.1",
    "port": 8000,
    "http_path": "/mcp/",
    "stateless_http": false
  }
}
```

You can also override startup settings on the command line:

```text
python -m qc_mcp --config quickcontext.json --transport http --host 127.0.0.1 --port 8000
```

## Running The Server

Local stdio mode:

```text
venv/Scripts/python.exe -m qc_mcp
```

HTTP mode:

```text
venv/Scripts/python.exe -m qc_mcp --config quickcontext.json --transport http --host 127.0.0.1 --port 8000
```

## Recommended Agent Workflow

For most agent tasks:

1. Call `project_info` for the target path.
2. If semantic retrieval is needed and the project is not indexed, call `index`.
3. Poll `index_status`.
4. Use `search` with `mode="auto"`.
5. Use `grep` or `symbol_lookup` when exact matching is a better fit.

## When To Use MCP

Use the MCP layer when:

- an agent runtime needs structured tools and typed responses
- you want quickcontext over stdio or HTTP
- you want the tool layer to stay aligned with the SDK

Use the SDK directly for native Python integrations.
