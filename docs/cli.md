# CLI

## Purpose

The CLI is the main terminal surface for quickcontext.

It is intended for:

- local setup
- indexing and refresh operations
- parser and retrieval debugging
- repository inspection
- LSP setup and verification
- collection maintenance
- local benchmarking

The CLI is built on top of the Python SDK. It does not implement its own indexing or retrieval system.

## Configuration

The CLI reads `quickcontext.json` by default.

You can override that with:

```text
python -m engine --config quickcontext.json <command>
```

The same JSON config used by the SDK controls:

- Qdrant connection
- embedding providers
- optional LLM description generation
- service binary path
- IPC transport settings

## Core Commands

State and maintenance:

- `status`
- `init`
- `reset`
- `repair`
- `gc`
- `compact`
- `audit`
- `list-projects`

Indexing:

- `index`
- `refresh`
- `watch`
- `warm`
- `cache-status`
- `cache-clear`

Parser, structure, and exact search:

- `parse`
- `extract-symbol`
- `grep`
- `symbol-lookup`
- `find-callers`
- `trace-call-graph`
- `skeleton`
- `import-graph`
- `find-importers`
- `text-search`
- `protocol-search`
- `pattern-search`
- `pattern-rewrite`

LSP:

- `lsp-setup`
- `lsp-check`
- `lsp-definition`
- `lsp-references`
- `lsp-hover`
- `lsp-symbols`
- `lsp-format`
- `lsp-diagnostics`
- `lsp-completion`
- `lsp-rename`
- `lsp-prepare-rename`
- `lsp-code-actions`
- `lsp-signature-help`
- `lsp-workspace-symbols`
- `lsp-sessions`
- `lsp-shutdown-all`

Editing:

- `file-read`
- `file-edit`
- `file-edit-revert`
- `symbol-edit`

Benchmarking and search guidance:

- `search`
- `benchmark-context`
- `benchmark-compare`
- `embed-test`
- `mcp-hints`

## Indexing

Recommended default indexing command:

```text
python -m engine index . --project quickcontext --no-descriptions
```

That profile keeps normal extraction and chunk coverage while skipping the main indexing cost, which is usually LLM description generation.

Important indexing flags:

- `--no-descriptions`
  Best default for most repos. Keeps normal extraction and uses lightweight fallback descriptions instead of LLM-generated ones.
- `--fast`
  Fastest indexing path. Disables descriptions and tightens chunk filtering defaults.
- `--no-skip-minified`
  Keeps likely minified or generated chunks. Important for dist-heavy frontend repos.
- `--incremental-resume`
  Continue indexing directly on the active collection instead of using the shadow-swap path.
- `--resume-batch-files`
  Control recovery batch sizing for resumable indexing.

Common flow:

```text
python -m engine init
python -m engine index <path> --project <name> --no-descriptions
python -m engine refresh <files...> --project <name>
python -m engine watch <path>
```

## Search And Retrieval

The CLI exposes several search styles.

Exact and lexical:

- `grep`
- `symbol-lookup`
- `text-search`

Semantic:

- `search`

Structured extraction:

- `protocol-search`
- `pattern-search`

The `search` command supports bias presets defined in `engine/src/search_modes.py`:

- `precise`
- `discovery`
- `implementation`
- `debug`
- `planning`

These presets tune vector-space choice, keyword weighting, reranking, and result expansion.

## LSP Commands

Use the CLI to plan, install, and verify likely language servers for a target project.

Examples:

```text
python -m engine lsp-setup "C:/path/to/project"
python -m engine lsp-setup "C:/path/to/project" --install
python -m engine lsp-check "C:/path/to/project"
python -m engine lsp-sessions
python -m engine lsp-shutdown-all
```

Windows PowerShell wrapper:

```powershell
.\scripts\setup_project_lsps.ps1 -Path "C:\path\to\project"
```

## Bootstrap

For first-time setup, prefer the bootstrap scripts instead of typing the manual steps yourself.

Windows:

```powershell
.\scripts\setup_quickcontext.ps1
```

Cross-platform:

```text
python scripts/bootstrap_quickcontext.py
```

If you want to test bootstrap without touching your main local config:

```text
python scripts/bootstrap_quickcontext.py --config _ignore/bootstrap.local.json
```

## When To Use The CLI

Use the CLI when:

- you want a terminal-first development surface
- you are validating setup or config
- you are debugging parser, search, or indexing behavior
- you are benchmarking retrieval quality or latency

Use the SDK when you are building another integration.
Use the MCP wrapper when an agent runtime needs a tool-based interface.
