# Contributing to quickcontext

## Overview

Contributions are welcome.

This project is still in progress, and there is room to improve:

- indexing quality
- retrieval and ranking
- parser and search performance
- protocol extraction quality
- cross-platform behavior
- future MCP support

## Before You Start

1. Read `README.md` for setup and architecture.
2. Read `AI_DOCS.md` if you are using an AI coding assistant.
3. Open an issue first if your change is large, architectural, or likely to affect multiple subsystems.

## Setup

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cargo build --release --manifest-path service/Cargo.toml
```

Linux:

```bash
sudo apt-get install -y build-essential
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cargo build --release --manifest-path service/Cargo.toml
```

## Contribution Guidelines

- Keep changes focused and easy to review.
- Do not mix unrelated refactors into feature work.
- Preserve existing CLI behavior unless the change is intentional and documented.
- Keep code modular.
- Keep imports minimal and organized.
- Avoid comments unless they explain non-obvious logic.
- Add useful docstrings for public or non-obvious Python functions.
- Do not hardcode secrets, tokens, or machine-specific paths.
- Keep runtime and transport changes cross-platform.
- Update `requirements.txt` when Python dependencies change.
- Update `service/Cargo.toml` and `service/Cargo.lock` when Rust dependencies change.
- Update `README.md` and `AI_DOCS.md` when setup or architecture changes.

## Validation

At minimum, run the checks that apply to your change:

```text
python -m py_compile engine/src/pipe.py engine/src/parsing.py engine/src/cli.py engine/__init__.py
python -m unittest engine.tests.test_vector_store engine.tests.test_regressions engine.tests.test_mcp_server
cargo check --manifest-path service/Cargo.toml
```

No database server is needed for development or tests; the vector store is an embedded SQLite file.

If your change affects indexing, search, or IPC behavior, include a short note in the pull request describing what you validated.

## Pull Requests

- Use a clear title and summary.
- Explain the problem, the change, and any tradeoffs.
- Mention follow-up work if the change is partial.
- Link the related issue when possible.

## Contribution License

By submitting code, documentation, or other material to this repository, you agree that your contribution will be licensed under the repository license.

## Scope

Good first contributions:

- small bug fixes
- doc improvements
- ranking and relevance tuning
- better error messages
- Linux and cross-platform polish
- tests and validation improvements
