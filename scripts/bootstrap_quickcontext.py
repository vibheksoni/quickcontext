from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _service_binary(root: Path, build_mode: str) -> Path:
    name = "quickcontext-service.exe" if os.name == "nt" else "quickcontext-service"
    profile = "release" if build_mode == "release" else "debug"
    return root / "service" / "target" / profile / name


def _resolved_service_path(root: Path, build_mode: str) -> Path:
    if build_mode != "skip":
        return _service_binary(root, build_mode)

    release_path = _service_binary(root, "release")
    if release_path.exists():
        return release_path
    debug_path = _service_binary(root, "debug")
    if debug_path.exists():
        return debug_path
    return debug_path


def _display(cmd: list[str]) -> str:
    return " ".join(cmd)


def _run(cmd: list[str], *, cwd: Path, dry_run: bool = False) -> None:
    print(f"+ {_display(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _load_or_seed_config(config_path: Path, example_path: Path) -> dict:
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return json.loads(example_path.read_text(encoding="utf-8"))


def _write_config(
    *,
    config_path: Path,
    example_path: Path,
    service_path: Path,
    dry_run: bool,
) -> None:
    data = _load_or_seed_config(config_path, example_path)
    legacy = data.pop("qdrant", None)
    store = data.setdefault("store", {})
    if isinstance(legacy, dict):
        for key in ("collection", "upsert_batch_size"):
            if key in legacy:
                store.setdefault(key, legacy[key])
    store.setdefault("path", None)

    service = data.setdefault("service", {})
    if not str(service.get("path", "") or "").strip():
        service["path"] = str(service_path)

    transport = data.setdefault("transport", {})
    transport.setdefault("windows_pipe_name", r"\\.\pipe\quickcontext")
    transport.setdefault("unix_socket_path", None)

    mcp = data.setdefault("mcp", {})
    mcp.setdefault("transport", "stdio")
    mcp.setdefault("host", "127.0.0.1")
    mcp.setdefault("port", 8000)
    mcp.setdefault("http_path", "/mcp/")
    mcp.setdefault("stateless_http", False)

    payload = json.dumps(data, indent=2) + "\n"
    print(f"+ write {config_path}")
    if dry_run:
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(payload, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python scripts/bootstrap_quickcontext.py")
    parser.add_argument("--profile", choices=["local", "cloud"], default="local")
    parser.add_argument("--service-build", choices=["debug", "release", "skip"], default="debug")
    parser.add_argument("--config", dest="config_path", default="quickcontext.json")
    parser.add_argument("--venv", dest="venv_path", default=".venv")
    parser.add_argument("--skip-docker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-init", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = _repo_root()
    venv_dir = (root / args.venv_path).resolve()
    venv_python = _venv_python(venv_dir)
    config_path = (root / args.config_path).resolve()
    example_name = "quickcontext.local.example.json" if args.profile == "local" else "quickcontext.example.json"
    example_path = (root / example_name).resolve()
    service_path = _resolved_service_path(root, args.service_build)

    print("quickcontext bootstrap")
    print(f"repo:   {root}")
    print(f"profile:{args.profile}")
    print(f"venv:   {venv_dir}")
    print(f"config: {config_path}")
    if config_path.exists():
        print("config: existing file detected, preserving current settings")

    if not venv_python.exists():
        _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=root, dry_run=args.dry_run)

    _run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], cwd=root, dry_run=args.dry_run)
    _run([str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"], cwd=root, dry_run=args.dry_run)

    if args.service_build != "skip":
        cargo_cmd = ["cargo", "build", "--manifest-path", "service/Cargo.toml"]
        if args.service_build == "release":
            cargo_cmd.insert(2, "--release")
        _run(cargo_cmd, cwd=root, dry_run=args.dry_run)

    _write_config(
        config_path=config_path,
        example_path=example_path,
        service_path=service_path,
        dry_run=args.dry_run,
    )

    if not args.skip_init:
        _run([str(venv_python), "-m", "engine", "--config", str(config_path), "init"], cwd=root, dry_run=args.dry_run)

    print()
    print("Bootstrap complete.")
    print(f"Config: {config_path}")
    print(f"Service binary: {service_path}")
    print("Vector store: embedded SQLite (no server or Docker needed)")
    if args.profile == "local":
        print("Local profile is ready for no-key indexing and search.")
    else:
        print("Cloud profile is configured. Replace placeholder API keys in quickcontext.json before indexing.")
    print()
    print("Next commands:")
    if os.name == "nt":
        print(rf"  {venv_python} -m engine status")
        print(rf"  {venv_python} -m engine index . --project quickcontext --no-descriptions")
        print(rf"  {venv_python} -m qc_mcp")
    else:
        print(f"  {venv_python} -m engine status")
        print(f"  {venv_python} -m engine index . --project quickcontext --no-descriptions")
        print(f"  {venv_python} -m qc_mcp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
