from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request


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


def _find_docker_compose() -> list[str]:
    if shutil.which("docker"):
        try:
            subprocess.run(["docker", "compose", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return ["docker", "compose"]
        except Exception:
            pass
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    raise RuntimeError("Docker Compose was not found. Install Docker Desktop or docker-compose first.")


def _inspect_qdrant_host_port(docker_binary: str, container_name: str) -> int | None:
    inspect = subprocess.run(
        [docker_binary, "inspect", container_name, "--format", "{{json .HostConfig.PortBindings}}"],
        text=True,
        capture_output=True,
    )
    if inspect.returncode != 0:
        return None
    try:
        bindings = json.loads(inspect.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return None
    entries = bindings.get("6333/tcp") or []
    for entry in entries:
        host_port = str(entry.get("HostPort", "")).strip()
        if host_port.isdigit():
            return int(host_port)
    return None


def _ensure_qdrant(*, root: Path, dry_run: bool) -> int:
    container_name = "quickcontext-qdrant"
    compose = _find_docker_compose()
    docker_binary = shutil.which("docker")
    if not docker_binary:
        raise RuntimeError("The docker CLI was not found. Install Docker Desktop first.")

    existing_host_port = _inspect_qdrant_host_port(docker_binary, container_name)
    if existing_host_port is not None:
        print(f"+ docker start {container_name}")
        if not dry_run:
            subprocess.run([docker_binary, "start", container_name], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return existing_host_port

    print(f"+ {_display([*compose, 'up', '-d', 'qdrant'])}")
    if dry_run:
        return 6333
    subprocess.run([*compose, "up", "-d", "qdrant"], cwd=str(root), check=True)
    return _inspect_qdrant_host_port(docker_binary, container_name) or 6333


def _load_or_seed_config(config_path: Path, example_path: Path) -> dict:
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return json.loads(example_path.read_text(encoding="utf-8"))


def _write_config(
    *,
    config_path: Path,
    example_path: Path,
    service_path: Path,
    qdrant_port: int,
    dry_run: bool,
) -> None:
    config_exists = config_path.exists()
    data = _load_or_seed_config(config_path, example_path)
    qdrant = data.setdefault("qdrant", {})
    if not config_exists and str(qdrant.get("host", "localhost")).strip().lower() in {"localhost", "127.0.0.1"}:
        qdrant["prefer_grpc"] = False
    if config_exists:
        qdrant.setdefault("port", int(qdrant_port))
    else:
        qdrant["port"] = int(qdrant_port)
    qdrant.setdefault("grpc_port", 6334)

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
    config_path.write_text(payload, encoding="utf-8")


def _wait_for_qdrant(port: int, timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://127.0.0.1:{int(port)}/collections"
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                if 200 <= response.status < 500:
                    return
        except urllib.error.URLError as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"Qdrant did not become ready at {url}: {last_error}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python scripts/bootstrap_quickcontext.py")
    parser.add_argument("--profile", choices=["local", "cloud"], default="local")
    parser.add_argument("--service-build", choices=["debug", "release", "skip"], default="debug")
    parser.add_argument("--config", dest="config_path", default="quickcontext.json")
    parser.add_argument("--venv", dest="venv_path", default=".venv")
    parser.add_argument("--skip-docker", action="store_true")
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
    qdrant_port = 6333

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

    if not args.skip_docker:
        qdrant_port = _ensure_qdrant(root=root, dry_run=args.dry_run)
        if not args.dry_run:
            _wait_for_qdrant(qdrant_port)

    _write_config(
        config_path=config_path,
        example_path=example_path,
        service_path=service_path,
        qdrant_port=qdrant_port,
        dry_run=args.dry_run,
    )

    if not args.skip_init:
        _run([str(venv_python), "-m", "engine", "--config", str(config_path), "init"], cwd=root, dry_run=args.dry_run)

    print()
    print("Bootstrap complete.")
    print(f"Config: {config_path}")
    print(f"Service binary: {service_path}")
    if args.profile == "local":
        print("Local profile is ready for no-key indexing and search.")
    else:
        print("Cloud profile is configured. Replace placeholder API keys in quickcontext.json before indexing.")
    print()
    print("Next commands:")
    if os.name == "nt":
        print(rf"  {venv_python} -m engine status")
        print(rf"  {venv_python} -m engine index . --project quickcontext --no-descriptions")
        print(rf"  {venv_python} -m quickcontext_mcp")
    else:
        print(f"  {venv_python} -m engine status")
        print(f"  {venv_python} -m engine index . --project quickcontext --no-descriptions")
        print(f"  {venv_python} -m quickcontext_mcp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
