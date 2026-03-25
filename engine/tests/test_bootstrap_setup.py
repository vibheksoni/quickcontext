import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


_BOOTSTRAP_PATH = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_quickcontext.py"
_SPEC = importlib.util.spec_from_file_location("bootstrap_quickcontext", _BOOTSTRAP_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Could not load bootstrap_quickcontext.py")
bootstrap = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bootstrap)


class BootstrapSetupTests(unittest.TestCase):
    def test_write_config_preserves_existing_runtime_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "quickcontext.json"
            example_path = root / "quickcontext.local.example.json"

            example_path.write_text(
                json.dumps(
                    {
                        "qdrant": {"host": "localhost", "port": 6333, "prefer_grpc": False},
                        "service": {"path": None},
                        "transport": {"windows_pipe_name": r"\\.\pipe\quickcontext"},
                        "mcp": {"transport": "stdio", "host": "127.0.0.1", "port": 8000, "http_path": "/mcp/", "stateless_http": False},
                    }
                ),
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "qdrant": {"host": "localhost", "port": 6333, "prefer_grpc": False},
                        "code_embedding": {"provider": "litellm", "api_key": "keep-me"},
                        "service": {"path": "C:/existing/quickcontext-service.exe"},
                    }
                ),
                encoding="utf-8",
            )

            bootstrap._write_config(
                config_path=config_path,
                example_path=example_path,
                service_path=Path("C:/new/quickcontext-service.exe"),
                qdrant_port=8803,
                dry_run=False,
            )

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["qdrant"]["port"], 6333)
            self.assertEqual(payload["service"]["path"], "C:/existing/quickcontext-service.exe")
            self.assertEqual(payload["code_embedding"]["provider"], "litellm")
            self.assertEqual(payload["code_embedding"]["api_key"], "keep-me")
            self.assertIn("transport", payload)
            self.assertIn("mcp", payload)

    def test_write_config_seeds_new_config_with_detected_qdrant_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "_ignore" / "bootstrap.local.json"
            example_path = root / "quickcontext.local.example.json"

            example_path.write_text(
                json.dumps(
                    {
                        "qdrant": {"host": "localhost", "port": 6333, "prefer_grpc": True},
                        "service": {"path": None},
                        "transport": {"windows_pipe_name": r"\\.\pipe\quickcontext", "unix_socket_path": None},
                        "mcp": {"transport": "stdio", "host": "127.0.0.1", "port": 8000, "http_path": "/mcp/", "stateless_http": False},
                    }
                ),
                encoding="utf-8",
            )

            bootstrap._write_config(
                config_path=config_path,
                example_path=example_path,
                service_path=Path("C:/svc/quickcontext-service.exe"),
                qdrant_port=8803,
                dry_run=False,
            )

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["qdrant"]["port"], 8803)
            self.assertFalse(payload["qdrant"]["prefer_grpc"])
            self.assertEqual(Path(payload["service"]["path"]), Path("C:/svc/quickcontext-service.exe"))


if __name__ == "__main__":
    unittest.main()
