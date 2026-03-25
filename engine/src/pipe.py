import json
import os
import socket
import struct
import subprocess
import time
from pathlib import Path
from typing import Optional

from engine.src.config import EngineConfig


IS_WINDOWS = os.name == "nt"
WINDOWS_PIPE_NAME = r"\\.\pipe\quickcontext"
SOCKET_PATH_ENV_VAR = "QC_SOCKET_PATH"
MAX_FRAME_SIZE = 256 * 1024 * 1024
_LAUNCHED_SERVER_PROCESSES: list[subprocess.Popen] = []
CONNECT_RETRY_SLEEP_SECONDS = 0.005
ENSURE_SERVER_PRECHECK_TIMEOUT_MS = 10
WAIT_NAMED_PIPE_POLL_MS = 10
LAUNCHED_SERVER_SHUTDOWN_WAIT_SECONDS = 1.5

if IS_WINDOWS:
    import ctypes
    import ctypes.wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    OPEN_EXISTING = 3
    INVALID_HANDLE_VALUE = ctypes.wintypes.HANDLE(-1).value
    PIPE_READMODE_BYTE = 0x00000000
    ERROR_PIPE_BUSY = 231

    kernel32.CreateFileW.argtypes = [
        ctypes.wintypes.LPCWSTR,
        ctypes.wintypes.DWORD,
        ctypes.wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.wintypes.DWORD,
        ctypes.wintypes.DWORD,
        ctypes.wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = ctypes.wintypes.HANDLE

    kernel32.SetNamedPipeHandleState.argtypes = [
        ctypes.wintypes.HANDLE,
        ctypes.POINTER(ctypes.wintypes.DWORD),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kernel32.SetNamedPipeHandleState.restype = ctypes.wintypes.BOOL

    kernel32.WaitNamedPipeW.argtypes = [ctypes.wintypes.LPCWSTR, ctypes.wintypes.DWORD]
    kernel32.WaitNamedPipeW.restype = ctypes.wintypes.BOOL

    kernel32.WriteFile.argtypes = [
        ctypes.wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.wintypes.DWORD,
        ctypes.POINTER(ctypes.wintypes.DWORD),
        ctypes.c_void_p,
    ]
    kernel32.WriteFile.restype = ctypes.wintypes.BOOL

    kernel32.ReadFile.argtypes = [
        ctypes.wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.wintypes.DWORD,
        ctypes.POINTER(ctypes.wintypes.DWORD),
        ctypes.c_void_p,
    ]
    kernel32.ReadFile.restype = ctypes.wintypes.BOOL

    kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
    kernel32.CloseHandle.restype = ctypes.wintypes.BOOL


def _default_unix_socket_path(config: EngineConfig | None = None) -> str:
    configured_path = ""
    if config is not None:
        configured_path = str(config.transport.unix_socket_path or "").strip()
    if configured_path:
        return configured_path

    env_path = os.environ.get(SOCKET_PATH_ENV_VAR, "").strip()
    if env_path:
        return env_path

    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if runtime_dir:
        return str(Path(runtime_dir) / "quickcontext.sock")

    user = (os.environ.get("USER") or os.environ.get("USERNAME") or "quickcontext").strip()
    safe_user = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in user)
    return str(Path("/tmp") / f"quickcontext-{safe_user}.sock")


def resolve_pipe_name(config: EngineConfig | None = None) -> str:
    if IS_WINDOWS:
        configured_pipe = ""
        if config is not None:
            configured_pipe = str(config.transport.windows_pipe_name or "").strip()
        return configured_pipe or WINDOWS_PIPE_NAME
    return _default_unix_socket_path(config)


PIPE_NAME = resolve_pipe_name()


class PipeError(Exception):
    """Base exception for pipe communication errors."""
    pass


class PipeConnectionError(PipeError):
    """Raised when the pipe connection fails or is lost."""
    pass


class PipeProtocolError(PipeError):
    """Raised on framing or serialization errors."""
    pass


def _reap_launched_server_processes(wait_seconds: float = LAUNCHED_SERVER_SHUTDOWN_WAIT_SECONDS) -> None:
    alive: list[subprocess.Popen] = []
    for proc in _LAUNCHED_SERVER_PROCESSES:
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=wait_seconds)
        except subprocess.TimeoutExpired:
            try:
                proc.terminate()
                proc.wait(timeout=0.5)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=0.5)
                except Exception:
                    alive.append(proc)
        except Exception:
            alive.append(proc)
    _LAUNCHED_SERVER_PROCESSES[:] = [proc for proc in alive if proc.poll() is None]


class PipeClient:
    """Client for the quickcontext Rust service.

    Protocol: 4-byte LE u32 length prefix + JSON payload.

    * pipe_name -- str, transport endpoint path
    * service_path -- Optional[str], path to quickcontext-service binary for auto-start
    """

    def __init__(self, pipe_name: Optional[str] = None, service_path: Optional[str] = None):
        self._pipe_name = pipe_name or PIPE_NAME
        self._service_path = service_path
        self._handle: Optional[int] = None
        self._socket: Optional[socket.socket] = None

    @property
    def connected(self) -> bool:
        """Returns True if the transport handle is open."""
        return self._handle is not None if IS_WINDOWS else self._socket is not None

    def connect(self, timeout_ms: int = 5000) -> None:
        """Open a connection to the service endpoint.

        * timeout_ms -- int, max wait time in milliseconds (default: 5000)

        Raises PipeConnectionError if the server is unreachable.
        """
        if self.connected:
            return

        if IS_WINDOWS:
            self._connect_windows(timeout_ms)
            return

        self._connect_unix(timeout_ms)

    def _connect_windows(self, timeout_ms: int) -> None:
        deadline = time.monotonic() + (timeout_ms / 1000.0)

        while True:
            handle = kernel32.CreateFileW(
                self._pipe_name,
                GENERIC_READ | GENERIC_WRITE,
                0,
                None,
                OPEN_EXISTING,
                0,
                None,
            )

            if handle != INVALID_HANDLE_VALUE:
                mode = ctypes.wintypes.DWORD(PIPE_READMODE_BYTE)
                kernel32.SetNamedPipeHandleState(
                    handle,
                    ctypes.byref(mode),
                    None,
                    None,
                )
                self._handle = handle
                return

            err = ctypes.get_last_error() or kernel32.GetLastError()

            if err == ERROR_PIPE_BUSY:
                if time.monotonic() >= deadline:
                    raise PipeConnectionError("pipe busy, timed out waiting")
                remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
                kernel32.WaitNamedPipeW(self._pipe_name, min(WAIT_NAMED_PIPE_POLL_MS, remaining_ms))
                continue

            if time.monotonic() >= deadline:
                raise PipeConnectionError(
                    f"failed to connect to {self._pipe_name} (win32 error {err})"
                )

            time.sleep(CONNECT_RETRY_SLEEP_SECONDS)

    def _connect_unix(self, timeout_ms: int) -> None:
        deadline = time.monotonic() + (timeout_ms / 1000.0)

        while True:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self._pipe_name)
                self._socket = sock
                return
            except OSError as exc:
                sock.close()
                if time.monotonic() >= deadline:
                    raise PipeConnectionError(
                        f"failed to connect to {self._pipe_name}: {exc}"
                    ) from exc
                time.sleep(CONNECT_RETRY_SLEEP_SECONDS)

    def close(self) -> None:
        """Close the transport handle."""
        if IS_WINDOWS:
            if self._handle is not None:
                kernel32.CloseHandle(self._handle)
                self._handle = None
            return

        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    def _write_frame(self, data: bytes) -> None:
        """Write a length-prefixed frame to the transport.

        * data -- bytes, the payload to send
        """
        frame = struct.pack("<I", len(data)) + data
        self._write_bytes(frame)

    def _write_bytes(self, data: bytes) -> None:
        if IS_WINDOWS:
            if self._handle is None:
                raise PipeConnectionError("not connected")

            written = ctypes.wintypes.DWORD(0)
            offset = 0

            while offset < len(data):
                chunk = data[offset:]
                ok = kernel32.WriteFile(
                    self._handle,
                    chunk,
                    len(chunk),
                    ctypes.byref(written),
                    None,
                )
                if not ok:
                    err = ctypes.get_last_error() or kernel32.GetLastError()
                    self.close()
                    raise PipeConnectionError(f"WriteFile failed (win32 error {err})")
                offset += written.value
            return

        if self._socket is None:
            raise PipeConnectionError("not connected")

        try:
            self._socket.sendall(data)
        except OSError as exc:
            self.close()
            raise PipeConnectionError(f"socket write failed: {exc}") from exc

    def _read_frame(self) -> bytes:
        """Read a length-prefixed frame from the transport.

        Returns the payload bytes (without the 4-byte length prefix).
        """
        len_buf = self._read_exact(4)
        length = struct.unpack("<I", len_buf)[0]

        if length > MAX_FRAME_SIZE:
            self.close()
            raise PipeProtocolError(f"frame too large: {length} bytes")

        return self._read_exact(length)

    def _read_exact(self, n: int) -> bytes:
        """Read exactly n bytes from the transport.

        * n -- int, number of bytes to read
        """
        if IS_WINDOWS:
            if self._handle is None:
                raise PipeConnectionError("not connected")

            buf = b""
            while len(buf) < n:
                remaining = n - len(buf)
                chunk = ctypes.create_string_buffer(remaining)
                read_count = ctypes.wintypes.DWORD(0)
                ok = kernel32.ReadFile(
                    self._handle,
                    chunk,
                    remaining,
                    ctypes.byref(read_count),
                    None,
                )
                if not ok:
                    err = ctypes.get_last_error() or kernel32.GetLastError()
                    self.close()
                    raise PipeConnectionError(f"ReadFile failed (win32 error {err})")
                if read_count.value == 0:
                    self.close()
                    raise PipeConnectionError("pipe closed by server")
                buf += chunk.raw[:read_count.value]
            return buf

        if self._socket is None:
            raise PipeConnectionError("not connected")

        buf = b""
        while len(buf) < n:
            remaining = n - len(buf)
            try:
                chunk = self._socket.recv(remaining)
            except OSError as exc:
                self.close()
                raise PipeConnectionError(f"socket read failed: {exc}") from exc
            if not chunk:
                self.close()
                raise PipeConnectionError("socket closed by server")
            buf += chunk
        return buf

    def request(self, payload: dict) -> dict:
        """Sends a JSON request and returns the JSON response.

        * payload -- dict, the request object (must have "method" key)

        Returns dict with "status" key ("ok" or "error").
        """
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._write_frame(data)
        response_bytes = self._read_frame()

        try:
            return json.loads(response_bytes)
        except json.JSONDecodeError as e:
            raise PipeProtocolError(f"invalid JSON response: {e}")

    def ping(self) -> str:
        """Sends a ping request. Returns "pong" on success."""
        resp = self.request({"method": "ping"})
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "unknown error"))
        return resp.get("data", "pong")

    def extract(
        self,
        path: str,
        compact: bool = False,
        stats_only: bool = False,
        respect_gitignore: bool = True,
    ) -> list[dict] | dict:
        """Extracts symbols from a file or directory.

        * path -- str, absolute path to file or directory
        * compact -- bool, strip source/docstring/params from symbols
        * stats_only -- bool, return only aggregate statistics
        * respect_gitignore -- bool, honor .gitignore rules

        Returns list of ExtractionResult dicts (normal mode),
        or dict with results+stats (compact mode),
        or dict with stats only (stats_only mode).
        """
        req = {"method": "extract", "path": path}
        if compact:
            req["compact"] = True
        if stats_only:
            req["stats_only"] = True
        if not respect_gitignore:
            req["respect_gitignore"] = False
        resp = self.request(req)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "extraction failed"))
        return resp.get("data", [])

    def extract_symbol(self, file: str, symbol: str) -> dict:
        """Extracts a specific symbol by name from a file.

        * file -- str, absolute path to the source file
        * symbol -- str, symbol name or "Parent.name" for disambiguation

        Returns dict with file_path, language, query, symbols, total_matches.
        """
        resp = self.request({
            "method": "extract_symbol",
            "file": file,
            "symbol": symbol,
        })
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "extract symbol failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid extract symbol response payload")
        return data

    def scan_files(self, path: str, respect_gitignore: bool = True) -> list[dict]:
        """Return supported source files with lightweight metadata.

        * path -- str, absolute path to file or directory
        * respect_gitignore -- bool, honor .gitignore rules

        Returns list of dictionaries with file path, language, size, and mtime.
        """
        payload: dict[str, object] = {
            "method": "scan_files",
            "path": path,
            "respect_gitignore": respect_gitignore,
        }
        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "scan files failed"))
        data = resp.get("data", [])
        if not isinstance(data, list):
            raise PipeProtocolError("invalid scan files response payload")
        return data

    def grep(self, query: str, path: Optional[str] = None, respect_gitignore: bool = True, limit: int = 200, before_context: int = 0, after_context: int = 0) -> dict:
        """Searches for literal text across files via Rust service.

        * query -- str, text to find
        * path -- Optional[str], file or directory path
        * respect_gitignore -- bool, apply gitignore rules when True
        * limit -- int, max number of matches to return
        * before_context -- int, number of lines before match to include
        * after_context -- int, number of lines after match to include

        Returns dict with matches and timing metadata.
        """
        payload: dict[str, object] = {
            "method": "grep",
            "query": query,
            "respect_gitignore": respect_gitignore,
            "limit": limit,
            "before_context": before_context,
            "after_context": after_context,
        }
        if path is not None:
            payload["path"] = path

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "grep failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid grep response payload")
        return data

    def symbol_lookup(
        self,
        query: str,
        path: Optional[str] = None,
        respect_gitignore: bool = True,
        limit: int = 50,
        intent_mode: bool = False,
        intent_level: int = 2,
    ) -> dict:
        """Looks up symbols by keyword using Rust in-memory index.

        * query -- str, symbol name/keyword query
        * path -- Optional[str], project root or search scope path
        * respect_gitignore -- bool, apply gitignore rules when True
        * limit -- int, max symbols to return
        * intent_mode -- bool, enable non-embedding intent expansion
        * intent_level -- int, intent expansion aggressiveness from 1..=3

        Returns dict with lookup results and index metadata.
        """
        payload: dict[str, object] = {
            "method": "symbol_lookup",
            "query": query,
            "respect_gitignore": respect_gitignore,
            "limit": limit,
            "intent_mode": bool(intent_mode),
            "intent_level": max(1, min(3, int(intent_level))),
        }
        if path is not None:
            payload["path"] = path

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "symbol lookup failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid symbol lookup response payload")
        return data

    def file_symbols(
        self,
        file: str,
        path: Optional[str] = None,
        respect_gitignore: bool = True,
        limit: int = 256,
    ) -> dict:
        """Lists indexed symbols for a specific file from the Rust symbol index.

        * file -- str, absolute file path
        * path -- Optional[str], project root or search scope path
        * respect_gitignore -- bool, apply gitignore rules when True
        * limit -- int, max symbols to return

        Returns dict with file symbol rows and index metadata.
        """
        payload: dict[str, object] = {
            "method": "file_symbols",
            "file": file,
            "respect_gitignore": respect_gitignore,
            "limit": limit,
        }
        if path is not None:
            payload["path"] = path

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "file symbols failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid file symbols response payload")
        return data

    def find_callers(self, symbol: str, path: Optional[str] = None, respect_gitignore: bool = True, limit: int = 100) -> dict:
        """Finds caller sites for a symbol using Rust call index.

        * symbol -- str, target symbol name
        * path -- Optional[str], project root or search scope path
        * respect_gitignore -- bool, apply gitignore rules when True
        * limit -- int, max caller rows to return

        Returns dict with caller matches and index metadata.
        """
        payload: dict[str, object] = {
            "method": "find_callers",
            "symbol": symbol,
            "respect_gitignore": respect_gitignore,
            "limit": limit,
        }
        if path is not None:
            payload["path"] = path

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "find callers failed"))
        data = resp.get("data", {})
        return data

    def trace_call_graph(
        self,
        symbol: str,
        path: Optional[str] = None,
        respect_gitignore: bool = True,
        direction: str = "both",
        max_depth: int = 5,
    ) -> dict:
        """Traces multi-hop call graph from a symbol via BFS.

        * symbol -- str, root symbol name to trace from
        * path -- Optional[str], project root or search scope path
        * respect_gitignore -- bool, apply gitignore rules when True
        * direction -- str, traversal direction: "upstream", "downstream", or "both"
        * max_depth -- int, max BFS hops (default: 5)

        Returns dict with nodes, edges, cycles, and index metadata.
        """
        payload: dict[str, object] = {
            "method": "trace_call_graph",
            "symbol": symbol,
            "respect_gitignore": respect_gitignore,
            "direction": direction,
            "max_depth": max_depth,
        }
        if path is not None:
            payload["path"] = path

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "trace call graph failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid trace call graph response payload")
        return data

    def skeleton(
        self,
        path: str,
        max_depth: int | None = None,
        include_signatures: bool = True,
        include_line_numbers: bool = False,
        collapse_threshold: int = 0,
        respect_gitignore: bool = True,
        format: str = "json",
    ) -> dict:
        """Generates a repo skeleton for a file or directory.

        * path -- str, absolute path to file or directory
        * max_depth -- int | None, max directory recursion depth
        * include_signatures -- bool, include function/class signatures
        * include_line_numbers -- bool, include line number ranges
        * collapse_threshold -- int, collapse dirs with fewer files than this
        * respect_gitignore -- bool, apply gitignore rules when True
        * format -- str, output format: "json" or "markdown"

        Returns dict with skeleton tree and metadata.
        """
        payload: dict[str, object] = {
            "method": "skeleton",
            "path": path,
            "include_signatures": include_signatures,
            "include_line_numbers": include_line_numbers,
            "collapse_threshold": collapse_threshold,
            "respect_gitignore": respect_gitignore,
            "format": format,
        }
        if max_depth is not None:
            payload["max_depth"] = max_depth

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "skeleton failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid skeleton response payload")
        return data

    def import_graph(self, file: str, path: Optional[str] = None, respect_gitignore: bool = True) -> dict:
        """Gets import dependencies for a file (outgoing edges).

        * file -- str, absolute path to the source file
        * path -- Optional[str], project root directory
        * respect_gitignore -- bool, apply gitignore rules when True

        Returns dict with import edges and graph metadata.
        """
        payload: dict[str, object] = {
            "method": "import_graph",
            "file": file,
            "respect_gitignore": respect_gitignore,
        }
        if path is not None:
            payload["path"] = path

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "import graph failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid import graph response payload")
        return data

    def import_neighbors(self, file: str, path: Optional[str] = None, respect_gitignore: bool = True) -> dict:
        """Gets both outgoing and incoming import edges for a file."""
        payload: dict[str, object] = {
            "method": "import_neighbors",
            "file": file,
            "respect_gitignore": respect_gitignore,
        }
        if path is not None:
            payload["path"] = path

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "import neighbors failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid import neighbors response payload")
        return data

    def find_importers(self, file: str, path: Optional[str] = None, respect_gitignore: bool = True) -> dict:
        """Finds files that import a given file (incoming edges).

        * file -- str, absolute path to the target file
        * path -- Optional[str], project root directory
        * respect_gitignore -- bool, apply gitignore rules when True

        Returns dict with importer edges and graph metadata.
        """
        payload: dict[str, object] = {
            "method": "find_importers",
            "file": file,
            "respect_gitignore": respect_gitignore,
        }
        if path is not None:
            payload["path"] = path

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "find importers failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid find importers response payload")
        return data

    def text_search(
        self,
        query: str,
        path: Optional[str] = None,
        respect_gitignore: bool = True,
        limit: int = 20,
        intent_mode: bool = False,
        intent_level: int = 2,
    ) -> dict:
        """Performs BM25 full-text search via Rust service.

        * query -- str, search query with optional operators (AND/OR/+/-/"exact")
        * path -- Optional[str], directory path to search in
        * respect_gitignore -- bool, apply gitignore rules when True
        * limit -- int, max number of results to return
        * intent_mode -- bool, enable non-embedding intent expansion
        * intent_level -- int, intent expansion aggressiveness from 1..=3

        Returns dict with ranked matches and search metadata.
        """
        payload: dict[str, object] = {
            "method": "text_search",
            "query": query,
            "respect_gitignore": respect_gitignore,
            "limit": limit,
            "intent_mode": bool(intent_mode),
            "intent_level": max(1, min(3, int(intent_level))),
        }
        if path is not None:
            payload["path"] = path

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "text search failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid text search response payload")
        return data

    def warm_project(
        self,
        path: str,
        respect_gitignore: bool = True,
    ) -> dict:
        """Warms persisted Rust indices for a project path."""
        payload: dict[str, object] = {
            "method": "warm_project",
            "path": path,
            "respect_gitignore": respect_gitignore,
        }

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "warm project failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid warm project response payload")
        return data

    def protocol_search(
        self,
        query: str,
        path: Optional[str] = None,
        respect_gitignore: bool = True,
        limit: int = 20,
        context_radius: Optional[int] = None,
        min_score: Optional[float] = None,
        include_markers: Optional[list[str]] = None,
        exclude_markers: Optional[list[str]] = None,
        max_input_fields: Optional[int] = None,
        max_output_fields: Optional[int] = None,
    ) -> dict:
        """Extracts protocol request/response contracts via Rust service.

        * query -- str, intent query for protocol extraction
        * path -- Optional[str], directory or file path to search in
        * respect_gitignore -- bool, apply gitignore rules when True
        * limit -- int, max number of contracts to return
        * context_radius -- Optional[int], context line radius per signal
        * min_score -- Optional[float], minimum score threshold
        * include_markers -- Optional[list[str]], extra markers to treat as protocol signals
        * exclude_markers -- Optional[list[str]], markers that skip candidate files
        * max_input_fields -- Optional[int], max inferred input fields per contract
        * max_output_fields -- Optional[int], max inferred output fields per contract

        Returns dict with protocol contracts and search metadata.
        """
        payload: dict[str, object] = {
            "method": "protocol_search",
            "query": query,
            "respect_gitignore": respect_gitignore,
            "limit": limit,
        }
        if path is not None:
            payload["path"] = path
        if context_radius is not None:
            payload["context_radius"] = int(context_radius)
        if min_score is not None:
            payload["min_score"] = float(min_score)
        if include_markers is not None:
            payload["include_markers"] = include_markers
        if exclude_markers is not None:
            payload["exclude_markers"] = exclude_markers
        if max_input_fields is not None:
            payload["max_input_fields"] = int(max_input_fields)
        if max_output_fields is not None:
            payload["max_output_fields"] = int(max_output_fields)

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "protocol search failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid protocol search response payload")
        return data

    def pattern_search(
        self,
        pattern: str,
        language: str,
        path: Optional[str] = None,
        respect_gitignore: bool = True,
        limit: int = 50,
    ) -> dict:
        """Searches for AST pattern matches via Rust service.

        * pattern -- str, code pattern with metavariables ($NAME, $$$, $_)
        * language -- str, target language name (python, rust, javascript, etc.)
        * path -- Optional[str], directory or file path to search in
        * respect_gitignore -- bool, apply gitignore rules when True
        * limit -- int, max number of matches to return

        Returns dict with matches and search metadata.
        """
        payload: dict[str, object] = {
            "method": "pattern_search",
            "pattern": pattern,
            "language": language,
            "respect_gitignore": respect_gitignore,
            "limit": limit,
        }
        if path is not None:
            payload["path"] = path

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "pattern search failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid pattern search response payload")
        return data

    def pattern_rewrite(
        self,
        pattern: str,
        replacement: str,
        language: str,
        path: Optional[str] = None,
        respect_gitignore: bool = True,
        limit: int = 50,
        dry_run: bool = True,
    ) -> dict:
        """Rewrites code matching an AST pattern via Rust service.

        * pattern -- str, code pattern with metavariables ($NAME, $$$, $_)
        * replacement -- str, replacement template with metavariable substitution
        * language -- str, target language name (python, rust, javascript, etc.)
        * path -- Optional[str], directory or file path to search in
        * respect_gitignore -- bool, apply gitignore rules when True
        * limit -- int, max number of files to rewrite
        * dry_run -- bool, compute edits without writing files when True

        Returns dict with rewrite results and edit metadata.
        """
        payload: dict[str, object] = {
            "method": "pattern_rewrite",
            "pattern": pattern,
            "replacement": replacement,
            "language": language,
            "respect_gitignore": respect_gitignore,
            "limit": limit,
            "dry_run": dry_run,
        }
        if path is not None:
            payload["path"] = path

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "pattern rewrite failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid pattern rewrite response payload")
        return data

    def file_read(
        self,
        file: str,
        start_line: int | None = None,
        end_line: int | None = None,
        max_bytes: int | None = None,
    ) -> dict:
        """Reads a file with optional line slicing.

        * file -- str, absolute path to the source file
        * start_line -- int | None, 1-based start line
        * end_line -- int | None, 1-based end line (None means read to end)
        * max_bytes -- int | None, optional byte cap for read

        Returns dict with line-numbered content and metadata.
        """
        payload: dict[str, object] = {
            "method": "file_read",
            "file": file,
        }
        if start_line is not None:
            payload["start_line"] = int(start_line)
        if end_line is not None:
            payload["end_line"] = int(end_line)
        if max_bytes is not None:
            payload["max_bytes"] = int(max_bytes)

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "file read failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid file read response payload")
        return data

    def file_edit(
        self,
        file: str,
        mode: str,
        edits: list[dict] | None = None,
        text: str | None = None,
        dry_run: bool = False,
        expected_hash: str | None = None,
        record_undo: bool = True,
    ) -> dict:
        """Applies line-based file edits with optional undo record.

        * file -- str, absolute path to the source file
        * mode -- str, edit mode: append|insert|replace|delete|batch
        * edits -- list[dict] | None, line-range edit objects
        * text -- str | None, fallback text for modes that need it
        * dry_run -- bool, compute result without writing file
        * expected_hash -- str | None, optimistic concurrency hash guard
        * record_undo -- bool, persist undo record for revert

        Returns dict with hash metadata and optional edit_id.
        """
        payload: dict[str, object] = {
            "method": "file_edit",
            "file": file,
            "mode": mode,
            "dry_run": bool(dry_run),
            "record_undo": bool(record_undo),
        }
        if edits is not None:
            payload["edits"] = edits
        if text is not None:
            payload["text"] = text
        if expected_hash is not None:
            payload["expected_hash"] = expected_hash

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "file edit failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid file edit response payload")
        return data

    def file_edit_revert(
        self,
        edit_id: str,
        dry_run: bool = False,
        expected_hash: str | None = None,
    ) -> dict:
        """Reverts a prior file_edit call by edit_id.

        * edit_id -- str, undo record ID returned from file_edit
        * dry_run -- bool, simulate revert without writing file
        * expected_hash -- str | None, optional pre-revert hash guard

        Returns dict with revert status and hash metadata.
        """
        payload: dict[str, object] = {
            "method": "file_edit_revert",
            "edit_id": edit_id,
            "dry_run": bool(dry_run),
        }
        if expected_hash is not None:
            payload["expected_hash"] = expected_hash

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "file edit revert failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid file edit revert response payload")
        return data

    def symbol_edit(
        self,
        file: str,
        symbol_name: str,
        new_source: str,
        dry_run: bool = False,
        expected_hash: str | None = None,
        record_undo: bool = True,
    ) -> dict:
        """Edits a symbol by name using AST extraction.

        * file -- str, absolute path to file
        * symbol_name -- str, name of symbol to replace
        * new_source -- str, new source code for the symbol
        * dry_run -- bool, simulate edit without writing
        * expected_hash -- str | None, optional pre-edit hash guard
        * record_undo -- bool, save undo record

        Returns dict with edit status, line range, and hash metadata.
        """
        payload: dict[str, object] = {
            "method": "symbol_edit",
            "file": file,
            "symbol_name": symbol_name,
            "new_source": new_source,
            "dry_run": bool(dry_run),
            "record_undo": bool(record_undo),
        }
        if expected_hash is not None:
            payload["expected_hash"] = expected_hash

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "symbol edit failed"))
        data = resp.get("data", {})
        if not isinstance(data, dict):
            raise PipeProtocolError("invalid symbol edit response payload")
        return data

    def lsp_definition(self, file: str, line: int, character: int) -> dict:
        """Go to definition of symbol at position via LSP.

        * file -- str, absolute path to the source file
        * line -- int, zero-based line number
        * character -- int, zero-based character offset

        Returns dict with definition location(s).
        """
        resp = self.request({
            "method": "lsp_definition",
            "file": file,
            "line": line,
            "character": character,
        })
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "lsp definition failed"))
        return resp.get("data", {})

    def lsp_references(
        self,
        file: str,
        line: int,
        character: int,
        include_declaration: bool = True,
    ) -> dict:
        """Find all references to symbol at position via LSP.

        * file -- str, absolute path to the source file
        * line -- int, zero-based line number
        * character -- int, zero-based character offset
        * include_declaration -- bool, include the declaration itself

        Returns dict with reference locations.
        """
        resp = self.request({
            "method": "lsp_references",
            "file": file,
            "line": line,
            "character": character,
            "include_declaration": include_declaration,
        })
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "lsp references failed"))
        return resp.get("data", {})

    def lsp_hover(self, file: str, line: int, character: int) -> dict:
        """Get hover information for symbol at position via LSP.

        * file -- str, absolute path to the source file
        * line -- int, zero-based line number
        * character -- int, zero-based character offset

        Returns dict with hover contents (type info, docs).
        """
        resp = self.request({
            "method": "lsp_hover",
            "file": file,
            "line": line,
            "character": character,
        })
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "lsp hover failed"))
        return resp.get("data", {})

    def lsp_symbols(self, file: str) -> dict:
        """Get document symbols (outline) for a file via LSP.

        * file -- str, absolute path to the source file

        Returns dict with hierarchical symbol tree.
        """
        resp = self.request({
            "method": "lsp_symbols",
            "file": file,
        })
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "lsp symbols failed"))
        return resp.get("data", {})

    def lsp_format(
        self,
        file: str,
        tab_size: int = 4,
        insert_spaces: bool = True,
    ) -> dict:
        """Format a document via LSP.

        * file -- str, absolute path to the source file
        * tab_size -- int, spaces per tab (default: 4)
        * insert_spaces -- bool, use spaces instead of tabs

        Returns dict with text edits to apply.
        """
        resp = self.request({
            "method": "lsp_format",
            "file": file,
            "tab_size": tab_size,
            "insert_spaces": insert_spaces,
        })
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "lsp format failed"))
        return resp.get("data", {})

    def lsp_diagnostics(self, file: str) -> dict:
        """Request diagnostics for a file via LSP.

        * file -- str, absolute path to the source file

        Returns dict with diagnostics data.
        """
        resp = self.request({
            "method": "lsp_diagnostics",
            "file": file,
        })
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "lsp diagnostics failed"))
        return resp.get("data", {})

    def lsp_completion(self, file: str, line: int, character: int) -> dict:
        """Request completions at position via LSP.

        * file -- str, absolute path to the source file
        * line -- int, zero-based line number
        * character -- int, zero-based character offset

        Returns dict with completion items.
        """
        resp = self.request({
            "method": "lsp_completion",
            "file": file,
            "line": line,
            "character": character,
        })
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "lsp completion failed"))
        return resp.get("data", {})

    def lsp_rename(self, file: str, line: int, character: int, new_name: str) -> dict:
        """Rename symbol at position via LSP.

        * file -- str, absolute path to the source file
        * line -- int, zero-based line number
        * character -- int, zero-based character offset
        * new_name -- str, new name for the symbol

        Returns dict with workspace edit (file changes).
        """
        resp = self.request({
            "method": "lsp_rename",
            "file": file,
            "line": line,
            "character": character,
            "new_name": new_name,
        })
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "lsp rename failed"))
        return resp.get("data", {})

    def lsp_prepare_rename(self, file: str, line: int, character: int) -> dict:
        """Validate rename and get symbol range at position via LSP.

        * file -- str, absolute path to the source file
        * line -- int, zero-based line number
        * character -- int, zero-based character offset

        Returns dict with rename range and placeholder text.
        """
        resp = self.request({
            "method": "lsp_prepare_rename",
            "file": file,
            "line": line,
            "character": character,
        })
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "lsp prepare rename failed"))
        return resp.get("data", {})

    def lsp_code_actions(
        self,
        file: str,
        start_line: int,
        start_character: int,
        end_line: int,
        end_character: int,
        diagnostics: list[dict] | None = None,
    ) -> dict:
        """Get available code actions for a range via LSP.

        * file -- str, absolute path to the source file
        * start_line -- int, zero-based start line
        * start_character -- int, zero-based start character
        * end_line -- int, zero-based end line
        * end_character -- int, zero-based end character
        * diagnostics -- list[dict] | None, diagnostics to include in context

        Returns dict with available code actions.
        """
        payload: dict[str, object] = {
            "method": "lsp_code_actions",
            "file": file,
            "start_line": start_line,
            "start_character": start_character,
            "end_line": end_line,
            "end_character": end_character,
        }
        if diagnostics is not None:
            payload["diagnostics"] = diagnostics

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "lsp code actions failed"))
        return resp.get("data", {})

    def lsp_signature_help(self, file: str, line: int, character: int) -> dict:
        """Get signature help at position via LSP.

        * file -- str, absolute path to the source file
        * line -- int, zero-based line number
        * character -- int, zero-based character offset

        Returns dict with active signature and parameter info.
        """
        resp = self.request({
            "method": "lsp_signature_help",
            "file": file,
            "line": line,
            "character": character,
        })
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "lsp signature help failed"))
        return resp.get("data", {})

    def lsp_workspace_symbols(self, query: str, file: str | None = None) -> dict:
        """Search for symbols across the workspace via LSP.

        * query -- str, symbol search query
        * file -- str | None, file path to identify which language server to use

        Returns dict with matching workspace symbols.
        """
        payload: dict[str, object] = {
            "method": "lsp_workspace_symbols",
            "query": query,
        }
        if file is not None:
            payload["file"] = file

        resp = self.request(payload)
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "lsp workspace symbols failed"))
        return resp.get("data", {})

    def lsp_sessions(self) -> dict:
        """List active LSP server sessions tracked by the service."""
        resp = self.request({"method": "lsp_sessions"})
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "lsp sessions failed"))
        return resp.get("data", {})

    def lsp_shutdown_all(self) -> dict:
        """Shut down all tracked LSP server sessions."""
        resp = self.request({"method": "lsp_shutdown_all"})
        if resp.get("status") == "error":
            raise PipeError(resp.get("message", "lsp shutdown all failed"))
        return resp.get("data", {})

    def shutdown(self) -> None:
        """Send a shutdown request to the server, then close the transport."""
        try:
            self.connect(timeout_ms=250)
        except PipeConnectionError:
            pass

        try:
            try:
                self.lsp_shutdown_all()
            except PipeError:
                pass
            self.request({"method": "shutdown"})
        except PipeConnectionError:
            pass
        finally:
            self.close()
            _reap_launched_server_processes()

    def ensure_server(self, timeout_ms: int = 10000) -> None:
        """Connects to the server, starting it if necessary.

        * timeout_ms -- int, max wait for server startup (default: 10000)

        If the server is not running and service_path was provided,
        launches it as a background process and waits for connection.
        """
        try:
            self.connect(timeout_ms=ENSURE_SERVER_PRECHECK_TIMEOUT_MS)
            return
        except PipeConnectionError:
            pass

        if not self._service_path:
            raise PipeConnectionError(
                "server not running and no service_path configured"
            )

        svc = Path(self._service_path)
        if not svc.exists():
            raise PipeConnectionError(f"service binary not found: {svc}")

        popen_kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }

        if IS_WINDOWS:
            popen_kwargs["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen([str(svc), "serve"], **popen_kwargs)
        _LAUNCHED_SERVER_PROCESSES[:] = [item for item in _LAUNCHED_SERVER_PROCESSES if item.poll() is None]
        _LAUNCHED_SERVER_PROCESSES.append(proc)

        self.connect(timeout_ms=timeout_ms)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()
