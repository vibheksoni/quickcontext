from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import hashlib
import math
import re

from engine.src.artifact_index import ARTIFACT_FALLBACK_ERROR
from engine.src.parsing import ExtractedSymbol, ExtractionResult


@dataclass(frozen=True, slots=True)
class CodeChunk:
    """
    Represents a chunk of code ready for embedding.

    Args:
        chunk_id: Deterministic hash-based ID for Qdrant point
        source: Raw source code text
        language: Programming language
        file_path: Absolute path to source file
        symbol_name: Name of the symbol (function/class/etc)
        symbol_kind: Type of symbol (function, class, method, etc)
        line_start: Starting line number
        line_end: Ending line number
        byte_start: Starting byte offset
        byte_end: Ending byte offset
        signature: Function/method signature if available
        docstring: Documentation string if available
        parent: Parent symbol name (class for methods, module for functions)
        visibility: Access modifier (public, private, etc)
        role: Architectural role classification if available
        file_hash: Hash of entire file content for change detection
    """
    chunk_id: str
    source: str
    language: str
    file_path: str
    symbol_name: str
    symbol_kind: str
    line_start: int
    line_end: int
    byte_start: int
    byte_end: int
    signature: Optional[str]
    docstring: Optional[str]
    parent: Optional[str]
    visibility: Optional[str]
    role: Optional[str]
    file_hash: str


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """
    Metadata for tracking chunk changes and indexing state.

    Args:
        chunk_id: Unique chunk identifier
        file_path: Source file path
        file_hash: Hash of file content
        indexed_at: Timestamp of last indexing
        token_count: Approximate token count for cost tracking
    """
    chunk_id: str
    file_path: str
    file_hash: str
    indexed_at: float
    token_count: int


class ChunkBuilder:
    """
    Converts extracted symbols into embedding-ready chunks.

    Strategy:
    - Per-symbol chunking (one symbol = one chunk)
    - Deterministic chunk IDs based on file path + symbol name + byte range
    - File-level fallback for files with no symbols
    - Large symbol truncation with configurable max size
    """

    def __init__(
        self,
        max_chunk_bytes: int = 32000,
        fallback_chunk_size: int = 8000,
        artifact_chunk_size: int = 24000,
        artifact_max_chunks: int = 8,
    ):
        """
        Args:
            max_chunk_bytes: Maximum bytes per chunk (default 32K for Qwen3-8B context)
            fallback_chunk_size: Chunk size for files with no symbols
            artifact_chunk_size: Approximate excerpt size for artifact fallback chunks
            artifact_max_chunks: Maximum coarse chunks emitted for one bundle artifact
        """
        self._max_chunk_bytes = max_chunk_bytes
        self._fallback_chunk_size = fallback_chunk_size
        self._artifact_chunk_size = max(1024, min(artifact_chunk_size, max_chunk_bytes))
        self._artifact_max_chunks = max(1, int(artifact_max_chunks))

    def build_chunks(self, results: list[ExtractionResult]) -> list[CodeChunk]:
        """
        Convert extraction results into chunks.

        results: list[ExtractionResult] — Extraction results from parser service.
        Returns: list[CodeChunk] — Code chunks ready for embedding.
        """
        chunks: list[CodeChunk] = []

        for result in results:
            file_path = result.file_path
            language = result.language

            if not result.symbols:
                if ARTIFACT_FALLBACK_ERROR in (result.errors or []):
                    chunks.extend(self._artifact_file_chunks(file_path, language, result.file_hash))
                    continue
                chunks.extend(self._fallback_chunks(file_path, language))
                continue

            if result.file_hash:
                file_hash = result.file_hash
            else:
                file_content = self._read_file(file_path)
                file_hash = self._hash_content(file_content)

            for symbol in result.symbols:
                chunk = self._symbol_to_chunk(symbol, file_hash)
                if chunk:
                    chunks.append(chunk)

        return chunks

    def _artifact_file_chunks(
        self,
        file_path: str,
        language: str,
        file_hash: Optional[str] = None,
    ) -> list[CodeChunk]:
        """
        Create a small number of coarse sampled chunks for generated/minified bundle artifacts.
        """
        try:
            content = self._read_file(file_path)
        except Exception:
            return []

        resolved_file_hash = file_hash or self._hash_content(content)
        content_bytes = content.encode("utf-8")
        total_bytes = len(content_bytes)
        if total_bytes == 0:
            return []

        desired_chunks = min(
            self._artifact_max_chunks,
            max(1, math.ceil(total_bytes / (512 * 1024))),
        )
        window_size = min(self._artifact_chunk_size, self._max_chunk_bytes)
        offsets = self._artifact_sample_offsets(total_bytes, window_size, desired_chunks)
        chunks: list[CodeChunk] = []

        for chunk_idx, offset in enumerate(offsets):
            target_end = min(total_bytes, offset + window_size)
            chunk_end = self._find_smart_chunk_end(content_bytes, offset, target_end)
            if chunk_end <= offset:
                chunk_end = target_end

            snippet_bytes = content_bytes[offset:chunk_end]
            snippet_text = snippet_bytes.decode("utf-8", errors="ignore")
            normalized_source = self._normalize_artifact_excerpt(
                file_path=file_path,
                chunk_index=chunk_idx,
                chunk_count=len(offsets),
                byte_start=offset,
                byte_end=chunk_end,
                source=snippet_text,
            )
            if not normalized_source.strip():
                continue

            line_start = content[:offset].count("\n") + 1
            line_end = content[:chunk_end].count("\n") + 1
            chunk_id = self._generate_chunk_id(
                file_path,
                f"<artifact_chunk_{chunk_idx}>",
                "file_artifact",
                None,
                None,
                offset,
                chunk_end,
            )
            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    source=self._truncate_source(normalized_source),
                    language=language,
                    file_path=file_path,
                    symbol_name=f"<artifact_chunk_{chunk_idx}>",
                    symbol_kind="file_artifact",
                    line_start=line_start,
                    line_end=line_end,
                    byte_start=offset,
                    byte_end=chunk_end,
                    signature=None,
                    docstring=None,
                    parent=None,
                    visibility=None,
                    role="generated",
                    file_hash=resolved_file_hash,
                )
            )

        return chunks

    def _artifact_sample_offsets(self, total_bytes: int, window_size: int, chunk_count: int) -> list[int]:
        """
        Pick evenly spaced excerpt offsets across a large artifact file.
        """
        if chunk_count <= 1 or total_bytes <= window_size:
            return [0]

        max_start = max(0, total_bytes - window_size)
        offsets = {
            int(round((max_start * idx) / max(1, chunk_count - 1)))
            for idx in range(chunk_count)
        }
        return sorted(offsets)

    def _normalize_artifact_excerpt(
        self,
        file_path: str,
        chunk_index: int,
        chunk_count: int,
        byte_start: int,
        byte_end: int,
        source: str,
    ) -> str:
        """
        Reformat a sampled artifact excerpt so it stays searchable without tripping minified-file heuristics.
        """
        normalized = source.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"([;{}])", r"\1\n", normalized)
        lines: list[str] = []
        for raw_line in normalized.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for offset in range(0, len(line), 240):
                lines.append(line[offset:offset + 240])
                if len(lines) >= 180:
                    break
            if len(lines) >= 180:
                break

        header = (
            f"Generated bundle artifact excerpt from {Path(file_path).name}\n"
            f"Chunk {chunk_index + 1} of {chunk_count}\n"
            f"Approx byte range {byte_start}-{byte_end}\n"
        )
        body = "\n".join(lines)
        return f"{header}\n{body}".strip()

    def _symbol_to_chunk(self, symbol: ExtractedSymbol, file_hash: str) -> Optional[CodeChunk]:
        """
        Convert a single symbol to a chunk.

        Args:
            symbol: Extracted symbol
            file_hash: Hash of source file

        Returns:
            CodeChunk or None if symbol should be skipped
        """
        source = symbol.source

        if len(source.encode('utf-8')) > self._max_chunk_bytes:
            source = self._truncate_source(source)

        chunk_id = self._generate_chunk_id(
            symbol.file_path,
            symbol.name,
            symbol.kind,
            symbol.parent,
            symbol.signature,
            symbol.byte_start,
            symbol.byte_end,
        )

        return CodeChunk(
            chunk_id=chunk_id,
            source=source,
            language=symbol.language,
            file_path=symbol.file_path,
            symbol_name=symbol.name,
            symbol_kind=symbol.kind,
            line_start=symbol.line_start,
            line_end=symbol.line_end,
            byte_start=symbol.byte_start,
            byte_end=symbol.byte_end,
            signature=symbol.signature,
            docstring=symbol.docstring,
            parent=symbol.parent,
            visibility=symbol.visibility,
            role=symbol.role,
            file_hash=file_hash,
        )

    def _fallback_chunks(self, file_path: str, language: str) -> list[CodeChunk]:
        """
        Create chunks for files with no extracted symbols.

        Args:
            file_path: Path to source file
            language: Programming language

        Returns:
            List of file-based chunks
        """
        try:
            content = self._read_file(file_path)
        except Exception:
            return []

        file_hash = self._hash_content(content)
        chunks: list[CodeChunk] = []

        content_bytes = content.encode('utf-8')
        total_bytes = len(content_bytes)

        if total_bytes <= self._max_chunk_bytes:
            chunk_id = self._generate_chunk_id(file_path, "<file>", "file", None, None, 0, total_bytes)
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                source=content,
                language=language,
                file_path=file_path,
                symbol_name="<file>",
                symbol_kind="file",
                line_start=1,
                line_end=content.count('\n') + 1,
                byte_start=0,
                byte_end=total_bytes,
                signature=None,
                docstring=None,
                parent=None,
                visibility=None,
                role=None,
                file_hash=file_hash,
            ))
        else:
            newline_offsets = [idx for idx, ch in enumerate(content) if ch == '\n']

            offset = 0
            chunk_idx = 0
            while offset < total_bytes:
                target_end = min(total_bytes, offset + self._fallback_chunk_size)
                chunk_end = self._find_smart_chunk_end(content_bytes, offset, target_end)
                chunk_bytes = content_bytes[offset:chunk_end]
                chunk_text = chunk_bytes.decode('utf-8', errors='ignore')

                chunk_id = self._generate_chunk_id(
                    file_path,
                    f"<file_chunk_{chunk_idx}>",
                    "file",
                    None,
                    None,
                    offset,
                    chunk_end,
                )
                line_start = bisect_right(newline_offsets, offset) + 1
                line_end = bisect_right(newline_offsets, chunk_end) + 1
                chunks.append(CodeChunk(
                    chunk_id=chunk_id,
                    source=chunk_text,
                    language=language,
                    file_path=file_path,
                    symbol_name=f"<file_chunk_{chunk_idx}>",
                    symbol_kind="file",
                    line_start=line_start,
                    line_end=line_end,
                    byte_start=offset,
                    byte_end=chunk_end,
                    signature=None,
                    docstring=None,
                    parent=None,
                    visibility=None,
                    role=None,
                    file_hash=file_hash,
                ))

                offset = chunk_end
                chunk_idx += 1

        return chunks

    def _find_smart_chunk_end(self, content_bytes: bytes, start: int, target_end: int) -> int:
        """
        Find a chunk end near target boundary while avoiding code-fence splits.

        content_bytes: bytes — Full file bytes.
        start: int — Chunk start offset.
        target_end: int — Initial target end offset.
        Returns: int — Chosen chunk end offset.
        """
        if target_end >= len(content_bytes):
            return len(content_bytes)

        window_start = max(start + 256, target_end - 1200)
        window = content_bytes[window_start:target_end]
        window_text = window.decode("utf-8", errors="ignore")

        candidate = self._pick_boundary_in_window(window_text)
        if candidate is None:
            return target_end

        proposed = window_start + candidate
        if proposed <= start:
            return target_end

        if self._inside_code_fence(content_bytes[start:proposed]):
            return target_end

        return proposed

    def _pick_boundary_in_window(self, window_text: str) -> Optional[int]:
        """
        Pick best boundary offset inside a window.

        window_text: str — UTF-8 decoded window text.
        Returns: Optional[int] — Byte-ish offset within decoded window text.
        """
        best_idx: Optional[int] = None
        best_score = -1

        patterns = [
            (re.compile(r"\n#{1,6}\s"), 100),
            (re.compile(r"\n```"), 90),
            (re.compile(r"\n\n+"), 60),
            (re.compile(r"\n[-*]\s"), 25),
            (re.compile(r"\n\d+\.\s"), 25),
            (re.compile(r"\n"), 1),
        ]

        text_len = max(1, len(window_text))
        for pattern, score in patterns:
            for match in pattern.finditer(window_text):
                idx = match.start()
                distance = text_len - idx
                adjusted = score - (distance / text_len) * 15
                if adjusted > best_score:
                    best_score = adjusted
                    best_idx = idx

        return best_idx

    def _inside_code_fence(self, text_bytes: bytes) -> bool:
        """
        Check if chunk slice ends while still inside a fenced code block.

        text_bytes: bytes — Chunk slice bytes.
        Returns: bool — True when odd fence count indicates open block.
        """
        decoded = text_bytes.decode("utf-8", errors="ignore")
        return decoded.count("\n```") % 2 == 1

    def _truncate_source(self, source: str) -> str:
        """
        Truncate source code to max chunk size.

        Args:
            source: Source code text

        Returns:
            Truncated source with marker
        """
        source_bytes = source.encode('utf-8')
        if len(source_bytes) <= self._max_chunk_bytes:
            return source

        truncated_bytes = source_bytes[:self._max_chunk_bytes - 100]
        truncated = truncated_bytes.decode('utf-8', errors='ignore')
        return truncated + "\n\n... [truncated]"

    def _generate_chunk_id(
        self,
        file_path: str,
        symbol_name: str,
        symbol_kind: str,
        parent: Optional[str] = None,
        signature: Optional[str] = None,
        byte_start: Optional[int] = None,
        byte_end: Optional[int] = None,
    ) -> str:
        """
        Generate deterministic chunk ID for Qdrant point.
        Uses symbol identity plus signature or byte span to avoid collisions.

        file_path: str — Source file path.
        symbol_name: str — Symbol name.
        symbol_kind: str — Symbol kind (function, class, method, etc).
        parent: Optional[str] — Parent symbol name (class for methods, None for top-level).
        signature: Optional[str] — Signature text when available.
        byte_start: Optional[int] — Byte start fallback for overload disambiguation.
        byte_end: Optional[int] — Byte end fallback for overload disambiguation.
        Returns: str — Hex-encoded SHA256 hash.
        """
        normalized_path = str(Path(file_path).resolve())
        parent_str = f"::{parent}" if parent else ""
        signature_str = f"::sig={signature}" if signature else ""
        span_str = (
            f"::span={int(byte_start)}-{int(byte_end)}"
            if byte_start is not None and byte_end is not None
            else ""
        )
        key = f"{normalized_path}::{symbol_name}::{symbol_kind}{parent_str}{signature_str}{span_str}"
        return hashlib.sha256(key.encode('utf-8')).hexdigest()

    def _hash_content(self, content: str) -> str:
        """
        Hash file content for change detection.

        Args:
            content: File content

        Returns:
            Hex-encoded SHA256 hash
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _read_file(self, file_path: str) -> str:
        """
        Read file content.

        Args:
            file_path: Path to file

        Returns:
            File content as string
        """
        return Path(file_path).read_text(encoding='utf-8', errors='ignore')
