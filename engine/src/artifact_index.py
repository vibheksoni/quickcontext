from dataclasses import dataclass
from pathlib import Path
import hashlib
import re


ARTIFACT_FALLBACK_ERROR = "__artifact_fallback__"

_HASHED_BUNDLE_PATTERN = re.compile(r"(?:^|[._-])[0-9a-f]{6,}(?:[._-]|$)")
_BUNDLE_PREFIXES = (
    "bundle.",
    "vendor.",
    "shared~",
    "loader.",
    "loaders.",
    "icons.",
    "modules.",
    "ondemand.",
)
_BUNDLE_INFIXES = (
    "~bundle.",
    "~loader.",
    "~ondemand.",
    "~icons.",
)


@dataclass(frozen=True, slots=True)
class ArtifactIndexProfile:
    """
    Lightweight file-level profile used to downgrade obvious bundle artifacts during fast indexing.
    """

    file_path: str
    language: str
    file_size: int
    file_mtime: int
    file_hash: str
    line_count: int
    max_line_length: int
    avg_line_length: float
    whitespace_ratio: float
    raw_minified_like: bool
    bundle_like_name: bool


def analyze_artifact_candidate(
    file_path: str | Path,
    language: str,
    file_size: int,
    file_mtime: int,
) -> ArtifactIndexProfile:
    """
    Read a likely large JS artifact once to compute a stable hash and cheap line-structure heuristics.
    """
    normalized_path = str(Path(file_path).resolve())
    path_obj = Path(normalized_path)
    hasher = hashlib.sha256()
    total_whitespace = 0
    total_bytes = 0
    line_count = 0
    total_line_length = 0
    max_line_length = 0

    with open(path_obj, "rb") as handle:
        for raw_line in handle:
            hasher.update(raw_line)
            total_bytes += len(raw_line)
            total_whitespace += (
                raw_line.count(b" ")
                + raw_line.count(b"\t")
                + raw_line.count(b"\r")
                + raw_line.count(b"\n")
            )
            line = raw_line.decode("utf-8", errors="ignore").rstrip("\r\n")
            line_length = len(line)
            line_count += 1
            total_line_length += line_length
            if line_length > max_line_length:
                max_line_length = line_length

    avg_line_length = (total_line_length / line_count) if line_count else 0.0
    whitespace_ratio = total_whitespace / max(1, total_bytes)
    lowered_name = path_obj.name.lower()
    bundle_like_name = (
        lowered_name.startswith(_BUNDLE_PREFIXES)
        or any(token in lowered_name for token in _BUNDLE_INFIXES)
        or ".min." in lowered_name
        or "~" in lowered_name
        or bool(_HASHED_BUNDLE_PATTERN.search(lowered_name))
    )
    raw_minified_like = (
        max_line_length >= 1200
        or (line_count <= 6 and avg_line_length >= 220)
        or (line_count <= 3 and whitespace_ratio < 0.04 and avg_line_length > 120)
    )

    return ArtifactIndexProfile(
        file_path=normalized_path,
        language=language,
        file_size=file_size,
        file_mtime=file_mtime,
        file_hash=hasher.hexdigest(),
        line_count=line_count,
        max_line_length=max_line_length,
        avg_line_length=avg_line_length,
        whitespace_ratio=whitespace_ratio,
        raw_minified_like=raw_minified_like,
        bundle_like_name=bundle_like_name,
    )


def should_downgrade_artifact_profile(profile: ArtifactIndexProfile) -> bool:
    """
    Decide whether a file should skip deep symbol extraction and use coarse artifact chunks instead.
    """
    language = profile.language.lower()
    if language not in {"javascript", "typescript", "tsx", "jsx"}:
        return False

    # Large hashed bundles are often transpiled/generated even when they are not
    # technically minified into a few giant lines. These files still explode the
    # symbol/chunk count while carrying relatively low retrieval value.
    if (
        profile.bundle_like_name
        and profile.file_size >= 256 * 1024
        and profile.line_count >= 6000
        and profile.avg_line_length >= 28
    ):
        return True

    if not profile.raw_minified_like:
        return False

    if profile.file_size >= 512 * 1024:
        return True

    return profile.bundle_like_name and profile.file_size >= 192 * 1024
