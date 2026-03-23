from dataclasses import dataclass
import re
from pathlib import Path


_SERVICE_PATTERN = re.compile(r'typeName\s*(?:=|:)\s*"([^"]+Service)"')
_TYPE_PATTERN = re.compile(r'typeName\s*(?:=|:)\s*"([^"]+)"')
_METHOD_NAME_PATTERN = re.compile(r'name:\s*"([A-Z][A-Za-z0-9_]{2,})"')
_CLIENT_METHOD_PATTERN = re.compile(r'this\.([a-z][A-Za-z0-9_]{2,})\s*=\s*async')
_FIELD_PATTERN = re.compile(r'name:\s*"([a-z][a-z0-9_]{2,})"\s*,\s*kind:')
_STRING_PATTERN = re.compile(r'"([^"\r\n]{6,120})"')
_CAMEL_BOUNDARY_PATTERN = re.compile(r"([a-z0-9])([A-Z])")
_GENERIC_TOKENS = {
    "async",
    "chunk",
    "constructor",
    "fields",
    "generated",
    "kind",
    "message",
    "messages",
    "method",
    "methods",
    "name",
    "names",
    "projection",
    "runtime",
    "service",
    "services",
    "source",
    "static",
    "super",
    "type",
    "types",
    "util",
}
_STRING_FOCUS_TOKENS = {
    "account",
    "auth",
    "billing",
    "credit",
    "email",
    "eligible",
    "eligibility",
    "limit",
    "login",
    "plan",
    "trial",
    "user",
}


@dataclass(frozen=True, slots=True)
class ArtifactSemanticSignals:
    services: list[str]
    methods: list[str]
    message_types: list[str]
    field_names: list[str]
    string_literals: list[str]
    keywords: list[str]


def _ordered_unique(values: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


def _split_tokens(text: str) -> list[str]:
    if not text:
        return []
    expanded = _CAMEL_BOUNDARY_PATTERN.sub(r"\1 \2", text)
    expanded = re.sub(r"[\\/.\-:()\[\],{}=]+", " ", expanded)
    raw_tokens = re.findall(r"[A-Za-z0-9_]+", expanded)
    tokens: list[str] = []
    for raw_token in raw_tokens:
        for part in raw_token.split("_"):
            token = part.lower().strip()
            if len(token) < 3:
                continue
            if token in _GENERIC_TOKENS:
                continue
            tokens.append(token)
    return tokens


def _extract_string_literals(source: str) -> list[str]:
    matches: list[str] = []
    for raw in _STRING_PATTERN.findall(source):
        lowered = raw.lower()
        if "http" in lowered or "/" in raw or "\\" in raw:
            continue
        if " " not in raw:
            continue
        tokens = set(_split_tokens(raw))
        if not tokens.intersection(_STRING_FOCUS_TOKENS):
            continue
        matches.append(raw.strip())
    return _ordered_unique(matches, limit=4)


def _extract_projection_items(source: str, label: str, limit: int) -> list[str]:
    match = re.search(rf"^{label}:\s*(.+)$", source, re.MULTILINE)
    if match is None:
        return []
    raw = match.group(1).strip()
    parts = [part.strip() for part in re.split(r"[|,]", raw) if part.strip()]
    return _ordered_unique(parts, limit=limit)


def extract_artifact_semantic_signals(source: str, file_name: str | None = None) -> ArtifactSemanticSignals:
    services = _ordered_unique(
        _extract_projection_items(source, "Services", limit=4) + _SERVICE_PATTERN.findall(source),
        limit=4,
    )
    message_types = _ordered_unique(
        _extract_projection_items(source, "Types", limit=6)
        + [match for match in _TYPE_PATTERN.findall(source) if not match.endswith("Service")],
        limit=6,
    )
    methods = _ordered_unique(
        _extract_projection_items(source, "Methods", limit=8)
        + _METHOD_NAME_PATTERN.findall(source)
        + _CLIENT_METHOD_PATTERN.findall(source),
        limit=8,
    )
    field_names = _ordered_unique(
        _extract_projection_items(source, "Fields", limit=10) + _FIELD_PATTERN.findall(source),
        limit=10,
    )
    string_literals = _ordered_unique(
        _extract_projection_items(source, "Strings", limit=4) + _extract_string_literals(source),
        limit=4,
    )

    keyword_candidates: list[str] = []
    if file_name:
        keyword_candidates.extend(_split_tokens(Path(file_name).name))
    keyword_candidates.extend(_extract_projection_items(source, "Keywords", limit=16))
    for value in [*services, *message_types, *methods, *field_names, *string_literals]:
        keyword_candidates.extend(_split_tokens(value))
    keywords = _ordered_unique(keyword_candidates, limit=16)

    return ArtifactSemanticSignals(
        services=services,
        methods=methods,
        message_types=message_types,
        field_names=field_names,
        string_literals=string_literals,
        keywords=keywords,
    )


def build_artifact_semantic_projection(
    source: str,
    file_path: str,
    chunk_index: int,
    chunk_count: int,
    byte_start: int,
    byte_end: int,
) -> str:
    signals = extract_artifact_semantic_signals(source, file_name=Path(file_path).name)
    lines = [
        f"Generated bundle artifact from {Path(file_path).name}",
        f"Chunk {chunk_index + 1} of {chunk_count}",
        f"Approx byte range {byte_start}-{byte_end}",
    ]
    if signals.services:
        lines.append("Services: " + ", ".join(signals.services[:3]))
    if signals.methods:
        lines.append("Methods: " + ", ".join(signals.methods[:6]))
    if signals.message_types:
        lines.append("Types: " + ", ".join(signals.message_types[:4]))
    if signals.field_names:
        lines.append("Fields: " + ", ".join(signals.field_names[:8]))
    if signals.keywords:
        lines.append("Keywords: " + ", ".join(signals.keywords[:12]))
    if signals.string_literals:
        lines.append("Strings: " + " | ".join(signals.string_literals[:3]))
    return "\n".join(lines).strip()
