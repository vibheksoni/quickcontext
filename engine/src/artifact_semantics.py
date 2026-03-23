from dataclasses import dataclass
import re
from pathlib import Path


_SERVICE_PATTERN = re.compile(r'typeName\s*(?:=|:)\s*"([^"]+Service)"')
_TYPE_PATTERN = re.compile(r'typeName\s*(?:=|:)\s*"([^"]+)"')
_METHOD_NAME_PATTERN = re.compile(r'name:\s*"([A-Z][A-Za-z0-9_]{2,})"')
_CLIENT_METHOD_PATTERN = re.compile(r'this\.([a-z][A-Za-z0-9_]{2,})\s*=\s*async')
_CALL_TARGET_PATTERN = re.compile(
    r'(?:(?:this|client|service|api|server|manager)\.)?([A-Za-z_$][A-Za-z0-9_$]{1,})\.([A-Za-z_$][A-Za-z0-9_$]{2,})\s*\('
)
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
    call_targets: list[str]
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


def _semantic_value_score(value: str) -> float:
    tokens = _split_tokens(value)
    if not tokens:
        return 0.0

    score = min(len(tokens), 6) * 0.85
    score += min(sum(len(token) for token in tokens) / 24.0, 1.6)

    uppercase_like = value.upper() == value and "_" in value
    if uppercase_like:
        score *= 0.45

    if "." in value:
        score += 0.2
    if "_" in value:
        score += 0.1
    return score


def _rank_semantic_values(values: list[str], limit: int) -> list[str]:
    unique = _ordered_unique(values, limit=max(limit * 8, limit))
    token_frequency: dict[str, int] = {}
    token_map: dict[str, list[str]] = {}
    for value in unique:
        tokens = list(dict.fromkeys(_split_tokens(value)))
        token_map[value] = tokens
        for token in tokens:
            token_frequency[token] = token_frequency.get(token, 0) + 1

    def _value_rank(value: str) -> tuple[float, str]:
        rarity_bonus = sum(1.0 / token_frequency[token] for token in token_map.get(value, []))
        return (_semantic_value_score(value) + rarity_bonus, value.lower())

    ranked = sorted(unique, key=lambda value: (-_value_rank(value)[0], _value_rank(value)[1]))
    return ranked[:limit]


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
    return _ordered_unique(matches, limit=32)


def _extract_call_targets(source: str) -> list[str]:
    targets = [f"{owner}.{name}" for owner, name in _CALL_TARGET_PATTERN.findall(source)]
    return _ordered_unique(targets, limit=64)


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
    message_types = _rank_semantic_values(
        _extract_projection_items(source, "Types", limit=6)
        + [match for match in _TYPE_PATTERN.findall(source) if not match.endswith("Service")],
        limit=6,
    )
    methods = _rank_semantic_values(
        _extract_projection_items(source, "Methods", limit=8)
        + _METHOD_NAME_PATTERN.findall(source)
        + _CLIENT_METHOD_PATTERN.findall(source),
        limit=8,
    )
    call_targets = _rank_semantic_values(
        _extract_projection_items(source, "CallTargets", limit=10) + _extract_call_targets(source),
        limit=10,
    )
    field_names = _rank_semantic_values(
        _extract_projection_items(source, "Fields", limit=10) + _FIELD_PATTERN.findall(source),
        limit=10,
    )
    string_literals = _rank_semantic_values(
        _extract_projection_items(source, "Strings", limit=4) + _extract_string_literals(source),
        limit=4,
    )

    keyword_candidates: list[str] = []
    if file_name:
        keyword_candidates.extend(_split_tokens(Path(file_name).name))
    keyword_candidates.extend(_extract_projection_items(source, "Keywords", limit=16))
    for value in [*services, *message_types, *methods, *call_targets, *field_names, *string_literals]:
        keyword_candidates.extend(_split_tokens(value))
    keywords = _ordered_unique(keyword_candidates, limit=16)

    return ArtifactSemanticSignals(
        services=services,
        methods=methods,
        call_targets=call_targets,
        message_types=message_types,
        field_names=field_names,
        string_literals=string_literals,
        keywords=keywords,
    )


def rank_artifact_signal_offsets(source: str, limit: int = 12) -> list[tuple[int, float]]:
    """
    Rank structurally interesting character offsets inside a generated artifact.
    """
    candidates: list[tuple[int, float, str]] = []
    patterns = (
        (_SERVICE_PATTERN, 3.5),
        (_TYPE_PATTERN, 2.6),
        (_CLIENT_METHOD_PATTERN, 3.2),
        (_METHOD_NAME_PATTERN, 2.8),
        (_FIELD_PATTERN, 2.2),
        (_STRING_PATTERN, 1.6),
    )
    for pattern, base_score in patterns:
        for match in pattern.finditer(source):
            text = match.group(1) if match.groups() else match.group(0)
            if pattern is _STRING_PATTERN:
                lowered = text.lower()
                if "http" in lowered or "/" in text or "\\" in text or " " not in text:
                    continue
                if not set(_split_tokens(text)).intersection(_STRING_FOCUS_TOKENS):
                    continue
            candidates.append((match.start(), base_score, text))

    token_frequency: dict[str, int] = {}
    token_map: list[tuple[int, float, str, list[str]]] = []
    for offset, base_score, text in candidates:
        tokens = list(dict.fromkeys(_split_tokens(text)))
        token_map.append((offset, base_score, text, tokens))
        for token in tokens:
            token_frequency[token] = token_frequency.get(token, 0) + 1

    scored: list[tuple[int, float]] = []
    for offset, base_score, text, tokens in token_map:
        rarity_bonus = sum(1.0 / token_frequency[token] for token in tokens)
        score = base_score + (_semantic_value_score(text) * 0.85) + rarity_bonus
        scored.append((offset, score))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:limit]


def find_artifact_signal_offsets(source: str, limit: int = 12) -> list[int]:
    """
    Find structurally interesting character offsets inside a generated artifact.
    """
    out: list[int] = []
    for offset, _score in rank_artifact_signal_offsets(source, limit=limit * 8):
        if any(abs(offset - existing) < 240 for existing in out):
            continue
        out.append(offset)
        if len(out) >= limit:
            break
    return out


def render_artifact_signal_packet(
    *,
    file_path: str,
    heading: str,
    signals: ArtifactSemanticSignals,
    extra_lines: list[str] | None = None,
) -> str:
    lines = [heading]
    if signals.services:
        lines.append("Services: " + ", ".join(signals.services[:3]))
    if signals.methods:
        lines.append("Methods: " + ", ".join(signals.methods[:6]))
    if signals.call_targets:
        lines.append("CallTargets: " + ", ".join(signals.call_targets[:6]))
    if signals.message_types:
        lines.append("Types: " + ", ".join(signals.message_types[:4]))
    if signals.field_names:
        lines.append("Fields: " + ", ".join(signals.field_names[:8]))
    if extra_lines:
        lines.extend(extra_lines)
    if signals.keywords:
        lines.append("Keywords: " + ", ".join(signals.keywords[:12]))
    if signals.string_literals:
        lines.append("Strings: " + " | ".join(signals.string_literals[:3]))
    return "\n".join(lines).strip()


def build_artifact_semantic_projection(
    source: str,
    file_path: str,
    chunk_index: int,
    chunk_count: int,
    byte_start: int,
    byte_end: int,
    file_signals: ArtifactSemanticSignals | None = None,
    neighbor_signals: ArtifactSemanticSignals | None = None,
) -> str:
    signals = extract_artifact_semantic_signals(source, file_name=Path(file_path).name)
    extra_lines: list[str] = [
        f"Chunk {chunk_index + 1} of {chunk_count}",
        f"Approx byte range {byte_start}-{byte_end}",
    ]
    if neighbor_signals is not None:
        related = _ordered_unique(
            [*neighbor_signals.methods, *neighbor_signals.call_targets, *neighbor_signals.field_names],
            limit=8,
        )
        if related:
            extra_lines.append("Nearby: " + ", ".join(related[:6]))
    if file_signals is not None:
        global_terms = _ordered_unique(
            [*file_signals.services, *file_signals.message_types, *file_signals.call_targets],
            limit=8,
        )
        if global_terms:
            extra_lines.append("FileWide: " + ", ".join(global_terms[:6]))
    return render_artifact_signal_packet(
        file_path=file_path,
        heading=f"Generated bundle artifact from {Path(file_path).name}",
        signals=signals,
        extra_lines=extra_lines,
    )
