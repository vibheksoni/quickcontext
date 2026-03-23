from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import asyncio
import re

from engine.src.artifact_semantics import extract_artifact_semantic_signals
from engine.src.chunker import CodeChunk


def _get_litellm():
    """
    Lazy-import litellm and suppress debug output.

    Returns: module — The litellm module.
    Raises: ImportError — If litellm is not installed.
    """
    import os
    import litellm

    os.environ.setdefault("LITELLM_LOG", "ERROR")
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    return litellm


@dataclass(frozen=True, slots=True)
class ChunkDescription:
    """
    Generated description and keywords for a code chunk.

    Args:
        chunk_id: Chunk identifier
        description: Natural language description (1-3 sentences)
        keywords: Extracted keywords for ranking/relevancy
        token_count: Tokens used for generation (cost tracking)
        cost_usd: Cost in USD for this generation
    """
    chunk_id: str
    description: str
    keywords: list[str]
    token_count: int
    cost_usd: float = 0.0


def _split_identifier_tokens(text: str, max_tokens: int = 10) -> list[str]:
    """
    Split symbol and path text into lowercase keyword tokens.
    """
    if not text:
        return []

    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    expanded = re.sub(r"[\\/.\-:()\[\],]", " ", expanded)
    raw_tokens = re.findall(r"[A-Za-z0-9_]+", expanded)

    seen: set[str] = set()
    out: list[str] = []
    for raw_token in raw_tokens:
        for part in raw_token.split("_"):
            lower = part.lower().strip()
            if len(lower) < 3 or lower in seen:
                continue
            seen.add(lower)
            out.append(lower)
            if len(out) >= max_tokens:
                return out

    return out


def _docstring_summary(docstring: Optional[str]) -> str:
    """
    Extract the first sentence-like summary from a docstring.
    """
    if not docstring:
        return ""

    cleaned = " ".join(docstring.strip().split())
    if not cleaned:
        return ""

    sentence, _, _ = cleaned.partition(". ")
    summary = sentence.strip()
    if summary and not summary.endswith("."):
        summary += "."
    return summary[:220]


def build_fallback_description(chunk: CodeChunk) -> ChunkDescription:
    """
    Build a lightweight local description and keywords without LLM calls.
    """
    path_tail = "/".join(Path(chunk.file_path).parts[-3:])
    docstring_summary = _docstring_summary(chunk.docstring)

    if chunk.symbol_kind == "file_artifact":
        signals = extract_artifact_semantic_signals(chunk.source, file_name=Path(chunk.file_path).name)
        summary_bits: list[str] = []
        if signals.services:
            summary_bits.append("services " + ", ".join(service.split(".")[-1] for service in signals.services[:2]))
        if signals.methods:
            summary_bits.append("methods " + ", ".join(signals.methods[:4]))
        if signals.field_names:
            summary_bits.append("fields " + ", ".join(signals.field_names[:5]))
        if signals.string_literals:
            summary_bits.append("messages " + " | ".join(signals.string_literals[:2]))

        description = f"Generated {chunk.language} bundle artifact"
        if path_tail:
            description += f" from {path_tail}"
        if summary_bits:
            description += " exposing " + "; ".join(summary_bits)
        description += "."

        keywords: list[str] = []
        for token in [
            chunk.language,
            "generated",
            "bundle",
            *(signals.keywords[:10]),
            *_split_identifier_tokens(path_tail, max_tokens=6),
        ]:
            if not token or token in keywords:
                continue
            keywords.append(token)
            if len(keywords) >= 12:
                break

        return ChunkDescription(
            chunk_id=chunk.chunk_id,
            description=description,
            keywords=keywords,
            token_count=0,
            cost_usd=0.0,
        )

    if docstring_summary:
        description = docstring_summary
        if path_tail:
            description = f"{description} Defined in {path_tail}."
    else:
        subject = f"{chunk.language} {chunk.symbol_kind} {chunk.symbol_name}"
        if chunk.parent:
            subject += f" in {chunk.parent}"
        description = subject
        if chunk.signature:
            description += f" with signature {chunk.signature}"
        if path_tail:
            description += f" from {path_tail}"
        description += "."

    keywords: list[str] = []
    for token in [
        chunk.language,
        chunk.symbol_kind,
        *_split_identifier_tokens(chunk.symbol_name, max_tokens=5),
        *_split_identifier_tokens(chunk.parent or "", max_tokens=4),
        *_split_identifier_tokens(chunk.signature or "", max_tokens=6),
        *_split_identifier_tokens(path_tail, max_tokens=6),
    ]:
        if not token or token in keywords:
            continue
        keywords.append(token)
        if len(keywords) >= 10:
            break

    return ChunkDescription(
        chunk_id=chunk.chunk_id,
        description=description,
        keywords=keywords,
        token_count=0,
        cost_usd=0.0,
    )


def build_fallback_descriptions(chunks: list[CodeChunk]) -> list[ChunkDescription]:
    """
    Build fallback descriptions for a batch of chunks.
    """
    return [build_fallback_description(chunk) for chunk in chunks]


class DescriptionGenerator:
    """
    Generates tiny NL descriptions and keywords for code chunks using LLM.

    Uses configured LLM (OpenRouter gpt-oss-20b) to generate:
    - 1-3 sentence description of what the code does
    - 3-7 keywords for semantic ranking
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        openrouter_provider: Optional[str] = None,
    ):
        """
        Args:
            model: LiteLLM model string (e.g., "openrouter/openai/gpt-oss-20b")
            api_key: API key for provider
            api_base: Optional API base URL
            max_tokens: Max tokens for description generation
            temperature: Sampling temperature (0.0 for deterministic)
            openrouter_provider: Force specific OpenRouter upstream (e.g., "Groq", "DeepInfra")
        """
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._openrouter_provider = openrouter_provider

    def generate(self, chunk: CodeChunk) -> ChunkDescription:
        """
        Generate description and keywords for a code chunk.

        Args:
            chunk: Code chunk to describe

        Returns:
            ChunkDescription with generated content
        """
        prompt = self._build_prompt(chunk)

        kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "api_key": self._api_key,
        }

        if self._api_base:
            kwargs["api_base"] = self._api_base

        if self._openrouter_provider:
            kwargs["extra_body"] = {
                "provider": {
                    "order": [self._openrouter_provider]
                }
            }

        try:
            response = _get_litellm().completion(**kwargs)
            content = response.choices[0].message.content.strip()
            token_count = response.usage.total_tokens
            parsed = self._parse_response(content)

            return ChunkDescription(
                chunk_id=chunk.chunk_id,
                description=parsed["description"],
                keywords=parsed["keywords"],
                token_count=token_count,
            )
        except Exception:
            return build_fallback_description(chunk)

    def generate_batch(
        self,
        chunks: list[CodeChunk],
        batch_size: int = 10,
        progress_callback=None,
    ) -> list[ChunkDescription]:
        """
        Generate descriptions for multiple chunks in batches using async concurrency.

        Args:
            chunks: List of code chunks
            batch_size: Number of concurrent requests (default: 10)

        Returns:
            List of chunk descriptions
        """
        return asyncio.run(self._generate_batch_async(chunks, batch_size, progress_callback=progress_callback))

    def generate_lightweight_metadata_batch(
        self,
        chunks: list[CodeChunk],
        request_batch_size: int = 4,
        max_tokens: Optional[int] = None,
        progress_callback=None,
    ) -> list[ChunkDescription]:
        """
        Generate compact descriptions and keywords for multiple chunks per request.
        """
        if not chunks:
            return []

        out: list[ChunkDescription] = []
        completed = 0
        batch_size = max(1, int(request_batch_size))
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            out.extend(self._generate_lightweight_metadata_once(batch, max_tokens=max_tokens))
            completed += len(batch)
            if progress_callback is not None:
                progress_callback(completed, len(chunks))
        return out

    async def _generate_batch_async(
        self,
        chunks: list[CodeChunk],
        batch_size: int,
        progress_callback=None,
    ) -> list[ChunkDescription]:
        """
        Async implementation of batch generation with concurrency control.

        Args:
            chunks: List of code chunks
            batch_size: Number of concurrent requests

        Returns:
            List of chunk descriptions
        """
        semaphore = asyncio.Semaphore(batch_size)
        tasks = [
            asyncio.create_task(self._generate_indexed_async(idx, chunk, semaphore))
            for idx, chunk in enumerate(chunks)
        ]
        results: list[ChunkDescription | None] = [None] * len(chunks)
        completed = 0
        for task in asyncio.as_completed(tasks):
            idx, description = await task
            results[idx] = description
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(chunks))
        return [item for item in results if item is not None]

    async def _generate_indexed_async(
        self,
        index: int,
        chunk: CodeChunk,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, ChunkDescription]:
        return index, await self._generate_async(chunk, semaphore)

    async def _generate_async(self, chunk: CodeChunk, semaphore: asyncio.Semaphore) -> ChunkDescription:
        """
        Async description generation for a single chunk with semaphore control.

        Args:
            chunk: Code chunk to describe
            semaphore: Semaphore for concurrency control

        Returns:
            ChunkDescription with generated content
        """
        async with semaphore:
            prompt = self._build_prompt(chunk)

            kwargs = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
                "api_key": self._api_key,
            }

            if self._api_base:
                kwargs["api_base"] = self._api_base

            if self._openrouter_provider:
                kwargs["extra_body"] = {
                    "provider": {
                        "order": [self._openrouter_provider]
                    }
                }

            try:
                response = await _get_litellm().acompletion(**kwargs)
                content = response.choices[0].message.content.strip()
                token_count = response.usage.total_tokens
                cost = _get_litellm().completion_cost(completion_response=response)
                parsed = self._parse_response(content)

                return ChunkDescription(
                    chunk_id=chunk.chunk_id,
                    description=parsed["description"],
                    keywords=parsed["keywords"],
                    token_count=token_count,
                    cost_usd=cost,
                )
            except Exception:
                return build_fallback_description(chunk)

    def _generate_lightweight_metadata_once(
        self,
        chunks: list[CodeChunk],
        max_tokens: Optional[int] = None,
    ) -> list[ChunkDescription]:
        """
        Generate compact metadata for a batch of chunks in one LLM request.
        """
        if not chunks:
            return []

        prompt = self._build_lightweight_batch_prompt(chunks)
        kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._lightweight_batch_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": self._temperature,
            "api_key": self._api_key,
        }

        if self._api_base:
            kwargs["api_base"] = self._api_base

        if self._openrouter_provider:
            kwargs["extra_body"] = {
                "provider": {
                    "order": [self._openrouter_provider]
                }
            }

        try:
            response = _get_litellm().completion(**kwargs)
            content = response.choices[0].message.content.strip()
            parsed = self._parse_lightweight_batch_response(content)
            try:
                cost = _get_litellm().completion_cost(completion_response=response)
            except Exception:
                cost = float(getattr(getattr(response, "usage", None), "cost", 0.0) or 0.0)
            token_count = int(getattr(response.usage, "total_tokens", 0) or 0)
        except Exception:
            return [build_fallback_description(chunk) for chunk in chunks]

        per_item_cost = cost / max(1, len(chunks))
        per_item_tokens = token_count // max(1, len(chunks))
        descriptions: list[ChunkDescription] = []
        for idx, chunk in enumerate(chunks):
            item = parsed.get(str(idx))
            if item is None:
                if len(chunks) > 1:
                    descriptions.extend(self._generate_lightweight_metadata_once([chunk], max_tokens=max_tokens))
                else:
                    descriptions.append(build_fallback_description(chunk))
                continue
            descriptions.append(
                ChunkDescription(
                    chunk_id=chunk.chunk_id,
                    description=item["description"],
                    keywords=item["keywords"],
                    token_count=per_item_tokens,
                    cost_usd=per_item_cost,
                )
            )
        return descriptions

    def _system_prompt(self) -> str:
        """
        System prompt for description generation.

        Returns:
            System prompt text
        """
        return """You are a code documentation assistant. Generate concise descriptions and keywords for code snippets.

Output format (JSON):
{
  "description": "1-3 sentence description of what the code does",
  "keywords": ["keyword1", "keyword2", "keyword3"]
}

Rules:
- Description: 1-3 sentences max, focus on WHAT and WHY, not HOW
- Keywords: 3-7 relevant terms for semantic search (language, domain, patterns, concepts)
- Keep it minimal to save tokens
- Output ONLY valid JSON, no markdown fences"""

    def _lightweight_batch_system_prompt(self) -> str:
        """
        System prompt for compact multi-chunk metadata generation.
        """
        return """You are a code indexing assistant. Generate compact descriptions and keywords for multiple generated-code packets in one response.

Output format (XML):
<items>
  <item id="0">
    <description>Short 1 sentence description.</description>
    <keywords>kw1|kw2|kw3|kw4</keywords>
  </item>
</items>

Rules:
- Return one `<item>` for every input id exactly once.
- Keep each description to 1 sentence.
- Keywords must be short, concrete, and reusable across projects.
- Infer behavior from the packet context, not just one identifier.
- Prefer protocol, UI, routing, auth, billing, search, indexing, or state-management concepts when present.
- Do not add explanations outside the XML.
- Output ONLY the XML."""

    def _build_prompt(self, chunk: CodeChunk) -> str:
        """
        Build user prompt for a code chunk.

        Args:
            chunk: Code chunk to describe

        Returns:
            Formatted prompt
        """
        context_parts = []

        context_parts.append(f"Language: {chunk.language}")
        context_parts.append(f"File: {chunk.file_path}")
        context_parts.append(f"Symbol: {chunk.symbol_name} ({chunk.symbol_kind})")

        if chunk.parent:
            context_parts.append(f"Parent: {chunk.parent}")

        if chunk.signature:
            context_parts.append(f"Signature: {chunk.signature}")

        if chunk.docstring:
            context_parts.append(f"Docstring: {chunk.docstring}")

        context = "\n".join(context_parts)

        source_preview = chunk.source
        if len(source_preview) > 2000:
            source_preview = source_preview[:2000] + "\n... [truncated]"

        return f"""{context}

Code:
```{chunk.language}
{source_preview}
```

Generate description and keywords."""

    def _build_lightweight_batch_prompt(self, chunks: list[CodeChunk]) -> str:
        """
        Build one prompt containing multiple generated-code packets for metadata generation.
        """
        items: list[str] = []
        for idx, chunk in enumerate(chunks):
            context_parts = [
                f"Language: {chunk.language}",
                f"File: {chunk.file_path}",
                f"Symbol: {chunk.symbol_name} ({chunk.symbol_kind})",
            ]
            if chunk.parent:
                context_parts.append(f"Parent: {chunk.parent}")
            if chunk.signature:
                context_parts.append(f"Signature: {chunk.signature}")

            source_preview = chunk.source
            marker = "\n\nRaw excerpt:\n"
            if marker in source_preview:
                source_preview = source_preview.split(marker, 1)[0]
            if len(source_preview) > 900:
                source_preview = source_preview[:900] + "\n... [truncated]"

            items.append(
                f"""<item id="{idx}">
<context>
{chr(10).join(context_parts)}
</context>
<packet>
{source_preview}
</packet>
</item>"""
            )

        return "<items>\n" + "\n".join(items) + "\n</items>"

    def _parse_response(self, content: str) -> dict[str, any]:
        """
        Parse LLM response into structured data.

        Args:
            content: Raw LLM response

        Returns:
            Dict with description and keywords
        """
        content = content.strip()

        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        try:
            parsed = json.loads(content)
            return {
                "description": parsed.get("description", ""),
                "keywords": parsed.get("keywords", []),
            }
        except json.JSONDecodeError:
            lines = content.split('\n')
            description = ""
            keywords = []

            for line in lines:
                line = line.strip()
                if line.startswith('"description"'):
                    description = line.split(':', 1)[1].strip(' ",')
                elif line.startswith('"keywords"'):
                    kw_part = line.split(':', 1)[1].strip(' []')
                    keywords = [k.strip(' "') for k in kw_part.split(',') if k.strip()]

            return {
                "description": description or "No description available",
                "keywords": keywords,
            }

    def _parse_lightweight_batch_response(self, content: str) -> dict[str, dict[str, object]]:
        """
        Parse XML-like multi-item metadata output.
        """
        content = content.strip()
        if content.startswith("```xml"):
            content = content[6:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        matches = re.findall(
            r"<item id=['\"]([^'\"]+)['\"]>\s*<description>(.*?)</description>\s*<keywords>(.*?)</keywords>\s*</item>",
            content,
            flags=re.DOTALL,
        )
        out: dict[str, dict[str, object]] = {}
        for item_id, description, keywords_text in matches:
            keywords = [
                keyword.strip()
                for keyword in re.split(r"[|,]", keywords_text)
                if keyword.strip()
            ]
            out[str(item_id)] = {
                "description": " ".join(description.strip().split()) or "No description available",
                "keywords": keywords[:8],
            }
        return out
