from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import random
from threading import Lock
import time
from typing import Optional, TYPE_CHECKING

from engine.src.chunker import CodeChunk
from engine.src.describer import ChunkDescription

if TYPE_CHECKING:
    from fastembed import TextEmbedding


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
class EmbeddedChunk:
    """
    Code chunk with dual embeddings (code + description vectors).

    Args:
        chunk_id: Chunk identifier
        code_vector: Embedding of raw source code
        desc_vector: Embedding of natural language description
        chunk: Original code chunk
        description: Generated description and keywords
        embedding_cost_usd: Cost in USD for embedding generation
    """
    chunk_id: str
    code_vector: list[float]
    desc_vector: list[float]
    chunk: CodeChunk
    description: ChunkDescription
    embedding_cost_usd: float = 0.0


def embed_length_sorted(model, texts: list[str], batch_size: int) -> list:
    """
    Embed texts with a local model in length-sorted batches, returning vectors in
    input order.

    Each batch is padded to its longest member, so mixing short and long inputs
    wastes most of the compute; sorting by length first produced identical vectors
    about 2x faster on real code chunks.
    """
    order = sorted(range(len(texts)), key=lambda idx: len(texts[idx]))
    embedded = list(model.embed([texts[idx] for idx in order], batch_size=max(1, int(batch_size))))
    vectors: list = [None] * len(texts)
    for position, idx in enumerate(order):
        vectors[idx] = embedded[position]
    return vectors


REMOTE_TOKENS_PER_SLOT = 512
REMOTE_MIN_TOKEN_BUDGET = 2048


def estimate_tokens(text: str) -> int:
    """
    Cheap token estimate for request packing (about 4 characters per token).
    """
    return len(text) // 4 + 1


def plan_remote_batches(texts: list[str], batch_size: int, token_budget: Optional[int]) -> list[list[int]]:
    """
    Group text indexes into embedding requests, longest texts first.

    Each request holds at most batch_size texts and, when token_budget is set, at
    most that many estimated tokens (a single oversized text still gets its own
    request). Scheduling the heaviest requests first and keeping request sizes
    even avoids slow stragglers at the end of a parallel run.
    """
    order = sorted(range(len(texts)), key=lambda idx: len(texts[idx]), reverse=True)
    batches: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for idx in order:
        tokens = estimate_tokens(texts[idx])
        too_many = len(current) >= max(1, batch_size)
        too_heavy = token_budget is not None and current and current_tokens + tokens > token_budget
        if current and (too_many or too_heavy):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(idx)
        current_tokens += tokens
    if current:
        batches.append(current)
    return batches


def contextual_code_text(chunk: CodeChunk) -> str:
    """
    Situate a code chunk before embedding: a short comment header with the
    trailing file path and qualified symbol name. A deterministic, zero-cost
    variant of contextual retrieval; chunks embedded in isolation otherwise
    lose which file and class they belong to.
    """
    parts = [part for part in Path(str(chunk.file_path).replace("\\\\?\\", "")).parts if part not in ("", "/", "\\")]
    lines = [f"// file: {'/'.join(parts[-4:])}"] if parts else []
    kind = (chunk.symbol_kind or "").strip()
    if kind and kind != "file_artifact" and chunk.symbol_name:
        qualified = f"{chunk.parent}.{chunk.symbol_name}" if chunk.parent else chunk.symbol_name
        lines.append(f"// {kind}: {qualified}")
    if not lines:
        return chunk.source
    return "\n".join(lines) + "\n" + (chunk.source or "")


@dataclass(frozen=True, slots=True)
class EmbeddingProviderStats:
    """
    Per-provider embedding execution statistics.

    request_count: int — Number of embedding HTTP requests issued.
    retry_count: int — Number of retries performed after initial failures.
    failed_request_count: int — Number of request batches that failed permanently.
    input_count: int — Number of input texts embedded for this provider.
    duration_seconds: float — End-to-end execution time in seconds.
    final_batch_size: int — Final batch size used by adaptive controller.
    batch_shrink_events: int — Number of adaptive batch-size decrease events.
    batch_grow_events: int — Number of adaptive batch-size increase events.
    """

    request_count: int
    retry_count: int
    failed_request_count: int
    input_count: int
    duration_seconds: float
    final_batch_size: int = 0
    batch_shrink_events: int = 0
    batch_grow_events: int = 0


@dataclass(frozen=True, slots=True)
class EmbeddingRunStats:
    """
    Aggregate embedding statistics for one embed_batch execution.

    code: EmbeddingProviderStats — Code embedding provider stats.
    description: EmbeddingProviderStats — Description embedding provider stats.
    """

    code: EmbeddingProviderStats
    description: EmbeddingProviderStats

    @property
    def request_count(self) -> int:
        """
        request_count: int — Total requests across code and description embeddings.
        """
        return self.code.request_count + self.description.request_count

    @property
    def retry_count(self) -> int:
        """
        retry_count: int — Total retry attempts across code and description embeddings.
        """
        return self.code.retry_count + self.description.retry_count

    @property
    def failed_request_count(self) -> int:
        """
        failed_request_count: int — Total permanently failed requests.
        """
        return self.code.failed_request_count + self.description.failed_request_count

    @property
    def input_count(self) -> int:
        """
        input_count: int — Total input texts processed.
        """
        return self.code.input_count + self.description.input_count

    @property
    def duration_seconds(self) -> float:
        """
        duration_seconds: float — Combined provider durations.
        """
        return self.code.duration_seconds + self.description.duration_seconds

    @property
    def batch_shrink_events(self) -> int:
        """
        batch_shrink_events: int — Total adaptive batch-size decrease events.
        """
        return self.code.batch_shrink_events + self.description.batch_shrink_events

    @property
    def batch_grow_events(self) -> int:
        """
        batch_grow_events: int — Total adaptive batch-size increase events.
        """
        return self.code.batch_grow_events + self.description.batch_grow_events

    @property
    def final_batch_size(self) -> int:
        """
        final_batch_size: int — Max of final code/description batch sizes.
        """
        return max(self.code.final_batch_size, self.description.final_batch_size)


class DualEmbedder:
    """
    Generates dual embeddings for code chunks.

    Strategy:
    - Code vector: Embed raw source code for exact semantic matching
    - Description vector: Embed NL description for conceptual/intent matching
    - Supports both local (fastembed) and cloud (litellm) providers
    """

    def __init__(
        self,
        code_provider: str,
        code_model: str,
        code_dimension: int,
        desc_provider: str,
        desc_model: str,
        desc_dimension: int,
        code_api_key: Optional[str] = None,
        desc_api_key: Optional[str] = None,
        code_api_base: Optional[str] = None,
        desc_api_base: Optional[str] = None,
        code_batch_size: int = 32,
        desc_batch_size: int = 32,
        code_min_batch_size: int = 8,
        desc_min_batch_size: int = 8,
        code_max_batch_size: int = 128,
        desc_max_batch_size: int = 128,
        code_adaptive_batching: bool = True,
        desc_adaptive_batching: bool = True,
        code_adaptive_target_latency_ms: int = 1500,
        desc_adaptive_target_latency_ms: int = 1500,
        code_concurrency: int = 16,
        desc_concurrency: int = 16,
        code_max_retries: int = 3,
        desc_max_retries: int = 3,
        code_retry_base_delay_ms: int = 250,
        desc_retry_base_delay_ms: int = 250,
        code_retry_max_delay_ms: int = 4000,
        desc_retry_max_delay_ms: int = 4000,
        code_request_timeout_seconds: Optional[float] = None,
        desc_request_timeout_seconds: Optional[float] = None,
        code_openrouter_provider: Optional[str] = None,
        desc_openrouter_provider: Optional[str] = None,
        code_contextual_header: bool = True,
        code_device: str = "auto",
        desc_device: str = "auto",
        code_max_input_tokens: Optional[int] = None,
        desc_max_input_tokens: Optional[int] = None,
    ):
        """
        Args:
            code_provider: Provider for code embeddings ("fastembed" or "litellm")
            code_model: Model name for code embeddings
            code_dimension: Expected dimension for code vectors
            desc_provider: Provider for description embeddings
            desc_model: Model name for description embeddings
            desc_dimension: Expected dimension for description vectors
            code_api_key: API key for code embedding provider (litellm only)
            desc_api_key: API key for description embedding provider (litellm only)
            code_api_base: Optional API base for code embeddings
            desc_api_base: Optional API base for description embeddings
            code_batch_size: Batch size for code embedding requests
            desc_batch_size: Batch size for description embedding requests
            code_min_batch_size: Lower bound for adaptive code batch sizing
            desc_min_batch_size: Lower bound for adaptive description batch sizing
            code_max_batch_size: Upper bound for adaptive code batch sizing
            desc_max_batch_size: Upper bound for adaptive description batch sizing
            code_adaptive_batching: Enable adaptive code batch sizing
            desc_adaptive_batching: Enable adaptive description batch sizing
            code_adaptive_target_latency_ms: Target latency for adaptive code batching
            desc_adaptive_target_latency_ms: Target latency for adaptive description batching
            code_concurrency: Max parallel request workers for code embeddings
            desc_concurrency: Max parallel request workers for description embeddings
            code_max_retries: Max retries per code embedding request batch
            desc_max_retries: Max retries per description embedding request batch
            code_retry_base_delay_ms: Base backoff delay in milliseconds for code retries
            desc_retry_base_delay_ms: Base backoff delay in milliseconds for description retries
            code_retry_max_delay_ms: Max backoff delay in milliseconds for code retries
            desc_retry_max_delay_ms: Max backoff delay in milliseconds for description retries
            code_request_timeout_seconds: Optional per-request timeout for code embeddings
            desc_request_timeout_seconds: Optional per-request timeout for description embeddings
            code_openrouter_provider: Force specific OpenRouter upstream for code embeddings
            desc_openrouter_provider: Force specific OpenRouter upstream for description embeddings
            code_contextual_header: Prepend a file/symbol header to code before embedding
            code_device: Local execution device for code embeddings (auto/cpu/gpu/cuda/directml)
            desc_device: Local execution device for description embeddings
            code_max_input_tokens: Optional token cap for local code embedding inputs
            desc_max_input_tokens: Optional token cap for local description embedding inputs
        """
        self._code_contextual_header = code_contextual_header
        self._code_provider = code_provider
        self._code_model = code_model
        self._code_dimension = code_dimension
        self._desc_provider = desc_provider
        self._desc_model = desc_model
        self._desc_dimension = desc_dimension
        self._code_api_key = code_api_key
        self._desc_api_key = desc_api_key
        self._code_api_base = code_api_base
        self._desc_api_base = desc_api_base
        self._code_batch_size = max(1, int(code_batch_size))
        self._desc_batch_size = max(1, int(desc_batch_size))
        self._code_min_batch_size = max(1, int(code_min_batch_size))
        self._desc_min_batch_size = max(1, int(desc_min_batch_size))
        self._code_max_batch_size = max(self._code_batch_size, int(code_max_batch_size))
        self._desc_max_batch_size = max(self._desc_batch_size, int(desc_max_batch_size))
        self._code_adaptive_batching = bool(code_adaptive_batching)
        self._desc_adaptive_batching = bool(desc_adaptive_batching)
        self._code_adaptive_target_latency_ms = max(100, int(code_adaptive_target_latency_ms))
        self._desc_adaptive_target_latency_ms = max(100, int(desc_adaptive_target_latency_ms))
        self._code_concurrency = max(1, int(code_concurrency))
        self._desc_concurrency = max(1, int(desc_concurrency))
        self._code_max_retries = max(0, int(code_max_retries))
        self._desc_max_retries = max(0, int(desc_max_retries))
        self._code_retry_base_delay_ms = max(1, int(code_retry_base_delay_ms))
        self._desc_retry_base_delay_ms = max(1, int(desc_retry_base_delay_ms))
        self._code_retry_max_delay_ms = max(1, int(code_retry_max_delay_ms))
        self._desc_retry_max_delay_ms = max(1, int(desc_retry_max_delay_ms))
        self._code_request_timeout_seconds = code_request_timeout_seconds
        self._desc_request_timeout_seconds = desc_request_timeout_seconds
        self._code_openrouter_provider = code_openrouter_provider
        self._desc_openrouter_provider = desc_openrouter_provider

        self._code_embedder: Optional["TextEmbedding"] = None
        self._desc_embedder: Optional["TextEmbedding"] = None

        self._last_run_stats = EmbeddingRunStats(
            code=EmbeddingProviderStats(0, 0, 0, 0, 0.0),
            description=EmbeddingProviderStats(0, 0, 0, 0, 0.0),
        )

        self._code_device = "cpu"
        self._desc_device = "cpu"
        if code_provider == "fastembed" or desc_provider == "fastembed":
            from engine.src.providers.local_models import get_local_model

            if code_provider == "fastembed":
                self._code_embedder, self._code_device = get_local_model(code_model, code_device, code_max_input_tokens)
            if desc_provider == "fastembed":
                self._desc_embedder, self._desc_device = get_local_model(desc_model, desc_device, desc_max_input_tokens)

    @staticmethod
    def _local_batch(batch_size: int, device: str) -> int:
        from engine.src.providers.local_models import effective_batch_size

        return effective_batch_size(batch_size, device)

    @property
    def devices(self) -> dict[str, str]:
        """
        devices: dict[str, str] — Resolved local execution device per vector space.
        """
        return {"code": self._code_device, "description": self._desc_device}

    @property
    def last_run_stats(self) -> EmbeddingRunStats:
        """
        last_run_stats: EmbeddingRunStats — Statistics from the most recent embed_batch call.
        """
        return self._last_run_stats

    def embed_batch(
        self,
        chunks: list[CodeChunk],
        descriptions: list[ChunkDescription],
        concurrency_override: Optional[int] = None,
        max_retries_override: Optional[int] = None,
        batch_size_override: Optional[int] = None,
        adaptive_batching_override: Optional[bool] = None,
        progress_callback=None,
    ) -> tuple[list[EmbeddedChunk], float]:
        """
        Generate dual embeddings for a batch of chunks.

        Args:
            chunks: List of code chunks
            descriptions: List of chunk descriptions (must match chunks by chunk_id)
            concurrency_override: Optional override for both providers' concurrency
            max_retries_override: Optional override for both providers' max retries
            batch_size_override: Optional override for initial batch size
            adaptive_batching_override: Optional override for adaptive batch resizing

        Returns:
            Tuple of (embedded chunks, total embedding cost in USD)
        """
        if len(chunks) != len(descriptions):
            raise ValueError("Chunks and descriptions must have same length")

        desc_map = {desc.chunk_id: desc for desc in descriptions}

        code_texts = [
            contextual_code_text(chunk) if self._code_contextual_header else chunk.source
            for chunk in chunks
        ]
        desc_texts = [desc_map[chunk.chunk_id].description for chunk in chunks]
        progress_state = {"code": 0, "description": 0}
        progress_lock = Lock() if self._code_provider == "litellm" or self._desc_provider == "litellm" else None

        def _provider_progress(provider_name: str):
            def _callback(completed: int, total: int) -> None:
                if progress_callback is None:
                    return
                if progress_lock is not None:
                    with progress_lock:
                        progress_state[provider_name] = completed
                        progress_callback(min(progress_state["code"], progress_state["description"]), total)
                else:
                    progress_state[provider_name] = completed
                    progress_callback(min(progress_state["code"], progress_state["description"]), total)
            return _callback

        if self._code_provider == "litellm" or self._desc_provider == "litellm":
            with ThreadPoolExecutor(max_workers=2) as executor:
                code_future = executor.submit(
                    self._embed_code,
                    code_texts,
                    concurrency_override=concurrency_override,
                    max_retries_override=max_retries_override,
                    batch_size_override=batch_size_override,
                    adaptive_batching_override=adaptive_batching_override,
                    progress_callback=_provider_progress("code"),
                )
                desc_future = executor.submit(
                    self._embed_descriptions,
                    desc_texts,
                    concurrency_override=concurrency_override,
                    max_retries_override=max_retries_override,
                    batch_size_override=batch_size_override,
                    adaptive_batching_override=adaptive_batching_override,
                    progress_callback=_provider_progress("description"),
                )
                code_vectors, code_cost, code_stats = code_future.result()
                desc_vectors, desc_cost, desc_stats = desc_future.result()
        else:
            code_vectors, code_cost, code_stats = self._embed_code(
                code_texts,
                concurrency_override=concurrency_override,
                max_retries_override=max_retries_override,
                batch_size_override=batch_size_override,
                adaptive_batching_override=adaptive_batching_override,
                progress_callback=_provider_progress("code"),
            )
            desc_vectors, desc_cost, desc_stats = self._embed_descriptions(
                desc_texts,
                concurrency_override=concurrency_override,
                max_retries_override=max_retries_override,
                batch_size_override=batch_size_override,
                adaptive_batching_override=adaptive_batching_override,
                progress_callback=_provider_progress("description"),
            )

        self._last_run_stats = EmbeddingRunStats(code=code_stats, description=desc_stats)

        total_embedding_cost = code_cost + desc_cost

        embedded: list[EmbeddedChunk] = []
        for i, chunk in enumerate(chunks):
            embedded.append(EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                code_vector=code_vectors[i],
                desc_vector=desc_vectors[i],
                chunk=chunk,
                description=desc_map[chunk.chunk_id],
                embedding_cost_usd=total_embedding_cost / len(chunks),
            ))

        return embedded, total_embedding_cost

    def _embed_code(
        self,
        texts: list[str],
        concurrency_override: Optional[int] = None,
        max_retries_override: Optional[int] = None,
        batch_size_override: Optional[int] = None,
        adaptive_batching_override: Optional[bool] = None,
        progress_callback=None,
    ) -> tuple[list[list[float]], float, EmbeddingProviderStats]:
        """
        Generate code embeddings.

        Args:
            texts: List of source code texts
            concurrency_override: Optional request concurrency override
            max_retries_override: Optional max retries override
            batch_size_override: Optional initial batch size override
            adaptive_batching_override: Optional adaptive batching behavior override

        Returns:
            Tuple of (code vectors, cost in USD, provider stats)
        """
        if self._code_provider == "fastembed":
            vectors, stats = self._embed_fastembed(texts, self._code_embedder, self._local_batch(self._code_batch_size, self._code_device), progress_callback=progress_callback)
            return vectors, 0.0, stats
        if self._code_provider == "litellm":
            return self._embed_litellm(
                texts=texts,
                model=self._code_model,
                api_key=self._code_api_key,
                api_base=self._code_api_base,
                openrouter_provider=self._code_openrouter_provider,
                batch_size=batch_size_override if batch_size_override is not None else self._code_batch_size,
                min_batch_size=self._code_min_batch_size,
                max_batch_size=self._code_max_batch_size,
                adaptive_batching=(
                    adaptive_batching_override
                    if adaptive_batching_override is not None
                    else self._code_adaptive_batching
                ),
                adaptive_target_latency_ms=self._code_adaptive_target_latency_ms,
                concurrency=concurrency_override or self._code_concurrency,
                max_retries=max_retries_override if max_retries_override is not None else self._code_max_retries,
                retry_base_delay_ms=self._code_retry_base_delay_ms,
                retry_max_delay_ms=self._code_retry_max_delay_ms,
                request_timeout_seconds=self._code_request_timeout_seconds,
                progress_callback=progress_callback,
            )
        raise ValueError(f"Unknown code provider: {self._code_provider}")

    def _embed_descriptions(
        self,
        texts: list[str],
        concurrency_override: Optional[int] = None,
        max_retries_override: Optional[int] = None,
        batch_size_override: Optional[int] = None,
        adaptive_batching_override: Optional[bool] = None,
        progress_callback=None,
    ) -> tuple[list[list[float]], float, EmbeddingProviderStats]:
        """
        Generate description embeddings.

        Args:
            texts: List of description texts
            concurrency_override: Optional request concurrency override
            max_retries_override: Optional max retries override
            batch_size_override: Optional initial batch size override
            adaptive_batching_override: Optional adaptive batching behavior override

        Returns:
            Tuple of (description vectors, cost in USD, provider stats)
        """
        if self._desc_provider == "fastembed":
            vectors, stats = self._embed_fastembed(texts, self._desc_embedder, self._local_batch(self._desc_batch_size, self._desc_device), progress_callback=progress_callback)
            return vectors, 0.0, stats
        if self._desc_provider == "litellm":
            return self._embed_litellm(
                texts=texts,
                model=self._desc_model,
                api_key=self._desc_api_key,
                api_base=self._desc_api_base,
                openrouter_provider=self._desc_openrouter_provider,
                batch_size=batch_size_override if batch_size_override is not None else self._desc_batch_size,
                min_batch_size=self._desc_min_batch_size,
                max_batch_size=self._desc_max_batch_size,
                adaptive_batching=(
                    adaptive_batching_override
                    if adaptive_batching_override is not None
                    else self._desc_adaptive_batching
                ),
                adaptive_target_latency_ms=self._desc_adaptive_target_latency_ms,
                concurrency=concurrency_override or self._desc_concurrency,
                max_retries=max_retries_override if max_retries_override is not None else self._desc_max_retries,
                retry_base_delay_ms=self._desc_retry_base_delay_ms,
                retry_max_delay_ms=self._desc_retry_max_delay_ms,
                request_timeout_seconds=self._desc_request_timeout_seconds,
                progress_callback=progress_callback,
            )
        raise ValueError(f"Unknown description provider: {self._desc_provider}")

    def _embed_fastembed(
        self,
        texts: list[str],
        embedder: "TextEmbedding",
        batch_size: int,
        progress_callback=None,
    ) -> tuple[list[list[float]], EmbeddingProviderStats]:
        """
        Generate embeddings using fastembed.

        Args:
            texts: List of texts to embed
            embedder: Fastembed TextEmbedding instance
            batch_size: Batch size for local embedding calls

        Returns:
            Tuple of (embedding vectors, provider stats)
        """
        started = time.time()
        if not texts:
            return [], EmbeddingProviderStats(0, 0, 0, 0, 0.0)

        vectors = [emb.tolist() for emb in embed_length_sorted(embedder, texts, batch_size)]
        duration = time.time() - started
        if progress_callback is not None:
            progress_callback(len(texts), len(texts))

        return vectors, EmbeddingProviderStats(
            request_count=1,
            retry_count=0,
            failed_request_count=0,
            input_count=len(texts),
            duration_seconds=duration,
        )

    def _embed_litellm(
        self,
        texts: list[str],
        model: str,
        api_key: Optional[str],
        api_base: Optional[str],
        openrouter_provider: Optional[str],
        batch_size: int,
        min_batch_size: int,
        max_batch_size: int,
        adaptive_batching: bool,
        adaptive_target_latency_ms: int,
        concurrency: int,
        max_retries: int,
        retry_base_delay_ms: int,
        retry_max_delay_ms: int,
        request_timeout_seconds: Optional[float],
        progress_callback=None,
    ) -> tuple[list[list[float]], float, EmbeddingProviderStats]:
        """
        Generate embeddings using parallel litellm calls with retry/backoff.

        Args:
            texts: List of texts to embed.
            model: Model name.
            api_key: API key for provider.
            api_base: Optional provider endpoint override.
            openrouter_provider: Optional OpenRouter upstream hint.
            batch_size: Max texts per request.
            min_batch_size: Lower bound for adaptive batch resizing.
            max_batch_size: Upper bound for adaptive batch resizing.
            adaptive_batching: Enable adaptive batch resizing when True.
            adaptive_target_latency_ms: Target request latency used by adaptive controller.
            concurrency: Max parallel request workers.
            max_retries: Max retries per failed request batch.
            retry_base_delay_ms: Exponential backoff base delay.
            retry_max_delay_ms: Exponential backoff max delay.
            request_timeout_seconds: Optional timeout passed to litellm.

        Returns:
            Tuple of (vectors, total_cost_usd, provider_stats).
        """
        started = time.time()
        if not texts:
            return [], 0.0, EmbeddingProviderStats(0, 0, 0, 0, 0.0)

        effective_concurrency = max(1, int(concurrency))
        effective_min_batch = max(1, int(min_batch_size))
        effective_max_batch = max(effective_min_batch, int(max_batch_size))
        current_batch_size = min(max(1, int(batch_size)), effective_max_batch)
        token_budget = (
            max(REMOTE_MIN_TOKEN_BUDGET, current_batch_size * REMOTE_TOKENS_PER_SLOT)
            if adaptive_batching
            else None
        )
        batches = plan_remote_batches(texts, current_batch_size, token_budget)

        vectors: list[list[float] | None] = [None] * len(texts)
        total_cost = 0.0
        total_retries = 0
        failed_requests = 0
        request_count = 0
        completed_inputs = 0

        with ThreadPoolExecutor(max_workers=min(effective_concurrency, len(batches))) as executor:
            future_map = {
                executor.submit(
                    self._embed_litellm_batch,
                    batch_texts=[texts[idx] for idx in indexes],
                    model=model,
                    api_key=api_key,
                    api_base=api_base,
                    openrouter_provider=openrouter_provider,
                    max_retries=max_retries,
                    retry_base_delay_ms=retry_base_delay_ms,
                    retry_max_delay_ms=retry_max_delay_ms,
                    request_timeout_seconds=request_timeout_seconds,
                ): indexes
                for indexes in batches
            }
            try:
                for future in as_completed(future_map):
                    indexes = future_map[future]
                    try:
                        batch_vectors, batch_cost, batch_retries, _ = future.result()
                    except Exception:
                        failed_requests += 1
                        raise
                    total_cost += batch_cost
                    total_retries += batch_retries
                    request_count += 1
                    completed_inputs += len(batch_vectors)
                    for idx, vector in zip(indexes, batch_vectors):
                        vectors[idx] = vector
                    if progress_callback is not None:
                        progress_callback(completed_inputs, len(texts))
            except BaseException:
                for pending in future_map:
                    pending.cancel()
                raise

        if any(vec is None for vec in vectors):
            raise RuntimeError("Embedding response missing vectors for one or more inputs")

        duration = time.time() - started

        return [vec for vec in vectors if vec is not None], total_cost, EmbeddingProviderStats(
            request_count=request_count,
            retry_count=total_retries,
            failed_request_count=failed_requests,
            input_count=len(texts),
            duration_seconds=duration,
            final_batch_size=current_batch_size,
        )

    def _embed_litellm_batch(
        self,
        batch_texts: list[str],
        model: str,
        api_key: Optional[str],
        api_base: Optional[str],
        openrouter_provider: Optional[str],
        max_retries: int,
        retry_base_delay_ms: int,
        retry_max_delay_ms: int,
        request_timeout_seconds: Optional[float],
    ) -> tuple[list[list[float]], float, int, float]:
        """
        Embed one request batch with retry and jittered exponential backoff.

        batch_texts: list[str] — Input texts for this request.
        model: str — Embedding model identifier.
        api_key: Optional[str] — API key for the provider.
        api_base: Optional[str] — Optional custom API endpoint.
        openrouter_provider: Optional[str] — Optional OpenRouter provider hint.
        max_retries: int — Max retry attempts after initial request.
        retry_base_delay_ms: int — Base backoff delay in milliseconds.
        retry_max_delay_ms: int — Max backoff delay in milliseconds.
        request_timeout_seconds: Optional[float] — Per-request timeout.
        Returns: tuple[list[list[float]], float, int, float] — Vectors, batch cost, retries used, request duration.
        """
        retries_used = 0
        litellm = _get_litellm()
        request_started = time.time()

        for attempt in range(max_retries + 1):
            kwargs = {
                "model": model,
                "input": batch_texts,
            }

            if api_key:
                kwargs["api_key"] = api_key
            if api_base:
                kwargs["api_base"] = api_base
            if request_timeout_seconds is not None:
                kwargs["timeout"] = float(request_timeout_seconds)
            if openrouter_provider:
                kwargs["extra_body"] = {
                    "provider": {
                        "order": [openrouter_provider],
                    }
                }

            try:
                response = litellm.embedding(**kwargs)
                vectors = self._vectors_from_response(batch_texts, response)

                cost = 0.0
                try:
                    cost = litellm.completion_cost(completion_response=response)
                except Exception:
                    pass

                return vectors, cost, retries_used, time.time() - request_started
            except Exception:
                if attempt >= max_retries:
                    raise

                retries_used += 1

                base_delay = max(1, int(retry_base_delay_ms)) / 1000.0
                max_delay = max(base_delay, int(retry_max_delay_ms) / 1000.0)
                backoff = min(max_delay, base_delay * (2 ** attempt))
                jitter = backoff * random.uniform(0.0, 0.2)
                time.sleep(backoff + jitter)

        raise RuntimeError("Embedding batch retry loop exited unexpectedly")

    def _vectors_from_response(self, batch_texts: list[str], response) -> list[list[float]]:
        """
        Extract vectors from a litellm embedding response while preserving input order.

        batch_texts: list[str] — Request batch input texts.
        response: object — litellm embedding response.
        Returns: list[list[float]] — Vectors aligned to batch input order.
        """
        response_data = getattr(response, "data", None)
        if not response_data:
            raise RuntimeError("Embedding response is missing data")

        if all(isinstance(item, dict) and "index" in item for item in response_data):
            ordered: list[list[float] | None] = [None] * len(batch_texts)
            for item in response_data:
                index = int(item["index"])
                if index < 0 or index >= len(batch_texts):
                    raise RuntimeError("Embedding response returned an out-of-range index")
                ordered[index] = item["embedding"]

            if any(vec is None for vec in ordered):
                raise RuntimeError("Embedding response omitted one or more expected vectors")

            return [vec for vec in ordered if vec is not None]

        vectors = [item["embedding"] for item in response_data]
        if len(vectors) != len(batch_texts):
            raise RuntimeError("Embedding response vector count does not match batch size")
        return vectors
