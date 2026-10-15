"""
Shared local (fastembed / ONNX Runtime) embedding models.

One model instance per (model, device, input cap) is loaded per process and
shared by indexing and query embedding. Devices resolve to ONNX Runtime
execution providers: CUDA when onnxruntime-gpu is installed, DirectML when
onnxruntime-directml is installed (any DX12 GPU on Windows), otherwise CPU.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

_LOGGER = logging.getLogger(__name__)
_LOCK = threading.Lock()
_MODELS: dict[tuple[str, str, Optional[int]], Any] = {}
_DEVICE_PROVIDERS = {
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
}
GPU_MIN_BATCH_SIZE = 64


def available_providers() -> list[str]:
    """
    ONNX Runtime execution providers usable in this environment.
    """
    try:
        import onnxruntime
    except ImportError:
        return []
    return list(onnxruntime.get_available_providers())


def _device_candidates(device: str) -> list[str]:
    requested = (device or "auto").strip().lower()
    if requested in ("cpu", ""):
        return ["cpu"]
    if requested in _DEVICE_PROVIDERS:
        return [requested, "cpu"]
    if requested in ("auto", "gpu"):
        providers = available_providers()
        candidates = [name for name, provider in _DEVICE_PROVIDERS.items() if provider in providers]
        return candidates + ["cpu"]
    raise ValueError(f"Unknown embedding device {device!r}; use auto, cpu, gpu, cuda, or directml")


def _create(model_name: str, device: str):
    from fastembed import TextEmbedding

    if device == "cpu":
        return TextEmbedding(model_name, providers=["CPUExecutionProvider"])
    if device == "cuda":
        import onnxruntime

        preload = getattr(onnxruntime, "preload_dlls", None)
        if callable(preload):
            preload()
    return TextEmbedding(model_name, providers=[_DEVICE_PROVIDERS[device], "CPUExecutionProvider"])


def _apply_input_cap(model: Any, max_input_tokens: Optional[int]) -> None:
    if not max_input_tokens:
        return
    tokenizer = getattr(getattr(model, "model", None), "tokenizer", None)
    if tokenizer is None:
        return
    current = getattr(tokenizer, "truncation", None) or {}
    limit = min(int(max_input_tokens), int(current.get("max_length") or max_input_tokens))
    tokenizer.enable_truncation(max_length=max(8, limit))


def get_local_model(model_name: str, device: str = "auto", max_input_tokens: Optional[int] = None) -> tuple[Any, str]:
    """
    Return (TextEmbedding, resolved_device) for a model, loading it once per process.

    A requested GPU that fails to initialise falls back to the next candidate
    (ending with CPU) instead of breaking indexing.
    """
    errors: list[str] = []
    for candidate in _device_candidates(device):
        key = (model_name, candidate, max_input_tokens or None)
        with _LOCK:
            cached = _MODELS.get(key)
            if cached is not None:
                return cached, candidate
            try:
                model = _create(model_name, candidate)
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                continue
            _apply_input_cap(model, max_input_tokens)
            _MODELS[key] = model
        if errors:
            _LOGGER.warning("Embedding device fallback for %s: %s", model_name, "; ".join(errors))
        return model, candidate
    raise RuntimeError(f"Could not load embedding model {model_name}: {'; '.join(errors)}")


def effective_batch_size(batch_size: int, device: str) -> int:
    """
    GPUs need larger batches to stay busy; CPU keeps the configured size.
    """
    batch = max(1, int(batch_size))
    return max(batch, GPU_MIN_BATCH_SIZE) if device != "cpu" else batch
