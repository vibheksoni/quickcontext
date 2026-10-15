# Embeddings

## Purpose

`code_embedding` and `desc_embedding` choose the models that turn code chunks, descriptions,
agent documents and queries into vectors. Two providers are supported:

- `fastembed`: local ONNX models, no API key, no network. Default for `quickcontext.local.example.json`.
- `litellm`: any hosted embedding API LiteLLM supports (OpenRouter, OpenAI, Voyage, Cohere,
  self-hosted TEI/vLLM through `api_base`). Default for `quickcontext.example.json`.

Changing a model or dimension changes the vector space. The next `index` run detects it and
rebuilds the project into a fresh collection.

## Profiles: Switch Local And Remote Without Editing The Config

A `profiles` map holds named overrides that merge over the base config:

```json
{
  "code_embedding": { "provider": "litellm", "model": "openrouter/qwen/qwen3-embedding-8b", "dimension": 4096, "api_key": "..." },
  "desc_embedding": { "provider": "litellm", "model": "openrouter/qwen/qwen3-embedding-8b", "dimension": 4096, "api_key": "..." },
  "profiles": {
    "local": {
      "code_embedding": { "provider": "fastembed", "model": "BAAI/bge-small-en-v1.5", "dimension": 384 },
      "desc_embedding": { "provider": "fastembed", "model": "BAAI/bge-small-en-v1.5", "dimension": 384 },
      "llm": null
    }
  }
}
```

Select a profile with any of:

- CLI: `python -m engine --profile local index . --project my-repo`
- MCP: `python -m qc_mcp --profile local`
- environment: `QC_PROFILE=local`
- config: `"profile": "local"` at the top level
- SDK: `EngineConfig.from_json(path, profile="local")`

Rules:

- A profile section is merged key by key over the base section, so `{"code_embedding": {"concurrency": 64}}`
  only changes concurrency.
- An embedding section that sets `model` or `provider` replaces the base section instead. API keys,
  query prefixes and other model-specific settings never carry over to a different model.
- `vectors` is re-derived from the new dimensions unless the profile sets it.
- Without an explicit `store.path`, each profile gets its own database (`store-<profile>.db`
  next to the default `store.db`), so switching profiles never overwrites another profile's index.

## Local Models (fastembed)

```json
"code_embedding": {
  "provider": "fastembed",
  "model": "BAAI/bge-small-en-v1.5",
  "dimension": 384,
  "batch_size": 16,
  "device": "auto",
  "max_input_tokens": null
}
```

- One model instance per process is shared by indexing, search and agent documents, so code and
  description spaces that use the same model load it once.
- Inputs are embedded in length-sorted batches, which roughly halves CPU time compared with
  input-order batches (padding is wasted work).
- `batch_size` is the main memory knob. 16 is a good CPU default.

### GPU (opt-in)

`device` accepts `auto` (CUDA, then DirectML, then CPU), `cpu`, `gpu`, `cuda` or `directml`.
GPU needs a GPU build of ONNX Runtime in place of the CPU one:

```powershell
pip uninstall -y onnxruntime
pip install onnxruntime-directml   # any DX12 GPU on Windows, ~300 MB
# or: pip install onnxruntime-gpu  # NVIDIA CUDA, ~3 GB with the CUDA/cuDNN wheels
```

A GPU that fails to initialise falls back to CPU with a warning. On a GPU device the batch size
is raised to at least 64.

Measured with bge-small-en-v1.5 on 512 code chunks: CPU 15.4 texts/s, DirectML 184 texts/s
(about 12x), with cosine similarity of at least 0.99999 between CPU and GPU vectors.

### Input Truncation

`max_input_tokens` caps the tokens embedded per input (default: the model's own limit, 512 for
bge-small). Smaller caps are faster but lose the tail of long chunks. On stealth-browser-mcp
(1,316 chunks, bge-small, CPU):

| Cap | Index time | Semantic file / symbol Hit@1 | Auto file / symbol Hit@1 |
| --- | --- | --- | --- |
| 512 (default) | 101 s | 33 / 26 of 36 | 34 / 31 of 36 |
| 256 | 79 s | 32 / 24 of 36 | 33 / 30 of 36 |

Leave it unset unless indexing speed or memory matters more than recall.

## Remote Models (litellm)

```json
"code_embedding": {
  "provider": "litellm",
  "model": "openrouter/qwen/qwen3-embedding-8b",
  "dimension": 4096,
  "batch_size": 32,
  "concurrency": 32,
  "api_key": "...",
  "query_prefix": "Instruct: Given a question about a codebase, retrieve the code that answers it\nQuery:"
}
```

- Requests run on a continuous queue: up to `concurrency` requests are in flight and a new one
  starts as soon as any finishes. Default `concurrency` is 16 (32 with `--fast`). Rate-limit and
  transient errors are retried with exponential backoff (`max_retries`, `retry_base_delay_ms`,
  `retry_max_delay_ms`).
- With `adaptive_batching: true` (default), requests are packed by estimated tokens, longest texts
  first, so request sizes stay even and no slow straggler holds up the end of the run. With
  `false`, every request holds exactly `batch_size` texts. `adaptive_target_latency_ms` is
  deprecated and ignored.
- `openrouter_provider` pins an OpenRouter upstream (for example `"Nebius"`) for indexing and queries.

Measured indexing of stealth-browser-mcp (1,316 chunks, qwen3-embedding-8b through OpenRouter):

| Setup | Index time |
| --- | --- |
| Old wave-based batching, concurrency 4 | 90.7 s |
| Continuous queue, concurrency 4 | 60.1 s |
| Continuous queue, concurrency 32 | 24.5 s |
| Continuous queue, concurrency 96 | 21.5 s |

None of these runs needed retries. Above about 32 the provider's own throughput is the limit.

### Query Prefix

Instruction-tuned embedding models (Qwen3-Embedding, E5, GTE-Qwen) expect queries to carry a task
instruction that documents do not. `query_prefix` is prepended to search queries only, never to
indexed text, so changing it does not require re-indexing.

On stealth-browser-mcp with qwen3-embedding-8b, the prefix above moved semantic search from
32/36 to 33/36 file Hit@1, and the default `auto` mode from 32/36 to 33/36 file and 29/36 to
30/36 symbol Hit@1. For bge-small, the BGE instruction ("Represent this sentence for searching
relevant passages: ") made results slightly worse (auto 34/31 to 33/30), so local configs leave
it unset.

## Choosing Between Local And Remote

On the same 36-question stealth-browser-mcp benchmark (`auto` retrieval):

| | Local bge-small (CPU) | Remote qwen3-embedding-8b |
| --- | --- | --- |
| File / symbol Hit@1 | 34 / 31 | 33 / 30 (with query prefix) |
| Index time | 101 s (embedding is ~12x faster with DirectML) | 21-25 s |
| Query latency (new query) | ~15-20 ms | ~0.3-2 s (API round trip) |
| Vector memory per 20k chunks | ~61 MB | ~655 MB |

Hybrid retrieval (text index, symbol index and file fusion) carries most of the quality, so a
small local model is enough for interactive agent use. Remote models make bulk indexing fast
without a GPU. A practical setup is a remote base config with a `local` profile, or the
reverse, selected per run with `--profile`.
