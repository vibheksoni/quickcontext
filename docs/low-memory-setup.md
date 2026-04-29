# Low Memory and Hardware Setup Guide

QuickContext is designed by default to maximize resource utilization to index your codebase as quickly as possible. It uses parallel workers and an **adaptive batching** mechanism that detects hardware speed and automatically scales batch sizes up to `128`. 

When running local models via `fastembed`, this aggressive scaling can easily consume upwards of 5GB+ of system memory, leading to potential Out-Of-Memory (OOM) kills on constrained environments.

If you are running on a machine with limited RAM, you can rigidly restrict memory usage through `quickcontext.json`.

## 1. Disable Adaptive Batching

Adaptive batching will dynamically override your `batch_size` flag if it thinks the system is fast enough. You must explicitly disable it via the JSON config to keep RAM usage predictable. 

Lock your batch limits for both code and description embeddings tightly (e.g., `8` or `16`):

```json
"code_embedding": {
  "adaptive_batching": false,
  "batch_size": 8,
  "max_batch_size": 8
}
```

## 2. Reduce Parallelism (Concurrency)

QuickContext uses multiple background threads for generating the vector embeddings and another set of threads for pushing data points into the Qdrant database.

Lowering the thread pool size limits memory allocations:

- `concurrency`: Controls the number of parallel workers generating embeddings.
- `upsert_concurrency`: Controls the number of concurrent writes to the database.

## 3. Tune Qdrant Docker Allocations

The default `docker-compose.yml` pre-allocates up to `4G` for Qdrant. If you're managing a small-to-medium codebase, you can restrict Qdrant's container to less memory within `docker-compose.yml`:

```yaml
    deploy:
      resources:
        limits:
          memory: 2G
```

---

## Example: Conservative `quickcontext.json`

Here is a full configuration snippet designed to run on constrained machines. It disables automatic scaling and forces strict, single-threaded processing and small chunk batches.

```json
{
  "qdrant": {
    "host": "localhost",
    "port": 6333,
    "upsert_batch_size": 50,
    "upsert_concurrency": 1
  },
  "code_embedding": {
    "provider": "fastembed",
    "adaptive_batching": false,
    "batch_size": 8,
    "max_batch_size": 8,
    "concurrency": 1
  },
  "desc_embedding": {
    "provider": "fastembed",
    "adaptive_batching": false,
    "batch_size": 8,
    "max_batch_size": 8,
    "concurrency": 1
  }
}
```

> **Performance Trade-off**: Enforcing these low memory limits will completely resolve the massive RAM hoarding/OOM kills, but your initial repository indexing phase will take noticeably longer to complete.
