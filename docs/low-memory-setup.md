# Low Memory and Hardware Setup Guide

Most indexing memory goes to the local embedding model (`fastembed`): its activations grow with
`batch_size` and with input length. Remote (`litellm`) embeddings use almost no local memory
beyond the vector store itself.

If you are running on a machine with limited RAM, restrict memory usage through `quickcontext.json`.

## 1. Use Small Local Batches

For `fastembed`, `batch_size` sets how many inputs go through the model at once. Inputs are
already sorted by length, so small batches stay efficient. Use `8` or `16` on constrained machines
and keep `device` on `cpu` (a GPU device raises the batch size to at least 64):

```json
"code_embedding": {
  "provider": "fastembed",
  "batch_size": 8,
  "device": "cpu"
}
```

## 2. Cap Input Length

`max_input_tokens` truncates each input before embedding. Activation memory and time grow with
input length, so `256` cuts both, at a small recall cost (see
[embeddings.md](embeddings.md#input-truncation) for measured numbers).

For remote embeddings, `concurrency` is the number of requests in flight. Lower it if your
provider rate-limits you; it has little effect on local memory.

## 3. Size The Vector Store

The vector store has no separate server process. Its memory cost is the in-memory
matrices of the collections you query:

```text
points x (code dimension + description dimension) x 4 bytes
```

For example, 20k chunks with 768 + 384 dimensions use about 92 MB, while 4096 + 4096
dimensions use about 655 MB. To keep memory low:

- prefer smaller embedding models (dimension is the main lever)
- lower `store.max_cached_collections` so fewer collections stay resident
- lower `store.upsert_batch_size` to reduce peak memory while indexing

---

## Example: Conservative `quickcontext.json`

```json
{
  "store": {
    "upsert_batch_size": 64,
    "max_cached_collections": 1
  },
  "code_embedding": {
    "provider": "fastembed",
    "batch_size": 8,
    "device": "cpu",
    "max_input_tokens": 256
  },
  "desc_embedding": {
    "provider": "fastembed",
    "batch_size": 8,
    "device": "cpu",
    "max_input_tokens": 256
  }
}
```

> **Trade-off**: These limits keep peak memory low. Small batches make indexing a little slower, and the 256-token cap costs a little recall (it also makes indexing faster). Drop `max_input_tokens` if recall matters more than memory.
