# Vector Store And Agent Documents

## Purpose

quickcontext keeps all vectors in an embedded store: one SQLite database file plus
in-memory numpy matrices. There is no database server, no Docker, and no network
hop on the query path.

The same store backs two things:

- code collections built by `index` / `refresh` (one per project)
- document collections that agents fill with `collect` and query with `recall`

## How It Works

- **Durable storage.** SQLite in WAL mode holds collections, aliases, point payloads, the
  large text field (`source` for code, `text` for documents) in a separate column, and
  each point's vectors as one float32 blob.
- **Exact search.** On the first query, a collection's vectors are loaded into one
  contiguous float32 matrix per vector space. A query is a single matrix-vector
  product plus a partial sort. Results are exact rather than approximate, so there is no
  recall loss from HNSW tuning.
- **Incremental freshness.** Every write bumps a per-collection revision and stamps the
  rows it touched, and deletes leave tombstones. Before each query the store runs one
  tiny revision lookup. If anything changed, from this process or any other, only the
  changed rows are applied to the in-memory matrix. Rows that are replaced or deleted are
  masked out until enough of them build up to justify compaction.
- **Filters.** Keyword fields (`language`, `file_path`, `symbol_kind`, `kind`, `source`,
  `tags`, or every metadata key for document collections) and path scopes are
  evaluated as numpy masks and cached per snapshot.
- **Full text.** Document collections also get an FTS5 BM25 index, so `recall` can fuse
  semantic and keyword ranking, or run keyword-only with no embedding model at all.
- **Snapshot isolation for re-indexing.** Code projects use versioned collections
  (`project_v1`, `project_v2`, ...) behind an alias. A full re-index builds a shadow
  version, copies unchanged points inside SQLite, and swaps the alias atomically.

## Measured Performance

Measured on a Windows desktop with Python 3.11 and numpy 2.4: 20,000 points with
768-d code vectors and 384-d description vectors.

| Operation | Time |
| --- | --- |
| Write 20,000 points (512 per transaction) | ~0.9 s |
| Cold load of the collection into memory | ~175 ms |
| Unfiltered top-20 search | ~2.5 ms |
| Filtered search (language + path scope), mask cached | ~3.5 ms |
| Reader process picks up a 14-row delete + 1 upsert and searches | ~15-20 ms |

Search cost grows linearly with `points x dimension`. Memory use is roughly
`points x (sum of dimensions) x 4 bytes`. For example, 20k points x (768 + 384) is about 92 MB,
and 20k points x (4096 + 4096) is about 655 MB. Smaller embedding models are the main
lever for very large indexes.

## Contextual Code Embeddings

Code chunks are embedded with a two-line header: the trailing file path, then
`kind: Parent.symbol`. This is a deterministic, zero-cost version of contextual
retrieval. An A/B run on this repository (bge-small for both vector spaces,
`--no-descriptions`, default hybrid search) measured:

| Case set | Hit@1 without -> with | MRR without -> with |
| --- | --- | --- |
| extended (20) | 14 -> 17 | 0.825 -> 0.900 |
| deep (16) | 13 -> 13 | 0.844 -> 0.875 |
| hard (12) | 10 -> 10 | 0.875 -> 0.917 |

Set `code_embedding.contextual_header: false` to turn it off. Re-index after changing it.

## Configuration

```json
{
  "store": {
    "path": null,
    "upsert_batch_size": 512,
    "busy_timeout_seconds": 30.0,
    "max_cached_collections": 4
  }
}
```

- `path`: database file. `null` uses `%LOCALAPPDATA%\quickcontext\store.db` on Windows,
  `$XDG_DATA_HOME/quickcontext/store.db` or `~/.local/share/quickcontext/store.db`
  elsewhere. Relative paths resolve against the config file's directory. `":memory:"`
  keeps everything in RAM. The `QC_STORE_PATH` environment variable also overrides the default.
- `max_cached_collections`: how many collections keep their matrices resident.
- `"store": null` disables vector features entirely. Parser, grep, symbol, and text
  tools keep working.

Older configs with a `"qdrant"` section still load. Its `collection` and
`upsert_batch_size` keys carry over, and the host and port settings are ignored.
Existing Qdrant data is not migrated, so re-run `index` once.

## Agent Documents API

```python
from engine import QuickContext

with QuickContext() as qc:
    qc.collect("research", [
        "Webhook deliveries retry with exponential backoff",
        {"text": "Auth tokens are cached in Redis for 15 minutes",
         "tags": ["auth"], "source": "docs/auth.md", "priority": 2},
    ], tags="sprint-12", kind="finding")

    for doc in qc.recall("research", "how long do auth tokens live?", limit=5):
        print(doc.score, doc.text, doc.tags, doc.source, doc.metadata)

    qc.recall("research", "redis", mode="keyword", tags=["auth"])
    qc.recall("research", "tokens", where={"priority": 2})
    qc.forget("research", tags="sprint-12")
```

- IDs are content hashes unless you pass `"id"`, so collecting the same text twice
  updates the existing document instead of duplicating it.
- `collect(..., dedupe_threshold=0.95)` merges a near-identical new document into the
  existing one (tags are unioned) instead of storing a near-duplicate.
- `mode="hybrid"` (default) fuses vector and BM25 rankings with weighted reciprocal
  rank fusion (`semantic_weight=0.7`, k=60, top 5x `limit` candidates from each side),
  `"semantic"` uses vectors only, and `"keyword"` uses BM25 only.
- BM25 indexes the text plus its `source`, `kind`, and `tags` ("contextual BM25"), and
  stopwords are dropped from keyword queries.
- `diversity=0.3` applies maximal marginal relevance so near-duplicate hits do not
  crowd out other results; `half_life_days=7` decays scores of older documents.
- Collections created while `desc_embedding` is unset, or with
  `qc.documents.create(name, semantic=False)`, are keyword-only and need no model.
- `qc.documents` also exposes `get`, `list`, `count`, `delete`, `drop`, and `collections`.

The same operations are available as CLI commands (`collect`, `recall`, `forget`,
`collections`) and MCP tools (`collect`, `recall`, `forget`, `collections`).

## Maintenance

- `python -m engine audit`: integrity check, per-collection counts, and vector
  dimensions compared with the active embedding config.
- `python -m engine gc`: delete orphaned shadow versions left by interrupted indexes.
- `python -m engine compact [--vacuum]`: merge FTS segments, checkpoint the WAL, and
  optionally reclaim disk space.
- `python -m engine repair`: re-point broken project aliases and remove stale versions.
