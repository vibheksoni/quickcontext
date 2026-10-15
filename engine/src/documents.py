"""
Agent-facing document collections: collect anything, recall it fast.

Documents are free text plus flat metadata (tags, source, kind, and any custom
keys) stored in the embedded vector store. Every collection gets BM25 full-text
search over the text plus its source, kind, and tags ("contextual BM25"); when a
description embedding provider is configured it also gets an exact vector index,
and recall fuses both with weighted reciprocal rank fusion. Optional MMR
diversification, recency decay, and near-duplicate merging cover the common
agent-memory failure modes (redundant hits, stale notes, duplicate writes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import time
from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, Sequence

import numpy as np

from engine.src.config import CollectionVectorConfig
from engine.src.searcher import embed_query
from engine.src.vector_store import (
    CollectionInfo,
    CollectionNotFoundError,
    PointRecord,
    SearchFilter,
    SearchRequest,
)

if TYPE_CHECKING:
    from engine.sdk import QuickContext


DOCUMENT_KIND = "documents"
TEXT_FIELD = "text"
VECTOR_NAME = "text"
RESERVED_FIELDS = ("text", "tags", "source", "kind", "added_at")
RRF_K = 60
DEFAULT_SEMANTIC_WEIGHT = 0.7
MIN_CANDIDATES = 50
RecallMode = Literal["hybrid", "semantic", "keyword"]


@dataclass(frozen=True, slots=True)
class Document:
    """
    One stored document, optionally with a recall score.

    id: str — Stable document ID (content hash unless supplied).
    text: str — Document body.
    score: float — Relevance score from recall (0.0 for plain reads).
    tags: list[str] — Free-form labels.
    source: Optional[str] — Where the content came from (URL, file, tool name...).
    kind: Optional[str] — Content type (note, finding, snippet, doc...).
    metadata: dict — Remaining custom fields.
    added_at: Optional[float] — Unix timestamp of the last write.
    """

    id: str
    text: str
    score: float = 0.0
    tags: list[str] = field(default_factory=list)
    source: Optional[str] = None
    kind: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    added_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "tags": list(self.tags),
            "source": self.source,
            "kind": self.kind,
            "metadata": dict(self.metadata),
            "added_at": self.added_at,
        }


def document_id(collection: str, text: str) -> str:
    """
    Deterministic content-addressed ID so re-collecting identical text is idempotent.
    """
    return hashlib.sha256(f"{collection}\0{text}".encode("utf-8")).hexdigest()[:32]


def _normalize_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        tags = [tags]
    seen: dict[str, None] = {}
    for tag in tags:
        value = str(tag).strip()
        if value:
            seen[value] = None
    return list(seen)


def _search_text(payload: dict) -> str:
    """
    Text indexed for BM25: the body plus the context that situates it.
    """
    context = [str(payload.get(key)) for key in ("source", "kind") if payload.get(key)]
    context.extend(str(tag) for tag in payload.get("tags") or [])
    if not context:
        return payload[TEXT_FIELD]
    return payload[TEXT_FIELD] + "\n" + " ".join(context)


def _mmr_order(
    candidates: list[tuple[str, float]],
    vectors: dict[str, np.ndarray],
    diversity: float,
    limit: int,
) -> list[tuple[str, float]]:
    """
    Maximal marginal relevance: greedily pick the candidate maximizing
    (1 - diversity) * relevance - diversity * max similarity to already-picked items.
    Relevance is min-max normalized so fused and raw scores behave the same.
    """
    if not candidates:
        return []
    scores = np.array([score for _, score in candidates], dtype=np.float64)
    span = scores.max() - scores.min()
    relevance = (scores - scores.min()) / span if span > 0 else np.ones_like(scores)
    picked: list[int] = []
    max_sim = np.zeros(len(candidates), dtype=np.float64)
    remaining = set(range(len(candidates)))
    while remaining and len(picked) < limit:
        best = max(remaining, key=lambda idx: (1.0 - diversity) * relevance[idx] - diversity * max_sim[idx])
        picked.append(best)
        remaining.discard(best)
        chosen = vectors.get(candidates[best][0])
        if chosen is None:
            continue
        for idx in remaining:
            other = vectors.get(candidates[idx][0])
            if other is not None:
                max_sim[idx] = max(max_sim[idx], float(np.dot(chosen, other)))
    return [candidates[idx] for idx in picked]


def _to_document(point_id: str, payload: dict, score: float = 0.0) -> Document:
    metadata = {key: value for key, value in payload.items() if key not in RESERVED_FIELDS}
    return Document(
        id=point_id,
        text=str(payload.get(TEXT_FIELD, "")),
        score=float(score),
        tags=list(payload.get("tags") or []),
        source=payload.get("source"),
        kind=payload.get("kind"),
        metadata=metadata,
        added_at=payload.get("added_at"),
    )


class DocumentStore:
    """
    Collect, recall, and forget documents in named collections.

    Obtain through QuickContext.documents, or use the QuickContext.collect /
    recall / forget shortcuts.
    """

    def __init__(self, qc: "QuickContext"):
        self._qc = qc

    @property
    def _store(self):
        return self._qc.store

    def _embedding_enabled(self) -> bool:
        return self._qc.config.desc_embedding is not None

    def _vector_spec(self) -> list[CollectionVectorConfig]:
        if not self._embedding_enabled():
            return []
        return [CollectionVectorConfig(name=VECTOR_NAME, dimension=self._qc.config.desc_embedding.dimension)]

    def _info(self, collection: str) -> CollectionInfo:
        info = self._store.get_collection(collection, with_count=False)
        if info is None:
            raise CollectionNotFoundError(collection)
        if info.kind != DOCUMENT_KIND:
            raise ValueError(f"Collection {collection!r} is a {info.kind!r} collection, not a document collection")
        return info

    def create(self, collection: str, *, semantic: Optional[bool] = None) -> CollectionInfo:
        """
        Create a document collection (no-op if it already exists).

        semantic: Optional[bool] — Build a vector index. Defaults to True when a
            description embedding provider is configured.
        """
        existing = self._store.get_collection(collection, with_count=False)
        if existing is not None:
            if existing.kind != DOCUMENT_KIND:
                raise ValueError(f"Collection {collection!r} already exists as a {existing.kind!r} collection")
            return existing
        use_vectors = self._embedding_enabled() if semantic is None else bool(semantic)
        if use_vectors and not self._embedding_enabled():
            raise RuntimeError("Semantic document collections need desc_embedding configured")
        return self._store.create_collection(
            collection,
            self._vector_spec() if use_vectors else [],
            kind=DOCUMENT_KIND,
            keyword_fields=("*",),
            text_field=TEXT_FIELD,
            fulltext=True,
        )

    def add(
        self,
        collection: str,
        items: str | dict | Iterable[str | dict],
        *,
        tags: Sequence[str] | str | None = None,
        source: Optional[str] = None,
        kind: Optional[str] = None,
        metadata: Optional[dict] = None,
        dedupe_threshold: Optional[float] = None,
    ) -> list[str]:
        """
        Add or replace documents. Creates the collection on first use.

        items: str | dict | Iterable — Text, or dicts with "text" plus optional
            "id", "tags", "source", "kind", "metadata", and any extra flat keys.
        tags/source/kind/metadata — Defaults applied to every item.
        dedupe_threshold: Optional[float] — When set (e.g. 0.95) and the collection has
            vectors, an item whose nearest existing document has cosine similarity at or
            above the threshold updates that document (tags are merged) instead of
            creating a near-duplicate.
        Returns: list[str] — Document IDs in input order.
        """
        if isinstance(items, (str, dict)):
            items = [items]
        info = self.create(collection)
        now = time.time()

        payloads: list[tuple[str, dict]] = []
        for item in items:
            entry = {"text": item} if isinstance(item, str) else dict(item)
            text = str(entry.pop("text", "") or "").strip()
            if not text:
                raise ValueError("Every document needs non-empty text")
            payload: dict[str, Any] = {}
            payload.update(metadata or {})
            payload.update(entry.pop("metadata", None) or {})
            point_id = str(entry.pop("id", None) or document_id(collection, text))
            item_tags = _normalize_tags(tags) + _normalize_tags(entry.pop("tags", None))
            payload.update(entry)
            payload.update({
                "text": text,
                "tags": list(dict.fromkeys(item_tags)),
                "source": entry.get("source", source),
                "kind": entry.get("kind", kind),
                "added_at": now,
            })
            payloads.append((point_id, {key: value for key, value in payload.items() if value is not None}))

        vectors: list[dict] = [{} for _ in payloads]
        if info.vectors:
            texts = [payload["text"] for _, payload in payloads]
            step = max(1, int(self._qc.config.desc_embedding.batch_size))
            embedded = []
            for offset in range(0, len(texts), step):
                embedded.extend(self._qc.desc_provider.embed(texts[offset:offset + step]))
            vectors = [{VECTOR_NAME: vector} for vector in embedded]

        if dedupe_threshold is not None and info.vectors:
            payloads = self._merge_near_duplicates(collection, payloads, vectors, float(dedupe_threshold))

        points = [
            PointRecord(id=point_id, payload=payload, vectors=vector, search_text=_search_text(payload))
            for (point_id, payload), vector in zip(payloads, vectors)
        ]
        batch_size = max(1, int(self._qc.config.store.upsert_batch_size))
        for offset in range(0, len(points), batch_size):
            self._store.upsert(collection, points[offset:offset + batch_size])
        return [point.id for point in points]

    def _merge_near_duplicates(
        self,
        collection: str,
        payloads: list[tuple[str, dict]],
        vectors: list[dict],
        threshold: float,
    ) -> list[tuple[str, dict]]:
        requests = [
            SearchRequest(vector=vector[VECTOR_NAME], using=VECTOR_NAME, limit=1, with_payload=["tags"])
            for vector in vectors
        ]
        nearest = self._store.search_batch(collection, requests)
        merged: list[tuple[str, dict]] = []
        for (point_id, payload), hits in zip(payloads, nearest):
            if hits and hits[0].id != point_id and hits[0].score >= threshold:
                existing_tags = list(hits[0].payload.get("tags") or [])
                payload = dict(payload)
                payload["tags"] = list(dict.fromkeys(existing_tags + list(payload.get("tags") or [])))
                point_id = hits[0].id
            merged.append((point_id, payload))
        return merged

    def search(
        self,
        collection: str,
        query: str,
        *,
        limit: int = 10,
        mode: RecallMode = "hybrid",
        tags: Sequence[str] | str | None = None,
        source: Optional[str] = None,
        kind: Optional[str] = None,
        where: Optional[dict] = None,
        min_score: Optional[float] = None,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        diversity: float = 0.0,
        half_life_days: Optional[float] = None,
    ) -> list[Document]:
        """
        Recall documents relevant to a query.

        mode: "hybrid" (vector + BM25 fused), "semantic" (vector only), or "keyword" (BM25 only).
            Collections without vectors always use keyword search.
        tags: Match documents carrying any of these tags.
        where: Extra exact-match filters on metadata keys (values may be lists = any of).
        min_score: Drop results scoring below this threshold (applied to the final score).
        semantic_weight: Share of the hybrid fusion given to the vector ranking (0..1);
            the rest goes to BM25.
        diversity: MMR trade-off in [0, 1]; 0 disables, ~0.3 removes near-duplicate hits.
        half_life_days: When set, multiply scores by 0.5 ** (age_days / half_life_days)
            so newer documents win ties against stale ones.
        """
        info = self._info(collection)
        flt = self._filter(tags=tags, source=source, kind=kind, where=where)
        limit = max(1, int(limit))
        semantic_ok = bool(info.vectors) and mode != "keyword"
        keyword_ok = mode != "semantic" or not info.vectors
        pool = max(limit * 5, MIN_CANDIDATES)
        candidate_fields: bool | list[str] = ["added_at"] if half_life_days else False

        vector_hits = []
        if semantic_ok:
            query_vector = embed_query(self._qc.desc_provider, query)
            vector_hits = self._store.search(
                collection,
                SearchRequest(vector=query_vector, using=VECTOR_NAME, limit=pool, filter=flt, with_payload=candidate_fields),
            )
        keyword_hits = (
            self._store.text_search(collection, query, limit=pool, flt=flt, with_payload=candidate_fields)
            if keyword_ok else []
        )

        light: dict[str, dict] = {}
        if vector_hits and keyword_hits:
            weight = min(max(float(semantic_weight), 0.0), 1.0)
            fused: dict[str, float] = {}
            for hits, list_weight in ((vector_hits, weight), (keyword_hits, 1.0 - weight)):
                for rank, hit in enumerate(hits):
                    fused[hit.id] = fused.get(hit.id, 0.0) + list_weight / (RRF_K + rank + 1)
                    light.setdefault(hit.id, hit.payload)
            ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        else:
            hits = vector_hits or keyword_hits
            light = {hit.id: hit.payload for hit in hits}
            ranked = [(hit.id, hit.score) for hit in hits]

        if half_life_days and half_life_days > 0:
            now = time.time()
            decayed = []
            for point_id, score in ranked:
                added_at = light[point_id].get("added_at")
                age_days = max(0.0, (now - float(added_at)) / 86400.0) if added_at else 0.0
                decayed.append((point_id, score * 0.5 ** (age_days / float(half_life_days))))
            ranked = sorted(decayed, key=lambda item: item[1], reverse=True)

        if diversity > 0 and info.vectors and len(ranked) > 1:
            candidates = ranked[: max(limit * 4, limit)]
            ranked = _mmr_order(
                candidates,
                self._store.vectors(collection, VECTOR_NAME, [point_id for point_id, _ in candidates]),
                min(max(float(diversity), 0.0), 1.0),
                limit,
            )

        if min_score is not None:
            ranked = [(point_id, score) for point_id, score in ranked if score >= min_score]
        ranked = ranked[:limit]
        records = {record.id: record.payload for record in self._store.retrieve(collection, [point_id for point_id, _ in ranked])}
        return [
            _to_document(point_id, records[point_id], score)
            for point_id, score in ranked
            if point_id in records
        ]

    def get(self, collection: str, ids: Sequence[str] | str) -> list[Document]:
        """
        Fetch documents by ID, in the order given; missing IDs are skipped.
        """
        self._info(collection)
        if isinstance(ids, str):
            ids = [ids]
        return [_to_document(record.id, record.payload) for record in self._store.retrieve(collection, list(ids))]

    def list(
        self,
        collection: str,
        *,
        tags: Sequence[str] | str | None = None,
        source: Optional[str] = None,
        kind: Optional[str] = None,
        where: Optional[dict] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """
        Page through documents in insertion order, optionally filtered.
        """
        self._info(collection)
        flt = self._filter(tags=tags, source=source, kind=kind, where=where)
        records = self._store.scroll(collection, flt=flt, limit=limit, offset=offset)
        return [_to_document(record.id, record.payload) for record in records]

    def count(
        self,
        collection: str,
        *,
        tags: Sequence[str] | str | None = None,
        source: Optional[str] = None,
        kind: Optional[str] = None,
        where: Optional[dict] = None,
    ) -> int:
        self._info(collection)
        return self._store.count(collection, self._filter(tags=tags, source=source, kind=kind, where=where))

    def delete(
        self,
        collection: str,
        ids: Sequence[str] | str | None = None,
        *,
        tags: Sequence[str] | str | None = None,
        source: Optional[str] = None,
        kind: Optional[str] = None,
        where: Optional[dict] = None,
    ) -> int:
        """
        Delete documents by ID or by filter. Returns the number removed.
        """
        self._info(collection)
        if ids is not None:
            return self._store.delete_points(collection, [ids] if isinstance(ids, str) else list(ids))
        flt = self._filter(tags=tags, source=source, kind=kind, where=where)
        if flt is None:
            raise ValueError("Pass ids or at least one filter; use drop() to remove a whole collection")
        return self._store.delete_by_filter(collection, flt)

    def drop(self, collection: str) -> bool:
        """
        Delete an entire document collection.
        """
        info = self._store.get_collection(collection, with_count=False)
        if info is None:
            return False
        if info.kind != DOCUMENT_KIND:
            raise ValueError(f"Refusing to drop {info.kind!r} collection {collection!r} through the document API")
        return self._store.delete_collection(info.name)

    def collections(self) -> list[dict]:
        """
        List document collections with point counts.
        """
        return [
            info.to_dict()
            for info in self._store.list_collections()
            if info.kind == DOCUMENT_KIND
        ]

    def _filter(
        self,
        *,
        tags: Sequence[str] | str | None,
        source: Optional[str],
        kind: Optional[str],
        where: Optional[dict],
    ) -> Optional[SearchFilter]:
        equals: dict[str, Any] = dict(where or {})
        tag_list = _normalize_tags(tags)
        if tag_list:
            equals["tags"] = tag_list
        if source is not None:
            equals["source"] = source
        if kind is not None:
            equals["kind"] = kind
        return SearchFilter.build(**equals)


__all__ = ["Document", "DocumentStore", "document_id"]
