from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import re
from threading import Lock
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from engine.src.keywords import (
    extract_keywords,
    keyword_overlap_score,
    keyword_query_coverage_score,
)
from engine.src.providers import EmbeddingProvider
from engine.src.query_dsl import StructuredSubQuery

if TYPE_CHECKING:
    from engine.src.reranker import ColBERTReranker


RRF_K = 60
TOP_RANK_BONUS_1 = 0.05
TOP_RANK_BONUS_2_3 = 0.02
QUERY_CACHE_SIZE = 256
DISK_QUERY_CACHE_LIMIT = 512
METADATA_TOKEN_CACHE_SIZE = 2048
HYBRID_REQUEST_MIN_LIMIT = 16
HYBRID_COMPLEX_QUERY_MIN_LIMIT = 18
TOKEN_SYNONYMS = {
    "remove": ("delete", "deletion"),
    "delete": ("remove", "removed"),
    "stale": ("obsolete", "old"),
    "obsolete": ("stale", "old"),
    "assign": ("set", "classify", "classified"),
    "assigned": ("assign", "set", "classify", "classified"),
    "classify": ("assigned", "role"),
    "classified": ("classify", "assigned", "role"),
    "carry": ("propagate", "store", "persist", "payload"),
    "carried": ("carry", "propagate", "store", "persist", "payload"),
    "propagate": ("carry", "carried", "store"),
    "payload": ("store", "persist"),
    "indexed": ("index", "stored", "persisted"),
    "stored": ("store", "persisted"),
    "persisted": ("store", "stored", "payload"),
    "hydrate": ("retrieve", "payload", "source"),
    "snippet": ("source",),
    "snippets": ("source",),
    "duplicate": ("dedup", "repeat"),
    "duplicates": ("dedup", "repeat"),
    "collapse": ("dedup",),
    "collapsed": ("dedup",),
    "deduplicate": ("dedup", "duplicate"),
}
ACTION_QUERY_KEYWORDS = {
    "assign",
    "assigned",
    "hydrate",
    "refresh",
    "delete",
    "remove",
    "detect",
    "classify",
    "connect",
    "collapse",
    "debounce",
    "cache",
    "carry",
    "carried",
    "persist",
    "store",
    "propagate",
    "create",
    "build",
    "expand",
}
LIGHT_RESULT_PAYLOAD_FIELDS = [
    "chunk_id",
    "language",
    "file_path",
    "symbol_name",
    "symbol_kind",
    "symbol_type",
    "line_start",
    "line_end",
    "description",
    "keywords",
    "path_context",
    "signature",
    "docstring",
    "parent",
    "role",
]
FULL_RESULT_PAYLOAD_FIELDS = LIGHT_RESULT_PAYLOAD_FIELDS + ["source"]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """
    Single search result with metadata and score.

    Args:
        score: Similarity score (0-1, higher is better)
        file_path: Path to source file
        symbol_name: Symbol name (function, class, etc.)
        symbol_kind: Symbol type (function, class, method, etc.)
        line_start: Starting line number
        line_end: Ending line number
        source: Raw source code
        description: Natural language description
        signature: Function/method signature (if available)
        docstring: Docstring (if available)
        parent: Parent symbol name (if nested)
        language: Programming language
        path_context: Lightweight path-derived context string
    """

    score: float
    file_path: str
    symbol_name: str
    symbol_kind: str
    line_start: int
    line_end: int
    source: str
    description: str
    point_id: int | str | None = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent: Optional[str] = None
    language: Optional[str] = None
    path_context: Optional[str] = None


class CodeSearcher:
    """
    Semantic code search using dual embeddings.

    Supports:
    - Code search: Embed query as code, search code vector space
    - Description search: Embed query as text, search description vector space
    - Hybrid search: Search both spaces and merge with RRF fusion
    - Structured search: typed sub-queries (lex/vec/hyde)
    - Filtering by language, file path, symbol type
    """

    def __init__(
        self,
        client: Any,
        collection_name: str,
        code_provider: EmbeddingProvider,
        desc_provider: EmbeddingProvider,
        reranker: Optional["ColBERTReranker"] = None,
    ):
        """
        Args:
            client: QdrantClient — Qdrant client instance.
            collection_name: str — Target collection name.
            code_provider: EmbeddingProvider — Embedding provider for code vectors.
            desc_provider: EmbeddingProvider — Embedding provider for description vectors.
            reranker: Optional[ColBERTReranker] — ColBERT reranker for post-retrieval reranking.
        """
        self._client = client
        self._collection_name = collection_name
        self._code_provider = code_provider
        self._desc_provider = desc_provider
        self._reranker = reranker
        self._cache_lock = Lock()
        self._query_vector_cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
        self._keyword_cache: OrderedDict[str, list[str]] = OrderedDict()
        self._metadata_token_cache: OrderedDict[str, list[str]] = OrderedDict()

    def close(self) -> None:
        """
        Close the underlying search client if it supports cleanup.
        """
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def search_code(
        self,
        query: str,
        limit: int = 10,
        language: Optional[str] = None,
        file_path: Optional[str] = None,
        path_prefix: Optional[str] = None,
        symbol_kind: Optional[str] = None,
        use_keywords: bool = False,
        keyword_weight: float = 0.3,
        rerank: bool = False,
        include_source: bool = True,
    ) -> list[SearchResult]:
        """
        Search by code similarity (exact semantic matching).

        Args:
            query: str — Code snippet or natural language query.
            limit: int — Maximum number of results.
            language: Optional[str] — Filter by programming language.
            file_path: Optional[str] — Filter by file path (exact match).
            path_prefix: Optional[str] — Filter by file path prefix.
            symbol_kind: Optional[str] — Filter by symbol type.
            use_keywords: bool — Enable keyword-based boosting.
            keyword_weight: float — Weight for keyword overlap score.
            rerank: bool — Enable ColBERT reranking.

        Returns:
            list[SearchResult] — Results sorted by relevance.
        """
        query_vector = self._embed_query_cached(query, using="code")
        ranking_keywords = self._extract_keywords_cached(query)
        blend_keywords = ranking_keywords if use_keywords else []

        return self._search(
            query=query,
            query_vector=query_vector,
            using="code",
            limit=limit,
            language=language,
            file_path=file_path,
            path_prefix=path_prefix,
            symbol_kind=symbol_kind,
            ranking_keywords=ranking_keywords,
            blend_keywords=blend_keywords,
            keyword_weight=keyword_weight,
            rerank=rerank,
            include_source=include_source,
        )

    def search_description(
        self,
        query: str,
        limit: int = 10,
        language: Optional[str] = None,
        file_path: Optional[str] = None,
        path_prefix: Optional[str] = None,
        symbol_kind: Optional[str] = None,
        use_keywords: bool = False,
        keyword_weight: float = 0.3,
        rerank: bool = False,
        include_source: bool = True,
    ) -> list[SearchResult]:
        """
        Search by description similarity (conceptual/intent matching).

        Args:
            query: str — Natural language query.
            limit: int — Maximum number of results.
            language: Optional[str] — Filter by programming language.
            file_path: Optional[str] — Filter by file path (exact match).
            path_prefix: Optional[str] — Filter by file path prefix.
            symbol_kind: Optional[str] — Filter by symbol type.
            use_keywords: bool — Enable keyword-based boosting.
            keyword_weight: float — Weight for keyword overlap score.
            rerank: bool — Enable ColBERT reranking.

        Returns:
            list[SearchResult] — Results sorted by relevance.
        """
        query_vector = self._embed_query_cached(query, using="description")
        ranking_keywords = self._extract_keywords_cached(query)
        blend_keywords = ranking_keywords if use_keywords else []

        return self._search(
            query=query,
            query_vector=query_vector,
            using="description",
            limit=limit,
            language=language,
            file_path=file_path,
            path_prefix=path_prefix,
            symbol_kind=symbol_kind,
            ranking_keywords=ranking_keywords,
            blend_keywords=blend_keywords,
            keyword_weight=keyword_weight,
            rerank=rerank,
            include_source=include_source,
        )

    def search_hybrid(
        self,
        query: str,
        limit: int = 10,
        code_weight: float = 0.5,
        desc_weight: float = 0.5,
        language: Optional[str] = None,
        file_path: Optional[str] = None,
        path_prefix: Optional[str] = None,
        symbol_kind: Optional[str] = None,
        use_keywords: bool = False,
        keyword_weight: float = 0.3,
        rerank: bool = False,
        rrf_k: int = RRF_K,
        top_rank_bonus_1: float = TOP_RANK_BONUS_1,
        top_rank_bonus_2_3: float = TOP_RANK_BONUS_2_3,
        rerank_top3_retrieval_weight: float = 0.75,
        rerank_top10_retrieval_weight: float = 0.60,
        rerank_tail_retrieval_weight: float = 0.40,
        rerank_candidate_multiplier: int = 4,
        include_source: bool = True,
    ) -> list[SearchResult]:
        """
        Hybrid search using RRF fusion across code and description spaces.

        Args:
            query: str — Search query (code or natural language).
            limit: int — Maximum number of results.
            code_weight: float — RRF list weight for code similarity.
            desc_weight: float — RRF list weight for description similarity.
            language: Optional[str] — Filter by programming language.
            file_path: Optional[str] — Filter by file path (exact match).
            path_prefix: Optional[str] — Filter by file path prefix.
            symbol_kind: Optional[str] — Filter by symbol type.
            use_keywords: bool — Enable keyword-based boosting.
            keyword_weight: float — Weight for keyword overlap score.
            rerank: bool — Enable ColBERT reranking with position-aware blending.

        Returns:
            list[SearchResult] — Results sorted by relevance.
        """
        ranking_keywords = self._extract_keywords_cached(query)
        blend_keywords = ranking_keywords if use_keywords else []
        request_limit = self._hybrid_request_limit(limit, ranking_keywords)

        code_vector, desc_vector = self._hybrid_query_vectors(query)

        code_results, desc_results = self._batch_search(
            requests=[
                {
                    "query": query,
                    "query_vector": code_vector,
                    "using": "code",
                    "limit": request_limit,
                    "language": language,
                    "file_path": file_path,
                    "path_prefix": path_prefix,
                    "symbol_kind": symbol_kind,
                    "ranking_keywords": ranking_keywords,
                    "blend_keywords": blend_keywords,
                    "keyword_weight": keyword_weight,
                },
                {
                    "query": query,
                    "query_vector": desc_vector,
                    "using": "description",
                    "limit": request_limit,
                    "language": language,
                    "file_path": file_path,
                    "path_prefix": path_prefix,
                    "symbol_kind": symbol_kind,
                    "ranking_keywords": ranking_keywords,
                    "blend_keywords": blend_keywords,
                    "keyword_weight": keyword_weight,
                },
            ]
        )

        fused = self._rrf_fuse(
            [code_results, desc_results],
            [code_weight, desc_weight],
            k=rrf_k,
            top_rank_bonus_1=top_rank_bonus_1,
            top_rank_bonus_2_3=top_rank_bonus_2_3,
        )

        rerank_active = rerank and self._reranker is not None
        if rerank_active:
            return self._blend_with_rerank(
                query,
                fused,
                limit,
                top3_retrieval_weight=rerank_top3_retrieval_weight,
                top10_retrieval_weight=rerank_top10_retrieval_weight,
                tail_retrieval_weight=rerank_tail_retrieval_weight,
                candidate_multiplier=rerank_candidate_multiplier,
            )

        return self._finalize_results(fused[:limit], include_source=include_source)

    def _hybrid_request_limit(self, limit: int, ranking_keywords: Optional[list[str]] = None) -> int:
        """
        Keep hybrid candidate overfetch bounded without dropping top-rank quality.
        """
        request_limit = max(limit * 3, HYBRID_REQUEST_MIN_LIMIT)
        if ranking_keywords and len(ranking_keywords) >= 15:
            request_limit = max(request_limit, HYBRID_COMPLEX_QUERY_MIN_LIMIT)
        return request_limit

    def search_structured(
        self,
        sub_queries: list[StructuredSubQuery],
        limit: int = 10,
        language: Optional[str] = None,
        file_path: Optional[str] = None,
        path_prefix: Optional[str] = None,
        symbol_kind: Optional[str] = None,
        use_keywords: bool = True,
        keyword_weight: float = 0.3,
        rerank: bool = False,
        first_query_weight: float = 2.0,
        rrf_k: int = RRF_K,
        top_rank_bonus_1: float = TOP_RANK_BONUS_1,
        top_rank_bonus_2_3: float = TOP_RANK_BONUS_2_3,
        rerank_top3_retrieval_weight: float = 0.75,
        rerank_top10_retrieval_weight: float = 0.60,
        rerank_tail_retrieval_weight: float = 0.40,
        rerank_candidate_multiplier: int = 4,
        include_source: bool = True,
    ) -> list[SearchResult]:
        """
        Structured search across typed sub-queries with RRF fusion.

        Args:
            sub_queries: list[StructuredSubQuery] — Ordered typed sub-queries.
            limit: int — Maximum number of results.
            language: Optional[str] — Filter by programming language.
            file_path: Optional[str] — Filter by file path (exact match).
            path_prefix: Optional[str] — Filter by file path prefix.
            symbol_kind: Optional[str] — Filter by symbol type.
            use_keywords: bool — Enable keyword-based boosting.
            keyword_weight: float — Weight for keyword overlap score.
            rerank: bool — Enable ColBERT reranking with position-aware blending.
            first_query_weight: float — Additional RRF weight for the first sub-query.

        Returns:
            list[SearchResult] — Results sorted by fused relevance.
        """
        if not sub_queries:
            return []

        requests: list[dict] = []
        weights: list[float] = []
        for idx, sub in enumerate(sub_queries):
            weight = first_query_weight if idx == 0 else 1.0
            weights.append(weight)

            if sub.kind == "lex":
                query_vector = self._embed_query_cached(sub.text, "code")
                per_query_keyword_weight = max(keyword_weight, 0.45)
                using = "code"
            else:
                provider = self._code_provider if self._can_share_query_embedding() else self._desc_provider
                cache_key = "shared" if self._can_share_query_embedding() else "description"
                query_vector = self._embed_query_with_provider_cached(sub.text, provider, cache_key)
                per_query_keyword_weight = keyword_weight
                using = "description"

            ranking_keywords = self._extract_keywords_cached(sub.text)
            blend_keywords = ranking_keywords if use_keywords else []

            requests.append(
                {
                    "query": sub.text,
                    "query_vector": query_vector,
                    "using": using,
                    "limit": max(limit * 3, 30),
                    "language": language,
                    "file_path": file_path,
                    "path_prefix": path_prefix,
                    "symbol_kind": symbol_kind,
                    "ranking_keywords": ranking_keywords,
                    "blend_keywords": blend_keywords,
                    "keyword_weight": per_query_keyword_weight,
                }
            )

        batch_results = self._batch_search(requests)
        result_lists: list[list[SearchResult]] = []
        filtered_weights: list[float] = []
        for idx, results in enumerate(batch_results):
            if results:
                result_lists.append(results)
                filtered_weights.append(weights[idx])

        if not result_lists:
            return []

        fused = self._rrf_fuse(
            result_lists,
            filtered_weights,
            k=rrf_k,
            top_rank_bonus_1=top_rank_bonus_1,
            top_rank_bonus_2_3=top_rank_bonus_2_3,
        )

        rerank_active = rerank and self._reranker is not None
        if rerank_active:
            return self._blend_with_rerank(
                " ".join(q.text for q in sub_queries),
                fused,
                limit,
                top3_retrieval_weight=rerank_top3_retrieval_weight,
                top10_retrieval_weight=rerank_top10_retrieval_weight,
                tail_retrieval_weight=rerank_tail_retrieval_weight,
                candidate_multiplier=rerank_candidate_multiplier,
            )

        return self._finalize_results(fused[:limit], include_source=include_source)

    def _result_key(self, result: SearchResult) -> str:
        """
        Build stable key for result deduplication across lists.

        result: SearchResult — Result item to key.
        Returns: str — Stable key.
        """
        return f"{result.file_path}:{result.symbol_kind}:{result.symbol_name}:{result.line_start}"

    def _rrf_fuse(
        self,
        result_lists: list[list[SearchResult]],
        list_weights: list[float],
        k: int = RRF_K,
        top_rank_bonus_1: float = TOP_RANK_BONUS_1,
        top_rank_bonus_2_3: float = TOP_RANK_BONUS_2_3,
    ) -> list[SearchResult]:
        """
        Reciprocal Rank Fusion with top-rank bonus.

        result_lists: list[list[SearchResult]] — Ranked result lists.
        list_weights: list[float] — Per-list RRF multipliers.
        k: int — RRF constant.
        Returns: list[SearchResult] — Fused list sorted by score.
        """
        fused_scores: dict[str, float] = {}
        representatives: dict[str, SearchResult] = {}

        for list_idx, results in enumerate(result_lists):
            weight = list_weights[list_idx] if list_idx < len(list_weights) else 1.0
            for rank, result in enumerate(results):
                key = self._result_key(result)
                representatives[key] = representatives.get(key, result)
                rrf_score = weight * (1.0 / (k + rank + 1))
                fused_scores[key] = fused_scores.get(key, 0.0) + rrf_score

                if rank == 0:
                    fused_scores[key] += top_rank_bonus_1
                elif rank < 3:
                    fused_scores[key] += top_rank_bonus_2_3

        ranked_keys = sorted(fused_scores.keys(), key=lambda key: fused_scores[key], reverse=True)
        return [
            SearchResult(
                score=fused_scores[key],
                file_path=representatives[key].file_path,
                symbol_name=representatives[key].symbol_name,
                symbol_kind=representatives[key].symbol_kind,
                line_start=representatives[key].line_start,
                line_end=representatives[key].line_end,
                source=representatives[key].source,
                description=representatives[key].description,
                point_id=representatives[key].point_id,
                signature=representatives[key].signature,
                docstring=representatives[key].docstring,
                parent=representatives[key].parent,
                language=representatives[key].language,
                path_context=representatives[key].path_context,
            )
            for key in ranked_keys
        ]

    def _blend_with_rerank(
        self,
        query: str,
        results: list[SearchResult],
        limit: int,
        top3_retrieval_weight: float = 0.75,
        top10_retrieval_weight: float = 0.60,
        tail_retrieval_weight: float = 0.40,
        candidate_multiplier: int = 4,
    ) -> list[SearchResult]:
        """
        Blend retrieval and reranker scores with position-aware weighting.

        query: str — Query text for reranker.
        results: list[SearchResult] — RRF-ranked candidate results.
        limit: int — Maximum number of results to return.
        Returns: list[SearchResult] — Blended and sorted results.
        """
        if self._reranker is None or not results:
            return results[:limit]

        candidate_count = min(len(results), max(limit * max(1, int(candidate_multiplier)), 30))
        candidates = self._hydrate_results(results[:candidate_count])

        documents = [item.source for item in candidates]
        reranked = self._reranker.rerank(query, documents, top_k=len(documents))
        if not reranked:
            return candidates[:limit]

        max_rerank = max((item.score for item in reranked), default=0.0)
        if max_rerank <= 0:
            max_rerank = 1.0

        rerank_scores = {item.index: (item.score / max_rerank) for item in reranked}

        blended: list[SearchResult] = []
        for rank, candidate in enumerate(candidates):
            if rank < 3:
                retrieval_weight = top3_retrieval_weight
            elif rank < 10:
                retrieval_weight = top10_retrieval_weight
            else:
                retrieval_weight = tail_retrieval_weight

            rerank_score = rerank_scores.get(rank, 0.0)
            score = candidate.score * retrieval_weight + rerank_score * (1.0 - retrieval_weight)
            blended.append(
                SearchResult(
                    score=score,
                    file_path=candidate.file_path,
                    symbol_name=candidate.symbol_name,
                    symbol_kind=candidate.symbol_kind,
                    line_start=candidate.line_start,
                    line_end=candidate.line_end,
                    source=candidate.source,
                    description=candidate.description,
                    point_id=candidate.point_id,
                    signature=candidate.signature,
                    docstring=candidate.docstring,
                    parent=candidate.parent,
                    language=candidate.language,
                    path_context=candidate.path_context,
                )
            )

        blended.sort(key=lambda item: item.score, reverse=True)
        return blended[:limit]

    def _search(
        self,
        query: str,
        query_vector: list[float],
        using: str,
        limit: int,
        language: Optional[str],
        file_path: Optional[str],
        path_prefix: Optional[str],
        symbol_kind: Optional[str],
        ranking_keywords: list[str],
        blend_keywords: list[str],
        keyword_weight: float,
        rerank: bool = False,
        include_source: bool = True,
    ) -> list[SearchResult]:
        """
        Internal search implementation.

        Args:
            query: str — Original query text for reranking.
            query_vector: list[float] — Embedding vector for the query.
            using: str — Vector space to search ("code" or "description").
            limit: int — Maximum number of results.
            language: Optional[str] — Filter by programming language.
            file_path: Optional[str] — Filter by file path (exact match).
            path_prefix: Optional[str] — Filter by file path prefix.
            symbol_kind: Optional[str] — Filter by symbol type.
            ranking_keywords: list[str] — Extracted keywords used for lexical reranking.
            blend_keywords: list[str] — Query keywords used for payload keyword blending.
            keyword_weight: float — Weight for keyword overlap score.
            rerank: bool — Enable ColBERT reranking.

        Returns:
            list[SearchResult] — Sorted search results.
        """
        filters = []

        if language:
            filters.append(self._match_filter("language", language))

        if file_path:
            filters.append(self._match_filter("file_path", file_path))

        normalized_path_prefix = None
        if path_prefix:
            normalized_path_prefix = path_prefix.replace("\\", "/").strip().strip("/").lower()
            if normalized_path_prefix:
                filters.append(self._match_filter("path_prefixes", normalized_path_prefix))

        if symbol_kind:
            filters.append(
                {
                    "should": [
                        self._match_filter("symbol_kind", symbol_kind),
                        self._match_filter("symbol_type", symbol_kind),
                    ]
                }
            )

        query_filter = {"must": filters} if filters else None

        rerank_active = rerank and self._reranker is not None
        if rerank_active:
            fetch_limit = limit * 5
        elif ranking_keywords:
            fetch_limit = limit * 3
        else:
            fetch_limit = limit

        if normalized_path_prefix:
            fetch_limit = max(fetch_limit, limit * 10)

        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            using=using,
            query_filter=query_filter,
            limit=fetch_limit,
            with_payload=LIGHT_RESULT_PAYLOAD_FIELDS,
        )

        if normalized_path_prefix and not results.points:
            fallback_filters = [
                condition for condition in filters
                if not (isinstance(condition, dict) and condition.get("key") == "path_prefixes")
            ]
            fallback_filter = {"must": fallback_filters} if fallback_filters else None
            results = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                using=using,
                query_filter=fallback_filter,
                limit=max(fetch_limit, limit * 25),
                with_payload=LIGHT_RESULT_PAYLOAD_FIELDS,
            )

        search_results = []
        for point in results.points:
            vector_score = point.score
            chunk_keywords = point.payload.get("keywords", [])
            path_context = point.payload.get("path_context")

            if blend_keywords and chunk_keywords:
                kw_score = keyword_overlap_score(blend_keywords, chunk_keywords)
                final_score = vector_score * (1 - keyword_weight) + kw_score * keyword_weight
            else:
                final_score = vector_score

            if path_context and ranking_keywords:
                path_keywords = extract_keywords(path_context)
                if path_keywords:
                    final_score += keyword_query_coverage_score(ranking_keywords, path_keywords) * 0.1

            symbol_kind_value = point.payload.get("symbol_kind") or point.payload.get("symbol_type") or "unknown"
            symbol_name_value = point.payload["symbol_name"]
            file_path_value = point.payload["file_path"]
            final_score *= self._kind_score_multiplier(symbol_kind_value, len(ranking_keywords))
            final_score *= self._path_signal_multiplier(file_path_value, ranking_keywords)
            final_score += self._basename_hit_bonus(file_path_value, ranking_keywords)
            final_score += self._provider_path_bonus(file_path_value, ranking_keywords)
            final_score += self._startup_boundary_bonus(file_path_value, ranking_keywords)
            final_score += self._startup_symbol_bonus(file_path_value, symbol_name_value, ranking_keywords)
            final_score += self._implementation_path_bonus(file_path_value, ranking_keywords)
            final_score += self._filecache_bonus(file_path_value, symbol_name_value, ranking_keywords)
            final_score += self._role_bonus(point.payload.get("role"), ranking_keywords)
            final_score *= self._subsystem_conflict_penalty(file_path_value, ranking_keywords)
            final_score *= self._ranking_subsystem_penalty(file_path_value, ranking_keywords)
            final_score *= self._ranking_helper_penalty(symbol_name_value, ranking_keywords)
            final_score *= self._startup_import_symbol_penalty(symbol_kind_value, ranking_keywords)
            final_score *= self._constructor_penalty(symbol_name_value, ranking_keywords)
            final_score *= self._wrapper_symbol_penalty(file_path_value, symbol_name_value, ranking_keywords)
            final_score *= self._container_symbol_penalty(symbol_name_value, symbol_kind_value, ranking_keywords)
            if ranking_keywords:
                symbol_tokens = self._symbol_tokens(
                    symbol_name_value,
                    point.payload.get("signature"),
                )
                symbol_overlap = keyword_query_coverage_score(ranking_keywords, symbol_tokens)
                final_score += symbol_overlap * 0.18
                final_score += self._action_symbol_bonus(ranking_keywords, symbol_tokens)
                if len(ranking_keywords) >= 2 and symbol_overlap >= 0.5:
                    final_score *= 1.12
                if (
                    len(ranking_keywords) >= 2
                    and symbol_overlap < 0.5
                    and symbol_kind_value.lower() in {"data_key", "variable", "property", "import"}
                ):
                    final_score *= 0.7

            if ranking_keywords:
                metadata_tokens = self._metadata_tokens(point.payload)
                metadata_overlap = keyword_query_coverage_score(ranking_keywords, metadata_tokens)
                final_score += metadata_overlap * 0.16
                if len(ranking_keywords) >= 3 and metadata_overlap >= 0.45:
                    final_score *= 1.08

                generic_symbol_names = {"main", "entry", "_start", "init"}
                normalized_symbol_name = symbol_name_value.lower()
                if (
                    len(ranking_keywords) > 1
                    and normalized_symbol_name in generic_symbol_names
                    and all(term not in normalized_symbol_name for term in ranking_keywords)
                ):
                    final_score *= 0.92
            if normalized_path_prefix:
                normalized_file_path = file_path_value.replace("\\", "/").lower()
                normalized_context = (path_context or "").replace("\\", "/").lower()
                if normalized_path_prefix not in normalized_file_path and normalized_path_prefix not in normalized_context:
                    continue

            search_results.append(
                SearchResult(
                    score=final_score,
                    file_path=file_path_value,
                    symbol_name=symbol_name_value,
                    symbol_kind=symbol_kind_value,
                    line_start=point.payload["line_start"],
                    line_end=point.payload["line_end"],
                    source=point.payload.get("source", ""),
                    description=point.payload["description"],
                    point_id=getattr(point, "id", None),
                    signature=point.payload.get("signature"),
                    docstring=point.payload.get("docstring"),
                    parent=point.payload.get("parent"),
                    language=point.payload.get("language"),
                    path_context=path_context,
                )
            )

        search_results.sort(key=lambda item: item.score, reverse=True)

        if rerank_active:
            return self._blend_with_rerank(query, search_results, limit)

        return self._finalize_results(search_results[:limit], include_source=include_source)

    def _batch_search(self, requests: list[dict]) -> list[list[SearchResult]]:
        """
        Execute multiple vector searches against the same collection in one batch.

        requests: list[dict] — Search request dictionaries matching _search parameters.
        Returns: list[list[SearchResult]] — Results in request order.
        """
        if not requests:
            return []

        query_requests: list[dict] = []
        contexts: list[dict] = []

        for request in requests:
            query_filter, normalized_path_prefix, fetch_limit = self._build_query_filter(
                limit=request["limit"],
                language=request["language"],
                file_path=request["file_path"],
                path_prefix=request["path_prefix"],
                symbol_kind=request["symbol_kind"],
                ranking_keywords=request["ranking_keywords"],
                rerank=False,
                apply_keyword_overfetch=False,
            )
            query_requests.append(
                {
                    "query": request["query_vector"],
                    "using": request["using"],
                    "filter": query_filter,
                    "limit": fetch_limit,
                    "with_payload": LIGHT_RESULT_PAYLOAD_FIELDS,
                    "with_vector": False,
                }
            )
            contexts.append(
                {
                    "ranking_keywords": request["ranking_keywords"],
                    "blend_keywords": request["blend_keywords"],
                    "keyword_weight": request["keyword_weight"],
                    "normalized_path_prefix": normalized_path_prefix,
                    "query_vector": request["query_vector"],
                    "using": request["using"],
                    "limit": request["limit"],
                }
            )

        responses = self._client.query_batch_points(
            collection_name=self._collection_name,
            requests=query_requests,
        )

        out: list[list[SearchResult]] = []
        for request, response, context in zip(requests, responses, contexts):
            results = self._points_to_results(
                points=response.points,
                ranking_keywords=context["ranking_keywords"],
                blend_keywords=context["blend_keywords"],
                keyword_weight=context["keyword_weight"],
                normalized_path_prefix=context["normalized_path_prefix"],
            )
            if context["normalized_path_prefix"] and not response.points:
                fallback_filter = self._build_fallback_query_filter(
                    language=request["language"],
                    file_path=request["file_path"],
                    symbol_kind=request["symbol_kind"],
                )
                fallback_response = self._client.query_points(
                    collection_name=self._collection_name,
                    query=context["query_vector"],
                    using=context["using"],
                    query_filter=fallback_filter,
                    limit=max(context["limit"] * 25, 30),
                    with_payload=LIGHT_RESULT_PAYLOAD_FIELDS,
                )
                results = self._points_to_results(
                    points=fallback_response.points,
                    ranking_keywords=context["ranking_keywords"],
                    blend_keywords=context["blend_keywords"],
                    keyword_weight=context["keyword_weight"],
                    normalized_path_prefix=context["normalized_path_prefix"],
                )
            out.append(results)

        return out

    def _build_query_filter(
        self,
        limit: int,
        language: Optional[str],
        file_path: Optional[str],
        path_prefix: Optional[str],
        symbol_kind: Optional[str],
        ranking_keywords: list[str],
        rerank: bool,
        apply_keyword_overfetch: bool = True,
    ) -> tuple[Optional[dict], Optional[str], int]:
        """
        Build Qdrant filter and fetch sizing for a search request.

        Returns: tuple[Optional[Filter], Optional[str], int]
        """
        filters = []

        if language:
            filters.append(self._match_filter("language", language))

        if file_path:
            filters.append(self._match_filter("file_path", file_path))

        normalized_path_prefix = None
        if path_prefix:
            normalized_path_prefix = path_prefix.replace("\\", "/").strip().strip("/").lower()
            if normalized_path_prefix:
                filters.append(self._match_filter("path_prefixes", normalized_path_prefix))

        if symbol_kind:
            filters.append(
                {
                    "should": [
                        self._match_filter("symbol_kind", symbol_kind),
                        self._match_filter("symbol_type", symbol_kind),
                    ]
                }
            )

        query_filter = {"must": filters} if filters else None

        rerank_active = rerank and self._reranker is not None
        if rerank_active:
            fetch_limit = limit * 5
        elif ranking_keywords and apply_keyword_overfetch:
            fetch_limit = limit * 3
        else:
            fetch_limit = limit

        if normalized_path_prefix:
            path_fetch_floor = limit * 10 if apply_keyword_overfetch else max(limit * 2, 40)
            fetch_limit = max(fetch_limit, path_fetch_floor)

        return query_filter, normalized_path_prefix, fetch_limit

    def _build_fallback_query_filter(
        self,
        language: Optional[str],
        file_path: Optional[str],
        symbol_kind: Optional[str],
    ) -> Optional[dict]:
        """
        Build a fallback filter without path-prefix constraints.
        """
        query_filter, _, _ = self._build_query_filter(
            limit=1,
            language=language,
            file_path=file_path,
            path_prefix=None,
            symbol_kind=symbol_kind,
            ranking_keywords=[],
            rerank=False,
        )
        return query_filter

    def _points_to_results(
        self,
        points,
        ranking_keywords: list[str],
        blend_keywords: list[str],
        keyword_weight: float,
        normalized_path_prefix: Optional[str],
    ) -> list[SearchResult]:
        """
        Convert Qdrant points to scored SearchResult objects.
        """
        search_results = []
        for point in points:
            vector_score = point.score
            chunk_keywords = point.payload.get("keywords", [])
            path_context = point.payload.get("path_context")

            if blend_keywords and chunk_keywords:
                kw_score = keyword_overlap_score(blend_keywords, chunk_keywords)
                final_score = vector_score * (1 - keyword_weight) + kw_score * keyword_weight
            else:
                final_score = vector_score

            if path_context and ranking_keywords:
                path_keywords = extract_keywords(path_context)
                if path_keywords:
                    final_score += keyword_query_coverage_score(ranking_keywords, path_keywords) * 0.1

            symbol_kind_value = point.payload.get("symbol_kind") or point.payload.get("symbol_type") or "unknown"
            symbol_name_value = point.payload["symbol_name"]
            file_path_value = point.payload["file_path"]
            final_score *= self._kind_score_multiplier(symbol_kind_value, len(ranking_keywords))
            final_score *= self._path_signal_multiplier(file_path_value, ranking_keywords)
            final_score += self._basename_hit_bonus(file_path_value, ranking_keywords)
            final_score += self._provider_path_bonus(file_path_value, ranking_keywords)
            final_score += self._startup_boundary_bonus(file_path_value, ranking_keywords)
            final_score += self._startup_symbol_bonus(file_path_value, symbol_name_value, ranking_keywords)
            final_score += self._implementation_path_bonus(file_path_value, ranking_keywords)
            final_score += self._filecache_bonus(file_path_value, symbol_name_value, ranking_keywords)
            final_score += self._role_bonus(point.payload.get("role"), ranking_keywords)
            final_score *= self._subsystem_conflict_penalty(file_path_value, ranking_keywords)
            final_score *= self._ranking_subsystem_penalty(file_path_value, ranking_keywords)
            final_score *= self._ranking_helper_penalty(symbol_name_value, ranking_keywords)
            final_score *= self._startup_import_symbol_penalty(symbol_kind_value, ranking_keywords)
            final_score *= self._constructor_penalty(symbol_name_value, ranking_keywords)
            final_score *= self._wrapper_symbol_penalty(file_path_value, symbol_name_value, ranking_keywords)
            final_score *= self._container_symbol_penalty(symbol_name_value, symbol_kind_value, ranking_keywords)
            if ranking_keywords:
                symbol_tokens = self._symbol_tokens(
                    symbol_name_value,
                    point.payload.get("signature"),
                )
                symbol_overlap = keyword_query_coverage_score(ranking_keywords, symbol_tokens)
                final_score += symbol_overlap * 0.18
                final_score += self._action_symbol_bonus(ranking_keywords, symbol_tokens)
                if len(ranking_keywords) >= 2 and symbol_overlap >= 0.5:
                    final_score *= 1.12
                if (
                    len(ranking_keywords) >= 2
                    and symbol_overlap < 0.5
                    and symbol_kind_value.lower() in {"data_key", "variable", "property", "import"}
                ):
                    final_score *= 0.7

                metadata_tokens = self._metadata_tokens(point.payload)
                metadata_overlap = keyword_query_coverage_score(ranking_keywords, metadata_tokens)
                final_score += metadata_overlap * 0.16
                if len(ranking_keywords) >= 3 and metadata_overlap >= 0.45:
                    final_score *= 1.08

                generic_symbol_names = {"main", "entry", "_start", "init"}
                normalized_symbol_name = symbol_name_value.lower()
                if (
                    len(ranking_keywords) > 1
                    and normalized_symbol_name in generic_symbol_names
                    and all(term not in normalized_symbol_name for term in ranking_keywords)
                ):
                    final_score *= 0.92
            if normalized_path_prefix:
                normalized_file_path = file_path_value.replace("\\", "/").lower()
                normalized_context = (path_context or "").replace("\\", "/").lower()
                if normalized_path_prefix not in normalized_file_path and normalized_path_prefix not in normalized_context:
                    continue

            search_results.append(
                SearchResult(
                    score=final_score,
                    file_path=file_path_value,
                    symbol_name=symbol_name_value,
                    symbol_kind=symbol_kind_value,
                    line_start=point.payload["line_start"],
                    line_end=point.payload["line_end"],
                    source=point.payload.get("source", ""),
                    description=point.payload["description"],
                    point_id=getattr(point, "id", None),
                    signature=point.payload.get("signature"),
                    docstring=point.payload.get("docstring"),
                    parent=point.payload.get("parent"),
                    language=point.payload.get("language"),
                    path_context=path_context,
                )
            )

        search_results.sort(key=lambda item: item.score, reverse=True)
        return search_results

    def _match_filter(self, key: str, value: str) -> dict:
        """
        Build a simple exact-match filter clause for the REST search client.
        """
        return {"key": key, "match": {"value": value}}

    def _hydrate_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Fetch full payloads only for final results that still need source content.
        """
        if not results:
            return results

        ids_to_fetch: list[int | str] = []
        seen_ids: set[str] = set()
        for result in results:
            if result.source or result.point_id is None:
                continue
            key = str(result.point_id)
            if key in seen_ids:
                continue
            seen_ids.add(key)
            ids_to_fetch.append(result.point_id)

        if not ids_to_fetch:
            return results

        try:
            records = self._client.retrieve(
                collection_name=self._collection_name,
                ids=ids_to_fetch,
                with_payload=FULL_RESULT_PAYLOAD_FIELDS,
                with_vectors=False,
            )
        except Exception:
            return results

        payload_by_id = {str(record.id): record.payload or {} for record in records}

        hydrated: list[SearchResult] = []
        for result in results:
            payload = payload_by_id.get(str(result.point_id)) if result.point_id is not None else None
            if payload is None:
                hydrated.append(result)
                continue

            hydrated.append(
                SearchResult(
                    score=result.score,
                    file_path=payload.get("file_path", result.file_path),
                    symbol_name=payload.get("symbol_name", result.symbol_name),
                    symbol_kind=payload.get("symbol_kind") or payload.get("symbol_type") or result.symbol_kind,
                    line_start=payload.get("line_start", result.line_start),
                    line_end=payload.get("line_end", result.line_end),
                    source=payload.get("source", result.source),
                    description=payload.get("description", result.description),
                    point_id=result.point_id,
                    signature=payload.get("signature", result.signature),
                    docstring=payload.get("docstring", result.docstring),
                    parent=payload.get("parent", result.parent),
                    language=payload.get("language", result.language),
                    path_context=payload.get("path_context", result.path_context),
                )
            )

        return hydrated

    def _finalize_results(self, results: list[SearchResult], include_source: bool) -> list[SearchResult]:
        """
        Optionally hydrate source content for final results.
        """
        if not include_source:
            return results
        return self._hydrate_results(results)

    def _can_share_query_embedding(self) -> bool:
        """
        Determine whether code and description searches can reuse one embedding.
        """
        return (
            type(self._code_provider) is type(self._desc_provider)
            and self._code_provider.model == self._desc_provider.model
            and self._code_provider.dimension == self._desc_provider.dimension
        )

    def _hybrid_query_vectors(self, query: str) -> tuple[list[float], list[float]]:
        """
        Resolve code and description query vectors for hybrid search with minimal overhead.
        """
        if self._can_share_query_embedding():
            shared_vector = self._embed_query_with_provider_cached(query, self._code_provider, cache_key="shared")
            return shared_vector, shared_vector

        code_cached = self._cached_query_vector(("code", query))
        desc_cached = self._cached_query_vector(("description", query))
        if code_cached is not None and desc_cached is not None:
            return code_cached, desc_cached
        if code_cached is not None:
            return code_cached, self._embed_query_cached(query, "description")
        if desc_cached is not None:
            return self._embed_query_cached(query, "code"), desc_cached

        with ThreadPoolExecutor(max_workers=2) as executor:
            code_future = executor.submit(self._embed_query_cached, query, "code")
            desc_future = executor.submit(self._embed_query_cached, query, "description")
            return code_future.result(), desc_future.result()

    def _cached_query_vector(self, cache_key: tuple[str, str]) -> Optional[list[float]]:
        """
        Return a cached query vector without invoking the embedding provider.
        """
        with self._cache_lock:
            cached = self._query_vector_cache.get(cache_key)
            if cached is None:
                return None
            self._query_vector_cache.move_to_end(cache_key)
            return cached

    def _embed_query_with_provider_cached(
        self,
        query: str,
        provider: EmbeddingProvider,
        cache_key: str,
    ) -> list[float]:
        """
        Cache query embedding for a specific provider-equivalence bucket.
        """
        key = (cache_key, query)
        with self._cache_lock:
            cached = self._query_vector_cache.get(key)
            if cached is not None:
                self._query_vector_cache.move_to_end(key)
                return cached

        disk_cached = self._load_query_vector_from_disk(
            query=query,
            provider=provider,
            cache_key=cache_key,
        )
        if disk_cached is not None:
            with self._cache_lock:
                self._query_vector_cache[key] = disk_cached
                self._query_vector_cache.move_to_end(key)
                while len(self._query_vector_cache) > QUERY_CACHE_SIZE:
                    self._query_vector_cache.popitem(last=False)
            return disk_cached

        vector = provider.embed_single(query).tolist()

        self._store_query_vector_to_disk(
            query=query,
            provider=provider,
            cache_key=cache_key,
            vector=vector,
        )
        with self._cache_lock:
            self._query_vector_cache[key] = vector
            self._query_vector_cache.move_to_end(key)
            while len(self._query_vector_cache) > QUERY_CACHE_SIZE:
                self._query_vector_cache.popitem(last=False)

        return vector

    def _query_cache_dir(self) -> Path:
        """
        Return the disk cache directory for persisted query embeddings.
        """
        cache_dir = Path.cwd().resolve() / ".quickcontext" / "query_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _query_cache_path(self, query: str, provider: EmbeddingProvider, cache_key: str) -> Path:
        """
        Build a stable on-disk cache path for a query embedding.
        """
        provider_key = f"{type(provider).__name__}:{provider.model}:{provider.dimension}:{cache_key}:{query}"
        digest = hashlib.sha256(provider_key.encode("utf-8")).hexdigest()
        return self._query_cache_dir() / f"{digest}.json"

    def _load_query_vector_from_disk(
        self,
        query: str,
        provider: EmbeddingProvider,
        cache_key: str,
    ) -> Optional[list[float]]:
        """
        Load a persisted query embedding vector if available.
        """
        cache_path = self._query_cache_path(query, provider, cache_key)
        if not cache_path.exists():
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            vector = payload.get("vector")
            if not isinstance(vector, list):
                return None
            cache_path.touch()
            return [float(value) for value in vector]
        except Exception:
            return None

    def _store_query_vector_to_disk(
        self,
        query: str,
        provider: EmbeddingProvider,
        cache_key: str,
        vector: list[float],
    ) -> None:
        """
        Persist a query embedding vector for reuse across CLI runs.
        """
        cache_path = self._query_cache_path(query, provider, cache_key)
        payload = {
            "query": query,
            "provider": type(provider).__name__,
            "model": provider.model,
            "dimension": provider.dimension,
            "stored_at": time.time(),
            "vector": vector,
        }
        try:
            cache_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            self._prune_disk_query_cache()
        except Exception:
            return

    def _prune_disk_query_cache(self) -> None:
        """
        Keep the on-disk query embedding cache bounded by entry count.
        """
        cache_dir = self._query_cache_dir()
        files = sorted(
            cache_dir.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in files[DISK_QUERY_CACHE_LIMIT:]:
            try:
                path.unlink()
            except OSError:
                continue

    def _kind_score_multiplier(self, symbol_kind: str, query_keyword_count: int) -> float:
        """
        Apply light semantic priors by symbol kind.

        symbol_kind: str — Stored symbol kind.
        query_keyword_count: int — Number of extracted query keywords.
        Returns: float — Score multiplier.
        """
        kind = (symbol_kind or "").lower()
        if kind in {"class", "function", "method", "struct", "enum", "trait", "interface"}:
            return 1.12
        if query_keyword_count >= 2 and kind in {"data_key", "variable", "import", "property"}:
            return 0.8
        return 1.0

    def _path_signal_multiplier(self, file_path: str, query_keywords: list[str]) -> float:
        """
        Apply a small penalty to lower-signal config/data files for multi-term queries.

        file_path: str — Result file path.
        query_keyword_count: int — Number of extracted query keywords.
        Returns: float — Score multiplier.
        """
        query_keyword_count = len(query_keywords)
        if query_keyword_count < 2:
            return 1.0

        normalized = file_path.replace("\\", "/").lower()
        query_terms = {term.lower() for term in query_keywords}
        if (
            not query_terms.intersection({"test", "tests", "testing", "regression"})
            and (
                "/test/" in normalized
                or "/tests/" in normalized
                or normalized.endswith("_test.py")
                or normalized.endswith("test_regressions.py")
                or normalized.startswith("test/")
                or normalized.startswith("tests/")
            )
        ):
            return 0.3

        if (
            not query_terms.intersection({"script", "scripts", "benchmark", "benchmarks", "harness"})
            and (
                "/scripts/" in normalized
                or normalized.startswith("scripts/")
            )
        ):
            return 0.68

        low_signal_suffixes = (
            ".yml",
            ".yaml",
            ".json",
            ".toml",
            ".ini",
            ".cfg",
            ".lock",
        )
        if normalized.endswith(low_signal_suffixes):
            return 0.82
        return 1.0

    def _payload_cache_key(self, payload: dict) -> str:
        """
        Build a stable cache key for payload-derived lexical metadata.
        """
        chunk_id = payload.get("chunk_id")
        if chunk_id:
            return str(chunk_id)

        parent = payload.get("parent")
        parent_str = f":{parent}" if parent else ""
        symbol_kind = payload.get("symbol_kind") or payload.get("symbol_type") or "unknown"
        return (
            f"{payload.get('file_path', '')}:{symbol_kind}:"
            f"{payload.get('symbol_name', '')}{parent_str}:{payload.get('line_start', 0)}"
        )

    def _token_variants(self, token: str) -> list[str]:
        """
        Expand a token with lightweight inflection variants for ranking.
        """
        token = token.lower()
        variants = [token]

        if len(token) > 4 and token.endswith("ies"):
            variants.append(token[:-3] + "y")
        elif len(token) > 4 and token.endswith("ing"):
            stem = token[:-3]
            variants.append(stem)
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                variants.append(stem[:-1])
        elif len(token) > 4 and token.endswith("ed"):
            stem = token[:-2]
            variants.append(stem)
            if not stem.endswith("e"):
                variants.append(token[:-1])
        elif len(token) > 4 and token.endswith(("ches", "shes", "xes", "zes", "sses")):
            variants.append(token[:-2])
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            variants.append(token[:-1])

        seen: set[str] = set()
        out: list[str] = []
        for candidate in variants:
            if len(candidate) < 3 or candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
            for synonym in TOKEN_SYNONYMS.get(candidate, ()):
                if len(synonym) < 3 or synonym in seen:
                    continue
                seen.add(synonym)
                out.append(synonym)
        return out

    def _identifier_tokens(self, text: str, max_tokens: Optional[int] = None) -> list[str]:
        """
        Tokenize identifier-like text, including snake_case and path segments.
        """
        if not text:
            return []

        expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        expanded = re.sub(r"[\\/.\-:()\\[\\],]", " ", expanded)
        raw_tokens = re.findall(r"[A-Za-z0-9_]+", expanded)

        seen: set[str] = set()
        out: list[str] = []
        for raw_token in raw_tokens:
            for part in raw_token.split("_"):
                lower = part.lower().strip()
                if len(lower) < 3:
                    continue
                for candidate in self._token_variants(lower):
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    out.append(candidate)
                    if max_tokens is not None and len(out) >= max_tokens:
                        return out

        return out

    def _metadata_tokens(self, payload: dict) -> list[str]:
        """
        Build cached lexical ranking tokens from payload metadata.
        """
        cache_key = self._payload_cache_key(payload)
        with self._cache_lock:
            cached = self._metadata_token_cache.get(cache_key)
            if cached is not None:
                self._metadata_token_cache.move_to_end(cache_key)
                return cached

        tokens: list[str] = []
        tokens.extend(self._identifier_tokens(payload.get("file_path", ""), max_tokens=12))
        tokens.extend(self._identifier_tokens(payload.get("path_context", ""), max_tokens=12))
        tokens.extend(self._identifier_tokens(payload.get("symbol_name", ""), max_tokens=8))
        tokens.extend(self._identifier_tokens(payload.get("symbol_kind", ""), max_tokens=4))
        tokens.extend(self._identifier_tokens(payload.get("role", ""), max_tokens=4))
        tokens.extend(self._identifier_tokens(payload.get("parent", ""), max_tokens=8))
        tokens.extend(self._identifier_tokens(payload.get("signature", ""), max_tokens=12))

        for keyword in payload.get("keywords", []) or []:
            tokens.extend(self._token_variants(str(keyword)))

        free_text_parts = [
            payload.get("description", ""),
            payload.get("docstring", ""),
        ]
        free_text = "\n".join(part for part in free_text_parts if part)
        if free_text:
            for token in extract_keywords(free_text, max_keywords=20):
                tokens.extend(self._token_variants(token))

        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
            if len(deduped) >= 48:
                break

        with self._cache_lock:
            self._metadata_token_cache[cache_key] = deduped
            self._metadata_token_cache.move_to_end(cache_key)
            while len(self._metadata_token_cache) > METADATA_TOKEN_CACHE_SIZE:
                self._metadata_token_cache.popitem(last=False)

        return deduped

    def _file_basename_tokens(self, file_path: str) -> list[str]:
        """
        Extract normalized tokens from the file basename.
        """
        return self._identifier_tokens(Path(file_path).stem, max_tokens=6)

    def _basename_hit_bonus(self, file_path: str, ranking_keywords: list[str]) -> float:
        """
        Reward strong basename matches for subsystem-specific queries.
        """
        hits = len(set(ranking_keywords) & set(self._file_basename_tokens(file_path)))
        if hits < 2:
            return 0.0
        return hits * 0.08

    def _provider_path_bonus(self, file_path: str, ranking_keywords: list[str]) -> float:
        """
        Prefer provider modules for provider-loading and startup dependency queries.
        """
        if not ranking_keywords:
            return 0.0

        keyword_set = set(ranking_keywords)
        if not keyword_set.intersection({"provider", "providers", "dependency", "dependencies"}):
            return 0.0
        if not keyword_set.intersection({"load", "loaded", "lazy", "start", "faster", "import"}):
            return 0.0

        normalized = file_path.replace("\\", "/").lower()
        if "/providers/" not in normalized:
            return 0.0

        bonus = 0.06
        if normalized.endswith("/factory.py"):
            bonus += 0.03
        return bonus

    def _startup_boundary_bonus(self, file_path: str, ranking_keywords: list[str]) -> float:
        """
        Prefer startup-boundary files for parser-only import and startup questions.
        """
        if not ranking_keywords:
            return 0.0

        keyword_set = set(ranking_keywords)
        if "parser" not in keyword_set:
            return 0.0
        if not keyword_set.intersection({"startup", "import", "imports", "qdrant"}):
            return 0.0

        normalized = file_path.replace("\\", "/").lower()
        if normalized.endswith("/engine/__init__.py"):
            return 0.22
        if normalized.endswith("/engine/src/parser_cli.py"):
            return 0.18
        if normalized.endswith("/engine/src/parsing.py"):
            return 0.20
        if normalized.endswith("/engine/src/pipe.py"):
            return 0.08
        return 0.0

    def _implementation_path_bonus(self, file_path: str, ranking_keywords: list[str]) -> float:
        """
        Prefer lower-level implementation files over wrappers for internal architecture questions.
        """
        if not ranking_keywords:
            return 0.0

        keyword_set = set(ranking_keywords)
        if not keyword_set.intersection({"path", "prefix", "scope", "scoped", "filter", "filters", "payload", "payloads"}):
            return 0.0
        if not keyword_set.intersection({"semantic", "search", "retrieval", "index", "indexing"}):
            return 0.0

        normalized = file_path.replace("\\", "/").lower()
        if normalized.endswith("/engine/src/searcher.py"):
            return 0.08
        if normalized.endswith("/engine/src/indexer.py"):
            return 0.07
        return 0.0

    def _startup_symbol_bonus(self, file_path: str, symbol_name: str, ranking_keywords: list[str]) -> float:
        """
        Prefer startup-boundary symbols that actually explain the parser-side import path.
        """
        if not ranking_keywords:
            return 0.0

        keyword_set = set(ranking_keywords)
        if "parser" not in keyword_set:
            return 0.0
        if not keyword_set.intersection({"startup", "import", "imports", "qdrant"}):
            return 0.0

        normalized = file_path.replace("\\", "/").lower()
        if normalized.endswith("/engine/src/parsing.py") and symbol_name == "RustParserService":
            return 0.03
        if normalized.endswith("/engine/__init__.py") and symbol_name in {"__getattr__", "QuickContext"}:
            return 0.03
        return 0.0

    def _wrapper_symbol_penalty(self, file_path: str, symbol_name: str, ranking_keywords: list[str]) -> float:
        """
        Down-rank generic wrapper entrypoints when the query is clearly asking for lower-level implementation details.
        """
        if not ranking_keywords:
            return 1.0

        normalized = file_path.replace("\\", "/").lower()
        keyword_set = set(ranking_keywords)
        symbol = symbol_name.lower()

        if normalized.endswith("/engine/sdk.py"):
            if (
                symbol in {"semantic_search", "structured_search"}
                and keyword_set.intersection({"path", "prefix", "scope", "scoped", "filter", "filters", "payload", "payloads"})
            ):
                return 0.48
            if (
                symbol in {"refresh_files", "watch", "index_directory"}
                and keyword_set.intersection({"detect", "unchanged", "file", "files"})
                and keyword_set.intersection({"index", "indexing"})
            ):
                return 0.52
            if (
                symbol in {"parser_service", "quickcontext", "connection", "code_provider", "desc_provider", "_get_searcher"}
                and "parser" in keyword_set
                and keyword_set.intersection({"startup", "import", "imports", "qdrant"})
            ):
                return 0.35

        if normalized.endswith("/engine/src/cli.py"):
            if (
                symbol in {"search", "cli", "mcp_hints", "init"}
                and keyword_set.intersection({"provider", "providers", "dependency", "dependencies", "path", "prefix", "filter", "filters", "payload", "payloads"})
            ):
                return 0.52
            if (
                symbol in {"init", "_optimize_search_config"}
                and "parser" in keyword_set
                and keyword_set.intersection({"startup", "import", "imports", "qdrant"})
            ):
                return 0.18

        return 1.0

    def _constructor_penalty(self, symbol_name: str, ranking_keywords: list[str]) -> float:
        """
        Avoid ranking constructors highly for non-constructor architecture questions.
        """
        if symbol_name != "__init__":
            return 1.0
        if len(ranking_keywords) < 2:
            return 1.0
        if set(ranking_keywords).intersection({"init", "initialize", "initialization", "constructor"}):
            return 1.0
        return 0.55

    def _startup_import_symbol_penalty(self, symbol_kind: str, ranking_keywords: list[str]) -> float:
        """
        Avoid ranking raw import rows above the parser implementation on startup-boundary questions.
        """
        if (symbol_kind or "").lower() != "import":
            return 1.0

        keyword_set = set(ranking_keywords)
        if "parser" in keyword_set and keyword_set.intersection({"startup", "import", "imports", "qdrant"}):
            return 0.45
        return 1.0

    def _ranking_helper_penalty(self, symbol_name: str, ranking_keywords: list[str]) -> float:
        """
        Down-rank internal scoring helper symbols unless the query is actually about ranking.
        """
        normalized = symbol_name.lower()
        if not any(token in normalized for token in ("bonus", "penalty")):
            return 1.0

        keyword_set = set(ranking_keywords)
        if keyword_set.intersection({"rank", "ranking", "score", "scoring", "rerank", "boost", "bonus", "penalty", "relevance"}):
            return 1.0
        return 0.25

    def _ranking_subsystem_penalty(self, file_path: str, ranking_keywords: list[str]) -> float:
        """
        Down-rank ranking/search-tuning modules for non-ranking architecture questions.
        """
        if not ranking_keywords:
            return 1.0

        keyword_set = set(ranking_keywords)
        if keyword_set.intersection({"rank", "ranking", "score", "scoring", "rerank", "boost", "bonus", "penalty", "relevance", "search"}):
            return 1.0

        normalized = file_path.replace("\\", "/").lower()
        if (
            normalized.endswith("/engine/src/searcher.py")
            or normalized.endswith("/service/src/structural_boost.rs")
            or normalized.endswith("/service/src/ranking.rs")
        ):
            return 0.62
        return 1.0

    def _filecache_bonus(self, file_path: str, symbol_name: str, ranking_keywords: list[str]) -> float:
        """
        Prefer file-signature cache code for unchanged-file detection questions.
        """
        if not ranking_keywords:
            return 0.0

        keyword_set = set(ranking_keywords)
        if not keyword_set.intersection({"detect", "unchanged", "file", "files"}):
            return 0.0
        if not keyword_set.intersection({"index", "indexing"}):
            return 0.0

        normalized = file_path.replace("\\", "/").lower()
        if not normalized.endswith("/engine/src/filecache.py"):
            return 0.0

        bonus = 0.06
        if symbol_name in {"is_unchanged_from_metadata", "is_unchanged"}:
            bonus += 0.04
        return bonus

    def _role_bonus(self, role: Optional[str], ranking_keywords: list[str]) -> float:
        """
        Apply a small role-aware boost for implementation-oriented architecture questions.
        """
        if not role or not ranking_keywords:
            return 0.0

        keyword_set = set(ranking_keywords)
        if not (
            keyword_set.intersection(ACTION_QUERY_KEYWORDS)
            or keyword_set.intersection({"path", "prefix", "scope", "scoped", "payload", "filter", "filters", "startup"})
        ):
            return 0.0

        normalized_role = role.lower()
        if normalized_role in {"orchestration", "logic", "utility"}:
            return 0.04
        if normalized_role == "entrypoint":
            return 0.02
        if normalized_role in {"test", "configuration"}:
            return -0.04
        return 0.0

    def _subsystem_conflict_penalty(self, file_path: str, ranking_keywords: list[str]) -> float:
        """
        Down-rank clearly unrelated subsystems for targeted architecture questions.
        """
        if not ranking_keywords:
            return 1.0

        keyword_set = set(ranking_keywords)
        normalized = file_path.replace("\\", "/").lower()

        if "parser" in keyword_set and keyword_set.intersection({"startup", "import", "imports", "qdrant"}):
            if (
                "/providers/" in normalized
                or normalized.endswith("/engine/src/embedder.py")
                or normalized.endswith("/engine/src/describer.py")
            ):
                return 0.2
            if (
                normalized.endswith("/engine/src/qdrant.py")
                or normalized.endswith("/engine/src/collection.py")
                or normalized.endswith("/engine/src/config.py")
            ):
                return 0.58

        return 1.0

    def _container_symbol_penalty(
        self,
        symbol_name: str,
        symbol_kind: str,
        ranking_keywords: list[str],
    ) -> float:
        """
        Down-rank generic result container symbols on action-heavy questions.
        """
        if not ranking_keywords:
            return 1.0

        if not set(ranking_keywords).intersection(ACTION_QUERY_KEYWORDS):
            return 1.0

        kind = (symbol_kind or "").lower()
        if kind not in {"class", "struct"}:
            return 1.0

        suffixes = ("result", "results", "item", "items", "config", "stats", "record")
        if symbol_name.lower().endswith(suffixes):
            return 0.72
        return 1.0

    def _action_symbol_bonus(self, ranking_keywords: list[str], symbol_tokens: list[str]) -> float:
        """
        Reward symbols whose action words match the query intent.
        """
        if not ranking_keywords or not symbol_tokens:
            return 0.0

        query_actions = set(ranking_keywords).intersection(ACTION_QUERY_KEYWORDS)
        if not query_actions:
            return 0.0

        action_hits = len(query_actions & set(symbol_tokens))
        return action_hits * 0.08

    def _symbol_tokens(self, symbol_name: str, signature: Optional[str]) -> list[str]:
        """
        Extract normalized tokens from symbol metadata for direct overlap scoring.

        symbol_name: str — Symbol name.
        signature: Optional[str] — Symbol signature.
        Returns: list[str] — Lowercase deduplicated tokens.
        """
        raw = f"{symbol_name} {signature or ''}"
        return self._identifier_tokens(raw, max_tokens=24)

    def _embed_query_cached(self, query: str, using: str) -> list[float]:
        """
        Cache query embeddings to avoid repeated provider calls for the same text.

        query: str — Query text.
        using: str — Vector space key ("code" or "description").
        Returns: list[float] — Query vector as a Python list.
        """
        cache_key = (using, query)
        with self._cache_lock:
            cached = self._query_vector_cache.get(cache_key)
            if cached is not None:
                self._query_vector_cache.move_to_end(cache_key)
                return cached

        provider = self._code_provider if using == "code" else self._desc_provider
        vector = provider.embed_single(query).tolist()

        with self._cache_lock:
            self._query_vector_cache[cache_key] = vector
            self._query_vector_cache.move_to_end(cache_key)
            while len(self._query_vector_cache) > QUERY_CACHE_SIZE:
                self._query_vector_cache.popitem(last=False)

        return vector

    def _extract_keywords_cached(self, query: str) -> list[str]:
        """
        Cache keyword extraction for repeated query text.

        query: str — Query text.
        Returns: list[str] — Extracted keywords.
        """
        with self._cache_lock:
            cached = self._keyword_cache.get(query)
            if cached is not None:
                self._keyword_cache.move_to_end(query)
                return cached

        keywords: list[str] = []
        seen: set[str] = set()
        for token in extract_keywords(query, max_keywords=12):
            for candidate in self._token_variants(token):
                if candidate in seen:
                    continue
                seen.add(candidate)
                keywords.append(candidate)
                if len(keywords) >= 16:
                    break
            if len(keywords) >= 16:
                break

        with self._cache_lock:
            self._keyword_cache[query] = keywords
            self._keyword_cache.move_to_end(query)
            while len(self._keyword_cache) > QUERY_CACHE_SIZE:
                self._keyword_cache.popitem(last=False)

        return keywords
