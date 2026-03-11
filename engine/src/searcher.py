from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from qdrant_client import QdrantClient, models

from engine.src.keywords import extract_keywords, keyword_overlap_score
from engine.src.providers import EmbeddingProvider
from engine.src.query_dsl import StructuredSubQuery

if TYPE_CHECKING:
    from engine.src.reranker import ColBERTReranker


RRF_K = 60
TOP_RANK_BONUS_1 = 0.05
TOP_RANK_BONUS_2_3 = 0.02


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
        client: QdrantClient,
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
        query_vector = self._code_provider.embed_single(query).tolist()
        query_keywords = extract_keywords(query) if use_keywords else []

        return self._search(
            query=query,
            query_vector=query_vector,
            using="code",
            limit=limit,
            language=language,
            file_path=file_path,
            path_prefix=path_prefix,
            symbol_kind=symbol_kind,
            query_keywords=query_keywords,
            keyword_weight=keyword_weight,
            rerank=rerank,
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
        query_vector = self._desc_provider.embed_single(query).tolist()
        query_keywords = extract_keywords(query) if use_keywords else []

        return self._search(
            query=query,
            query_vector=query_vector,
            using="description",
            limit=limit,
            language=language,
            file_path=file_path,
            path_prefix=path_prefix,
            symbol_kind=symbol_kind,
            query_keywords=query_keywords,
            keyword_weight=keyword_weight,
            rerank=rerank,
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
        code_results = self.search_code(
            query=query,
            limit=max(limit * 3, 30),
            language=language,
            file_path=file_path,
            path_prefix=path_prefix,
            symbol_kind=symbol_kind,
            use_keywords=use_keywords,
            keyword_weight=keyword_weight,
            rerank=False,
        )
        desc_results = self.search_description(
            query=query,
            limit=max(limit * 3, 30),
            language=language,
            file_path=file_path,
            path_prefix=path_prefix,
            symbol_kind=symbol_kind,
            use_keywords=use_keywords,
            keyword_weight=keyword_weight,
            rerank=False,
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

        return fused[:limit]

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

        result_lists: list[list[SearchResult]] = []
        weights: list[float] = []

        for idx, sub in enumerate(sub_queries):
            weight = first_query_weight if idx == 0 else 1.0

            if sub.kind == "lex":
                results = self.search_code(
                    query=sub.text,
                    limit=max(limit * 3, 30),
                    language=language,
                    file_path=file_path,
                    path_prefix=path_prefix,
                    symbol_kind=symbol_kind,
                    use_keywords=use_keywords,
                    keyword_weight=max(keyword_weight, 0.45),
                    rerank=False,
                )
            elif sub.kind == "vec":
                results = self.search_description(
                    query=sub.text,
                    limit=max(limit * 3, 30),
                    language=language,
                    file_path=file_path,
                    path_prefix=path_prefix,
                    symbol_kind=symbol_kind,
                    use_keywords=use_keywords,
                    keyword_weight=keyword_weight,
                    rerank=False,
                )
            else:
                results = self.search_description(
                    query=sub.text,
                    limit=max(limit * 3, 30),
                    language=language,
                    file_path=file_path,
                    path_prefix=path_prefix,
                    symbol_kind=symbol_kind,
                    use_keywords=use_keywords,
                    keyword_weight=keyword_weight,
                    rerank=False,
                )

            if results:
                result_lists.append(results)
                weights.append(weight)

        if not result_lists:
            return []

        fused = self._rrf_fuse(
            result_lists,
            weights,
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

        return fused[:limit]

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
        candidates = results[:candidate_count]

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
        query_keywords: list[str],
        keyword_weight: float,
        rerank: bool = False,
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
            query_keywords: list[str] — Extracted keywords from query.
            keyword_weight: float — Weight for keyword overlap score.
            rerank: bool — Enable ColBERT reranking.

        Returns:
            list[SearchResult] — Sorted search results.
        """
        filters = []

        if language:
            filters.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(value=language),
                )
            )

        if file_path:
            filters.append(
                models.FieldCondition(
                    key="file_path",
                    match=models.MatchValue(value=file_path),
                )
            )

        normalized_path_prefix = None
        if path_prefix:
            normalized_path_prefix = path_prefix.replace("\\", "/").strip().strip("/").lower()
            if normalized_path_prefix:
                filters.append(
                    models.FieldCondition(
                        key="path_prefixes",
                        match=models.MatchValue(value=normalized_path_prefix),
                    )
                )

        if symbol_kind:
            filters.append(
                models.Filter(
                    should=[
                        models.FieldCondition(
                            key="symbol_kind",
                            match=models.MatchValue(value=symbol_kind),
                        ),
                        models.FieldCondition(
                            key="symbol_type",
                            match=models.MatchValue(value=symbol_kind),
                        ),
                    ]
                )
            )

        query_filter = models.Filter(must=filters) if filters else None

        rerank_active = rerank and self._reranker is not None
        if rerank_active:
            fetch_limit = limit * 5
        elif query_keywords:
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
            with_payload=True,
        )

        if normalized_path_prefix and not results.points:
            fallback_filters = [
                condition for condition in filters
                if not (
                    isinstance(condition, models.FieldCondition)
                    and getattr(condition, "key", None) == "path_prefixes"
                )
            ]
            fallback_filter = models.Filter(must=fallback_filters) if fallback_filters else None
            results = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                using=using,
                query_filter=fallback_filter,
                limit=max(fetch_limit, limit * 25),
                with_payload=True,
            )

        search_results = []
        for point in results.points:
            vector_score = point.score
            chunk_keywords = point.payload.get("keywords", [])
            path_context = point.payload.get("path_context")

            if query_keywords and chunk_keywords:
                kw_score = keyword_overlap_score(query_keywords, chunk_keywords)
                final_score = vector_score * (1 - keyword_weight) + kw_score * keyword_weight
            else:
                final_score = vector_score

            if path_context and query_keywords:
                path_keywords = extract_keywords(path_context)
                if path_keywords:
                    final_score += keyword_overlap_score(query_keywords, path_keywords) * 0.1

            symbol_kind_value = point.payload.get("symbol_kind") or point.payload.get("symbol_type") or "unknown"
            symbol_name_value = point.payload["symbol_name"]
            file_path_value = point.payload["file_path"]

            if query_keywords:
                generic_symbol_names = {"main", "entry", "_start"}
                normalized_symbol_name = symbol_name_value.lower()
                if (
                    len(query_keywords) > 1
                    and normalized_symbol_name in generic_symbol_names
                    and all(term not in normalized_symbol_name for term in query_keywords)
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
                    source=point.payload["source"],
                    description=point.payload["description"],
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

        return search_results[:limit]
