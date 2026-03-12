from dataclasses import dataclass

import requests

from engine.src.config import QdrantConfig


@dataclass(frozen=True, slots=True)
class SearchPoint:
    id: int | str | None
    score: float
    payload: dict


@dataclass(frozen=True, slots=True)
class SearchResponse:
    points: list[SearchPoint]


@dataclass(frozen=True, slots=True)
class SearchRecord:
    id: int | str | None
    payload: dict


class RestQdrantSearchClient:
    """
    Lightweight REST-only client for Qdrant search and retrieve operations.
    """

    def __init__(self, config: QdrantConfig):
        self._config = config
        self._base_url = f"http://{config.host}:{config.port}"
        self._timeout = config.timeout
        self._session = requests.Session()
        if config.api_key:
            self._session.headers["api-key"] = config.api_key

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        using: str,
        query_filter: dict | None,
        limit: int,
        with_payload,
    ) -> SearchResponse:
        body = {
            "query": query,
            "using": using,
            "limit": limit,
            "with_payload": with_payload,
            "with_vector": False,
        }
        if query_filter:
            body["filter"] = query_filter

        payload = self._post_json(f"/collections/{collection_name}/points/query", body)
        points = payload.get("result", {}).get("points", [])
        return SearchResponse(points=[self._point_from_dict(item) for item in points])

    def query_batch_points(self, collection_name: str, requests: list[dict]) -> list[SearchResponse]:
        payload = self._post_json(
            f"/collections/{collection_name}/points/query/batch",
            {"searches": requests},
        )
        results = payload.get("result", [])
        return [
            SearchResponse(points=[self._point_from_dict(item) for item in result.get("points", [])])
            for result in results
        ]

    def retrieve(
        self,
        collection_name: str,
        ids: list[int | str],
        with_payload,
        with_vectors: bool,
    ) -> list[SearchRecord]:
        body = {
            "ids": ids,
            "with_payload": with_payload,
            "with_vector": with_vectors,
        }
        payload = self._post_json(f"/collections/{collection_name}/points", body)
        return [
            SearchRecord(id=item.get("id"), payload=item.get("payload") or {})
            for item in payload.get("result", [])
        ]

    def close(self) -> None:
        self._session.close()

    def _post_json(self, path: str, body: dict) -> dict:
        response = self._session.post(
            f"{self._base_url}{path}",
            json=body,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def _point_from_dict(self, item: dict) -> SearchPoint:
        return SearchPoint(
            id=item.get("id"),
            score=float(item.get("score", 0.0)),
            payload=item.get("payload") or {},
        )
