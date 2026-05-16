from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Reranker:

    def __init__(self, model_name: str) -> None:
        # Import here to avoid loading at import time when reranker is disabled
        from sentence_transformers import CrossEncoder  # noqa: WPS433

        logger.info("loading_reranker", model=model_name)
        self._model = CrossEncoder(model_name)
        self.model_name = model_name
        logger.info("reranker_ready", model=model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores: list = self._model.predict(pairs).tolist()

        scored: List[Dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            enriched = {**candidate, "rerank_score": float(score)}
            scored.append(enriched)

        # Sort descending by cross-encoder score
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.debug(
            "reranking_complete",
            num_candidates=len(candidates),
            top_n=top_n,
            top_score=scored[0]["rerank_score"] if scored else None,
        )

        return scored[:top_n]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    settings = get_settings()
    return Reranker(model_name=settings.reranker_model)
