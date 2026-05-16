from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.embedding_service import EmbeddingService
from app.services.vector_store_service import VectorStoreService

logger = get_logger(__name__)

_RRF_K = 60  # Standard RRF constant from Cormack et al. (2009)


class HybridRetriever:

    def __init__(
        self,
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service
        self._settings = get_settings()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:

        settings = self._settings
        # Fetch more candidates than needed before fusion
        fetch_k = min(top_k * 4, 40)

        # ── Dense retrieval ────────────────────────────────────────────────────
        query_embedding = self._embedding_service.embed_query(query)
        dense_results = self._vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            where=filters,
        )
        logger.debug("dense_retrieved", count=len(dense_results))

        # ── Keyword retrieval (hybrid only) ────────────────────────────────────
        keyword_results: List[Dict[str, Any]] = []
        if settings.hybrid_alpha < 1.0:
            try:
                keyword_results = self._vector_store.keyword_search(
                    query=query,
                    top_k=fetch_k,
                    where=filters,
                )
                logger.debug("keyword_retrieved", count=len(keyword_results))
            except Exception as exc:
                # Keyword search degradation: fall back to dense-only.
                logger.warning("keyword_search_failed", error=str(exc))

        # ── Fusion ─────────────────────────────────────────────────────────────
        if keyword_results:
            fused = _reciprocal_rank_fusion(
                dense_results=dense_results,
                keyword_results=keyword_results,
                alpha=settings.hybrid_alpha,
                top_k=top_k,
            )
            for r in fused:
                r["retrieval_source"] = "hybrid"
        else:
            fused = dense_results[:top_k]
            for r in fused:
                r["retrieval_source"] = "dense"

        logger.info(
            "retrieval_complete",
            query_preview=query[:60],
            num_results=len(fused),
            hybrid=bool(keyword_results),
        )
        return fused


def _reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    alpha: float = 0.7,
    top_k: int = 5,
) -> List[Dict[str, Any]]:

    scores: Dict[str, float] = {}
    # Map from chunk id → result dict (dense result takes precedence)
    id_to_result: Dict[str, Dict[str, Any]] = {}

    # Dense contributions (weight = alpha)
    for rank, result in enumerate(dense_results, start=1):
        cid = result["id"]
        scores[cid] = scores.get(cid, 0.0) + alpha * (1.0 / (_RRF_K + rank))
        id_to_result[cid] = result

    # Keyword contributions (weight = 1 - alpha)
    keyword_weight = 1.0 - alpha
    for rank, result in enumerate(keyword_results, start=1):
        cid = result["id"]
        scores[cid] = scores.get(cid, 0.0) + keyword_weight * (1.0 / (_RRF_K + rank))
        if cid not in id_to_result:
            id_to_result[cid] = result

    # Sort by RRF score, descending
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    merged: List[Dict[str, Any]] = []
    for cid in sorted_ids[:top_k]:
        result = {**id_to_result[cid], "rrf_score": scores[cid]}
        merged.append(result)

    return merged