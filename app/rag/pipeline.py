from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import (
    JobChunk,
    QueryResponse,
    RetrievalMetadata,
    SourceReference,
)
from app.rag.prompts import SYSTEM_PROMPT, build_answer_prompt, format_context
from app.rag.retriever import HybridRetriever
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.vector_store_service import VectorStoreService
from app.utils.preprocessor import truncate_text

logger = get_logger(__name__)


class RAGPipeline:

    def __init__(
        self,
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        enable_reranker: bool = True,
    ) -> None:
        self._retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_service=embedding_service,
        )
        self._llm = llm_service
        self._embedding_service = embedding_service
        self._settings = get_settings()
        self._enable_reranker = enable_reranker and self._settings.enable_reranker

        if self._enable_reranker:
            from app.rag.reranker import get_reranker  # noqa: WPS433 — lazy import
            self._reranker = get_reranker()
        else:
            self._reranker = None

        logger.info(
            "rag_pipeline_ready",
            reranker_enabled=self._enable_reranker,
            hybrid_alpha=self._settings.hybrid_alpha,
        )

    def run(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> QueryResponse:
        t_start = time.perf_counter()

        fetch_k = (
            min(top_k * 4, self._settings.default_top_k * 4)
            if self._enable_reranker
            else top_k
        )

        raw_candidates = self._retriever.retrieve(
            query=query,
            top_k=fetch_k,
            filters=filters,
        )

        hybrid_applied = any(
            r.get("retrieval_source") == "hybrid" for r in raw_candidates
        )

        reranker_applied = False
        if self._enable_reranker and self._reranker and raw_candidates:
            final_candidates = self._reranker.rerank(
                query=query,
                candidates=raw_candidates,
                top_n=min(top_k, self._settings.reranker_top_n),
            )
            reranker_applied = True
        else:
            final_candidates = raw_candidates[:top_k]

        context = format_context(final_candidates)
        user_prompt = build_answer_prompt(query=query, context=context)

        # ── Step 4: Generate answer ────────────────────────────────────────────
        try:
            answer = self._llm.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.error("llm_generation_failed", error=str(exc))
            answer = (
                "I found relevant job listings but was unable to generate a "
                "synthesised answer at this time. Please review the results below."
            )

        # ── Step 5: Assemble response ──────────────────────────────────────────
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        job_chunks = [_to_job_chunk(r) for r in final_candidates]
        sources = _deduplicate_sources(final_candidates)

        metadata = RetrievalMetadata(
            total_chunks_retrieved=len(raw_candidates),
            reranker_applied=reranker_applied,
            hybrid_search_applied=hybrid_applied,
            latency_ms=round(latency_ms, 2),
            embedding_model=self._embedding_service.model_name,
            llm_model=f"{self._llm.provider}:{self._llm.model_name}",
        )

        logger.info(
            "pipeline_complete",
            query_preview=query[:60],
            latency_ms=round(latency_ms, 2),
            num_results=len(job_chunks),
            reranker_applied=reranker_applied,
            hybrid_applied=hybrid_applied,
            llm_provider=self._llm.provider,
            llm_model=self._llm.model_name,
        )

        return QueryResponse(
            query=query,
            answer=answer,
            results=job_chunks,
            sources=sources,
            retrieval_metadata=metadata,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_job_chunk(result: Dict[str, Any]) -> JobChunk:
    """Convert a raw retrieval result dict to a JobChunk Pydantic model."""
    meta = result.get("metadata", {})
    return JobChunk(
        chunk_id=result["id"],
        job_id=meta.get("job_id", ""),
        job_title=meta.get("job_title", ""),
        company_name=meta.get("company_name", ""),
        job_category=meta.get("job_category", ""),
        job_level=meta.get("job_level", ""),
        job_location=meta.get("job_location", ""),
        publication_date=meta.get("publication_date"),
        tags=meta.get("tags"),
        text_snippet=truncate_text(result.get("text", ""), max_chars=300),
        similarity_score=round(result.get("similarity_score", 0.0), 4),
        rerank_score=result.get("rerank_score"),
    )


def _deduplicate_sources(results: List[Dict[str, Any]]) -> List[SourceReference]:
    seen: set = set()
    sources: List[SourceReference] = []
    for result in results:
        meta = result.get("metadata", {})
        job_id = meta.get("job_id", "")
        if job_id and job_id not in seen:
            seen.add(job_id)
            sources.append(
                SourceReference(
                    job_id=job_id,
                    job_title=meta.get("job_title", ""),
                    company_name=meta.get("company_name", ""),
                    job_location=meta.get("job_location", ""),
                    job_level=meta.get("job_level", ""),
                )
            )
    return sources
