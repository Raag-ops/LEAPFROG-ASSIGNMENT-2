from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request models ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=3,
        max_length=512,
        description="Natural-language job search query",
        examples=["remote python jobs in fintech"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of job chunks to retrieve before reranking",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional metadata filters applied at the vector-store level. "
            "Supported keys: job_category, job_level, company_name."
        ),
        examples=[{"job_level": "Senior Level", "job_category": "Engineering"}],
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank or whitespace-only")
        return v.strip()


# ── Sub-models for response ────────────────────────────────────────────────────

class JobChunk(BaseModel):
    """A single retrieved job chunk with its metadata and similarity score."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    job_id: str = Field(..., description="Source job listing ID (e.g. LF0001)")
    job_title: str
    company_name: str
    job_category: str
    job_level: str
    job_location: str
    publication_date: Optional[str] = None
    tags: Optional[str] = None
    text_snippet: str = Field(
        ..., description="First 300 characters of the matched chunk"
    )
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity (dense retrieval)"
    )
    rerank_score: Optional[float] = Field(
        default=None, description="Cross-encoder relevance score (if reranker enabled)"
    )


class SourceReference(BaseModel):
    """Compact citation for the LLM answer's source jobs."""

    job_id: str
    job_title: str
    company_name: str
    job_location: str
    job_level: str


# ── Response models ────────────────────────────────────────────────────────────

class QueryResponse(BaseModel):

    query: str
    answer: str = Field(..., description="LLM-synthesised answer grounded in retrieved chunks")
    results: List[JobChunk] = Field(..., description="Retrieved and (optionally) reranked chunks")
    sources: List[SourceReference] = Field(
        ..., description="Deduplicated list of source jobs cited in the answer"
    )
    retrieval_metadata: RetrievalMetadata


class RetrievalMetadata(BaseModel):
    total_chunks_retrieved: int
    reranker_applied: bool
    hybrid_search_applied: bool
    latency_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    embedding_model: str
    llm_model: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Health check ───────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    vector_store_ready: bool
    embedding_model: str
    llm_model: str
