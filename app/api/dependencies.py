from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings, get_settings
from app.rag.pipeline import RAGPipeline
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.llm_service import LLMService
from app.services.vector_store_service import VectorStoreService


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreService:
    embedding_service = get_embedding_service()
    return VectorStoreService(embedding_service=embedding_service)


@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    settings = get_settings()
    return RAGPipeline(
        vector_store=get_vector_store(),
        embedding_service=get_embedding_service(),
        llm_service=get_llm_service(),
        enable_reranker=settings.enable_reranker,
    )


# ── Type aliases for route annotations ────────────────────────────────────────
SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
VectorStoreDep = Annotated[VectorStoreService, Depends(get_vector_store)]
LLMServiceDep = Annotated[LLMService, Depends(get_llm_service)]
RAGPipelineDep = Annotated[RAGPipeline, Depends(get_rag_pipeline)]
