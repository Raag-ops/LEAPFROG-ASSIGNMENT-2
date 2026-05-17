from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import RAGPipelineDep, SettingsDep, VectorStoreDep
from app.core.logging import get_logger
from app.models.schemas import HealthResponse, QueryRequest, QueryResponse

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the RAG pipeline",
    description=(
        "Submit a natural-language job search query. The pipeline retrieves "
        "relevant job listing chunks, optionally reranks them, and generates "
        "a synthesised answer grounded exclusively in the retrieved context."
    ),
    responses={
        200: {"description": "Successful RAG response"},
        422: {"description": "Validation error in request body"},
        503: {"description": "Vector store not ready (run the ingest script first)"},
        500: {"description": "Internal pipeline error"},
    },
)
async def query(
    request: QueryRequest,
    pipeline: RAGPipelineDep,
    vector_store: VectorStoreDep,
) -> QueryResponse:

    # Guard: reject requests when the vector store is empty
    if not vector_store.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Vector store is empty. Run `python scripts/ingest.py` to "
                "populate it before querying."
            ),
        )

    logger.info(
        "query_received",
        query=request.query,
        top_k=request.top_k,
        filters=request.filters,
    )

    try:
        response = pipeline.run(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
        )
    except Exception as exc:
        logger.error("query_pipeline_error", error=str(exc), query=request.query)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {type(exc).__name__}. Check server logs.",
        ) from exc

    return response


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Returns application liveness status and vector store readiness.",
)
async def health(
    settings: SettingsDep,
    vector_store: VectorStoreDep,
) -> HealthResponse:

    return HealthResponse(
        status="ok",
        version=settings.app_version,
        vector_store_ready=vector_store.is_ready(),
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
    )
