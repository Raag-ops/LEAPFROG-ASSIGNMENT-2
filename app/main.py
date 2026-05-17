from __future__ import annotations

import traceback
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.api.dependencies import get_embedding_service, get_vector_store, get_llm_service
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:

    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info("startup_begin", app=settings.app_name, version=settings.app_version)

    # Warm up singletons — these are cached via lru_cache, so subsequent
    # calls return the already-initialised instances.
    get_embedding_service()
    get_vector_store()
    get_llm_service()

    logger.info("startup_complete")

    yield  # Application is running

    logger.info("shutdown_begin")
    # No explicit cleanup needed: ChromaDB flushes on process exit,
    # and SentenceTransformer holds no external connections.
    logger.info("shutdown_complete")


def create_app() -> FastAPI:

    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production RAG pipeline for job data retrieval. "
            "Combines hybrid search (dense + keyword) with optional "
            "cross-encoder reranking and LLM answer synthesis."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ───────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler ───────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:

        logger.error(
            "unhandled_exception",
            path=str(request.url),
            method=request.method,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please check server logs.",
            },
        )

    # ── Routes ─────────────────────────────────────────────────────────────────
    app.include_router(router, prefix="/api", tags=["RAG"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
