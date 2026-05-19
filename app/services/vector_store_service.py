from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


class VectorStoreService:

    def __init__(self, embedding_service: EmbeddingService) -> None:
        settings = get_settings()
        self._embedding_service = embedding_service
        self._keyword_cache: Optional[List[Dict[str, Any]]] = None

        logger.info("initialising_chroma", persist_dir=settings.chroma_persist_dir)
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "chroma_ready",
            collection=settings.chroma_collection_name,
            num_documents=self._collection.count(),
        )

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 256,
    ) -> None:

        total = len(chunks)
        logger.info("adding_chunks", total=total, batch_size=batch_size)

        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            ids = [c["id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [c["metadata"] for c in batch]

            embeddings = self._embedding_service.embed_documents(
                texts, batch_size=batch_size
            )

            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.debug("batch_upserted", start=start, end=start + len(batch))

        logger.info("chunks_added", total=total)
        self._keyword_cache = None

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        try:
            raw = self._collection.query(**query_kwargs)
        except Exception as exc:
            logger.error("chroma_query_failed", error=str(exc))
            raise

        results: List[Dict[str, Any]] = []
        ids = raw["ids"][0]
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        dists = raw["distances"][0]

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            results.append(
                {
                    "id": cid,
                    "text": doc,
                    "metadata": meta,
                    # Chroma returns cosine distance [0, 2]; convert to similarity [0, 1]
                    "similarity_score": max(0.0, 1.0 - dist / 2.0),
                }
            )

        return results

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:

        query_kwargs: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        try:
            raw = self._collection.query(**query_kwargs)
        except Exception as exc:
            logger.error("chroma_keyword_query_failed", error=str(exc))
            raise

        results: List[Dict[str, Any]] = []
        ids = raw["ids"][0]
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        dists = raw["distances"][0]

        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            results.append(
                {
                    "id": cid,
                    "text": doc,
                    "metadata": meta,
                    "similarity_score": max(0.0, 1.0 - dist / 2.0),
                }
            )

        return results

    # ── Utilities ──────────────────────────────────────────────────────────────

    @property
    def document_count(self) -> int:
        return self._collection.count()

    def is_ready(self) -> bool:
        try:
            return self._collection.count() > 0
        except Exception:
            return False
