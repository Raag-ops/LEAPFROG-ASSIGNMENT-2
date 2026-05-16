from __future__ import annotations

import hashlib
import uuid
from typing import Any, Dict, List

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback for older langchain versions (<0.2)
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[no-redef]

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _make_chunk_id(job_id: str, chunk_index: int) -> str:
    raw = f"{job_id}__chunk_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


class JobDescriptionChunker:
    def __init__(self) -> None:
        settings = get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            add_start_index=True,
        )
        logger.info(
            "chunker_initialised",
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def chunk_document(
        self,
        document_text: str,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        job_id = metadata.get("job_id", str(uuid.uuid4()))
        raw_chunks = self._splitter.split_text(document_text)
        chunks: List[Dict[str, Any]] = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = _make_chunk_id(job_id, idx)
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": idx,
                        "total_chunks": len(raw_chunks),
                    },
                }
            )

        logger.debug(
            "document_chunked",
            job_id=job_id,
            num_chunks=len(chunks),
        )
        return chunks

    def chunk_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        all_chunks: List[Dict[str, Any]] = []
        for doc in documents:
            chunks = self.chunk_document(
                document_text=doc["text"],
                metadata=doc["metadata"],
            )
            all_chunks.extend(chunks)

        logger.info("batch_chunked", num_documents=len(documents), total_chunks=len(all_chunks))
        return all_chunks
