from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# BGE asymmetric retrieval prefix (see https://huggingface.co/BAAI/bge-small-en-v1.5)
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingService:

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        logger.info("loading_embedding_model", model=model_name, device=device)
        self._model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info(
            "embedding_model_ready",
            model=model_name,
            dimension=self.dimension,
        )

    def embed_query(self, query: str) -> List[float]:

        prefixed = _QUERY_PREFIX + query.strip()
        vector: np.ndarray = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> List[List[float]]:

        logger.info("embedding_documents", num_texts=len(texts), batch_size=batch_size)
        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return vectors.tolist()

    def embed_single(self, text: str) -> List[float]:

        vector: np.ndarray = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:

    settings = get_settings()
    return EmbeddingService(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )