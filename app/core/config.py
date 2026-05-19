from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(path: str) -> str:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)
    return str((PROJECT_ROOT / path_obj).resolve())


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_name: str = "Job RAG Pipeline"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── LLM provider ───────────────────────────────────────────────────────────
    # Supports OpenAI, Groq, or any OpenAI-compatible endpoint.
    llm_provider: Literal["openai", "groq", "ollama"] = "openai"
    openai_api_key: str = Field(default="", description="OpenAI (or compatible) API key")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL — override for Groq/Ollama/Azure",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model name passed to the LLM provider",
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=64, le=8192)

    # ── Embeddings ─────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"  # "cuda" if GPU available
    embedding_batch_size: int = 64

    # ── Vector store ───────────────────────────────────────────────────────────
    chroma_persist_dir: str = str(PROJECT_ROOT / "data" / "chroma_db")
    chroma_collection_name: str = "job_listings"

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=64, ge=0, le=256)

    # ── Retrieval ──────────────────────────────────────────────────────────────
    default_top_k: int = Field(default=5, ge=1, le=20)
    hybrid_alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for dense retrieval (1-alpha = BM25 weight)",
    )
    enable_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = Field(default=3, ge=1, le=10)

    # ── Data ───────────────────────────────────────────────────────────────────
    data_path: str = str(PROJECT_ROOT / "data" / "LFJobs.csv")

    @field_validator("chroma_persist_dir", "data_path")
    @classmethod
    def resolve_paths_from_project_root(cls, v: str) -> str:
        return _resolve_project_path(v)

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_lt_chunk_size(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @field_validator("llm_model")
    @classmethod
    def validate_model_for_provider(cls, v: str, info) -> str:
        provider = info.data.get("llm_provider", "openai")
        groq_models = {"llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"}
        if provider == "groq" and v not in groq_models:
            # Allow it but warn — don't block.
            pass
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
