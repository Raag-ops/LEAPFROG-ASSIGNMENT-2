#!/usr/bin/env python3
"""
Data ingestion script.

Reads the job listings CSV, preprocesses and chunks each job description,
generates embeddings, and upserts everything into the ChromaDB vector store.

Usage
-----
    python scripts/ingest.py                        # default data path from .env
    python scripts/ingest.py --data-path data/jobs.csv
    python scripts/ingest.py --data-path data/jobs.csv --batch-size 128 --dry-run

This script is idempotent: re-running it upserts (not duplicates) chunks
because chunk IDs are deterministic hashes of (job_id, chunk_index).

Expected CSV columns
--------------------
    ID, Job Category, Job Title, Company Name, Publication Date,
    Job Location, Job Level, Tags, Job Description
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Allow running from project root without installing as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.embedding_service import get_embedding_service
from app.services.vector_store_service import VectorStoreService
from app.utils.chunker import JobDescriptionChunker
from app.utils.preprocessor import build_chunk_document

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest job listings into ChromaDB")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the jobs CSV file (defaults to DATA_PATH in .env)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding + upsert batch size (default: 128)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk the data without writing to the vector store",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only ingest the first N rows (useful for testing)",
    )
    return parser.parse_args()


def load_dataframe(data_path: str) -> pd.DataFrame:
    """
    Load and minimally validate the jobs CSV.

    Args:
        data_path: Path to the CSV file.

    Returns:
        DataFrame with required columns.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")

    df = pd.read_csv(path, dtype=str)
    df.fillna("", inplace=True)

    required_columns = {
        "ID",
        "Job Category",
        "Job Title",
        "Company Name",
        "Publication Date",
        "Job Location",
        "Job Level",
        "Job Description",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    logger.info("csv_loaded", rows=len(df), columns=list(df.columns))
    return df


def build_documents(df: pd.DataFrame) -> list[dict]:
    """
    Convert each CSV row into a document dict with text + metadata.

    Args:
        df: Jobs DataFrame.

    Returns:
        List of document dicts ready for the chunker.
    """
    documents = []
    for _, row in df.iterrows():
        doc_text = build_chunk_document(dict(row))
        metadata = {
            "job_id": str(row.get("ID", "")),
            "job_title": str(row.get("Job Title", "")),
            "company_name": str(row.get("Company Name", "")),
            "job_category": str(row.get("Job Category", "")),
            "job_level": str(row.get("Job Level", "")),
            "job_location": str(row.get("Job Location", "")),
            "publication_date": str(row.get("Publication Date", "")),
            "tags": str(row.get("Tags", "")),
        }
        documents.append({"text": doc_text, "metadata": metadata})
    return documents


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = parse_args()

    data_path = args.data_path or settings.data_path
    t_start = time.perf_counter()

    logger.info(
        "ingestion_start",
        data_path=data_path,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    # -- Load -----------------------------------------------------------------
    df = load_dataframe(data_path)
    if args.limit:
        df = df.head(args.limit)
        logger.info("limit_applied", limit=args.limit)

    # -- Preprocess + chunk ---------------------------------------------------
    documents = build_documents(df)
    chunker = JobDescriptionChunker()
    all_chunks = chunker.chunk_batch(documents)

    logger.info(
        "chunking_complete",
        num_jobs=len(documents),
        total_chunks=len(all_chunks),
        avg_chunks_per_job=round(len(all_chunks) / max(len(documents), 1), 1),
    )

    if args.dry_run:
        logger.info("dry_run_complete - no data written to vector store")
        # Print a sample chunk for inspection
        if all_chunks:
            sample = all_chunks[0]
            print("\n=== Sample chunk (job 0, chunk 0) ===")
            print(f"ID: {sample['id']}")
            print(f"Metadata: {sample['metadata']}")
            print(f"Text preview:\n{sample['text'][:400]}")
        return

    # -- Embed + upsert -------------------------------------------------------
    embedding_service = get_embedding_service()
    vector_store = VectorStoreService(embedding_service=embedding_service)

    before_count = vector_store.document_count
    vector_store.add_chunks(all_chunks, batch_size=args.batch_size)
    after_count = vector_store.document_count

    elapsed = time.perf_counter() - t_start
    logger.info(
        "ingestion_complete",
        chunks_before=before_count,
        chunks_after=after_count,
        new_chunks=after_count - before_count,
        elapsed_seconds=round(elapsed, 1),
    )
    print(
        f"\nIngestion complete in {elapsed:.1f}s\n"
        f"  Jobs processed : {len(documents)}\n"
        f"  Total chunks   : {len(all_chunks)}\n"
        f"  Store size     : {after_count} chunks\n"
    )


if __name__ == "__main__":
    main()

