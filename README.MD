# Job RAG Pipeline

The project ingests job listings from `data/LFJobs.csv`, cleans the HTML job descriptions, chunks the text, embeds chunks with a SentenceTransformer model, stores vectors in ChromaDB, retrieves relevant chunks for a user query, optionally reranks them, and generates an answer with a local Hugging Face LLM.

## What This Project Does

This app lets a user ask natural-language questions such as:

```text
remote python developer jobs with machine learning experience
```

The system then:

1. Converts the query into an embedding.
2. Searches the local ChromaDB vector index.
3. Optionally combines semantic search with keyword search.
4. Optionally reranks candidate chunks with a cross-encoder.
5. Injects the retrieved job chunks into a grounded prompt.
6. Uses a Hugging Face instruction model to generate a concise answer.
7. Returns the answer, retrieved chunks, source jobs, and retrieval metadata.

## Architecture Overview

```text
CSV job data
  -> ingestion script
  -> HTML cleaning
  -> metadata enrichment
  -> chunking
  -> embedding generation
  -> ChromaDB vector store

User query
  -> FastAPI endpoint
  -> query validation
  -> query embedding
  -> dense vector retrieval
  -> optional keyword fusion
  -> optional reranking
  -> prompt construction
  -> Hugging Face LLM
  -> grounded JSON response
```

## Main Technologies

- **FastAPI**: HTTP API and Swagger docs.
- **Pydantic**: request/response validation and settings.
- **SentenceTransformers**: embedding and reranking models.
- **BAAI/bge-small-en-v1.5**: default embedding model.
- **ChromaDB**: persistent local vector database.
- **Hugging Face Transformers**: local LLM inference.
- **Qwen/Qwen2.5-3B-Instruct**: default effective generation model.
- **LangChain text splitters**: recursive character chunking.
- **BeautifulSoup + lxml**: HTML job-description cleaning.
- **Plotly + scikit-learn**: notebook evaluation and 3D vector-space visualization.

## Folder Structure

```text
.
├── app/
│   ├── main.py                    # FastAPI app factory, middleware, docs, startup
│   ├── api/
│   │   ├── routes.py              # /api/query and /api/health endpoints
│   │   └── dependencies.py        # cached service dependency providers
│   ├── core/
│   │   ├── config.py              # central settings and path resolution
│   │   └── logging.py             # structlog / stdlib logging setup
│   ├── models/
│   │   └── schemas.py             # Pydantic API schemas
│   ├── rag/
│   │   ├── pipeline.py            # end-to-end RAG orchestration
│   │   ├── prompts.py             # system prompt and context formatting
│   │   ├── retriever.py           # dense + keyword hybrid retrieval
│   │   └── reranker.py            # optional cross-encoder reranking
│   ├── services/
│   │   ├── embedding_service.py   # SentenceTransformer embedding wrapper
│   │   ├── llm_service.py         # Hugging Face generation wrapper
│   │   └── vector_store_service.py# ChromaDB wrapper
│   └── utils/
│       ├── chunker.py             # recursive chunk splitting
│       └── preprocessor.py        # HTML cleaning and document construction
├── scripts/
│   └── ingest.py                  # CLI ingestion/indexing script
├── data/
│   ├── LFJobs.csv                 # source job listing dataset
│   └── chroma_db/                 # persistent ChromaDB index
├── notebook/
│   ├── ingestion.ipynb            # interactive ingestion workflow
│   ├── edaBook.ipynb              # exploratory data analysis
│   └── evalRAG.ipynb              # retrieval evaluation and embedded plots
├── docs/
│   ├── RAG_SYSTEM_DOCUMENTATION.md# full architecture documentation
│   └── *.png                      # EDA/report figures
├── requirements.txt               # runtime dependencies
└── requirements-dev.txt           # notebook/test/dev dependencies
```

## Recommended Python Version

Use **Python 3.12** for the smoothest compatibility, especially if you want CUDA-enabled PyTorch.

Python 3.13 can work for CPU-only usage if compatible wheels are available in your environment, but CUDA PyTorch wheels may not be available for Python 3.13.

## Installation

### Option A: Conda Environment

```powershell
conda create -n job-rag python=3.12 -y
conda activate job-rag
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

### Option B: Python venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## CPU vs CUDA Setup

The default `requirements.txt` installs CPU-compatible PyTorch:

```text
torch==2.6.0
```

For CUDA on Windows:

1. Use Python 3.12.
2. Install a compatible NVIDIA driver.
3. Replace CPU PyTorch with a CUDA wheel:

```powershell
pip uninstall -y torch
pip install torch==2.6.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Confirm CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

To use GPU in the app, set:

```text
EMBEDDING_DEVICE=cuda
```

The current code uses `embedding_device` to choose the Hugging Face pipeline device as well.

## Configuration

Settings are defined in `app/core/config.py`. You can override them with environment variables or a `.env` file in the project root.

Common settings:

```text
APP_NAME=Job RAG Pipeline
APP_VERSION=1.0.0
LOG_LEVEL=INFO

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=64

CHROMA_PERSIST_DIR=data/chroma_db
CHROMA_COLLECTION_NAME=job_listings

CHUNK_SIZE=512
CHUNK_OVERLAP=64

DEFAULT_TOP_K=5
HYBRID_ALPHA=0.7
ENABLE_RERANKER=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_TOP_N=3

HUGGINGFACE_MODEL=Qwen/Qwen2.5-3B-Instruct
HUGGINGFACE_TASK=text-generation
```

The LLM service uses Hugging Face Transformers. If no `HUGGINGFACE_MODEL` is set and the configured `LLM_MODEL` is the legacy default `gpt-4o-mini`, the service resolves it to:

```text
Qwen/Qwen2.5-3B-Instruct
```

## Index the Dataset

The repo already contains a Chroma index in `data/chroma_db`, but you can rebuild it at any time.

Run ingestion:

```powershell
python scripts/ingest.py
```

Run a small dry run without writing to Chroma:

```powershell
python scripts/ingest.py --dry-run --limit 5
```

Use a custom CSV:

```powershell
python scripts/ingest.py --data-path data/LFJobs.csv --batch-size 128
```

Expected CSV columns:

```text
ID
Job Category
Job Title
Company Name
Publication Date
Job Location
Job Level
Tags
Job Description
```

## Run the API

```powershell
python -m uvicorn app.main:app --reload
```

Open Swagger docs:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
GET http://127.0.0.1:8000/api/health
```

Query endpoint:

```text
POST http://127.0.0.1:8000/api/query
```

Example request body:

```json
{
  "query": "remote python developer jobs with machine learning experience",
  "top_k": 5
}
```

Optional metadata filters:

```json
{
  "query": "senior backend engineer",
  "top_k": 5,
  "filters": {
    "job_level": "Senior Level"
  }
}
```

PowerShell query example:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/api/query" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"remote python developer jobs","top_k":5}'
```

## API Response Shape

`POST /api/query` returns:

- `query`: normalized user query.
- `answer`: LLM-generated answer grounded in retrieved chunks.
- `results`: retrieved chunks with metadata and scores.
- `sources`: deduplicated source jobs.
- `retrieval_metadata`: latency, model names, reranker flag, hybrid flag, timestamp.

## Notebooks

Start Jupyter from the activated environment:

```powershell
jupyter lab
```

If you do not have JupyterLab installed:

```powershell
pip install jupyterlab
```

Notebooks:

- `notebook/ingestion.ipynb`: interactive ingestion flow.
- `notebook/edaBook.ipynb`: exploratory data analysis.
- `notebook/evalRAG.ipynb`: retrieval evaluation and embedded interactive Plotly charts.

The evaluation notebook requires:

```text
plotly
scikit-learn
nbformat
ipykernel
```

These are included in `requirements-dev.txt`.

## Evaluation Workflow

Open:

```text
notebook/evalRAG.ipynb
```

The notebook:

1. Loads the same app services used by the API.
2. Builds pseudo-labelled evaluation cases from `data/LFJobs.csv`.
3. Computes retrieval metrics:
   - hit@k
   - MRR
   - precision@k
   - term overlap
   - latency
4. Builds an interactive 3D vector similarity plot embedded in the notebook.
5. Builds an embedded similarity distribution chart.

To change the focus query for the 3D plot, edit:

```python
FOCUS_QUERY = "remote python developer jobs with machine learning experience"
```

## Full Technical Documentation

The detailed architecture and methodology report is here:

```text
docs/RAG_SYSTEM_DOCUMENTATION.md
```

It explains:

- architecture
- methodology
- ingestion
- chunking
- embeddings
- vector database
- retrieval
- reranking
- prompts
- LLM integration
- API lifecycle
- evaluation
- security and limitations
- future improvements

## Troubleshooting

### `Vector store is empty`

Run:

```powershell
python scripts/ingest.py
```

### Plotly says `nbformat>=4.2.0` is missing

Install dev requirements in the same environment used by your notebook kernel:

```powershell
pip install -r requirements-dev.txt
```

Or install directly:

```powershell
pip install nbformat
```

Restart the notebook kernel after installing.

### Hugging Face model loading is slow

The first run downloads and loads models. Later runs use the local Hugging Face cache. CPU generation can still be slow for `Qwen/Qwen2.5-3B-Instruct`.

For faster experimentation:

```powershell
$env:HUGGINGFACE_MODEL="google/flan-t5-base"
$env:HUGGINGFACE_TASK="text2text-generation"
```

### CUDA wheel fails to install

Use Python 3.12 for CUDA PyTorch. CUDA wheels may not be available for Python 3.13.

### Port 8000 is already in use

Run on another port:

```powershell
python -m uvicorn app.main:app --reload --port 8001
```

## Development Notes

Run a syntax check:

```powershell
python -m py_compile app/main.py app/api/routes.py scripts/ingest.py
```

Run tests if tests are added:

```powershell
pytest
```

Format/lint if configured:

```powershell
ruff check .
```

## Current Limitations

- No custom frontend beyond FastAPI Swagger docs.
- No authentication or authorization.
- No conversational memory.
- Keyword retrieval is a lightweight custom implementation, not BM25.
- ChromaDB is local and single-node.
- Prompt-injection protection is prompt-based only.
- LLM generation is not streamed.

## Suggested Next Improvements

- Add a dedicated frontend for job search and result browsing.
- Add true BM25 or sparse-vector hybrid retrieval.
- Separate LLM device config from embedding device config.
- Add streaming responses.
- Add authentication.
- Add source-linked citation UI.
- Add prompt-injection filtering.
- Add automated answer faithfulness evaluation.
- Add Docker support.
