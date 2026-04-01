"""
TEAR — FastAPI Backend
Endpoints:
  POST /api/query          — full pipeline query
  GET  /api/stream/{query} — streaming response
  POST /api/index          — trigger dataset indexing
  GET  /api/status         — system status
  GET  /api/stats          — pipeline statistics
  POST /api/evaluate       — run evaluation
"""

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from config import config
from pipeline import TEARPipeline
from modules.dataset_loader import DatasetLoader


# ── Pydantic Models ───────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    token_budget: Optional[int] = Field(None, ge=100, le=4096)
    use_compression: Optional[bool] = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    complexity_label: str
    complexity_score: float
    k_used: int
    docs_retrieved: int
    docs_in_context: int
    tokens_retrieved: int
    tokens_in_context: int
    tokens_saved: int
    token_budget: int
    prompt_tokens: int
    total_tokens: int
    latency_seconds: float
    passages: list
    complexity_features: dict


class IndexRequest(BaseModel):
    datasets: Optional[list] = None    # None = all configured
    max_docs: Optional[int] = 50000


class EvalRequest(BaseModel):
    n_samples: int = Field(default=200, ge=10, le=2000)
    systems: list = Field(default=["tear", "standard_rag", "no_rag"])


class SystemStatus(BaseModel):
    status: str
    index_ready: bool
    llm_loaded: bool
    index_doc_count: int
    model_id: str
    vector_store: str


# ── App State ─────────────────────────────────────────────────

class AppState:
    pipeline: Optional[TEARPipeline] = None
    qa_pairs: list = []
    index_ready: bool = False
    indexing_in_progress: bool = False


state = AppState()


# ── Lifespan ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup."""
    logger.info("Starting TEAR API...")
    state.pipeline = TEARPipeline(config, lazy_llm=False)

    # Try loading pre-built index
    try:
        state.pipeline.retrieval_engine.load_index()
        state.index_ready = True
        logger.info("Pre-built index loaded.")
    except FileNotFoundError:
        logger.warning("No pre-built index found. POST /api/index to build one.")

    yield

    logger.info("TEAR API shutting down.")


# ── FastAPI App ───────────────────────────────────────────────

app = FastAPI(
    title="TEAR API",
    description="Token-Efficient Adaptive Retrieval for Scalable RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins + ["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """Health check and system status."""
    pipeline = state.pipeline
    doc_count = 0
    if pipeline and pipeline.retrieval_engine.faiss_index:
        doc_count = pipeline.retrieval_engine.faiss_index.ntotal

    return SystemStatus(
        status="ready" if state.index_ready else "index_not_built",
        index_ready=state.index_ready,
        llm_loaded=pipeline is not None and pipeline.generator is not None,
        index_doc_count=doc_count,
        model_id=config.llm_model_id,
        vector_store=config.vector_store,
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Run full TEAR pipeline on a query."""
    if not state.index_ready:
        raise HTTPException(
            status_code=503,
            detail="Index not ready. POST /api/index to build."
        )

    # Override token budget if provided
    if req.token_budget:
        state.pipeline.budget_selector.token_budget = req.token_budget

    try:
        result = state.pipeline.run(req.query)
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        complexity_label=result.complexity_label,
        complexity_score=result.complexity_score,
        k_used=result.k_used,
        docs_retrieved=result.docs_retrieved,
        docs_in_context=result.docs_in_context,
        tokens_retrieved=result.tokens_retrieved,
        tokens_in_context=result.tokens_in_context,
        tokens_saved=result.tokens_saved,
        token_budget=result.token_budget,
        prompt_tokens=result.prompt_tokens,
        total_tokens=result.total_tokens,
        latency_seconds=result.latency_seconds,
        passages=result.passages,
        complexity_features=result.complexity_features,
    )


@app.get("/api/stream")
async def stream_query(q: str):
    """Server-Sent Events streaming response."""
    if not state.index_ready:
        raise HTTPException(status_code=503, detail="Index not ready")

    def event_generator():
        try:
            for token in state.pipeline.run_stream(q):
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/index")
async def build_index(req: IndexRequest, background_tasks: BackgroundTasks):
    """Trigger dataset loading and index building (runs in background)."""
    if state.indexing_in_progress:
        return {"status": "already_indexing"}

    def _build():
        state.indexing_in_progress = True
        try:
            loader_config = config
            if req.max_docs:
                loader_config.max_docs_per_dataset = req.max_docs
            if req.datasets:
                loader_config.datasets_to_load = req.datasets

            loader = DatasetLoader(loader_config)
            documents, qa_pairs = loader.load_all()
            state.qa_pairs = qa_pairs

            texts = [d.text for d in documents]
            doc_ids = [d.doc_id for d in documents]
            sources = [d.source for d in documents]

            state.pipeline.retrieval_engine.index_documents(texts, doc_ids, sources)
            state.index_ready = True
            logger.info(f"Index built: {len(texts)} documents")
        except Exception as e:
            logger.exception(f"Indexing failed: {e}")
        finally:
            state.indexing_in_progress = False

    background_tasks.add_task(_build)
    return {"status": "indexing_started", "message": "Check /api/status for progress"}


@app.post("/api/evaluate")
async def run_evaluation(req: EvalRequest, background_tasks: BackgroundTasks):
    """Run evaluation pipeline (async, results saved to disk)."""
    if not state.index_ready:
        raise HTTPException(status_code=503, detail="Index not ready")
    if not state.qa_pairs:
        raise HTTPException(status_code=400, detail="No QA pairs loaded. Index first.")

    def _eval():
        from modules.evaluator import TEAREvaluator
        evaluator = TEAREvaluator(state.pipeline)
        for system in req.systems:
            metrics = evaluator.evaluate(state.qa_pairs, system=system,
                                         n_samples=req.n_samples)
            logger.info(metrics.summary())

    background_tasks.add_task(_eval)
    return {"status": "evaluation_started",
            "message": "Results will be saved to results_*.json"}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── Run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=config.api_host,
        port=config.api_port,
        reload=False,
        workers=1,
    )
