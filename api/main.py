import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import asyncio, time, json
from pathlib import Path
from query_engine import query_all
from ragas_eval import compute_ragas_all_strategies
from config import (RETRIEVAL_K, LLM_TEMPERATURE, LLM_MAX_TOKENS,
                    EMBED_MODEL, RETRIEVAL_METRIC, DEDUP_THRESHOLD,
                    MAX_CONTEXT_CHARS, MAX_CONTEXT_TOKENS, CONTEXT_MODE,
                    SPACY_MODEL, CHUNKING_STRATEGIES, INDEX_BASE_DIR,
                    PHASE1_FROZEN, PHASE2_FROZEN, PHASE_INDEX_DIRS)

app = FastAPI(title="HAE-RAG Benchmark", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    question:              str
    mode:                  str   = "phase2"   # "phase1" | "phase2" | "demo"
    k:                     int   = RETRIEVAL_K
    context_mode:          str   = CONTEXT_MODE
    context_budget_tokens: int   = MAX_CONTEXT_TOKENS
    context_budget_chars:  int   = MAX_CONTEXT_CHARS
    temperature:           float = LLM_TEMPERATURE
    max_tokens:            int   = LLM_MAX_TOKENS
    dedup:                 bool  = True
    dedup_threshold:       Optional[float] = None
    strategies:            Optional[list[str]] = None


class RagasRequest(BaseModel):
    question:         str
    mode:             str  = "phase2"
    strategies:       Optional[list[str]] = None
    strategy_results: Optional[dict] = None


class EvalQuestion(BaseModel):
    q_id:        str
    question:    str
    category:    Optional[str] = "single_fact"
    needle_type: Optional[str] = "corpus"
    gold_answer: Optional[str] = None
    gold_span:   Optional[str] = None
    gold_span_2: Optional[str] = None
    source_doc:  Optional[str] = None
    difficulty:  Optional[str] = None


class EvalRequest(BaseModel):
    questions:             list[EvalQuestion]
    mode:                  str   = "phase2"   # "phase1" | "phase2" | "demo"
    strategies:            list[str] = ["S1", "S2", "S3", "S4", "B0"]
    k:                     int   = RETRIEVAL_K
    context_mode:          str   = CONTEXT_MODE
    context_budget_tokens: int   = MAX_CONTEXT_TOKENS
    temperature:           float = LLM_TEMPERATURE
    max_tokens:            int   = LLM_MAX_TOKENS
    dedup:                 bool  = True


def _norm_doc(d):
    if not d: return d
    if d.endswith(".pdf"): d = d[:-4]
    return d.replace("_shallow", "").replace("_deep", "")

def check_hit(source_doc, chunks, k):
    if not source_doc or not chunks: return 0
    src = _norm_doc(source_doc)
    return int(any(_norm_doc(c.get("doc_id","")) == src for c in chunks[:k]))

def reciprocal_rank(source_doc, chunks):
    if not source_doc or not chunks: return 0.0
    src = _norm_doc(source_doc)
    for i, c in enumerate(chunks):
        if _norm_doc(c.get("doc_id","")) == src:
            return 1.0 / (i + 1)
    return 0.0


@app.get("/health")
async def health():
    return {"status": "ok", "version": "4.0.0"}


@app.post("/query")
async def query(req: QueryRequest):
    if not req.question.strip(): raise HTTPException(400, "Empty question")
    if len(req.question) > 1000: raise HTTPException(400, "Question too long")
    return await query_all(
        question=req.question, k=req.k,
        context_mode=req.context_mode,
        context_budget_tokens=req.context_budget_tokens,
        context_budget_chars=req.context_budget_chars,
        temperature=req.temperature,
        max_tokens=req.max_tokens, dedup=req.dedup,
        dedup_threshold=req.dedup_threshold,
        strategies=req.strategies,
        mode=req.mode,
    )


@app.post("/evaluate")
async def evaluate(req: EvalRequest):
    if len(req.questions) > 200:
        raise HTTPException(400, "Maximum 200 questions per batch")
    all_results = []
    t_start = time.time()
    for q in req.questions:
        result = await query_all(
            question=q.question, k=req.k,
            context_mode=req.context_mode,
            context_budget_tokens=req.context_budget_tokens,
            temperature=req.temperature, max_tokens=req.max_tokens,
            dedup=req.dedup, strategies=req.strategies,
            mode=req.mode,
        )
        q_result = {
            "q_id": q.q_id, "question": q.question,
            "category": q.category, "needle_type": q.needle_type,
            "gold_span": q.gold_span, "gold_answer": q.gold_answer,
            "strategies": {}
        }
        for sid, r in result["results"].items():
            chunks = r["retrieved_chunks"]
            has_source = q.source_doc is not None
            q_result["strategies"][sid] = {
                "answer":         r["answer"],
                "latency_s":      r["latency_s"],
                "context_tokens": r.get("context_tokens_est", 0),
                "hit_at_1":  check_hit(q.source_doc, chunks, 1) if has_source else None,
                "hit_at_3":  check_hit(q.source_doc, chunks, 3) if has_source else None,
                "hit_at_5":  check_hit(q.source_doc, chunks, 5) if has_source else None,
                "mrr":       reciprocal_rank(q.source_doc, chunks) if has_source else None,
                "top_chunks": [{"rank": c["rank"], "doc_id": c["doc_id"],
                                "text": c["text"][:300]} for c in chunks[:3]],
            }
        all_results.append(q_result)
        await asyncio.sleep(0.3)

    scored = [r for r in all_results if r.get("source_doc")]
    aggregate = {}
    for sid in req.strategies:
        if sid == "B0":
            aggregate[sid] = {"hit_at_5": None, "mrr": None}
            continue
        h5  = [r["strategies"][sid]["hit_at_5"]  for r in scored if r["strategies"].get(sid)]
        mrr = [r["strategies"][sid]["mrr"]        for r in scored if r["strategies"].get(sid)]
        aggregate[sid] = {
            "n":       len(h5),
            "hit_at_5": round(sum(h5) / len(h5), 3) if h5  else None,
            "mrr":      round(sum(mrr) / len(mrr), 3) if mrr else None,
        }

    return {
        "total_questions":  len(all_results),
        "scored_questions": len(scored),
        "elapsed_s":        round(time.time() - t_start, 1),
        "aggregate":        aggregate,
        "results":          all_results,
    }


@app.post("/ragas")
async def ragas_endpoint(req: RagasRequest):
    if not req.question.strip():
        raise HTTPException(400, "Empty question")

    if req.strategy_results:
        strategy_results = req.strategy_results
    else:
        strats = req.strategies or ["S1", "S2", "S3", "S4", "B0"]
        query_resp = await query_all(
            question=req.question, mode=req.mode,
            strategies=strats,
        )
        strategy_results = query_resp["results"]

    ragas_scores = await compute_ragas_all_strategies(req.question, strategy_results)

    return {
        "question": req.question,
        "mode":     req.mode,
        "ragas":    ragas_scores,
    }


@app.get("/config/frozen")
async def frozen_config(mode: str = "phase2"):
    frozen = PHASE1_FROZEN if mode == "phase1" else PHASE2_FROZEN
    return {
        "mode":               mode,
        "embed_model":        EMBED_MODEL,
        "distance_metric":    RETRIEVAL_METRIC,
        "dedup_threshold":    frozen["dedup_threshold"],
        "dedup_timing":       "query-time",
        "context_mode":       frozen["context_mode"],
        "max_context_tokens": frozen["context_budget_tokens"],
        "context_truncation": "rank-order, whole-chunk, budget=1800tok",
        "k":                  frozen["k"],
        "sentence_splitter":  f"spacy:{SPACY_MODEL}",
        "index_configs":      frozen["index_configs"],
        "chunking_strategies": {
            sid: {"name": v["name"], "description": v["description"]}
            for sid, v in CHUNKING_STRATEGIES.items()
        },
    }


@app.get("/index-stats")
async def index_stats(mode: str = "phase2"):
    from query_engine import _collections_phase1, _collections_phase2, _collections_demo
    cols = _collections_phase1 if mode == "phase1" else \
           _collections_phase2 if mode == "phase2" else _collections_demo
    stats = {}
    for sid, col in cols.items():
        try:
            stats[sid] = {
                "strategy":        sid,
                "name":            CHUNKING_STRATEGIES[sid]["name"],
                "total_chunks":    col.count(),
                "total_documents": 162,
            }
        except Exception:
            pass
    return stats


_ui_dist = Path(__file__).resolve().parent.parent / "ui" / "dist"
if _ui_dist.exists():
    app.mount("/", StaticFiles(directory=str(_ui_dist), html=True), name="static")
