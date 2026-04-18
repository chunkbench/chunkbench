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
from config import (RETRIEVAL_K, LLM_TEMPERATURE, LLM_MAX_TOKENS,
                    EMBED_MODEL, RETRIEVAL_METRIC, DEDUP_THRESHOLD,
                    MAX_CONTEXT_CHARS, MAX_CONTEXT_TOKENS, CONTEXT_MODE,
                    SPACY_MODEL, CHUNKING_STRATEGIES, INDEX_BASE_DIR)

app = FastAPI(title="HAE-RAG Benchmark", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    question:              str
    k:                     int   = RETRIEVAL_K
    context_mode:          str   = CONTEXT_MODE
    context_budget_tokens: int   = MAX_CONTEXT_TOKENS
    context_budget_chars:  int   = MAX_CONTEXT_CHARS
    temperature:           float = LLM_TEMPERATURE
    max_tokens:            int   = LLM_MAX_TOKENS
    dedup:                 bool  = True
    dedup_threshold:       Optional[float] = None
    strategies:            Optional[list[str]] = None


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
    strategies:            list[str] = ["S1", "S2", "S3", "S4", "B0"]
    k:                     int   = RETRIEVAL_K
    context_mode:          str   = CONTEXT_MODE
    context_budget_tokens: int   = MAX_CONTEXT_TOKENS
    temperature:           float = LLM_TEMPERATURE
    max_tokens:            int   = LLM_MAX_TOKENS
    dedup:                 bool  = True


def check_hit(gold_span, chunks, k):
    if not gold_span or not chunks: return 0
    gold = " ".join(gold_span.lower().split())[:120]
    return int(any(gold in " ".join(c["text"].lower().split()) for c in chunks[:k]))

def reciprocal_rank(gold_span, chunks):
    if not gold_span or not chunks: return 0.0
    gold = " ".join(gold_span.lower().split())[:120]
    for i, c in enumerate(chunks):
        if gold in " ".join(c["text"].lower().split()): return 1.0 / (i + 1)
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
        )
        q_result = {
            "q_id": q.q_id, "question": q.question,
            "category": q.category, "needle_type": q.needle_type,
            "gold_span": q.gold_span, "gold_answer": q.gold_answer,
            "strategies": {}
        }
        for sid, r in result["results"].items():
            chunks = r["retrieved_chunks"]
            has_gold = q.gold_span is not None
            q_result["strategies"][sid] = {
                "answer":         r["answer"],
                "latency_s":      r["latency_s"],
                "context_tokens": r.get("context_tokens_est", 0),
                "hit_at_1":  check_hit(q.gold_span, chunks, 1) if has_gold else None,
                "hit_at_3":  check_hit(q.gold_span, chunks, 3) if has_gold else None,
                "hit_at_5":  check_hit(q.gold_span, chunks, 5) if has_gold else None,
                "mrr":       reciprocal_rank(q.gold_span, chunks) if has_gold else None,
                "top_chunks": [{"rank": c["rank"], "doc_id": c["doc_id"],
                                "text": c["text"][:300]} for c in chunks[:3]],
            }
        all_results.append(q_result)
        await asyncio.sleep(0.3)

    scored = [r for r in all_results if r["gold_span"]]
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


@app.get("/config/frozen")
async def frozen_config():
    return {
        "embed_model":        EMBED_MODEL,
        "distance_metric":    RETRIEVAL_METRIC,
        "dedup_threshold":    DEDUP_THRESHOLD,
        "dedup_timing":       "query-time",
        "context_mode":       CONTEXT_MODE,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "context_truncation": "rank-order, whole-chunk, budget=1800tok",
        "phase1_k":           RETRIEVAL_K,
        "sentence_splitter":  f"spacy:{SPACY_MODEL}",
        "chunking_strategies": {
            sid: {"name": v["name"], "description": v["description"]}
            for sid, v in CHUNKING_STRATEGIES.items()
        },
    }


@app.get("/index-stats")
async def index_stats():
    stats = {}
    for sid in ("S1", "S2", "S3", "S4"):
        f = Path(INDEX_BASE_DIR) / f"chroma_{sid.lower()}" / "index_stats.json"
        if f.exists(): stats[sid] = json.loads(f.read_text())
    return stats


_ui_dist = Path(__file__).resolve().parent.parent / "ui" / "dist"
if _ui_dist.exists():
    app.mount("/", StaticFiles(directory=str(_ui_dist), html=True), name="static")
