import asyncio, os
import openai
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextPrecisionWithoutReference,
    ContextUtilization,
    ContextRelevance,
    ResponseGroundedness,
)

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        client = openai.AsyncOpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com/v1",
        )
        _llm = llm_factory("deepseek-chat", client=client)
    return _llm


_metrics_cache = None


def _get_metrics():
    global _metrics_cache
    if _metrics_cache is None:
        llm = _get_llm()
        _metrics_cache = {
            "faithfulness":          Faithfulness(llm=llm),
            "context_precision":     ContextPrecisionWithoutReference(llm=llm),
            "context_utilization":   ContextUtilization(llm=llm),
            "context_relevance":     ContextRelevance(llm=llm),
            "response_groundedness": ResponseGroundedness(llm=llm),
        }
    return _metrics_cache


async def _safe_score(metric, **kwargs):
    try:
        result = await metric.ascore(**kwargs)
        return round(float(result.value), 3)
    except Exception:
        return None


async def compute_ragas(question: str, answer: str, retrieved_chunks: list[dict]) -> dict:
    contexts = [c["text"] for c in retrieved_chunks if c.get("text")]

    null_result = {
        "faithfulness": None, "answer_relevancy": None,
        "context_precision": None, "context_utilization": None,
        "context_relevance": None, "response_groundedness": None,
        "ragas_score": None,
    }

    if not contexts or not answer:
        return null_result

    m = _get_metrics()

    faith, cp, cu, cr, rg = await asyncio.gather(
        _safe_score(m["faithfulness"],          user_input=question, response=answer, retrieved_contexts=contexts),
        _safe_score(m["context_precision"],     user_input=question, response=answer, retrieved_contexts=contexts),
        _safe_score(m["context_utilization"],   user_input=question, response=answer, retrieved_contexts=contexts),
        _safe_score(m["context_relevance"],     user_input=question,                  retrieved_contexts=contexts),
        _safe_score(m["response_groundedness"],                       response=answer, retrieved_contexts=contexts),
    )

    available = [v for v in [faith, cp, cu, cr, rg] if v is not None]
    ragas_score = round(sum(available) / len(available), 3) if available else None

    return {
        "faithfulness":          faith,
        "answer_relevancy":      None,
        "context_precision":     cp,
        "context_utilization":   cu,
        "context_relevance":     cr,
        "response_groundedness": rg,
        "ragas_score":           ragas_score,
    }


async def compute_ragas_all_strategies(question: str, strategy_results: dict) -> dict:
    out = {}
    for sid, r in strategy_results.items():
        chunks = r.get("retrieved_chunks", [])
        answer = r.get("answer", "")
        if sid == "B0" or not chunks:
            out[sid] = {
                "faithfulness": None, "answer_relevancy": None,
                "context_precision": None, "context_utilization": None,
                "context_relevance": None, "response_groundedness": None,
                "ragas_score": None,
            }
            continue
        out[sid] = await compute_ragas(question, answer, chunks)
    return out
