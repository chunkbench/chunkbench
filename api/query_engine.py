import asyncio, os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import chromadb
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
import openai
from config import (
    LLM_MODEL, LLM_API_BASE, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    EMBED_MODEL, EMBED_NORMALIZE, RETRIEVAL_K, CONTEXT_MODE,
    MAX_CONTEXT_TOKENS, MAX_CONTEXT_CHARS,
    PROMPT_TEMPLATE_RAG, PROMPT_TEMPLATE_NO_RAG,
    CHUNKING_STRATEGIES, INDEX_BASE_DIR,
)
from retrieval import (
    set_embed_model, deduplicate_chunks,
    truncate_context, build_context_string, context_token_estimate,
)

print("Loading BGE-M3 for queries...")
_embed = BGEM3FlagModel(EMBED_MODEL, use_fp16=False, device="cpu")
set_embed_model(_embed)

_llm = openai.AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url=LLM_API_BASE,
)

_collections = {}
for _sid in ("S1", "S2", "S3", "S4"):
    _path = Path(INDEX_BASE_DIR) / f"chroma_{_sid.lower()}"
    if _path.exists():
        _client = chromadb.PersistentClient(path=str(_path))
        try:
            _collections[_sid] = _client.get_collection(f"hae_{_sid.lower()}")
        except Exception:
            print(f"  WARNING: Collection hae_{_sid.lower()} not found")

print(f"Query engine ready. Collections: {list(_collections.keys())}")


def embed_query(question: str) -> list[float]:
    return _embed.encode([question], batch_size=1, max_length=512)["dense_vecs"][0].tolist()


async def _run_strategy(
    strategy_id: str, question: str, query_embedding: list[float],
    k: int = RETRIEVAL_K,
    context_mode: str = CONTEXT_MODE,
    context_budget_tokens: int = MAX_CONTEXT_TOKENS,
    context_budget_chars: int = MAX_CONTEXT_CHARS,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    dedup: bool = True,
    dedup_threshold: float = None,
) -> dict:
    t0 = time.time()

    if strategy_id == "B0":
        retrieved_chunks = []
        prompt = PROMPT_TEMPLATE_NO_RAG.format(question=question)
        context_tokens = 0
    else:
        if strategy_id not in _collections:
            return {
                "strategy_id": strategy_id,
                "strategy_name": CHUNKING_STRATEGIES[strategy_id]["name"],
                "answer": "Index not built yet.",
                "retrieved_chunks": [], "context_tokens_est": 0,
                "latency_s": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "error": "index_not_found"
            }

        raw = _collections[strategy_id].query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = [
            {"chunk_id": raw["ids"][0][i], "text": raw["documents"][0][i],
             "doc_id": raw["metadatas"][0][i]["doc_id"],
             "distance": float(raw["distances"][0][i]), "rank": i + 1}
            for i in range(len(raw["ids"][0]))
        ]

        if dedup:
            chunks = deduplicate_chunks(chunks, threshold=dedup_threshold)

        chunks = truncate_context(
            chunks, context_mode=context_mode,
            max_tokens=context_budget_tokens,
            max_chars=context_budget_chars,
        )

        context_str = build_context_string(chunks)
        context_tokens = context_token_estimate(chunks)
        prompt = PROMPT_TEMPLATE_RAG.format(context=context_str, question=question)
        retrieved_chunks = chunks

    response = await _llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    answer = response.choices[0].message.content.strip()

    return {
        "strategy_id":        strategy_id,
        "strategy_name":      CHUNKING_STRATEGIES[strategy_id]["name"],
        "answer":             answer,
        "retrieved_chunks":   retrieved_chunks,
        "context_tokens_est": context_tokens,
        "latency_s":          round(time.time() - t0, 2),
        "prompt_tokens":      response.usage.prompt_tokens,
        "completion_tokens":  response.usage.completion_tokens,
    }


async def query_all(
    question: str,
    k: int = RETRIEVAL_K,
    context_mode: str = CONTEXT_MODE,
    context_budget_tokens: int = MAX_CONTEXT_TOKENS,
    context_budget_chars: int = MAX_CONTEXT_CHARS,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    dedup: bool = True,
    dedup_threshold: float = None,
    strategies: list[str] = None,
) -> dict:
    if strategies is None:
        strategies = ["S1", "S2", "S3", "S4", "B0"]

    query_embedding = embed_query(question)

    results = await asyncio.gather(*[
        _run_strategy(sid, question, query_embedding, k=k,
                      context_mode=context_mode,
                      context_budget_tokens=context_budget_tokens,
                      context_budget_chars=context_budget_chars,
                      temperature=temperature, max_tokens=max_tokens,
                      dedup=dedup, dedup_threshold=dedup_threshold)
        for sid in strategies
    ])

    is_study_mode = (
        k == RETRIEVAL_K and context_mode == CONTEXT_MODE and
        context_budget_tokens == MAX_CONTEXT_TOKENS and
        temperature == LLM_TEMPERATURE and max_tokens == LLM_MAX_TOKENS and
        dedup is True and strategies is None
    )

    return {
        "question":      question,
        "is_study_mode": is_study_mode,
        "results":       {r["strategy_id"]: r for r in results},
    }
