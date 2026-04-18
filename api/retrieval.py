import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import DEDUP_ENABLED, DEDUP_THRESHOLD, MAX_CONTEXT_CHARS, MAX_CONTEXT_TOKENS

_embed_model = None

def set_embed_model(model):
    global _embed_model
    _embed_model = model


def deduplicate_chunks(chunks: list[dict], threshold: float = None) -> list[dict]:
    if threshold is None:
        threshold = DEDUP_THRESHOLD
    if not DEDUP_ENABLED or len(chunks) <= 1:
        return chunks

    texts = [c["text"] for c in chunks]
    embs = _embed_model.encode(texts, batch_size=len(texts), max_length=512)["dense_vecs"]

    keep = [True] * len(chunks)
    for i in range(len(chunks)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(chunks)):
            if not keep[j]:
                continue
            if float(np.dot(embs[i], embs[j])) >= threshold:
                keep[j] = False

    return [c for c, k in zip(chunks, keep) if k]


def truncate_context_chars(chunks: list[dict], max_chars: int = None) -> list[dict]:
    if max_chars is None:
        max_chars = MAX_CONTEXT_CHARS
    kept, total = [], 0
    for chunk in chunks:
        if total + len(chunk["text"]) <= max_chars:
            kept.append(chunk)
            total += len(chunk["text"])
        else:
            break
    return kept


def truncate_context_tokens(chunks: list[dict], max_tokens: int = None) -> list[dict]:
    """1 token ≈ 4 chars."""
    if max_tokens is None:
        max_tokens = MAX_CONTEXT_TOKENS
    kept, total_tokens = [], 0
    for chunk in chunks:
        chunk_tokens = len(chunk["text"]) // 4
        if total_tokens + chunk_tokens <= max_tokens:
            kept.append(chunk)
            total_tokens += chunk_tokens
        else:
            break
    return kept


def truncate_context(chunks: list[dict], context_mode: str = "fixed-budget",
                     max_tokens: int = None, max_chars: int = None) -> list[dict]:
    if context_mode == "fixed-budget":
        return truncate_context_tokens(chunks, max_tokens=max_tokens)
    return truncate_context_chars(chunks, max_chars=max_chars)


def build_context_string(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {c['doc_id']} | Rank {c['rank']}]\n{c['text']}"
        for c in chunks
    )


def context_token_estimate(chunks: list[dict]) -> int:
    return sum(len(c["text"]) for c in chunks) // 4
