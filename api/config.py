import os
from pathlib import Path
from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

LLM_MODEL       = "deepseek-chat"
LLM_API_BASE    = "https://api.deepseek.com/v1"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 512

EMBED_MODEL      = "BAAI/bge-m3"

RETRIEVAL_K      = 20
RETRIEVAL_METRIC = "cosine"

DEDUP_ENABLED   = True
DEDUP_THRESHOLD = 0.95

CONTEXT_MODE       = "fixed-budget"
MAX_CONTEXT_TOKENS = 1500
MAX_CONTEXT_CHARS  = 6000

SPACY_MODEL = "en_core_web_sm"

PHASE_INDEX_DIRS = {
    "phase1": {
        "S1": str(_project_root / "indexes" / "chroma_s1_phase1_backup"),
        "S2": str(_project_root / "indexes" / "chroma_s2_phase1_backup"),
        "S3": str(_project_root / "indexes" / "chroma_s3_phase1_backup"),
        "S4": str(_project_root / "indexes" / "chroma_s4"),
    },
    "phase2": {
        "S1": str(_project_root / "indexes" / "grid_s1_256_10pct"),
        "S2": str(_project_root / "indexes" / "grid_s2_256_00pct"),
        "S3": str(_project_root / "indexes" / "chroma_s3_S3_t70_max1000"),
        "S4": str(_project_root / "indexes" / "chroma_s4"),
    },
}

_FROZEN_BASE = {
    "k": 20,
    "context_mode": "fixed-budget",
    "context_budget_tokens": 1500,
    "temperature": 0.0,
    "max_tokens": 512,
    "dedup": True,
    "dedup_threshold": 0.95,
}

PHASE1_FROZEN = {
    **_FROZEN_BASE,
    "index_configs": {
        "S1": "512 tokens · 10% overlap · 17,886 chunks",
        "S2": "512 characters · 10% overlap · 18,543 chunks",
        "S3": "percentile/95 · no length cap · 3,920 chunks",
        "S4": "atomic propositions (DeepSeek-V3, T=0) · 84,693 propositions",
    },
}

PHASE2_FROZEN = {
    **_FROZEN_BASE,
    "index_configs": {
        "S1": "256 tokens · 10% overlap (grid-search best)",
        "S2": "256 characters · 0% overlap (grid-search best)",
        "S3": "t70 threshold · 1,000-token max · 11,232 chunks (grid-search best)",
        "S4": "atomic propositions (unchanged) · 84,693 propositions",
    },
}

CHUNKING_STRATEGIES = {
    "S1": {
        "name":        "Fixed-size",
        "description": "Fixed-length token blocks",
        "method":      "fixed",
    },
    "S2": {
        "name":        "Recursive",
        "description": "Hierarchical separator splitting (paragraph → sentence → word)",
        "method":      "recursive",
    },
    "S3": {
        "name":        "Semantic",
        "description": "Embedding-based breakpoints (spaCy sentences, BAAI/bge-m3)",
        "method":      "semantic",
    },
    "S4": {
        "name":        "Proposition",
        "description": "LLM-decomposed atomic propositions (DeepSeek-V3, T=0)",
        "method":      "proposition",
    },
    "B0": {
        "name":        "No-RAG Baseline",
        "description": "No retrieval — LLM answers from parametric memory only",
        "method":      "no_rag",
    },
}

PROMPT_TEMPLATE_RAG = """You are a clinical expert in hereditary angioedema (HAE).
Answer the following question based ONLY on the provided context.
If the context does not contain sufficient information, state exactly:
"The provided context does not contain sufficient information to answer this question."
Do not use any knowledge outside the provided context.

Context:
{context}

Question: {question}

Answer:"""

PROMPT_TEMPLATE_NO_RAG = """You are a clinical expert in hereditary angioedema (HAE).
Answer the following question based on your knowledge.

Question: {question}

Answer:"""
