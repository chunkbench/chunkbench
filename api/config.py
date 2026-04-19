import os
from pathlib import Path
from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# LLM
LLM_MODEL         = "deepseek-chat"
LLM_API_BASE      = "https://api.deepseek.com/v1"
LLM_TEMPERATURE   = 0.0
LLM_MAX_TOKENS    = 512
LLM_THINKING_MODE = False

# Embedding (Tier-1 FROZEN)
EMBED_MODEL       = "BAAI/bge-m3"
EMBED_DEVICE      = "cpu"
EMBED_BATCH_SIZE  = 32
EMBED_NORMALIZE   = True

# Retrieval
RETRIEVAL_K       = 10
RETRIEVAL_METRIC  = "cosine"

# Deduplication (query-time, Tier-1 FROZEN)
DEDUP_ENABLED     = True
DEDUP_THRESHOLD   = 0.95

# Context assembly
CONTEXT_MODE       = "fixed-budget"
MAX_CONTEXT_TOKENS = 1800
MAX_CONTEXT_CHARS  = 6000

# Sentence splitter for S3 (Tier-1 FROZEN)
SPACY_MODEL       = "en_core_web_sm"

# Corpus
CORPUS_PDF_DIR    = str(_project_root / "corpus" / "pdfs")
CORPUS_MANIFEST   = str(_project_root / "data" / "corpus_manifest.csv")
INDEXED_ROLES     = {"haystack", "needle_sentinel", "needle_holdout", "haystack_distractor"}
INDEX_BASE_DIR = str(_project_root / "indexes")
RESULTS_DIR    = str(_project_root / "data" / "results")

# Per-strategy index paths by phase
PHASE_INDEX_DIRS = {
    "phase1": {
        "S1": str(_project_root / "indexes" / "chroma_s1_phase1_backup"),
        "S2": str(_project_root / "indexes" / "chroma_s2_phase1_backup"),
        "S3": str(_project_root / "indexes" / "chroma_s3_phase1_backup"),
        "S4": str(_project_root / "indexes" / "chroma_s4"),
    },
    "phase2": {
        "S1": str(_project_root / "indexes" / "chroma_s1_S1_512_20pct"),
        "S2": str(_project_root / "indexes" / "chroma_s2_S2_1024_10pct"),
        "S3": str(_project_root / "indexes" / "chroma_s3"),
        "S4": str(_project_root / "indexes" / "chroma_s4"),
    },
}

PHASE1_FROZEN = {
    "k": 10, "context_mode": "fixed-budget", "context_budget_tokens": 1800,
    "temperature": 0.0, "max_tokens": 512, "dedup": True, "dedup_threshold": 0.95,
    "index_configs": {
        "S1": "size=512, overlap=50 (10%) — 17,886 chunks",
        "S2": "size=512, overlap=50 (10%) — 18,543 chunks",
        "S3": "percentile/95 — 3,920 chunks",
        "S4": "proposition/default — 85,537 chunks",
    },
}

PHASE2_FROZEN = {
    "k": 10, "context_mode": "fixed-budget", "context_budget_tokens": 1800,
    "temperature": 0.0, "max_tokens": 512, "dedup": True, "dedup_threshold": 0.95,
    "index_configs": {
        "S1": "size=512, overlap=102 (20%) — 20,141 chunks",
        "S2": "size=1024, overlap=102 (10%) — 9,638 chunks",
        "S3": "t85_max2000 — 11,305 chunks",
        "S4": "proposition/default — 85,537 chunks",
    },
}

# Chunking strategies
CHUNKING_STRATEGIES = {
    "S1": {
        "name":          "Fixed-size",
        "description":   "Pure chunking — fixed character boundaries",
        "chunk_size":    512,
        "chunk_overlap": 50,
        "method":        "fixed"
    },
    "S2": {
        "name":          "Recursive",
        "description":   "Pure chunking — respects paragraph/sentence structure",
        "chunk_size":    512,
        "chunk_overlap": 50,
        "separators":    ["\n\n", "\n", ". ", " "],
        "method":        "recursive"
    },
    "S3": {
        "name":                        "Semantic",
        "description":                 "Pure chunking — embedding-space breakpoints",
        "breakpoint_threshold_type":   "percentile",
        "breakpoint_threshold_amount": 95,
        "sentence_splitter":           "spacy",
        "spacy_model":                 "en_core_web_sm",
        "method":                      "semantic"
    },
    "S4": {
        "name":        "Proposition",
        "description": "Retrieval unit construction — LLM decomposes text into atomic propositions",
        "llm_model":   "deepseek-chat",
        "temperature": 0.0,
        "method":      "proposition"
    },
    "B0": {
        "name":        "No-RAG baseline",
        "description": "No retrieval — LLM answers from parametric memory only",
        "method":      "no_rag"
    }
}

QUESTION_CATEGORIES = {
    "single_fact": "Cat-1: Single-fact retrieval",
    "numerical":   "Cat-2: Numerical precision",
    "multi_hop":   "Cat-3: Multi-hop (2 gold spans required)",
    "negation":    "Cat-4: Negation / exclusion",
    "sentinel":    "Cat-5: Sentinel (contamination control)",
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
