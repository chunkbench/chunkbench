# HAE-RAG Chunking Benchmark

Replication code for the two-phase RAG chunking benchmark described in:

> Dolu, K.O. *Chunking Strategy Comparison for Retrieval-Augmented Generation in a Domain-Homogeneous Medical Corpus: A Benchmark Study on Hereditary Angioedema Literature.* (2026)

Live demo: [chunkbench.tech](https://chunkbench.tech)

---

## Overview

This repository implements a controlled benchmark comparing four document chunking strategies for RAG over a 162-document hereditary angioedema (HAE) corpus:

| ID | Strategy | Description |
|----|----------|-------------|
| S1 | Fixed-size | Character-boundary segmentation (512 chars, 50-char overlap) |
| S2 | Recursive | Structure-aware segmentation (paragraph → sentence → word) |
| S3 | Semantic | Embedding-space breakpoint detection (SemanticChunker, percentile=85, max=2000) |
| S4 | Proposition | LLM-based atomic proposition extraction (DeepSeek-V3) |
| B0 | No-RAG | Parametric baseline (no retrieval) |

All strategies share identical query-time parameters: BAAI/bge-m3 embeddings, k=10, cosine similarity, deduplication threshold=0.95, fixed 1800-token context budget, DeepSeek-V3 generator (T=0.0).

---

## Requirements

- Python 3.10 or later
- 16 GB RAM (BGE-M3 runs on CPU; 8 GB minimum with reduced batch size)
- [spaCy model](https://spacy.io/models/en): `en_core_web_sm`
- API keys: `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY` (required only for S3/S4 index construction)

---

## Installation

```bash
git clone https://github.com/chunkbench/hae-rag-chunk.git
cd hae-rag-chunk

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m spacy download en_core_web_sm
```

Create a `.env` file in the project root:

```
DEEPSEEK_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

---

## Repository Structure

```
api/                    FastAPI backend
  config.py             Frozen protocol parameters
  main.py               REST endpoints (/query, /evaluate, /config/frozen)
  query_engine.py       Parallel strategy execution
  retrieval.py          Deduplication and context assembly

build_s3_correct.py     S3 index construction (SemanticChunker + OpenRouter BGE-M3)
run_phase1_vps.py       Phase 1 evaluation (90 questions × 5 strategies)
run_phase2_grid.py      Phase 2 grid search (S1/S2/S3 parameter optimisation)
run_phase2_eval.py      Phase 2 full evaluation with best configurations
run_s3_partial_grid.py  S3-only grid (2-config subset)

corpus_manifest.csv     Document metadata (doc_id, role, language)
App.jsx                 React UI (Study Mode + Demo Mode)
```

---

## Usage

### Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8007
```

### Query all strategies

```bash
curl -X POST http://localhost:8007/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the mechanism of bradykinin release in HAE?"}'
```

### Run Phase 1 evaluation

Place the question set at `data/questions_v1_FROZEN.json`, then:

```bash
python run_phase1_vps.py
```

Results are written to `data/results/phase1/phase1_results.xlsx` (checkpoint-based; safe to interrupt and resume).

### Run Phase 2 grid search

```bash
python run_phase2_grid.py   # grid search on 20-question tuning set
python run_phase2_eval.py   # full 90-question evaluation with best configs
```

---

## Corpus

The 162-document corpus consists of peer-reviewed HAE publications. Document identifiers and corpus roles (haystack / needle_sentinel / needle_holdout / haystack_distractor) are listed in `corpus_manifest.csv`. Raw PDFs and parsed text files are not redistributed.

---

## License

MIT
