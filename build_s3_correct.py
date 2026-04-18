#!/usr/bin/env python3
import json, os, sys, time, gc
import numpy as np
import requests
from pathlib import Path

import spacy
import chromadb
from chromadb.config import Settings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings

SPACY_MODEL    = "en_core_web_sm"
INDEX_DIR      = Path("indexes/chroma_s3")
PARSED_DIR     = Path("data/parsed")
OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
EMBED_API      = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL    = "baai/bge-m3"
REMOVE_DOCS    = {"HAE_010", "HAE_012", "HAE_099"}


class OpenRouterBGEM3(Embeddings):
    def embed_documents(self, texts):
        all_embs = []
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = [t[:8000] for t in texts[i:i+batch_size]]
            for attempt in range(3):
                try:
                    resp = requests.post(
                        EMBED_API,
                        headers={"Authorization": f"Bearer {OPENROUTER_KEY}",
                                 "Content-Type": "application/json"},
                        json={"model": EMBED_MODEL, "input": batch},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    sorted_data = sorted(data["data"], key=lambda x: x["index"])
                    all_embs.extend([d["embedding"] for d in sorted_data])
                    break
                except Exception as e:
                    print(f"    API error (attempt {attempt+1}): {e}", flush=True)
                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        raise
            time.sleep(0.3)
        return all_embs

    def embed_query(self, text):
        return self.embed_documents([text])[0]


print("Loading spaCy...", flush=True)
nlp = spacy.load(SPACY_MODEL)
nlp.max_length = 2000000
embeddings = OpenRouterBGEM3()
print("Ready.\n", flush=True)

INDEX_DIR.mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=str(INDEX_DIR), settings=Settings(anonymized_telemetry=False))
try:
    collection = client.get_collection("hae_s3")
    existing_meta = collection.get(limit=10000, include=["metadatas"])
    existing_docs = set(m["doc_id"] for m in existing_meta["metadatas"])
    print(f"Resuming: {len(existing_docs)} docs already indexed, {collection.count()} chunks", flush=True)
except:
    collection = client.create_collection(name="hae_s3", metadata={"hnsw:space": "cosine"})
    existing_docs = set()
    print("Starting fresh.", flush=True)

parsed_docs = sorted(PARSED_DIR.glob("*.txt"))
print(f"[S3] Semantic (LangChain SemanticChunker + API) -- {len(parsed_docs)} documents\n", flush=True)

chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)

total_chunks = 0
total_docs = 0
for idx_d, doc_path in enumerate(parsed_docs):
    doc_id = doc_path.stem
    if doc_id in REMOVE_DOCS:
        print(f"  [{idx_d+1}/{len(parsed_docs)}] {doc_id}: SKIP (non-English)", flush=True)
        continue
    if doc_id in existing_docs:
        print(f"  [{idx_d+1}/{len(parsed_docs)}] {doc_id}: SKIP (already indexed)", flush=True)
        total_docs += 1
        continue

    text = doc_path.read_text(encoding="utf-8")
    t0 = time.time()

    sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    rejoined = "\n".join(sentences)

    try:
        chunk_texts = chunker.split_text(rejoined)
    except Exception as e:
        print(f"  [{idx_d+1}/{len(parsed_docs)}] {doc_id}: ERROR chunking: {e}", flush=True)
        continue

    chunks = []
    for i, t in enumerate(chunk_texts):
        t = t.strip()
        if len(t) >= 100:
            chunks.append({
                "chunk_id":    f"{doc_id}_S3_{i:04d}",
                "doc_id":      doc_id,
                "strategy":    "S3",
                "chunk_index": i,
                "text":        t,
            })

    if not chunks:
        print(f"  [{idx_d+1}/{len(parsed_docs)}] {doc_id}: 0 chunks", flush=True)
        continue

    chunk_embs = embeddings.embed_documents([c["text"] for c in chunks])

    for i in range(0, len(chunks), 50):
        batch = chunks[i:i+50]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=chunk_embs[i:i+50],
            documents=[c["text"] for c in batch],
            metadatas=[{"doc_id": c["doc_id"], "strategy": c["strategy"],
                        "chunk_index": c["chunk_index"], "char_len": len(c["text"])} for c in batch],
        )

    total_chunks += len(chunks)
    total_docs += 1
    elapsed = time.time() - t0
    print(f"  [{idx_d+1}/{len(parsed_docs)}] {doc_id}: {len(chunks)} chunks (total: {total_chunks}) [{elapsed:.0f}s]", flush=True)

    if (idx_d + 1) % 20 == 0:
        gc.collect()

stats = {"strategy": "S3", "name": "Semantic", "total_documents": total_docs, "total_chunks": total_chunks}
(INDEX_DIR / "index_stats.json").write_text(json.dumps(stats, indent=2))
print(f"\n[S3] Complete -- {total_docs} docs, {total_chunks} chunks", flush=True)
