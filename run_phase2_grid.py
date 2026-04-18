#!/usr/bin/env python3
import json, os, sys, time, gc, requests
import numpy as np
from pathlib import Path
from datetime import datetime

TUNING_FILE    = "data/questions_tuning.json"
PARSED_DIR     = Path("data/parsed")
INDEX_BASE     = Path("indexes")
RESULTS_DIR    = Path("data/results/phase2_tuning")
API_BASE       = "http://127.0.0.1:8007"
CHECKPOINT     = RESULTS_DIR / "grid_checkpoint.json"

OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
EMBED_API      = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL    = "baai/bge-m3"

EVAL_PARAMS = {
    "k": 10, "context_mode": "fixed-budget",
    "context_budget_tokens": 1800, "temperature": 0.0, "dedup": True,
}

PHASE2_GRID = {
    "S1": [
        {"chunk_size": 512,  "chunk_overlap": 51,  "label": "S1_512_10pct"},
        {"chunk_size": 512,  "chunk_overlap": 102, "label": "S1_512_20pct"},
        {"chunk_size": 1024, "chunk_overlap": 102, "label": "S1_1024_10pct"},
        {"chunk_size": 1024, "chunk_overlap": 205, "label": "S1_1024_20pct"},
        {"chunk_size": 2048, "chunk_overlap": 205, "label": "S1_2048_10pct"},
        {"chunk_size": 2048, "chunk_overlap": 410, "label": "S1_2048_20pct"},
    ],
    "S2": [
        {"chunk_size": 512,  "chunk_overlap": 51,  "label": "S2_512_10pct"},
        {"chunk_size": 512,  "chunk_overlap": 102, "label": "S2_512_20pct"},
        {"chunk_size": 1024, "chunk_overlap": 102, "label": "S2_1024_10pct"},
        {"chunk_size": 1024, "chunk_overlap": 205, "label": "S2_1024_20pct"},
        {"chunk_size": 2048, "chunk_overlap": 205, "label": "S2_2048_10pct"},
        {"chunk_size": 2048, "chunk_overlap": 410, "label": "S2_2048_20pct"},
    ],
    "S3": [
        {"threshold": 70, "max_chunk_size": 1000, "label": "S3_t70_max1000"},
        {"threshold": 70, "max_chunk_size": 2000, "label": "S3_t70_max2000"},
        {"threshold": 85, "max_chunk_size": 1000, "label": "S3_t85_max1000"},
        {"threshold": 85, "max_chunk_size": 2000, "label": "S3_t85_max2000"},
        {"threshold": 95, "max_chunk_size": 1000, "label": "S3_t95_max1000"},
        {"threshold": 95, "max_chunk_size": 2000, "label": "S3_t95_max2000"},
    ],
}

REMOVE_DOCS = {"HAE_010", "HAE_012", "HAE_099"}


def embed_api(texts, retries=3):
    all_embs = []
    for i in range(0, len(texts), 20):
        batch = [t[:8000] for t in texts[i:i+20]]
        for attempt in range(retries):
            try:
                resp = requests.post(EMBED_API,
                    headers={"Authorization": f"Bearer {OPENROUTER_KEY}",
                             "Content-Type": "application/json"},
                    json={"model": EMBED_MODEL, "input": batch}, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                all_embs.extend([d["embedding"] for d in sorted_data])
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(3)
                else:
                    raise
        time.sleep(0.3)
    return all_embs


def build_s1_s2(strategy_id, chunk_size, chunk_overlap, label):
    import chromadb
    from chromadb.config import Settings

    idx_dir = INDEX_BASE / f"chroma_{strategy_id.lower()}_{label}"
    idx_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(idx_dir), settings=Settings(anonymized_telemetry=False))
    try: client.delete_collection(f"hae_{strategy_id.lower()}")
    except: pass
    coll = client.create_collection(f"hae_{strategy_id.lower()}", metadata={"hnsw:space": "cosine"})

    total = 0
    for doc_path in sorted(PARSED_DIR.glob("*.txt")):
        doc_id = doc_path.stem
        if doc_id in REMOVE_DOCS:
            continue
        text = doc_path.read_text(encoding="utf-8")

        chunks = []
        if strategy_id == "S1":
            start, idx = 0, 0
            while start < len(text):
                t = text[start:start+chunk_size].strip()
                if t:
                    chunks.append(t)
                    idx += 1
                start += chunk_size - chunk_overlap
        else:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " "], length_function=len)
            chunks = [c.strip() for c in splitter.split_text(text) if c.strip()]

        if not chunks:
            continue

        embs = embed_api(chunks)
        for i in range(0, len(chunks), 100):
            batch = chunks[i:i+100]
            coll.add(
                ids=[f"{doc_id}_{strategy_id}_{j:04d}" for j in range(i, i+len(batch))],
                embeddings=embs[i:i+100],
                documents=batch,
                metadatas=[{"doc_id": doc_id, "strategy": strategy_id,
                            "chunk_index": i+j, "char_len": len(c)} for j, c in enumerate(batch)])
        total += len(chunks)

    print(f"      {total} chunks", flush=True)
    return str(idx_dir), total


def build_s3(threshold, max_chunk_size, label):
    import spacy, chromadb
    from chromadb.config import Settings
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_core.embeddings import Embeddings

    class APIEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return embed_api(texts)
        def embed_query(self, text):
            return embed_api([text])[0]

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000
    chunker = SemanticChunker(APIEmbeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=threshold)

    idx_dir = INDEX_BASE / f"chroma_s3_{label}"
    idx_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(idx_dir), settings=Settings(anonymized_telemetry=False))
    try: client.delete_collection("hae_s3")
    except: pass
    coll = client.create_collection("hae_s3", metadata={"hnsw:space": "cosine"})

    total = 0
    for doc_path in sorted(PARSED_DIR.glob("*.txt")):
        doc_id = doc_path.stem
        if doc_id in REMOVE_DOCS:
            continue
        text = doc_path.read_text(encoding="utf-8")
        sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        rejoined = "\n".join(sentences)

        try:
            raw_chunks = chunker.split_text(rejoined)
        except:
            continue

        chunks = []
        for chunk in raw_chunks:
            if len(chunk) <= max_chunk_size:
                if chunk.strip():
                    chunks.append(chunk.strip())
            else:
                sub_sents = [s.text.strip() for s in nlp(chunk).sents]
                current, clen = [], 0
                for s in sub_sents:
                    if clen + len(s) > max_chunk_size and current:
                        chunks.append(" ".join(current))
                        current, clen = [s], len(s)
                    else:
                        current.append(s)
                        clen += len(s)
                if current:
                    chunks.append(" ".join(current))

        chunks = [c for c in chunks if c and len(c) >= 100]
        if not chunks:
            continue

        embs = embed_api(chunks)
        for i in range(0, len(chunks), 100):
            batch = chunks[i:i+100]
            coll.add(
                ids=[f"{doc_id}_S3_{j:04d}" for j in range(i, i+len(batch))],
                embeddings=embs[i:i+100], documents=batch,
                metadatas=[{"doc_id": doc_id, "strategy": "S3",
                            "chunk_index": i+j, "char_len": len(c)} for j, c in enumerate(batch)])
        total += len(chunks)

    print(f"      {total} chunks", flush=True)
    return str(idx_dir), total


def swap_and_restart(strategy_id, new_index_dir):
    import subprocess, shutil

    prod_dir = INDEX_BASE / f"chroma_{strategy_id.lower()}"
    backup_dir = INDEX_BASE / f"chroma_{strategy_id.lower()}_phase1_backup"

    if not backup_dir.exists() and prod_dir.exists():
        shutil.copytree(str(prod_dir), str(backup_dir))

    if prod_dir.exists():
        shutil.rmtree(str(prod_dir))
    shutil.copytree(new_index_dir, str(prod_dir))

    import chromadb
    from chromadb.config import Settings
    c = chromadb.PersistentClient(path=str(prod_dir), settings=Settings(anonymized_telemetry=False))
    co = c.get_collection(f"hae_{strategy_id.lower()}")
    stats = {"strategy": strategy_id, "name": strategy_id, "total_documents": 162, "total_chunks": co.count()}
    (prod_dir / "index_stats.json").write_text(json.dumps(stats, indent=2))

    subprocess.run(["systemctl", "restart", "chunkbench"], check=False)
    time.sleep(15)
    print(f"      API restarted with new {strategy_id} index", flush=True)


def eval_tuning(strategy_id):
    with open(TUNING_FILE) as f:
        tuning_qs = json.load(f)

    results = []
    for q in tuning_qs:
        payload = {
            "questions": [{
                "q_id": q["q_id"], "question": q["question"],
                "category": q.get("category", "single_fact"),
                "gold_span": q.get("gold_span", ""),
                "gold_answer": q.get("gold_answer", ""),
            }],
            "strategies": [strategy_id],
            **EVAL_PARAMS,
        }
        try:
            r = requests.post(f"{API_BASE}/evaluate", json=payload, timeout=180)
            res = r.json()["results"][0]
            sr = res["strategies"].get(strategy_id, {})
            results.append({
                "q_id": q["q_id"],
                "hit_at_5": sr.get("hit_at_5", 0),
                "mrr": sr.get("mrr", 0),
            })
        except Exception as e:
            print(f"      ERROR {q['q_id']}: {e}", flush=True)
            results.append({"q_id": q["q_id"], "hit_at_5": 0, "mrr": 0})
        time.sleep(0.5)

    hit5 = sum(r["hit_at_5"] or 0 for r in results) / max(len(results), 1)
    mrr = sum(r["mrr"] or 0 for r in results) / max(len(results), 1)
    return hit5, mrr, results


def load_checkpoint():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"done_configs": [], "grid_results": {}}

def save_checkpoint(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "w") as f:
        json.dump(data, f, indent=2)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cp = load_checkpoint()
    done = set(cp["done_configs"])
    grid_results = cp["grid_results"]

    for sid in ["S1", "S2", "S3"]:
        print(f"\n{'='*60}", flush=True)
        print(f"{sid} GRID SEARCH", flush=True)
        print(f"{'='*60}", flush=True)

        if sid not in grid_results:
            grid_results[sid] = []

        for cfg in PHASE2_GRID[sid]:
            label = cfg["label"]
            if label in done:
                print(f"  [{label}] SKIP (already done)", flush=True)
                continue

            print(f"\n  [{label}]", flush=True)
            print(f"    Building index...", end=" ", flush=True)
            t0 = time.time()
            if sid in ("S1", "S2"):
                idx_dir, n_chunks = build_s1_s2(sid, cfg["chunk_size"], cfg["chunk_overlap"], label)
            else:
                idx_dir, n_chunks = build_s3(cfg["threshold"], cfg["max_chunk_size"], label)
            build_time = round(time.time() - t0)
            print(f"{build_time}s", flush=True)

            print(f"    Swapping index...", flush=True)
            swap_and_restart(sid, idx_dir)

            print(f"    Evaluating tuning set (20q)...", end=" ", flush=True)
            t0 = time.time()
            hit5, mrr, details = eval_tuning(sid)
            eval_time = round(time.time() - t0)
            print(f"{eval_time}s | Hit@5={hit5:.3f} MRR={mrr:.3f}", flush=True)

            result = {**cfg, "hit5": round(hit5, 4), "mrr": round(mrr, 4),
                      "n_chunks": n_chunks, "build_time_s": build_time}
            grid_results[sid].append(result)
            done.add(label)
            save_checkpoint({"done_configs": list(done), "grid_results": grid_results})
            gc.collect()

    print(f"\n{'='*60}", flush=True)
    print("RESTORING PHASE 1 INDEXES", flush=True)
    import shutil, subprocess
    for sid in ["S1", "S2", "S3"]:
        backup = INDEX_BASE / f"chroma_{sid.lower()}_phase1_backup"
        prod = INDEX_BASE / f"chroma_{sid.lower()}"
        if backup.exists():
            if prod.exists():
                shutil.rmtree(str(prod))
            shutil.copytree(str(backup), str(prod))
            print(f"  {sid}: restored from backup", flush=True)
    subprocess.run(["systemctl", "restart", "chunkbench"], check=False)
    time.sleep(10)

    print(f"\n{'='*60}", flush=True)
    print("BEST CONFIG SELECTION", flush=True)
    print(f"{'='*60}", flush=True)
    best_configs = {}
    for sid in ["S1", "S2", "S3"]:
        configs = grid_results[sid]
        best = sorted(configs, key=lambda x: (x["hit5"], x["mrr"]), reverse=True)[0]
        best_configs[sid] = best
        print(f"  {sid}: {best['label']} | Hit@5={best['hit5']:.3f} MRR={best['mrr']:.3f}", flush=True)
    best_configs["S4"] = {"label": "S4_default", "note": "No tuning by design"}
    print(f"  S4: default (no tuning)", flush=True)

    with open(RESULTS_DIR / "best_configs.json", "w") as f:
        json.dump(best_configs, f, indent=2)

    with open(RESULTS_DIR / "full_grid.json", "w") as f:
        json.dump(grid_results, f, indent=2)

    print(f"\nGrid search complete!", flush=True)
    print(f"Best configs: {RESULTS_DIR / 'best_configs.json'}", flush=True)


if __name__ == "__main__":
    main()
