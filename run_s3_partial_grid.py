#!/usr/bin/env python3
import json, os, sys, time, gc, shutil, subprocess, requests
from pathlib import Path

TUNING_FILE    = "data/questions_tuning.json"
PARSED_DIR     = Path("data/parsed")
INDEX_BASE     = Path("indexes")
RESULTS_DIR    = Path("data/results/phase2_tuning")
API_BASE       = "http://127.0.0.1:8007"
CHECKPOINT     = RESULTS_DIR / "s3_partial_checkpoint.json"

OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
EMBED_API      = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL    = "baai/bge-m3"

EVAL_PARAMS = {
    "k": 10, "context_mode": "fixed-budget",
    "context_budget_tokens": 1800, "temperature": 0.0, "dedup": True,
}

S3_CONFIGS = [
    {"threshold": 70, "max_chunk_size": 1000, "label": "S3_t70_max1000"},
    {"threshold": 85, "max_chunk_size": 2000, "label": "S3_t85_max2000"},
]

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
    if idx_dir.exists():
        shutil.rmtree(str(idx_dir))
    idx_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(idx_dir), settings=Settings(anonymized_telemetry=False))
    try: client.delete_collection("hae_s3")
    except: pass
    coll = client.create_collection("hae_s3", metadata={"hnsw:space": "cosine"})

    total = 0
    doc_count = 0
    parsed_docs = sorted(PARSED_DIR.glob("*.txt"))
    n_total = len(parsed_docs)
    for di, doc_path in enumerate(parsed_docs, 1):
        doc_id = doc_path.stem
        if doc_id in REMOVE_DOCS:
            continue
        text = doc_path.read_text(encoding="utf-8")
        sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        rejoined = "\n".join(sentences)

        try:
            raw_chunks = chunker.split_text(rejoined)
        except Exception as e:
            print(f"    {doc_id}: chunk error: {e}", flush=True)
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
        doc_count += 1
        if doc_count % 20 == 0:
            print(f"    [{doc_count}/{n_total - len(REMOVE_DOCS)}] {doc_id}: {len(chunks)} chunks (total: {total})", flush=True)
            gc.collect()

    print(f"    Built: {doc_count} docs, {total} chunks", flush=True)
    stats = {"strategy": "S3", "name": "Semantic", "total_documents": doc_count, "total_chunks": total}
    (idx_dir / "index_stats.json").write_text(json.dumps(stats, indent=2))
    return str(idx_dir), total


def swap_and_restart(new_index_dir):
    prod_dir = INDEX_BASE / "chroma_s3"
    if prod_dir.exists():
        shutil.rmtree(str(prod_dir))
    shutil.copytree(new_index_dir, str(prod_dir))
    subprocess.run(["systemctl", "restart", "chunkbench"], check=False)
    time.sleep(20)


def eval_tuning():
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
            "strategies": ["S3"], **EVAL_PARAMS,
        }
        try:
            r = requests.post(f"{API_BASE}/evaluate", json=payload, timeout=240)
            res = r.json()["results"][0]
            sr = res["strategies"].get("S3", {})
            results.append({
                "q_id": q["q_id"],
                "hit_at_5": sr.get("hit_at_5", 0),
                "mrr": sr.get("mrr", 0),
            })
        except Exception as e:
            print(f"    ERROR {q['q_id']}: {e}", flush=True)
            results.append({"q_id": q["q_id"], "hit_at_5": 0, "mrr": 0})
        time.sleep(0.5)

    hit5 = sum(r["hit_at_5"] or 0 for r in results) / max(len(results), 1)
    mrr = sum(r["mrr"] or 0 for r in results) / max(len(results), 1)
    return hit5, mrr, results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    backup_dir = INDEX_BASE / "chroma_s3_phase1_backup"
    prod_dir = INDEX_BASE / "chroma_s3"
    if not backup_dir.exists() and prod_dir.exists():
        print("Backing up current S3 index...", flush=True)
        shutil.copytree(str(prod_dir), str(backup_dir))

    results = []
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            results = json.load(f)
    done = set(r["label"] for r in results)

    for cfg in S3_CONFIGS:
        label = cfg["label"]
        if label in done:
            print(f"\n[{label}] already done, skipping", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"[{label}] threshold={cfg['threshold']} max_chunk_size={cfg['max_chunk_size']}", flush=True)
        print(f"{'='*60}", flush=True)

        print(f"  Building index...", flush=True)
        t0 = time.time()
        idx_dir, n_chunks = build_s3(cfg["threshold"], cfg["max_chunk_size"], label)
        build_time = round(time.time() - t0)
        print(f"  Build done: {build_time}s, {n_chunks} chunks", flush=True)

        print(f"  Swapping S3 to {label}...", flush=True)
        swap_and_restart(idx_dir)

        print(f"  Evaluating tuning set...", flush=True)
        t0 = time.time()
        hit5, mrr, details = eval_tuning()
        eval_time = round(time.time() - t0)
        print(f"  Hit@5={hit5:.3f} MRR={mrr:.3f} ({eval_time}s)", flush=True)

        results.append({
            **cfg, "hit5": round(hit5, 4), "mrr": round(mrr, 4),
            "n_chunks": n_chunks, "build_time_s": build_time,
        })
        with open(CHECKPOINT, "w") as f:
            json.dump(results, f, indent=2)
        gc.collect()

    print("\nRestoring Phase 1 S3 index...", flush=True)
    if backup_dir.exists():
        if prod_dir.exists():
            shutil.rmtree(str(prod_dir))
        shutil.copytree(str(backup_dir), str(prod_dir))
        subprocess.run(["systemctl", "restart", "chunkbench"], check=False)

    print(f"\n{'='*60}", flush=True)
    print("S3 PARTIAL GRID RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for r in results:
        print(f"  {r['label']}: Hit@5={r['hit5']:.3f} MRR={r['mrr']:.3f} chunks={r['n_chunks']}", flush=True)

    best = sorted(results, key=lambda x: (x["hit5"], x["mrr"]), reverse=True)[0]
    print(f"\n  Best: {best['label']} Hit@5={best['hit5']:.3f}", flush=True)


if __name__ == "__main__":
    main()
