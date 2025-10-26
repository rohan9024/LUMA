# scripts/eval_folder_group.py
import argparse, glob, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CFG
from index.manager import MultiTierIndex
from memory.policy import StreamingMemoryPolicy
from ingest.pipeline import Ingestor
from retrieval.engine import RetrievalEngine

def base_group(path):
    stem = Path(path).stem
    if "_aug" in stem:
        stem = stem.split("_aug")[0]
    return stem.lower()

def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    gains = (2**rels - 1)
    discounts = 1.0 / np.log2(np.arange(2, 2 + len(rels)))
    return float(np.sum(gains * discounts))
def ndcg_at_k(rels, k):
    ideal = np.sort(rels)[::-1]; idcg = dcg_at_k(ideal, k)
    return 0.0 if idcg == 0.0 else dcg_at_k(rels, k) / idcg
def mrr(rels):
    for i, r in enumerate(rels, 1):
        if r > 0: return 1.0 / i
    return 0.0
def recall_at_k(rels, k):
    rels = np.asarray(rels)[:k]; return float((rels > 0).any())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/inbox_aug")
    ap.add_argument("--captions", default="data/captions_aug.txt")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--hot_budget", type=int, default=500)
    args = ap.parse_args()

    CFG.memory.budget_hot = int(args.hot_budget)

    files = []
    for e in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"):
        files += glob.glob(os.path.join(args.dir, e))
    files = sorted(files, key=lambda p: os.path.basename(p).lower())
    if not files:
        print("[group-eval] no images found"); return

    # read captions csv or lines
    lines = [ln.strip() for ln in Path(args.captions).read_text(encoding="utf-8").splitlines() if ln.strip()]
    m = {}
    for ln in lines:
        if "," in ln:
            a,b = ln.split(",",1)
            m[a.strip()] = b.strip()
    captions = [m.get(os.path.basename(p), "") for p in files]

    # Build
    idx = MultiTierIndex(CFG.models.d, CFG.index)
    pol = StreamingMemoryPolicy(CFG.memory, CFG.models.d)
    ing = Ingestor(CFG, idx, pol)
    eng = RetrievalEngine(CFG, idx, ing)

    metas = [{"source": files[i], "caption": captions[i], "text": captions[i], "modality":"image", "ts": time.time()} for i in range(len(files))]
    ing.ingest_images_with_captions(files, captions, metas, use_caption_alignment=False)

    # map sources to groups
    groups = [base_group(p) for p in files]

    ks = [1,5,args.k]
    ndcgs = {k: [] for k in ks}; recs = {k: [] for k in ks}
    mrrs = []; qlat = []
    for i, cap in enumerate(captions):
        qv = eng.embed_query(cap, modality="text")
        D, I, tel = eng.search_with_telemetry(qv, k=args.k, modality="text")
        qlat.append(tel["lat_ms"])
        rpaths = [idx.id2meta.get(int(x), {}).get("source") for x in I]
        rgroups = [base_group(p) if p else "" for p in rpaths]
        rels = [1 if g == groups[i] else 0 for g in rgroups]
        for k in ks:
            ndcgs[k].append(ndcg_at_k(rels, k))
            recs[k].append(recall_at_k(rels, k))
        mrrs.append(mrr(rels))

    print("\n=== Group-aware Summary ===")
    for k in ks:
        print(f"nDCG@{k}: {np.mean(ndcgs[k]):.4f}   Recall@{k}: {np.mean(recs[k]):.4f}")
    print(f"MRR: {np.mean(mrrs):.4f}")
    print(f"Query latency p50: {np.percentile(qlat,50):.2f} ms, p95: {np.percentile(qlat,95):.2f} ms")
    print(f"Hot size: {idx.hot.ntotal}  Warm trained: {bool(idx.warm and idx.warm.is_trained())}")
if __name__ == "__main__":
    main()