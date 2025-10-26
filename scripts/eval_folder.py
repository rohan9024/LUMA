# scripts/eval_folder.py
import argparse, time, glob, os, sys, traceback
from pathlib import Path
import numpy as np

def log(*a): print(*a, flush=True)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CFG
from index.manager import MultiTierIndex
from memory.policy import StreamingMemoryPolicy
from ingest.pipeline import Ingestor
from retrieval.engine import RetrievalEngine

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

def read_captions_any(caps_path: Path, files: list[str]) -> list[str]:
    lines = [ln.strip() for ln in caps_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    csv = any("," in ln for ln in lines)
    if not csv:
        return lines[:len(files)]
    name2cap = {}
    for ln in lines:
        parts = ln.split(",", 1)
        if len(parts) >= 2:
            name2cap[parts[0].strip()] = parts[1].strip()
    out = []
    for p in files:
        out.append(name2cap.get(os.path.basename(p), ""))
    return out

def main():
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--dir", default="data/inbox")
        ap.add_argument("--captions", default="data/captions.txt")
        ap.add_argument("--k", type=int, default=10)
        ap.add_argument("--align", action="store_true")
        ap.add_argument("--no-align", dest="align", action="store_false")
        ap.set_defaults(align=True)
        ap.add_argument("--hot_budget", type=int, default=2000)
        args = ap.parse_args()

        CFG.memory.budget_hot = int(args.hot_budget)

        files = []
        for e in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"):
            files += glob.glob(os.path.join(args.dir, e))
        files = sorted({os.path.basename(f).lower(): f for f in files}.values(), key=lambda p: os.path.basename(p).lower())
        log(f"[info] Found {len(files)} unique images in {args.dir}")
        if not files: return

        caps_path = Path(args.captions)
        captions = read_captions_any(caps_path, files) if caps_path.exists() else [""]*len(files)
        m = min(len(files), len(captions))
        files, captions = files[:m], captions[:m]
        log(f"[info] Using {m} image-caption pairs")

        idx = MultiTierIndex(CFG.models.d, CFG.index)
        pol = StreamingMemoryPolicy(CFG.memory, CFG.models.d)
        ing = Ingestor(CFG, idx, pol)
        eng = RetrievalEngine(CFG, idx, ing)
        log("[info] Ready. Ingesting ...")

        metas = [{"source": files[i], "caption": captions[i], "text": captions[i], "modality":"image", "ts": time.time()} for i in range(m)]
        t0 = time.time()
        ids = ing.ingest_images_with_captions(files, captions, metas, use_caption_alignment=args.align)
        log(f"[info] Ingest avg latency: {(time.time()-t0)*1000.0/m:.2f} ms/item")

        ks = [1,5,args.k]
        ndcgs = {k: [] for k in ks}; recs  = {k: [] for k in ks}
        mrrs, safe_flags, qlat = [], [], []
        log(f"[info] Querying {m} captions ...")
        for cap, rid in zip(captions, ids):
            qv = eng.embed_query(cap, modality="text")
            D, I, tel = eng.search_with_telemetry(qv, k=args.k, modality="text")
            qlat.append(tel["lat_ms"])
            rels = [1 if int(x) == int(rid) else 0 for x in I]
            for k in ks:
                ndcgs[k].append(ndcg_at_k(rels, k))
                recs[k].append(recall_at_k(rels, k))
            mrrs.append(mrr(rels))
            safe_flags.append(1.0 if tel["safe_top1"] else 0.0)

        log("\n=== Summary (folder eval) ===")
        for k in ks:
            log(f"nDCG@{k}: {np.mean(ndcgs[k]):.4f}   Recall@{k}: {np.mean(recs[k]):.4f}")
        log(f"MRR: {np.mean(mrrs):.4f}")
        log(f"Safe@1: {np.mean(safe_flags):.4f}")
        log(f"Query latency p50: {np.percentile(qlat,50):.2f} ms, p95: {np.percentile(qlat,95):.2f} ms")
        log(f"Hot size: {idx.hot.ntotal}  Warm trained: {bool(idx.warm and idx.warm.is_trained())}  ZetaPQ: {idx.zeta_estimate():.4f}")
    except Exception:
        print("\n[error] Unexpected exception:", flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()