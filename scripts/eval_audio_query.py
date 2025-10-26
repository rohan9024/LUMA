# scripts/eval_audio_query.py
# (Only the main evaluation loop changes to pass modality="audio")
import argparse, os, glob, sys, time
from pathlib import Path
import numpy as np

# --- Keep all the helper functions and imports as they are ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CFG
from index.manager import MultiTierIndex
from memory.policy import StreamingMemoryPolicy
from ingest.pipeline import Ingestor
from retrieval.engine import RetrievalEngine

def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]; gains = (2**rels - 1); discounts = 1.0 / np.log2(np.arange(2, 2 + len(rels))); return float(np.sum(gains * discounts))
def ndcg_at_k(rels, k):
    ideal = np.sort(rels)[::-1]; idcg = dcg_at_k(ideal, k); return 0.0 if idcg == 0.0 else dcg_at_k(rels, k) / idcg
def mrr(rels):
    for i, r in enumerate(rels, 1):
        if r > 0: return 1.0 / i
    return 0.0
def recall_at_k(rels, k):
    rels = np.asarray(rels)[:k]; return float((rels > 0).any())

def read_captions_csv(caps_path, files):
    lines = [ln.strip() for ln in Path(caps_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    m = {};
    for ln in lines:
        if "," in ln: a,b = ln.split(",",1); m[a.strip()] = b.strip()
    return [m.get(os.path.basename(p), "") for p in files]
# --- End of helpers ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", default="data/inbox")
    ap.add_argument("--captions", default="data/captions_inbox.txt")
    ap.add_argument("--audio_dir", default="data/audio_inbox")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    # --- Setup code is the same ---
    files = [];
    for e in ("*.jpg","*.jpeg","*.png"): files += glob.glob(os.path.join(args.img_dir, e))
    files = sorted({os.path.basename(f).lower(): f for f in files}.values(), key=lambda p: os.path.basename(p).lower())
    captions = read_captions_csv(args.captions, files)
    audios = [str(Path(args.audio_dir) / (Path(p).stem + ".wav")) for p in files]
    valids = [i for i,a in enumerate(audios) if Path(a).exists()]
    if not valids: print("[audio-eval] no audio files found."); return

    idx = MultiTierIndex(CFG.models.d, CFG.index); pol = StreamingMemoryPolicy(CFG.memory, CFG.models.d)
    ing = Ingestor(CFG, idx, pol); eng = RetrievalEngine(CFG, idx, ing)
    metas = [{"source": files[i], "caption": captions[i]} for i in range(len(files))]
    # Ingesting images also trains the CLAP->CLIP bridge from captions
    image_ids = ing.ingest_images_with_captions(files, captions, metas, use_caption_alignment=False)

    # --- Evaluation loop with modality="audio" ---
    ks = [1,5,args.k]; ndcgs = {k: [] for k in ks}; recs = {k: [] for k in ks}
    mrrs, safe_flags, qlat = [], [], []

    print("[audio-eval] Querying with audio...")
    for j in valids:
        qv = eng.embed_query(audios[j], modality="audio")
        # THIS IS THE KEY CHANGE: pass modality="audio" to telemetry
        D, I, tel = eng.search_with_telemetry(qv, k=args.k, modality="audio")
        qlat.append(tel["lat_ms"])
        rpaths = [idx.id2meta.get(int(x), {}).get("source") for x in I]
        rels = [1 if (rp is not None and rp == files[j]) else 0 for rp in rpaths]
        for k in ks:
            ndcgs[k].append(ndcg_at_k(rels, k)); recs[k].append(recall_at_k(rels, k))
        mrrs.append(mrr(rels)); safe_flags.append(1.0 if tel["safe_top1"] else 0.0)

    # --- Summary printing is the same ---
    print("\n=== Audio→Image Summary ===")
    for k in ks: print(f"nDCG@{k}: {np.mean(ndcgs[k]):.4f}   Recall@{k}: {np.mean(recs[k]):.4f}")
    print(f"MRR: {np.mean(mrrs):.4f}"); print(f"Safe@1: {np.mean(safe_flags):.4f}")
    print(f"Query latency p50: {np.percentile(qlat,50):.2f} ms, p95: {np.percentile(qlat,95):.2f} ms")
    print(f"eps_align: {ing.alignment_eps_for_modality('audio'):.4f}   zeta_pq: {idx.zeta_estimate():.4f}")

if __name__ == "__main__":
    main()