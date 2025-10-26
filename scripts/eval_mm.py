import argparse, time, os
from pathlib import Path
import numpy as np
from PIL import Image
import sys, pathlib

# Make repo root importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset
from config import CFG
from index.manager import MultiTierIndex
from memory.policy import StreamingMemoryPolicy
from ingest.pipeline import Ingestor
from retrieval.engine import RetrievalEngine
from eval.metrics import Meter
def save_images(tmp_dir, images):
    paths = []
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(images):
        p = tmp_dir / f"img_{i}.jpg"
        im.save(p)
        paths.append(str(p))
    return paths

def build_system():
    idx = MultiTierIndex(CFG.models.d, CFG.index)
    pol = StreamingMemoryPolicy(CFG.memory, CFG.models.d)
    ing = Ingestor(CFG, idx, pol)
    eng = RetrievalEngine(CFG, idx, ing)
    return idx, pol, ing, eng

def run_eval(n=1000, qn=200, batch=64, align=True, hot_budget=2000, seed=42):
    np.random.seed(seed)
    # Tighten hot budget for the memory test
    CFG.memory.budget_hot = int(hot_budget)

    print("Loading COCO (small subset)...")
    ds = load_dataset("coco_captions", "2017", split="validation")
    ds = ds.shuffle(seed=seed).select(range(n))
    images = [x["image"] for x in ds]
    captions = [x["captions"][0]["caption"] if x["captions"] else "" for x in ds]  # one caption per image
    paths = save_images(Path("data/coco_tmp"), images)

    idx, pol, ing, eng = build_system()

    print(f"Ingesting {n} images in batches of {batch} (alignment={'ON' if align else 'OFF'}) ...")
    t0 = time.time()
    all_ids = []
    for i in range(0, n, batch):
        b_imgs = paths[i:i+batch]
        b_caps = captions[i:i+batch]
        metas = [{"source": b_imgs[j], "caption": b_caps[j], "text": b_caps[j], "modality":"image", "ts": time.time()} for j in range(len(b_imgs))]
        before = time.time()
        ids = ing.ingest_images_with_captions(b_imgs, b_caps, metas, use_caption_alignment=align)
        all_ids.extend([int(x) for x in ids])
        per_item = (time.time() - before) * 1000.0 / max(1, len(b_imgs))
        print(f"  batch {i:5d}-{i+len(b_imgs)-1:5d}: {per_item:.2f} ms/item")
    t_ing = time.time() - t0
    print(f"Total ingest time: {t_ing:.2f}s  ({t_ing*1000/n:.2f} ms/item avg)")

    # Build queries: pick qn random images and use their captions as text queries
    q_idx = np.random.choice(n, size=min(qn, n), replace=False)
    q_caps = [captions[i] for i in q_idx]
    q_rel = [all_ids[i] for i in q_idx]  # id of the matching image
    meter = Meter()
    lat = []

    print("Querying...")
    for cap, rel_id in zip(q_caps, q_rel):
        qv = eng.embed_query(cap, modality="text")
        t1 = time.time()
        D, I, tel = eng.search_with_telemetry(qv, k=10)
        lat.append((time.time() - t1) * 1000.0)
        # Build binary relevance list
        rels = [1 if int(i) == int(rel_id) else 0 for i in I]
        meter.add(rels, margin_top2=tel["margin_top2"], eps=tel["eps_align"], zeta=tel["zeta_pq"])

    summ = meter.summary()
    summ["Query_p50_ms"] = float(np.percentile(lat, 50)) if lat else 0.0
    summ["Query_p95_ms"] = float(np.percentile(lat, 95)) if lat else 0.0
    summ["Hot_size"] = idx.hot.ntotal
    summ["Warm_trained"] = bool(idx.warm and idx.warm.is_trained())
    summ["ZetaPQ"] = idx.zeta_estimate()

    print("\n=== Summary ===")
    for k, v in summ.items():
        print(f"{k:>14}: {v:.4f}" if isinstance(v, float) else f"{k:>14}: {v}")
    return summ

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="num images to index")
    ap.add_argument("--qn", type=int, default=200, help="num queries")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--align", action="store_true", help="enable streaming caption alignment")
    ap.add_argument("--no-align", dest="align", action="store_false")
    ap.set_defaults(align=True)
    ap.add_argument("--hot_budget", type=int, default=2000)
    args = ap.parse_args()

    run_eval(n=args.n, qn=args.qn, batch=args.batch, align=args.align, hot_budget=args.hot_budget)

if __name__ == "__main__":
    main()