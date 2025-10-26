# scripts/eval_latency.py
import time, numpy as np
from config import CFG
from index.manager import MultiTierIndex
from memory.policy import StreamingMemoryPolicy
from ingest.pipeline import Ingestor
from retrieval.engine import RetrievalEngine
from PIL import Image
from pathlib import Path
import glob

def main():
    # small folder of images (e.g., data/inbox/*.jpg)
    files = sorted(glob.glob("data/inbox/*.jpg"))[:256]
    if not files:
        print("Put some images into data/inbox/*.jpg first.")
        return

    idx = MultiTierIndex(CFG.models.d, CFG.index)
    pol = StreamingMemoryPolicy(CFG.memory, CFG.models.d)
    ing = Ingestor(CFG, idx, pol)
    eng = RetrievalEngine(CFG, idx, ing)

    # ingest timing
    imgs = [Image.open(p).convert("RGB") for p in files]
    caps = [""] * len(imgs)
    metas = [{"source": files[i], "text": "", "modality":"image", "ts": time.time()} for i in range(len(files))]

    t0 = time.time()
    ids = ing.ingest_images_with_captions(files, caps, metas, use_caption_alignment=False)
    per_item = (time.time() - t0) * 1000.0 / max(1, len(files))
    print(f"Ingest latency: {per_item:.2f} ms/item over {len(files)} items")

    # query timing
    qv = eng.embed_query("a test query", modality="text")
    times = []
    for _ in range(200):
        s = time.time(); eng.search(qv, k=10); times.append((time.time()-s)*1000.0)
    print(f"Query latency p50={np.percentile(times,50):.2f} ms, p95={np.percentile(times,95):.2f} ms")

if __name__ == "__main__":
    main()