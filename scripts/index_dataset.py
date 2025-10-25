# scripts/index_dataset.py
# Placeholder to index a folder with images and a captions.json mapping
import json, time
from pathlib import Path
from PIL import Image
from config import CFG
from index.manager import MultiTierIndex
from memory.policy import StreamingMemoryPolicy
from ingest.pipeline import Ingestor

def main(data_dir="./data/sample"):
    idx = MultiTierIndex(CFG.models.d, CFG.index)
    pol = StreamingMemoryPolicy(CFG.memory, CFG.models.d)
    ing = Ingestor(CFG, idx, pol)
    p = Path(data_dir)
    caps = json.loads((p / "captions.json").read_text())
    image_paths, captions, metas = [], [], []
    for fn, cap in caps.items():
        image_paths.append(str(p/fn))
        captions.append(cap)
        metas.append({"source": fn, "ts": time.time(), "modality":"image"})
    ing.ingest_images_with_captions(image_paths, captions, metas)
    print("Indexed", len(image_paths), "images")

if __name__ == "__main__":
    main()