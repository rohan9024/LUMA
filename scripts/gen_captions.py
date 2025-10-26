import argparse, os, sys, time, glob
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/inbox", help="folder with images")
    ap.add_argument("--out", default="data/captions.txt", help="output captions file (CSV: file, caption)")
    ap.add_argument("--device", default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    use_cuda = (args.device in ("auto","cuda")) and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"[gen] device={device}")

    files = []
    for e in ("*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"):
        files += glob.glob(os.path.join(args.dir, e))
    files = sorted({os.path.basename(f): f for f in files}.values(), key=lambda p: os.path.basename(p).lower())
    if not files:
        print("[gen] no images found"); return

    print("[gen] loading BLIP model ...")
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    caps = []
    t0 = time.time()
    for i in range(0, len(files), args.batch):
        batch = files[i:i+args.batch]
        images = [Image.open(p).convert("RGB") for p in batch]
        inputs = proc(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=25)
        texts = proc.batch_decode(out, skip_special_tokens=True)
        for p, c in zip(batch, texts):
            nm = os.path.basename(p)
            caps.append(f"{nm}, {c}")
        print(f"[gen] {i+len(batch)}/{len(files)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(caps), encoding="utf-8")
    print(f"[gen] wrote {len(caps)} captions to {args.out} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()