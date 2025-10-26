# scripts/augment_copies.py
import argparse, os, glob, random
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance

def aug(im):
    w,h = im.size
    cw,ch = int(w*0.9), int(h*0.9)
    x = random.randint(0, max(0,w-cw)); y = random.randint(0, max(0,h-ch))
    im = im.crop((x,y,x+cw,y+ch)).resize((w,h), Image.BICUBIC)
    if random.random()<0.5: im = ImageOps.mirror(im)
    im = ImageEnhance.Contrast(im).enhance(0.9 + 0.2*random.random())
    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/inbox")
    ap.add_argument("--out", default="data/inbox_aug")
    ap.add_argument("--copies", type=int, default=20)
    args = ap.parse_args()

    files = []
    for e in ("*.jpg","*.jpeg","*.png"): files += glob.glob(os.path.join(args.src, e))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    count = 0
    for p in files:
        im = Image.open(p).convert("RGB")
        base = Path(p).stem
        for i in range(args.copies):
            im2 = aug(im)
            im2.save(Path(args.out)/f"{base}_aug{i}.jpg", quality=92)
            count += 1
    print("wrote", count, "augmented images to", args.out)

if __name__ == "__main__":
    from PIL import Image
    main()