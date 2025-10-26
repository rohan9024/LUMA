# scripts/gen_tts_audio.py
import argparse, os, sys, time
from pathlib import Path
import pyttsx3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="data/captions.txt", help="CSV 'file, caption' or one caption per line (aligned to sorted images).")
    ap.add_argument("--audio_dir", default="data/audio", help="Output folder for wav files")
    args = ap.parse_args()

    Path(args.audio_dir).mkdir(parents=True, exist_ok=True)
    lines = [ln.strip() for ln in Path(args.captions).read_text(encoding="utf-8").splitlines() if ln.strip()]
    # detect CSV
    csv = any("," in ln for ln in lines)
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)

    print(f"[tts] generating {len(lines)} wav files to {args.audio_dir} ...")
    if csv:
        for ln in lines:
            name, cap = ln.split(",", 1)
            name = name.strip()
            cap = cap.strip()
            out = Path(args.audio_dir) / (Path(name).stem + ".wav")
            engine.save_to_file(cap, str(out))
        engine.runAndWait()
    else:
        # one caption per line; we just index by number
        for i, cap in enumerate(lines):
            out = Path(args.audio_dir) / f"caption_{i:05d}.wav"
            engine.save_to_file(cap, str(out))
        engine.runAndWait()
    print("[tts] done.")

if __name__ == "__main__":
    main()