# ingest/pipeline.py
from pathlib import Path
import time
import numpy as np
from PIL import Image
from embedders.image_embedder import ImageEmbedder
from embedders.text_embedder import TextEmbedder
from embedders.video_embedder import VideoEmbedder
from embedders.audio_embedder import AudioEmbedder
from alignment.incremental_procrustes import IncrementalProcrustes

class Ingestor:
    def __init__(self, cfg, index_mgr, policy):
        self.cfg = cfg
        self.idx = index_mgr
        mcfg = cfg.models

        self.img = ImageEmbedder(mcfg.device, mcfg.clip_name, mcfg.clip_pretrained)
        self.txt = TextEmbedder(self.img)
        self.vid = VideoEmbedder(self.img, fps=mcfg.frame_sample_rate)
        try:
            self.aud = AudioEmbedder(device=mcfg.device)
        except Exception:
            self.aud = None

        d = mcfg.d
        self.align = {
            "clap2clip": IncrementalProcrustes(d, ema=0.999),  # CLAP text -> CLIP text bridge
        }
        self.canonical = {"text", "image", "video"}
        self.policy = policy

    def _align_vec(self, v, modality):
        v = v.astype(np.float32) / (np.linalg.norm(v) + 1e-9)
        if modality in self.canonical:
            return v
        return self.align[modality].align(v)

    def alignment_eps_for_modality(self, modality: str) -> float:
        if modality in self.canonical:
            return 0.0
        if modality == "audio" and "clap2clip" in self.align:
            return float(self.align["clap2clip"].alignment_error())
        return 2.0 # Default to max error if no alignment found

    def ingest_images_with_captions(self, image_paths, captions, metas, use_caption_alignment=True):
        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        X_img = self.img.embed_image(imgs)
        X_clip_txt = self.txt.embed(captions)

        # Train CLAP->CLIP bridge using captions
        if self.aud is not None and captions:
            X_clap_txt = self.aud.embed_text(captions)
            for xc, xt in zip(X_clap_txt, X_clip_txt):
                self.align["clap2clip"].update_pair(xc, xt)
            self.align["clap2clip"].refresh()

        ids = self.idx.add(X_img, metas, hot=True)
        for vid, x, meta in zip(ids, X_img, metas):
            self.policy.score(int(vid), x, usage=0, ts=meta.get("ts", time.time()))
        self.idx.rebalance(self.policy)
        return ids

    # Keep other ingest methods as they are
    def ingest_text(self, texts, metas):
        X_clip_txt = self.txt.embed(texts)
        if self.aud is not None and texts:
            X_clap_txt = self.aud.embed_text(texts)
            for xc, xt in zip(X_clap_txt, X_clip_txt):
                self.align["clap2clip"].update_pair(xc, xt)
            self.align["clap2clip"].refresh()
        ids = self.idx.add(X_clip_txt, metas, hot=True)
        for vid, x, meta in zip(ids, X_clip_txt, metas):
            self.policy.score(int(vid), x, usage=0, ts=meta.get("ts", time.time()))
        self.idx.rebalance(self.policy)
        return ids