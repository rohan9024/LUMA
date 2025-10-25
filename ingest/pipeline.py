# luma/ingest/pipeline.py
from pathlib import Path
import json, time
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
        self.txt = TextEmbedder(mcfg.device, mcfg.clip_name, mcfg.clip_pretrained)
        self.vid = VideoEmbedder(mcfg.device, mcfg.frame_sample_rate, mcfg.clip_name, mcfg.clip_pretrained)
        try:
            self.aud = AudioEmbedder(mcfg.clap_repo, mcfg.device)
        except Exception:
            self.aud = None
        d = cfg.models.d
        # Alignment: image/text are canonical; audio -> canonical; video uses image, so canonical too
        self.align = {
            "text": IncrementalProcrustes(d, ema=0.999),
            "image": IncrementalProcrustes(d, ema=0.999),
            "video": IncrementalProcrustes(d, ema=0.999),
            "audio": IncrementalProcrustes(d, ema=0.999),
        }
        # Initialize canonical as identity; only apply transforms to non-canonical if needed
        self.canonical = {"text", "image", "video"}  # audio aligned to text

        self.policy = policy

    def _align_vec(self, v, modality):
        v = v.astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        if modality in self.canonical:
            return v
        return self.align[modality].align(v)

    def ingest_text(self, texts, metas):
        X = self.txt.embed(texts)  # already in canonical
        # optional: refresh alignment for text vs image if you want dual adaptation
        ids = self.idx.add(X, metas, hot=True)
        for vid, x, meta in zip(ids, X, metas):
            self.policy.score(int(vid), x, usage=0, ts=meta.get("ts", time.time()))
        return ids

    def ingest_images_with_captions(self, image_paths, captions, metas):
        imgs = [Image.open(p).convert("RGB") for p in image_paths]
        X_img = self.img.embed_image(imgs)  # canonical
        X_txt = self.txt.embed(captions)    # canonical
        # Update cross-modal alignment stats (image<->text): here canonical, but maintain stability
        for xi, yt in zip(X_img, X_txt):
            self.align["image"].update_pair(xi, yt)
        self.align["image"].refresh()

        ids = self.idx.add(X_img, metas, hot=True)
        for vid, x, meta in zip(ids, X_img, metas):
            self.policy.score(int(vid), x, usage=0, ts=meta.get("ts", time.time()))
        return ids

    def ingest_videos_with_transcripts(self, video_paths, transcripts, metas):
        X_vid = self.vid.embed(video_paths)   # canonical via image frames
        X_txt = self.txt.embed(transcripts)   # canonical
        # Update image/video alignment if desired (video vs text)
        for xv, yt in zip(X_vid, X_txt):
            self.align["video"].update_pair(xv, yt)
        self.align["video"].refresh()

        ids = self.idx.add(X_vid, metas, hot=True)
        for vid, x, meta in zip(ids, X_vid, metas):
            self.policy.score(int(vid), x, usage=0, ts=meta.get("ts", time.time()))
        return ids

    def ingest_audio_with_captions(self, audio_paths, captions, metas):
        if self.aud is None:
            raise RuntimeError("Audio model not available.")
        X_aud = self.aud.embed(audio_paths)            # not canonical
        X_txt = self.txt.embed(captions)               # canonical
        # Update audio->text alignment
        for xa, yt in zip(X_aud, X_txt):
            self.align["audio"].update_pair(xa, yt)
        self.align["audio"].refresh()
        # Align audio vectors into canonical
        X_al = np.vstack([self._align_vec(xa, "audio") for xa in X_aud])
        ids = self.idx.add(X_al, metas, hot=True)
        for vid, x, meta in zip(ids, X_al, metas):
            self.policy.score(int(vid), x, usage=0, ts=meta.get("ts", time.time()))
        return ids