# luma/embedders/video_embedder.py
import cv2, numpy as np, torch
from PIL import Image
from .image_embedder import ImageEmbedder

class VideoEmbedder:
    def __init__(self, device="cuda", fps=2, clip_name="ViT-L-14", pretrained="openai"):
        self.fps = fps
        self.img = ImageEmbedder(device, clip_name, pretrained)

    def _sample_frames(self, path):
        cap = cv2.VideoCapture(str(path))
        fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(int(round(fps_native / self.fps)), 1)
        frames = []
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if i % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            i += 1
        cap.release()
        return frames

    @torch.no_grad()
    def embed(self, paths):
        out = []
        for p in paths:
            frames = self._sample_frames(p)
            if not frames:
                out.append(np.zeros((self.img.model.text_projection.shape[1],), dtype=np.float32))
                continue
            embs = self.img.embed_image(frames)
            v = embs.mean(axis=0)
            v = v / (np.linalg.norm(v) + 1e-9)
            out.append(v.astype(np.float32))
        return np.vstack(out)