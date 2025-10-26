# embedders/video_embedder.py
import cv2, numpy as np, torch
from PIL import Image
from .image_embedder import ImageEmbedder

class VideoEmbedder:
    def __init__(self, image_embedder: ImageEmbedder | None = None,
                 device="auto", fps=2, clip_name="ViT-L-14", pretrained="openai"):
        self.fps = fps
        self.img = image_embedder if image_embedder is not None else ImageEmbedder(device, clip_name, pretrained)

    def _sample_frames(self, path):
        cap = cv2.VideoCapture(str(path))
        fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(int(round(fps_native / self.fps)), 1)
        frames, i = [], 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if i % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            i += 1
        cap.release()
        return frames

    @torch.inference_mode()
    def embed(self, paths, frame_bs=64):
        out = []
        for p in paths:
            frames = self._sample_frames(p)
            if not frames:
                out.append(np.zeros((self.img.model.text_projection.shape[1],), dtype=np.float32))
                continue
            embs = []
            for i in range(0, len(frames), frame_bs):
                chunk = frames[i:i+frame_bs]
                e = self.img.embed_image(chunk, batch_size=len(chunk))
                embs.append(e)
            v = np.vstack(embs).mean(axis=0)
            v = v / (np.linalg.norm(v) + 1e-9)
            out.append(v.astype(np.float32))
        return np.vstack(out)