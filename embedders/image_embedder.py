# embedders/image_embedder.py
import torch, open_clip
from PIL import Image
import numpy as np

class ImageEmbedder:
    def __init__(self, device="auto", clip_name="ViT-L-14", pretrained="openai"):
        use_cuda = (device in ("auto","cuda")) and torch.cuda.is_available() and (torch.version.cuda is not None)
        self.device = "cuda" if use_cuda else "cpu"
        self.precision = "fp16" if self.device == "cuda" else "fp32"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_name, pretrained=pretrained, device=self.device, precision=self.precision
        )
        self.tokenizer = open_clip.get_tokenizer(clip_name)
        torch.backends.cudnn.benchmark = True
        print(f"[ImageEmbedder] Using device: {self.device}, precision: {self.precision}")

    @torch.inference_mode()
    def embed_image(self, pil_images, batch_size=32):
        outs = []
        model_dtype = next(self.model.parameters()).dtype  # expect float16 on CUDA
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i+batch_size]
            t = torch.stack([self.preprocess(im) for im in batch])  # CPU, float32
            # Move to device first, then force dtype to match model
            t = t.to(self.device, non_blocking=True)
            if t.dtype != model_dtype:
                t = t.to(dtype=model_dtype)

            # Debug (optional): uncomment to verify
            # print("embed_image dtypes -> input:", t.dtype, "model:", model_dtype, "device:", t.device)

            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = self.model.encode_image(t)
            else:
                feats = self.model.encode_image(t)

            feats = torch.nn.functional.normalize(feats, dim=-1)
            outs.append(feats.float().cpu().numpy())
        return np.vstack(outs)

    @torch.inference_mode()
    def embed_text(self, texts, batch_size=64):
        outs = []
        for i in range(0, len(texts), batch_size):
            toks = self.tokenizer(texts[i:i+batch_size]).to(self.device)
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = self.model.encode_text(toks)
            else:
                feats = self.model.encode_text(toks)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            outs.append(feats.float().cpu().numpy())
        return np.vstack(outs)