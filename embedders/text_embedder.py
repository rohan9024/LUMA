# luma/embedders/text_embedder.py
# For diversity, reuse CLIP text or plug SentenceTransformers. We'll default to CLIP text for joint space.
from .image_embedder import ImageEmbedder
class TextEmbedder:
    def __init__(self, device="cuda", clip_name="ViT-L-14", pretrained="openai"):
        self.clip = ImageEmbedder(device, clip_name, pretrained)
    def embed(self, texts):
        return self.clip.embed_text(texts)