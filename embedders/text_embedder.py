# embedders/text_embedder.py
# Reuse the same CLIP model as ImageEmbedder for a shared space.
from .image_embedder import ImageEmbedder

class TextEmbedder:
    def __init__(self, image_embedder: ImageEmbedder | None = None,
                 device="auto", clip_name="ViT-L-14", pretrained="openai"):
        self.clip = image_embedder if image_embedder is not None else ImageEmbedder(device, clip_name, pretrained)

    def embed(self, texts, batch_size=64):
        return self.clip.embed_text(texts, batch_size=batch_size)