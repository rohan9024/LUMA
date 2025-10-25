# luma/retrieval/engine.py
import numpy as np
from embedders.image_embedder import ImageEmbedder
from embedders.text_embedder import TextEmbedder
from embedders.audio_embedder import AudioEmbedder
from embedders.video_embedder import VideoEmbedder

class RetrievalEngine:
    def __init__(self, cfg, index_mgr, ingestor):
        self.cfg = cfg
        self.idx = index_mgr
        self.ingestor = ingestor  # for embedders and alignment

    def embed_query(self, query, modality="text", extra=None):
        if modality == "text":
            v = self.ingestor.txt.embed([query])[0]
        elif modality == "image":
            v = self.ingestor.img.embed_image([query])[0]  # query is PIL.Image
        elif modality == "audio":
            v = self.ingestor.aud.embed([query])[0]
            v = self.ingestor._align_vec(v, "audio")
        elif modality == "video":
            v = self.ingestor.vid.embed([query])[0]
        else:
            raise ValueError("unknown modality")
        return v.astype(np.float32)

    def search(self, query_vec, k=10):
        D, I = self.idx.search(query_vec[None, :], k=k)
        return D[0], I[0]