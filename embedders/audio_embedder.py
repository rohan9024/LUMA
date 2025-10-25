# luma/embedders/audio_embedder.py
import torch, numpy as np, librosa
from transformers import ClapProcessor, ClapModel

class AudioEmbedder:
    def __init__(self, repo="laion/clap-htsat-unfused", device="auto"):
        use_cuda = (device in ("auto","cuda")) and torch.cuda.is_available() and (torch.version.cuda is not None)
        self.device = "cuda" if use_cuda else "cpu"
        self.processor = ClapProcessor.from_pretrained(repo)
        self.model = ClapModel.from_pretrained(repo).to(self.device)
        print(f"[AudioEmbedder] Using device: {self.device}")

    @torch.no_grad()
    def embed(self, wav_paths):
        feats = []
        for p in wav_paths:
            y, sr = librosa.load(p, sr=48000, mono=True)
            inputs = self.processor(audios=y, sampling_rate=48000, return_tensors="pt").to(self.device)
            emb = self.model.get_audio_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            feats.append(emb.squeeze(0).float().cpu().numpy())
        return np.vstack(feats)