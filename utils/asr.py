# utils/asr.py
import torch

class ASRTranscriber:
    def __init__(self, model_size="small", device_auto="auto"):
        from faster_whisper import WhisperModel
        device = "cuda" if (device_auto in ("auto","cuda") and torch.cuda.is_available()) else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_text(self, path: str) -> str:
        segments, info = self.model.transcribe(path, beam_size=1, vad_filter=True)
        text = " ".join([s.text for s in segments]).strip()
        return text