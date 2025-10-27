# 🌌 LUMA-RAG: Lifelong Multimodal Retrieval-Augmented Agent

> **Trustworthy, low-latency RAG** that keeps learning from text, images, video, and audio — without re-indexing.

LUMA maintains a **cheap, streaming cross-modal alignment** (CLAP→CLIP bridge), **prioritizes memory under a budget**, and exposes a **provable safety signal** for each answer.

---

## 🧠 Overview

| Component | Description |
|------------|--------------|
| **Canonical space** | CLIP (ViT-B/32, d=512) |
| **Modalities** | Text, Images, Video (frames), Audio (CLAP) |
| **Indexing** | Hot HNSW (RAM) + Warm IVFPQ (compressed, adaptive) |
| **Alignment** | Incremental Procrustes (streaming CLAP→CLIP bridge) |
| **Safety** | Safe@k = margin_top2 vs (ε alignment drift + ζ PQ distortion) |
| **UI** | Streamlit app with live ingestion, retrieval, citations, telemetry |

---

## 🚀 Why LUMA is Different

- 🧩 **Streaming alignment:** integrates new modalities online (no re-index).
- 🛡️ **Safety you can see:** every query shows `margin_top2`, `ε`, `ζ`, `Safe@1`.
- 🧮 **Real-world memory:** adaptive hot/warm tiers with IVFPQ compression.
- ⚖️ **Provable stability:** Safe@1 ensures the top-1 result won't flip under drift.

---

## ⚙️ Quick Start

### 1️⃣ Prerequisites

- Python **3.10** or **3.11**
- OS: Windows / Linux / macOS  
- GPU recommended (NVIDIA)
- FFmpeg (required for video/audio)

```bash
# Windows
choco install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

---

### 2️⃣ Create Environment & Install Dependencies

```bash
# Clone the repository
git clone https://github.com/rohan9024/LUMA.git
cd LUMA

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install PyTorch (choose one based on your setup)
# CUDA 11.8 build (recommended for NVIDIA GPUs):
pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio

# OR CPU-only (slower):
# pip install torch torchvision torchaudio

# Install core dependencies
pip install open-clip-torch faiss-cpu transformers sentencepiece opencv-python ffmpeg-python librosa pydub numpy scipy scikit-learn streamlit tqdm orjson fastapi uvicorn accelerate pyttsx3
```

---

### 3️⃣ Configure

Create or edit `config.py` in the root directory:

```python
# config.py
class ModelConfig:
    device = "auto"  # "cuda", "cpu", or "auto"
    clip_name = "ViT-B-32"
    clip_pretrained = "openai"
    d = 512  # embedding dimension
    
    clap_model = "laion/clap-htsat-unfused"
    clap_d = 512

class MemoryConfig:
    budget_hot = 2000  # max items in hot (HNSW) tier
    budget_warm = 10000  # max items in warm (IVFPQ) tier
    
    # HNSW parameters
    hnsw_M = 16
    hnsw_ef_construction = 200
    hnsw_ef_search = 50
    
    # IVFPQ parameters
    ivf_nlist = 32
    pq_m = 8
    pq_nbits = 8

class AlignmentConfig:
    refresh_interval = 100  # refresh alignment every N audio queries
    epsilon_threshold = 0.1  # alignment drift warning threshold

class SafetyConfig:
    k = 10  # retrieve top-k results
    safety_multiplier = 2.0  # Safe@1 = margin_top2 > 2*(ε + ζ)
```

---

### 4️⃣ Run the App

```bash
# Start the Streamlit web interface
streamlit run web/app.py

# OR
python -m streamlit run web/app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`).

**Features in the UI:**
- 📥 **Ingest** text, images, video, audio
- 🔍 **Query** across all modalities
- 📊 **View telemetry**: `margin_top2`, `ε`, `ζ`, `Safe@1`
- 📚 **Citations** with source tracking

---

## 🧩 Usage Examples

### A) Text → Image Retrieval

```bash
# 1. Generate captions for your images using BLIP
pip install accelerate
python -m scripts.gen_captions --dir data/inbox --out data/captions_inbox.txt

# 2. Evaluate retrieval performance
python -u -m scripts.eval_folder \
  --dir data/inbox \
  --captions data/captions_inbox.txt \
  --no-align \
  --hot_budget 1000
```

**Expected Results:**
- Recall@10 ≈ 0.94
- MRR ≈ 0.59
- Safe@1 ≈ 1.00

---

### B) Memory Spill (Hot/Warm Tiers)

```bash
# 1. Augment dataset with copies to simulate large-scale data
python -m scripts.augment_copies \
  --src data/inbox \
  --out data/inbox_aug \
  --copies 20

# 2. Generate captions for augmented data
python -m scripts.gen_captions \
  --dir data/inbox_aug \
  --out data/captions_aug.txt

# 3. Evaluate with memory budget constraints
python -u -m scripts.eval_folder_group \
  --dir data/inbox_aug \
  --captions data/captions_aug.txt \
  --hot_budget 500
```

**Expected Results:**
- Shows adaptive IVFPQ training
- Group-aware Recall@10 ≈ 0.53
- Demonstrates hot/warm tier switching

---

### C) Audio → Image (CLAP→CLIP Bridge)

```bash
# 1. Generate TTS audio from captions
pip install pyttsx3
python -m scripts.gen_tts_audio \
  --captions data/captions_inbox.txt \
  --audio_dir data/audio_inbox

# 2. Evaluate audio-to-image retrieval
python -u -m scripts.eval_audio_query \
  --img_dir data/inbox \
  --captions data/captions_inbox.txt \
  --audio_dir data/audio_inbox \
  --k 10
```

**Expected Results:**
- Recall@10 ≈ 0.42
- ε ≈ 0.00 after refresh
- Safe@1 ≈ 1.00

---

## 📁 Repository Structure

```
luma/
├── config.py                    # Configuration settings
├── luma/
│   ├── embedders/
│   │   ├── image_embedder.py   # CLIP image encoder
│   │   ├── text_embedder.py    # CLIP text encoder
│   │   ├── audio_embedder.py   # CLAP audio encoder
│   │   └── video_embedder.py   # Video frame processing
│   ├── alignment/
│   │   └── incremental_procrustes.py  # CLAP→CLIP alignment
│   ├── index/
│   │   ├── hnsw.py             # Hot tier (FAISS HNSW)
│   │   ├── ivfpq.py            # Warm tier (FAISS IVFPQ)
│   │   └── manager.py          # Multi-tier index manager
│   ├── memory/
│   │   └── policy.py           # Memory budget management
│   ├── ingest/
│   │   └── pipeline.py         # Data ingestion pipeline
│   ├── retrieval/
│   │   └── engine.py           # Retrieval with safety metrics
│   └── rag/
│       └── generator.py        # RAG response generation
├── web/
│   └── app.py                  # Streamlit web interface
├── scripts/
│   ├── run_server.py           # FastAPI server
│   ├── gen_captions.py         # BLIP caption generation
│   ├── gen_tts_audio.py        # TTS audio generation
│   ├── eval_folder.py          # Single-folder evaluation
│   ├── eval_folder_group.py    # Multi-tier evaluation
│   ├── eval_audio_query.py     # Audio retrieval evaluation
│   ├── augment_copies.py       # Dataset augmentation
│   └── profile_latency.py      # Performance profiling
├── data/
│   ├── inbox/                  # Your image data
│   ├── audio_inbox/            # Your audio data
│   └── captions_inbox.txt      # Caption file
└── README.md
```

---

## 🧮 Safety Telemetry (Safe@k)

For each query, LUMA computes:

| Metric | Meaning |
|--------|---------|
| **margin_top2** | Similarity gap between top-1 and top-2 results |
| **ε** | Alignment drift (spectral norm change since last refresh) |
| **ζ** | PQ distortion (reconstruction error for warm tier) |
| **Safe@1** | `True` if `margin_top2 > 2*(ε + ζ)` |

**Interpretation:**
- ✅ **Green (Safe@1=True)**: Top-1 result is stable and trustworthy
- ⚠️ **Yellow/Red**: Consider checking more results (broader k)

---

## 📊 Benchmark Results

| Task | Dataset | Metrics |
|------|---------|---------|
| **Text→Image** | 31 images, BLIP captions | nDCG@10 ≈ 0.67, Recall@10 ≈ 0.94, Safe@1 = 1.00 |
| **Memory/PQ Spill** | 620 images, hot_budget=500 | Recall@10 ≈ 0.53, MRR ≈ 0.47 |
| **Audio→Image** | TTS audio | Recall@10 ≈ 0.42, ε ≈ 0.00, Safe@1 ≈ 1.00 |

---

## 🧰 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Torch not compiled with CUDA** | Reinstall with CUDA wheel: `pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio` |
| **Paging file too small (Windows)** | Use ViT-B/32 or increase Windows pagefile size |
| **Relative import error** | Run from repo root with absolute imports: `python -m scripts.eval_folder` |
| **FFmpeg not found** | Add FFmpeg to PATH or reinstall |
| **FAISS "nx ≥ k" error** | Reduce `ivf_nlist` in config or ingest more data |
| **0.00 ms latency** | Ignore (Windows timer granularity issue) |
| **QuickGELU mismatch warning** | Harmless, can be ignored |
| **Missing captions** | Ensure `--captions` path is correct |

---

## 🧪 Research Claims

**LUMA is a streaming multimodal RAG system with:**
1. **Online CLAP→CLIP alignment bridge** (no re-indexing needed)
2. **Provable per-query stability guarantee** (Safe@k metric)
3. **Budgeted multi-tier memory** (hot HNSW + warm IVFPQ)

### Citation

If you use or extend LUMA, please cite:

```bibtex
@misc{luma2025,
  title  = {LUMA-RAG: Lifelong Multimodal Agents with Provably Stable Streaming Alignment},
  author = {Rohan Wandre},
  year   = {2025},
  note   = {https://github.com/rohan9024/luma}
}
```

---

## 🧭 Roadmap

- [ ] RL-based memory policy (learned retention)
- [ ] Non-linear hyper-network alignment
- [ ] Public multimodal benchmarks (Flickr30k, MSR-VTT, AudioCaps)
- [ ] Faithfulness & hallucination auditing
- [ ] Docker container for easy deployment
- [ ] Support for more audio formats (MP3, FLAC, etc.)
- [ ] Video understanding with temporal modeling
- [ ] Multi-user support with isolated memory spaces

---


## 🙏 Acknowledgements

- **OpenCLIP** team for CLIP models
- **CLAP** team for audio embeddings
- **FAISS** for efficient similarity search
- **BLIP** for image captioning
- **PyTorch**, **HuggingFace Transformers**, **Streamlit**, **scikit-learn** communities

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/rohan9024/LUMA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rohan9024/LUMA/discussions)
- **Email**: rohanwandre24@gmail.com

---

**Built with ❤️ for the multimodal AI community**
