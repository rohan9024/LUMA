from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    device: str = "auto"
    d: int = 512                 # 512 for ViT-B/32; keep this so CLAP↔CLIP dims match
    clip_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    text_maxlen: int = 256
    frame_sample_rate: int = 2

@dataclass
class IndexConfig:
    hnsw_m: int = 64
    hnsw_efc: int = 200
    hnsw_efs: int = 128
    use_ivfpq: bool = True
    ivf_nlist: int = 4096
    pq_m: int = 16
    pq_bits: int = 8

@dataclass
class MemoryConfig:
    budget_hot: int = 2000
    budget_warm: int = 5_000_000
    decay_tau_s: float = 7 * 24 * 3600
    w_recency: float = 0.35
    w_usage: float = 0.25
    w_novelty: float = 0.25
    w_coverage: float = 0.15

@dataclass
class Paths:
    root: Path = Path("./data")
    media_inbox: Path = Path("./data/inbox")
    store: Path = Path("./data/store")
    hnsw_path: Path = Path("./data/store/hnsw.index")
    ivfpq_path: Path = Path("./data/store/ivfpq.index")
    meta_path: Path = Path("./data/store/meta.parquet")

@dataclass
class AppConfig:
    models: ModelConfig = ModelConfig()
    index: IndexConfig = IndexConfig()
    memory: MemoryConfig = MemoryConfig()
    paths: Paths = Paths()

CFG = AppConfig()