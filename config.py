# luma/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    device: str = "cuda"  # "cpu" if no GPU
    d: int = 768  # shared embedding dim
    clip_name: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    clap_repo: str = "laion/clap-htsat-unfused"
    text_maxlen: int = 256
    frame_sample_rate: int = 2  # frames per second

@dataclass
class IndexConfig:
    hnsw_m: int = 64
    hnsw_efc: int = 200
    hnsw_efs: int = 128
    use_ivfpq: bool = True
    ivf_nlist: int = 4096
    pq_m: int = 16  # subquantizers
    pq_bits: int = 8

@dataclass
class MemoryConfig:
    budget_hot: int = 200_000
    budget_warm: int = 5_000_000
    decay_tau_s: float = 7 * 24 * 3600  # 1 week half-life
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