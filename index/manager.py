# luma/index/manager.py
import numpy as np
from .hnsw import HNSWIndex
from .ivfpq import IVFPQIndex

class MultiTierIndex:
    def __init__(self, dim, cfg):
        self.hot = HNSWIndex(dim, m=cfg.hnsw_m, efc=cfg.hnsw_efc, efs=cfg.hnsw_efs)
        self.warm = IVFPQIndex(dim, nlist=cfg.ivf_nlist, m=cfg.pq_m, nbits=cfg.pq_bits) if cfg.use_ivfpq else None
        self.id2meta = {}  # map global id -> metadata keys
        self.next_id = 0

    def train_warm(self, samples):
        if self.warm:
            self.warm.train(samples)

    def add(self, vecs, metas, hot=True):
        # Assign global ids
        ids = np.arange(self.next_id, self.next_id + len(vecs), dtype=np.int64)
        self.next_id += len(vecs)
        if hot:
            hot_ids = self.hot.add(vecs)
            # map hot local ids to global ids (simplified: assume aligned)
        elif self.warm:
            self.warm.add(vecs)
        for i, m in zip(ids, metas):
            self.id2meta[int(i)] = m
        return ids

    def search(self, q, k=10):
        D1, I1 = self.hot.search(q, k)
        if self.warm:
            D2, I2 = self.warm.search(q, k)
            # Merge
            D = np.concatenate([D1, D2], axis=1)
            I = np.concatenate([I1, I2], axis=1)
            # Keep top-k per row
            outD, outI = [], []
            for drow, irow in zip(D, I):
                idx = np.argsort(-drow)[:k]
                outD.append(drow[idx])
                outI.append(irow[idx])
            return np.stack(outD), np.stack(outI)
        return D1, I1