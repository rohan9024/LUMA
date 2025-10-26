import numpy as np
from typing import Dict, List
from .hnsw import HNSWIndex
from .ivfpq import IVFPQIndex

class MultiTierIndex:
    def __init__(self, dim, cfg):
        self.dim = dim
        self.cfg = cfg
        self.hot = HNSWIndex(dim, m=cfg.hnsw_m, efc=cfg.hnsw_efc, efs=cfg.hnsw_efs)
        self.warm = IVFPQIndex(dim, nlist=cfg.ivf_nlist, m=cfg.pq_m, nbits=cfg.pq_bits) if cfg.use_ivfpq else None

        self.id2meta: Dict[int, dict] = {}
        self.id2vec: Dict[int, np.ndarray] = {}
        self.id2tier: Dict[int, str] = {}
        self.train_buf: List[np.ndarray] = []
        self.warm_zeta_sum = 0.0
        self.warm_zeta_cnt = 0
        self.next_id = 0

    def _assign_ids(self, n): 
        ids = np.arange(self.next_id, self.next_id + n, dtype=np.int64); self.next_id += n; return ids

    def train_warm_if_needed(self):
        if not self.warm or self.warm.is_trained(): return
        if len(self.train_buf) < 512: return
        samples = np.vstack(self.train_buf[: min(len(self.train_buf), 50000)])
        self.warm.train(samples)

    def add(self, vecs, metas, hot=True):
        vecs = vecs.astype(np.float32)
        ids = self._assign_ids(len(vecs))
        self.hot.add(vecs, ids)
        for i, v, m in zip(ids, vecs, metas):
            self.id2meta[int(i)] = m
            self.id2vec[int(i)] = v
            self.id2tier[int(i)] = "hot"
            self.train_buf.append(v)
        if len(self.train_buf) > 100_000: self.train_buf = self.train_buf[-100_000:]
        return ids

    def _rebuild_hot(self, keep_ids: List[int]):
        new_hot = HNSWIndex(self.dim, m=self.cfg.hnsw_m, efc=self.cfg.hnsw_efc, efs=self.cfg.hnsw_efs)
        if keep_ids:
            vecs = np.vstack([self.id2vec[i] for i in keep_ids if i in self.id2vec])
            ids = np.asarray(keep_ids, dtype=np.int64)
            new_hot.add(vecs, ids)
        self.hot = new_hot

    def spill_to_warm(self, spill_ids: List[int], keep_ids: List[int] | None = None):
        if not self.warm or not spill_ids: return
        self.train_warm_if_needed()
        if not self.warm.is_trained():
            pool = [self.id2vec[i] for i in spill_ids if i in self.id2vec]
            if self.train_buf: pool += self.train_buf[: min(50000, len(self.train_buf))]
            samples = np.vstack(pool) if len(pool) > 0 else np.vstack(self.train_buf[:2048])
            self.warm.train(samples)

        vecs = np.vstack([self.id2vec[i] for i in spill_ids if i in self.id2vec])
        ids_arr = np.asarray(spill_ids, dtype=np.int64)
        if len(ids_arr) > 0: self.warm.add(vecs, ids_arr)

        removed = 0
        try: removed = self.hot.remove(ids_arr)
        except Exception: removed = 0
        if keep_ids is not None and removed < len(ids_arr):
            self._rebuild_hot(keep_ids)

        for i in spill_ids:
            try:
                orig = self.id2vec[i]; rec = self.warm.reconstruct(i)
                o = orig / (np.linalg.norm(orig) + 1e-9); r = rec / (np.linalg.norm(rec) + 1e-9)
                self.warm_zeta_sum += float(np.linalg.norm(o - r)); self.warm_zeta_cnt += 1
            except Exception: pass
            self.id2tier[i] = "warm"

    def rebalance(self, policy):
        hot_ids = [i for i, t in self.id2tier.items() if t == "hot"]
        if not hot_ids: return
        keep, spill = policy.select_hot(hot_ids)
        self.spill_to_warm(spill, keep_ids=keep)

    def search(self, q, k=10):
        D1, I1 = self.hot.search(q, k=max(k, 50))
        if self.warm and self.warm.is_trained():
            D2, I2 = self.warm.search(q, k=max(k, 50))
            merged: Dict[int, float] = {}
            for d, i in zip(D1[0], I1[0]):
                if i < 0: continue
                merged[int(i)] = max(merged.get(int(i), -1e9), float(d))
            for d, i in zip(D2[0], I2[0]):
                if i < 0: continue
                merged[int(i)] = max(merged.get(int(i), -1e9), float(d))
            items = sorted(merged.items(), key=lambda t: -t[1])[:k]
            if not items:
                return np.zeros((1, k), dtype=np.float32), -np.ones((1, k), dtype=np.int64)
            Ds = np.array([[s for _, s in items]], dtype=np.float32)
            Is = np.array([[i for i, _ in items]], dtype=np.int64)
            return Ds, Is
        return D1[:, :k], I1[:, :k]

    def mark_access(self, ids: List[int], policy):
        for i in ids:
            v = self.id2vec.get(int(i))
            if v is not None:
                policy.bump_usage(int(i), v)

    def zeta_estimate(self) -> float:
        if self.warm_zeta_cnt == 0: return 0.0
        return float(self.warm_zeta_sum / self.warm_zeta_cnt)