# index/hnsw.py
import faiss, numpy as np

class HNSWIndex:
    def __init__(self, dim, m=64, efc=200, efs=128, metric=faiss.METRIC_INNER_PRODUCT):
        self.dim = dim
        base = faiss.IndexHNSWFlat(dim, m, metric)
        base.hnsw.efConstruction = efc
        base.hnsw.efSearch = efs
        self.index = faiss.IndexIDMap2(base)

    @property
    def ntotal(self):
        return self.index.ntotal

    def add(self, vecs, ids):
        vecs = vecs.astype(np.float32)
        faiss.normalize_L2(vecs)
        ids = ids.astype(np.int64)
        self.index.add_with_ids(vecs, ids)

    def remove(self, ids: np.ndarray):
        try:
            ids = np.asarray(ids, dtype=np.int64)
            sel = faiss.IDSelectorArray(ids.size, faiss.swig_ptr(ids))
            removed = self.index.remove_ids(sel)
            return int(removed)
        except Exception:
            return 0  # fallback: no removal

    def search(self, q, k=10):
        q = q.astype(np.float32)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        return D, I

    def save(self, path):
        faiss.write_index(self.index, str(path))

    @staticmethod
    def load(path):
        obj = HNSWIndex.__new__(HNSWIndex)
        obj.index = faiss.read_index(str(path))
        obj.dim = obj.index.d
        return obj