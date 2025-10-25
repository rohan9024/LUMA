# luma/index/hnsw.py
import faiss, numpy as np

class HNSWIndex:
    def __init__(self, dim, m=64, efc=200, efs=128, metric=faiss.METRIC_INNER_PRODUCT):
        self.dim = dim
        self.index = faiss.IndexHNSWFlat(dim, m, metric)
        self.index.hnsw.efConstruction = efc
        self.index.hnsw.efSearch = efs
        self.ntotal = 0

    def add(self, vecs):
        vecs = vecs.astype(np.float32)
        faiss.normalize_L2(vecs)  # ensure normalized for dot-product
        self.index.add(vecs)
        ids = np.arange(self.ntotal, self.ntotal + vecs.shape[0], dtype=np.int64)
        self.ntotal += vecs.shape[0]
        return ids

    def search(self, q, k=10):
        q = q.astype(np.float32)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        return D, I

    def save(self, path):
        faiss.write_index(self.index, str(path))

    @staticmethod
    def load(path):
        idx = HNSWIndex.__new__(HNSWIndex)
        idx.index = faiss.read_index(str(path))
        idx.dim = idx.index.d
        idx.ntotal = idx.index.ntotal
        return idx