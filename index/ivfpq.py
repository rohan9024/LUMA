# luma/index/ivfpq.py
import faiss, numpy as np

class IVFPQIndex:
    def __init__(self, dim, nlist=4096, m=16, nbits=8, metric=faiss.METRIC_INNER_PRODUCT):
        self.quantizer = faiss.IndexHNSWFlat(dim, 32, metric)
        self.index = faiss.IndexIVFPQ(self.quantizer, dim, nlist, m, nbits, metric)
        self.trained = False

    def train(self, samples):
        samples = samples.astype(np.float32)
        faiss.normalize_L2(samples)
        self.index.train(samples)
        self.trained = True

    def add(self, vecs):
        if not self.trained:
            raise RuntimeError("Train IVFPQ before add.")
        vecs = vecs.astype(np.float32)
        faiss.normalize_L2(vecs)
        self.index.add(vecs)

    def search(self, q, k=10):
        q = q.astype(np.float32)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        return D, I

    def save(self, path):
        faiss.write_index(self.index, str(path))