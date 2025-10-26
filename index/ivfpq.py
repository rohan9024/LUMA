import faiss, numpy as np

class IVFPQIndex:
    def __init__(self, dim, nlist=4096, m=16, nbits=8, metric=faiss.METRIC_INNER_PRODUCT):
        self.dim = int(dim)
        self.metric = metric
        self.target_nlist = int(nlist)
        self.m = int(m)
        self.nbits = int(nbits)
        self.index = None
        self.trained = False
        self._build_index(self.target_nlist)

    def _build_index(self, nlist):
        quant = faiss.IndexHNSWFlat(self.dim, 32, self.metric)
        base = faiss.IndexIVFPQ(quant, self.dim, int(nlist), self.m, self.nbits, self.metric)
        self.index = faiss.IndexIDMap2(base)
        self.trained = False

    def _choose_nlist(self, n_train: int) -> int:
        if n_train <= 0:
            return min(32, self.target_nlist)
        nlist = int(max(32, 4 * np.sqrt(n_train)))
        nlist = min(nlist, int(n_train))
        nlist = min(nlist, self.target_nlist)
        return max(8, nlist)

    def is_trained(self):
        return self.trained or (self.index is not None and self.index.is_trained)

    def train(self, samples):
        samples = samples.astype(np.float32)
        faiss.normalize_L2(samples)
        n_train = samples.shape[0]
        nlist = self._choose_nlist(n_train)
        inner = self.index.index
        cur_nlist = getattr(inner, "nlist", None)
        if cur_nlist is None or cur_nlist > nlist:
            self._build_index(nlist)
        self.index.train(samples)
        self.trained = True

    def add(self, vecs, ids):
        if not self.is_trained():
            raise RuntimeError("Train IVFPQ before add.")
        vecs = vecs.astype(np.float32)
        faiss.normalize_L2(vecs)
        ids = ids.astype(np.int64)
        self.index.add_with_ids(vecs, ids)

    def search(self, q, k=10):
        q = q.astype(np.float32)
        faiss.normalize_L2(q)
        return self.index.search(q, k)

    def reconstruct(self, id_: int):
        return self.index.reconstruct(int(id_)).astype(np.float32)

    def save(self, path):
        faiss.write_index(self.index, str(path))