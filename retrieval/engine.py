# retrieval/engine.py
import time
import numpy as np

class RetrievalEngine:
    def __init__(self, cfg, index_mgr, ingestor):
        self.cfg = cfg
        self.idx = index_mgr
        self.ingestor = ingestor

    def embed_query(self, query, modality="text"):
        if modality == "text":
            v = self.ingestor.txt.embed([query])[0]
        elif modality == "image":
            v = self.ingestor.img.embed_image([query])[0]
        elif modality == "audio":
            # Preferred: CLAP audio -> CLAP->CLIP bridge -> CLIP space
            v = self.ingestor.aud.embed([query])[0]
            if "clap2clip" in self.ingestor.align:
                v = self.ingestor.align["clap2clip"].align(v)
        elif modality == "video":
            v = self.ingestor.vid.embed([query])[0]
        else:
            raise ValueError(f"Unknown modality: {modality}")
        return v.astype(np.float32)

    @staticmethod
    def _margin_top2(sims_row):
        if sims_row.size < 2: return 0.0
        s = np.sort(sims_row)[::-1]
        return float(s[0] - s[1])

    def search_with_telemetry(self, query_vec, k=10, modality="text"):
        t0 = time.perf_counter() # Use perf_counter for more accurate timing
        D, I = self.idx.search(query_vec[None, :], k=max(k, 50))
        lat_ms = (time.perf_counter() - t0) * 1000.0

        sims = D[0]
        margin = self._margin_top2(sims)
        eps = self.ingestor.alignment_eps_for_modality(modality)
        zeta = self.idx.zeta_estimate()
        safe = margin > 2.0 * (eps + zeta)

        self.idx.mark_access([int(i) for i in I[0][:k] if i >= 0], self.ingestor.policy)
        tel = {"margin_top2": margin, "eps_align": eps, "zeta_pq": zeta, "safe_top1": bool(safe), "lat_ms": lat_ms}
        return D[0][:k], I[0][:k], tel