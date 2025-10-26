# memory/policy.py
import time, numpy as np

class StreamingMemoryPolicy:
    def __init__(self, cfg, d):
        self.cfg = cfg
        self.d = d
        self.centroid = np.zeros((d,), dtype=np.float32)
        self.count = 0
        self.items = {}  # id -> dict(score, recency, usage, novelty, ts)
        self.budget_hot = cfg.budget_hot
        self.budget_warm = cfg.budget_warm

    def _novelty(self, x):
        if self.count == 0:
            return 1.0
        c = self.centroid / (np.linalg.norm(self.centroid) + 1e-9)
        return float(1.0 - np.dot(x, c))

    def _update_centroid(self, x):
        self.centroid = (self.centroid * self.count + x) / (self.count + 1)
        self.count += 1

    def score(self, item_id, x, usage=0, ts=None, coverage_gain=0.0):
        now = time.time()
        ts = ts or now
        recency = np.exp(-(now - ts) / self.cfg.decay_tau_s)
        novelty = self._novelty(x)
        s = (self.cfg.w_recency * recency +
             self.cfg.w_usage * np.sqrt(usage + 1) +
             self.cfg.w_novelty * novelty +
             self.cfg.w_coverage * coverage_gain)
        self.items[item_id] = dict(score=s, recency=recency, usage=usage, novelty=novelty, ts=ts)
        self._update_centroid(x)
        return s

    def bump_usage(self, item_id, x=None):
        if item_id not in self.items:
            if x is None: return
            return self.score(item_id, x, usage=1)
        it = self.items[item_id]
        it["usage"] += 1
        # recompute score with updated usage
        s = (self.cfg.w_recency * it["recency"] +
             self.cfg.w_usage * np.sqrt(it["usage"] + 1) +
             self.cfg.w_novelty * it["novelty"] +
             self.cfg.w_coverage * 0.0)
        it["score"] = s
        self.items[item_id] = it
        return s

    def select_hot(self, hot_ids):
        # Choose top by score under budget
        scored = sorted(((i, self.items.get(i, {}).get("score", 0.0)) for i in hot_ids),
                        key=lambda t: t[1], reverse=True)
        keep = [i for i,_ in scored[:self.budget_hot]]
        spill = [i for i,_ in scored[self.budget_hot:]]
        return keep, spill