# eval/metrics.py
import numpy as np
from dataclasses import dataclass, field

def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    gains = (2**rels - 1)
    discounts = 1.0 / np.log2(np.arange(2, 2 + len(rels)))
    return float(np.sum(gains * discounts))

def ndcg_at_k(rels, k):
    ideal = np.sort(rels)[::-1]
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(rels, k) / idcg

def mrr(rels):
    # rels is a list where rels[i] is 1 if ith ranked item is relevant else 0
    for i, r in enumerate(rels, 1):
        if r > 0:
            return 1.0 / i
    return 0.0

def recall_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    return float((rels > 0).any())

@dataclass
class Meter:
    ks: tuple = (1, 5, 10)
    ndcg: dict = field(default_factory=dict)
    rec: dict = field(default_factory=dict)
    mrrs: list = field(default_factory=list)
    safe1: list = field(default_factory=list)

    def __post_init__(self):
        for k in self.ks:
            self.ndcg[k] = []
            self.rec[k] = []

    def add(self, rels, margin_top2=None, eps=0.0, zeta=0.0):
        for k in self.ks:
            self.ndcg[k].append(ndcg_at_k(rels, k))
            self.rec[k].append(recall_at_k(rels, k))
        self.mrrs.append(mrr(rels))
        if margin_top2 is not None:
            self.safe1.append(1.0 if margin_top2 > 2.0 * (eps + zeta) else 0.0)

    def summary(self):
        out = {}
        for k in self.ks:
            out[f"nDCG@{k}"] = float(np.mean(self.ndcg[k])) if self.ndcg[k] else 0.0
            out[f"Recall@{k}"] = float(np.mean(self.rec[k])) if self.rec[k] else 0.0
        out["MRR"] = float(np.mean(self.mrrs)) if self.mrrs else 0.0
        if self.safe1:
            out["Safe@1"] = float(np.mean(self.safe1))
        return out