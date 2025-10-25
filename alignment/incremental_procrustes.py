# luma/alignment/incremental_procrustes.py
import numpy as np
from numpy.linalg import svd, norm

class IncrementalProcrustes:
    def __init__(self, d=768, ema=0.999):
        self.d = d
        self.S = np.zeros((d, d), dtype=np.float32)  # cross-covariance
        self.T = np.eye(d, dtype=np.float32)         # alignment
        self.ema = ema                                # decay for non-stationary streams
        self.n = 0

    def update_pair(self, x, y):
        # x, y are L2-normalized d-dim vectors from modality m and reference
        self.S = self.ema * self.S + (1 - self.ema) * np.outer(x, y)
        self.n += 1

    def refresh(self):
        # Compute T = argmin ||X T - Y||_F with T orthogonal => T = UV^T
        U, _, Vt = svd(self.S, full_matrices=False)
        self.T = (U @ Vt).astype(np.float32)
        # Stabilize: ensure det ~ +1 (proper rotation). If negative, flip.
        if np.linalg.det(self.T) < 0:
            U[:, -1] *= -1
            self.T = (U @ Vt).astype(np.float32)
        return self.T

    def align(self, x):
        return (x @ self.T).astype(np.float32)

    def alignment_error(self):
        # ||I - T||_2 proxy via spectral norm (upper bound using Frobenius)
        return min(norm(np.eye(self.d, dtype=np.float32) - self.T, ord=2),
                   norm(np.eye(self.d, dtype=np.float32) - self.T, 'fro'))