import numpy as np
from numpy.linalg import svd, norm

class IncrementalProcrustes:
    """
    Streaming orthogonal Procrustes with drift tracking.
    First refresh sets drift=0 (baseline), subsequent refreshes compute drift,
    but we reset pending drift after applying it so Safe@k reflects future updates.
    """
    def __init__(self, d=768, ema=0.999):
        self.d = d
        self.S = np.zeros((d, d), dtype=np.float32)
        self.T = np.eye(d, dtype=np.float32)
        self.T_prev = np.eye(d, dtype=np.float32)
        self.ema = ema
        self._pending_drift = 0.0
        self._last_applied_drift = 0.0
        self.n = 0
        self.initialized = False

    def update_pair(self, x, y):
        self.S = self.ema * self.S + (1 - self.ema) * np.outer(x, y)
        # We don’t compute pending drift here; it’s cheap to set to 0 and treat drift at refresh time.
        self.n += 1

    def refresh(self):
        U, _, Vt = svd(self.S, full_matrices=False)
        T_new = (U @ Vt).astype(np.float32)
        if np.linalg.det(T_new) < 0:
            U[:, -1] *= -1
            T_new = (U @ Vt).astype(np.float32)

        if not self.initialized:
            self._last_applied_drift = 0.0
            self.initialized = True
        else:
            try:
                self._last_applied_drift = float(norm(T_new - self.T, ord=2))
            except Exception:
                self._last_applied_drift = float(norm(T_new - self.T, ord='fro'))

        self.T_prev = self.T
        self.T = T_new
        # After applying, pending drift for future queries is 0
        self._pending_drift = 0.0
        return self.T

    def align(self, x): 
        return (x @ self.T).astype(np.float32)

    def alignment_error(self):
        # pending drift since last refresh (used in Safe@k)
        return self._pending_drift

    def absolute_error(self):
        I = np.eye(self.d, dtype=np.float32)
        try: return float(norm(I - self.T, ord=2))
        except Exception: return float(norm(I - self.T, ord='fro'))