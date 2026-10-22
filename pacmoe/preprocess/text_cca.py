"""
Text CCA (Section 2.2.2): FinBERT 768 -> 256 dimensions, fitted on the training split.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

# Implementation details not stated in the paper.
CCA_MAX_ITER = 500
CCA_TOL = 1e-6
PCA_RANDOM_STATE = 42
VARIANCE_EPS = 1e-8


def _clean(a) -> np.ndarray:
    a = a.detach().cpu().float().numpy() if isinstance(a, torch.Tensor) else np.asarray(a, dtype=np.float32)
    return np.nan_to_num(a.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def _standardize_fit(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Columns with variance > eps; their mean and std (std <= eps -> 1)."""
    mask = np.asarray(np.nanvar(a, axis=0) > VARIANCE_EPS, dtype=bool)
    sel = a[:, mask].astype(np.float32, copy=False)
    mean = sel.mean(axis=0).astype(np.float32)
    scale = sel.std(axis=0).astype(np.float32)
    return mask, mean, np.where(scale > VARIANCE_EPS, scale, 1.0).astype(np.float32)


class TextCCA:
    """fit(train pooled FinBERT (N, 768), train structured view (N, 57)); transform(any (M, 768))."""

    def __init__(self, output_dim: int = 256) -> None:
        self.output_dim = int(output_dim)

    def fit(self, x_view, structured_view) -> "TextCCA":
        """ for CCA plus for PCA."""
        from sklearn.cross_decomposition import CCA
        from sklearn.decomposition import PCA

        x, y = _clean(x_view), _clean(structured_view)
        ok = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x, y = x[ok], y[ok]
        self.x_mask, self.x_mean, self.x_scale = _standardize_fit(x)
        y_mask, y_mean, y_scale = _standardize_fit(y)
        xs = ((x[:, self.x_mask] - self.x_mean) / self.x_scale).astype(np.float32)
        ys = ((y[:, y_mask] - y_mean) / y_scale).astype(np.float32)
        n_cca = min(self.output_dim, x.shape[0] - 1, xs.shape[1], ys.shape[1])
        self.cca = CCA(n_components=int(n_cca), scale=False, max_iter=CCA_MAX_ITER, tol=CCA_TOL).fit(xs, ys) if n_cca > 0 else None
        n_pca = min(self.output_dim - int(n_cca), x.shape[0], xs.shape[1])
        self.pca = PCA(n_components=int(n_pca), random_state=PCA_RANDOM_STATE).fit(xs) if n_pca > 0 else None
        z = self._raw(x)
        self.out_mean = z.mean(axis=0).astype(np.float32)
        scale = z.std(axis=0).astype(np.float32)
        self.out_scale = np.where(scale > VARIANCE_EPS, scale, 1.0).astype(np.float32)
        return self

    def _raw(self, x: np.ndarray) -> np.ndarray:
        xs = ((x[:, self.x_mask].astype(np.float32, copy=False) - self.x_mean) / self.x_scale).astype(np.float32)
        parts = [np.asarray(m.transform(xs), dtype=np.float32) for m in (self.cca, self.pca) if m is not None]
        z = np.concatenate(parts, axis=1).astype(np.float32) if parts else np.zeros((x.shape[0], 0), dtype=np.float32)
        if z.shape[1] < self.output_dim:
            z = np.concatenate([z, np.zeros((z.shape[0], self.output_dim - z.shape[1]), dtype=np.float32)], axis=1)
        return np.nan_to_num(z[:, : self.output_dim].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def transform(self, x) -> torch.Tensor:
        """(M, 768) -> (M, 256)."""
        x = _clean(x)
        if x.ndim != 2 or x.shape[0] == 0:
            return torch.zeros((0, self.output_dim), dtype=torch.float32)
        return torch.tensor(((self._raw(x) - self.out_mean) / self.out_scale).astype(np.float32), dtype=torch.float32)

    def transform_sequences(self, sequences: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Per-sample sentence embeddings (L_i, 768) -> (L_i, 256)."""
        return [self.transform(s) for s in sequences]
