"""
Residual CCA adapters between hidden representations (implementation detail of the encoders).

price_tech_cca: aligns the OHLCV representation with the technical-indicator representation.
graph_view_cca: aligns the graph representation with its own views.

The CCA directions are estimated without gradients and applied as a residual:
    out = LayerNorm(anchor + w * delta),  delta = 0.5 * (proj_x(x) + proj_y_in_x(y)).
stat_mode = "batch":   the statistics are those of the current batch (setup "main").
stat_mode = "running": an EMA of past training batches is used; the current batch is added
                       only after it has been transformed, and evaluation never updates it
                       (setups "ablation" and "walk_forward").
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from pacmoe import constants as K

Solution = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]


class _CCARunningStats(nn.Module):
    """EMA of the first and second moments per cache key (stored as extra_state)."""

    def __init__(self, momentum: float) -> None:
        super().__init__()
        self.momentum = float(momentum)
        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.counts: Dict[str, int] = {}

    def update_count(self, key: str) -> int:
        return int(self.counts.get(str(key), 0))

    @torch.no_grad()
    def update(self, key: str, x: torch.Tensor, y: torch.Tensor) -> None:
        """Blend the batch moments into the EMA."""
        key = str(key)
        x64 = torch.nan_to_num(x.detach().double(), nan=0.0, posinf=0.0, neginf=0.0)
        y64 = torch.nan_to_num(y.detach().double(), nan=0.0, posinf=0.0, neginf=0.0)
        n = float(x64.shape[0])
        new = {
            "x_mean": x64.mean(dim=0, keepdim=True),
            "y_mean": y64.mean(dim=0, keepdim=True),
            "xx": (x64.T @ x64) / n,
            "yy": (y64.T @ y64) / n,
            "xy": (x64.T @ y64) / n,
        }
        count = self.update_count(key)
        old = self.stats.get(key)
        if old is None:
            self.stats[key] = new
        else:
            rate = max(self.momentum, 1.0 / float(count + 1))
            self.stats[key] = {n_: old[n_].to(device=v.device) * (1.0 - rate) + v * rate for n_, v in new.items()}
        self.counts[key] = count + 1

    def covariances(self, key: str, device: torch.device):
        """Means and covariances (float32) of a key, or None."""
        stat = self.stats.get(str(key))
        if stat is None:
            return None
        x_mean, y_mean = stat["x_mean"].to(device=device), stat["y_mean"].to(device=device)
        cxx = stat["xx"].to(device=device) - x_mean.T @ x_mean
        cyy = stat["yy"].to(device=device) - y_mean.T @ y_mean
        cxy = stat["xy"].to(device=device) - x_mean.T @ y_mean
        return x_mean.float(), y_mean.float(), cxx.float(), cyy.float(), cxy.float()

    def get_extra_state(self):
        return {
            "counts": dict(self.counts),
            "stats": {k: {n: v.detach().cpu().clone() for n, v in s.items()} for k, s in self.stats.items()},
        }

    def set_extra_state(self, state) -> None:
        if not isinstance(state, dict):
            return
        self.counts = {str(k): int(v) for k, v in dict(state.get("counts", {})).items()}
        self.stats = {str(k): {str(n): v.clone() for n, v in dict(s).items()} for k, s in dict(state.get("stats", {})).items()}


class BatchCCAAdapter(nn.Module):
    """Two-view CCA residual adapter."""

    def __init__(self, dim: int, residual_weight: float, stat_mode: str, input_norm: bool, output_norm: bool) -> None:
        super().__init__()
        self.dim = int(dim)
        self.residual_weight = float(residual_weight)
        self.ridge = K.CCA_RIDGE
        self.min_batch = K.CCA_MIN_BATCH
        self.stat_mode = stat_mode
        self.stat_warmup_steps = K.CCA_STAT_WARMUP_STEPS
        self.input_norm_x = nn.LayerNorm(self.dim) if input_norm else nn.Identity()
        self.input_norm_y = nn.LayerNorm(self.dim) if input_norm else nn.Identity()
        self.output_norm = nn.LayerNorm(self.dim) if output_norm else nn.Identity()
        if stat_mode == "running":
            self.running_stats = _CCARunningStats(momentum=K.CCA_STAT_MOMENTUM)

    def _solve(self, x_mean, y_mean, cxx, cyy, cxy, sample_rank_limit: Optional[int], device, dtype) -> Optional[Solution]:
        """Whitened SVD of the cross-covariance."""
        with torch.no_grad():
            try:
                eval_x, evec_x = torch.linalg.eigh(cxx)
                eval_y, evec_y = torch.linalg.eigh(cyy)
                invsqrt_x = evec_x @ torch.diag(torch.rsqrt(eval_x.clamp_min(self.ridge))) @ evec_x.T
                invsqrt_y = evec_y @ torch.diag(torch.rsqrt(eval_y.clamp_min(self.ridge))) @ evec_y.T
                u, _s, vh = torch.linalg.svd(invsqrt_x @ cxy @ invsqrt_y, full_matrices=False)
            except RuntimeError:
                return None
            rank_limit = min(int(cxx.shape[0]), u.shape[1], vh.shape[0])
            if sample_rank_limit is not None:
                rank_limit = min(int(sample_rank_limit), rank_limit)
            components = min(self.dim, max(0, int(rank_limit)))
            if components <= 0:
                return None
            wx = (invsqrt_x @ u[:, :components]).to(device=device, dtype=dtype)
            wy = (invsqrt_y @ vh.T[:, :components]).to(device=device, dtype=dtype)
        return x_mean.to(device=device, dtype=dtype), y_mean.to(device=device, dtype=dtype), wx, wy, components

    def _fit_batch(self, x: torch.Tensor, y: torch.Tensor) -> Optional[Solution]:
        """CCA from the statistics of the current batch."""
        batch, dim = x.shape
        with torch.no_grad():
            x_fit = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            y_fit = torch.nan_to_num(y.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            x_mean, y_mean = x_fit.mean(dim=0, keepdim=True), y_fit.mean(dim=0, keepdim=True)
            x_c, y_c = x_fit - x_mean, y_fit - y_mean
            denom = float(max(batch - 1, 1))
            eye = torch.eye(dim, device=x.device, dtype=torch.float32)
            cxx = (x_c.T @ x_c) / denom + self.ridge * eye
            cyy = (y_c.T @ y_c) / denom + self.ridge * eye
            cxy = (x_c.T @ y_c) / denom
        return self._solve(x_mean, y_mean, cxx, cyy, cxy, batch - 1, x.device, x.dtype)

    @staticmethod
    def _apply_solution(x: torch.Tensor, y: torch.Tensor, solution: Solution) -> torch.Tensor:
        """delta = 0.5 * (P_x x + P_x^T-mapped y)."""
        x_mean, y_mean, wx, wy, _ = solution
        zx = (x - x_mean) @ wx
        zy = (y - y_mean) @ wy
        return 0.5 * (zx @ wx.T + zy @ wx.T)

    def cca_delta(self, x: torch.Tensor, y: torch.Tensor, cache_key: str = "default") -> Tuple[torch.Tensor, bool]:
        """Return (delta, enabled)."""
        if self.stat_mode == "running":
            return self._running_delta(x, y, cache_key)
        if x.shape[0] < self.min_batch or y.shape != x.shape:
            return torch.zeros_like(x), False
        solution = self._fit_batch(x, y)
        if solution is None:
            return torch.zeros_like(x), False
        return self._apply_solution(x, y, solution), True

    def _running_delta(self, x: torch.Tensor, y: torch.Tensor, key: str) -> Tuple[torch.Tensor, bool]:
        """Use the EMA of previous batches; update after use (training only)."""
        stats = self.running_stats
        solution = None
        if stats.update_count(key) >= self.stat_warmup_steps:
            cov = stats.covariances(key, device=x.device)
            if cov is not None:
                x_mean, y_mean, cxx, cyy, cxy = cov
                eye = torch.eye(x.shape[1], device=x.device, dtype=torch.float32)
                solution = self._solve(x_mean, y_mean, cxx + self.ridge * eye, cyy + self.ridge * eye, cxy, None, x.device, x.dtype)
        delta = torch.zeros_like(x) if solution is None else self._apply_solution(x, y, solution)
        if self.training and x.shape[0] >= self.min_batch:
            stats.update(key, x, y)
        return delta, solution is not None

    def forward(self, x: torch.Tensor, y: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """(B, D), (B, D), (B, D) -> (B, D)."""
        delta, _ = self.cca_delta(self.input_norm_x(x), self.input_norm_y(y))
        return self.output_norm(anchor + self.residual_weight * delta)


class MultiViewCCAAdapter(nn.Module):
    """
    Anchor-to-view CCA adapter; the deltas of all views are averaged.

    In setup "main" the graph representation is passed twice as a view of itself (the views are
    copies made by nan_to_num, so they are not skipped). exclude_anchor_views=True (setups
    "ablation" and "walk_forward") skips such self-views.
    """

    def __init__(self, dim: int, residual_weight: float, stat_mode: str, exclude_anchor_views: bool) -> None:
        super().__init__()
        self.dim = int(dim)
        self.residual_weight = float(residual_weight)
        self.exclude_anchor_views = bool(exclude_anchor_views)
        self.input_norm = nn.LayerNorm(self.dim)
        self.pair_adapter = BatchCCAAdapter(dim, 1.0, stat_mode, input_norm=False, output_norm=False)
        self.output_norm = nn.LayerNorm(self.dim)

    def forward(self, views: Iterable[Optional[torch.Tensor]], anchor: torch.Tensor) -> torch.Tensor:
        """views: list of (B, D) or None; anchor (B, D) -> (B, D)."""
        batch = int(anchor.shape[0])
        valid: List[torch.Tensor] = []
        alias: List[bool] = []
        for value in views:
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape != (batch, self.dim):
                continue
            valid.append(torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0).to(device=anchor.device, dtype=anchor.dtype))
            alias.append(self.exclude_anchor_views and (value is anchor or (value.data_ptr() == anchor.data_ptr() and value.shape == anchor.shape)))
        if len(valid) < 2:
            return anchor
        anchor_cca = self.input_norm(anchor)
        deltas = []
        for index, view in enumerate(valid):
            if alias[index] or view.data_ptr() == anchor.data_ptr():
                continue
            delta, enabled = self.pair_adapter.cca_delta(anchor_cca, self.input_norm(view), cache_key=f"view_{index}")
            if enabled:
                deltas.append(delta)
        if not deltas:
            return anchor
        return self.output_norm(anchor + self.residual_weight * torch.stack(deltas, dim=0).mean(dim=0))
