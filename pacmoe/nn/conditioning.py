"""
Small modules that condition the representations and the router (implementation details).
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from pacmoe import constants as K

logger = logging.getLogger(__name__)


class TickerEmbedding(nn.Module):
    """h + E[ticker_id]; E is zero-initialized."""

    def __init__(self, num_tickers: int, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_tickers, dim))

    def forward(self, h: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        """(B, D), (B,) -> (B, D)."""
        return h + F.embedding(ticker_ids.to(device=h.device, dtype=torch.long), self.weight).to(dtype=h.dtype)


class MissingModalityEmbedding(nn.Module):
    """Replace rows whose modality is unavailable with a learned vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        """(B, D), (B,) -> (B, D)."""
        mask = available.to(device=x.device, dtype=x.dtype).unsqueeze(-1) > 0.5
        return torch.where(mask, x, self.embedding.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand_as(x))


class PreNormMLPBlock(nn.Module):
    """x + FFN(LayerNorm(x))."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D) -> (B, D)."""
        return x + self.ffn(self.norm(x))


class _EMAReliability(nn.Module):
    """Reliability score sigmoid(1 - ||(h - mu) / sigma|| / sqrt(D)) with EMA mu, sigma^2."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("ema_mean", torch.zeros(dim))
        self.register_buffer("ema_var", torch.ones(dim))
        self.register_buffer("_initialized", torch.tensor(False))

    @torch.no_grad()
    def _update(self, h: torch.Tensor) -> None:
        """EMA update of the batch mean and variance (training only)."""
        h_f = h.detach().float().reshape(-1, self.dim)
        b_mean = h_f.mean(0)
        b_var = h_f.var(0, unbiased=False).clamp_min(1e-8)
        if not (torch.isfinite(b_mean).all() and torch.isfinite(b_var).all()):
            return
        if not self._initialized.item():
            self.ema_mean.copy_(b_mean)
            self.ema_var.copy_(b_var)
            self._initialized.fill_(True)
        else:
            d = K.QUALITY_EMA_DECAY
            self.ema_mean.mul_(d).add_(b_mean * (1.0 - d))
            self.ema_var.mul_(d).add_(b_var * (1.0 - d))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(B, D) -> (B,)."""
        if self.training:
            self._update(h)
        if not self._initialized.item():
            return torch.ones(h.shape[0], device=h.device, dtype=h.dtype)
        mean = self.ema_mean.to(device=h.device, dtype=torch.float32)
        std = self.ema_var.to(device=h.device, dtype=torch.float32).clamp_min(1e-8).sqrt()
        z = ((h.float() - mean) / std).norm(dim=-1) / (self.dim ** 0.5)
        return torch.sigmoid(-z + 1.0).to(dtype=h.dtype)


class _Consistency(nn.Module):
    """(cos(P h_price, T h_text) + 1) / 2."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        proj_dim = max(dim // 4, 16)
        self.price_proj = nn.Linear(dim, proj_dim, bias=False)
        self.text_proj = nn.Linear(dim, proj_dim, bias=False)

    def forward(self, h_price: torch.Tensor, h_text: torch.Tensor) -> torch.Tensor:
        """(B, D), (B, D) -> (B,)."""
        p = F.normalize(self.price_proj(h_price.float()), dim=-1)
        t = F.normalize(self.text_proj(h_text.to(device=h_price.device, dtype=h_price.dtype).float()), dim=-1)
        return (((p * t).sum(dim=-1) + 1.0) / 2.0).to(dtype=h_price.dtype)


class QualityGate(nn.Module):
    """quality_m = clamp(a * q_m + b * reliability_m + c * consistency), (a, b, c) = softmax(learned)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.price_cal = _EMAReliability(dim)
        self.text_cal = _EMAReliability(dim)
        self.graph_cal = _EMAReliability(dim)
        self.consistency = _Consistency(dim)
        w = K.QUALITY_CONSISTENCY_WEIGHT
        self.blend_logits = nn.Parameter(torch.tensor([(1.0 - w) / 2.0, (1.0 - w) / 2.0, w], dtype=torch.float32).log())

    def forward(self, h_price: torch.Tensor, h_text: torch.Tensor, h_graph: torch.Tensor, quality_base: torch.Tensor) -> torch.Tensor:
        """quality_base (B, 3) = availability of price/text/graph -> (B, 3)."""
        dtype = h_price.dtype
        h_t = h_text.to(h_price.device, dtype)
        h_g = h_graph.to(h_price.device, dtype)
        alpha, beta, gamma = torch.softmax(self.blend_logits.float(), dim=0)
        scores = [self.price_cal(h_price), self.text_cal(h_t), self.graph_cal(h_g)]
        consist = self.consistency(h_price, h_t)
        qb = quality_base.to(h_price.device, dtype)
        out = [(alpha * qb[:, i].clamp(0.0, 1.0) + beta * scores[i] + gamma * consist).clamp(0.0, 1.0) for i in range(3)]
        return torch.stack(out, dim=-1)


class GCCACondition(nn.Module):
    """Agreement (cosine) of price/text, price/graph and text/graph in a GCCA space estimated with
    EMA covariances; raw cosine of LayerNorm-ed vectors is used until the first solve."""

    _VIEWS = ("p", "t", "g")

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.d_model = dim
        self.k = min(K.GCCA_K, dim)
        self.norm = nn.LayerNorm(dim)
        for name in self._VIEWS:
            self.register_buffer(f"ema_mean_{name}", torch.zeros(dim))
            self.register_buffer(f"ema_cov_{name}", torch.eye(dim))
            self.register_buffer(f"proj_{name}", torch.zeros(dim, self.k))
        for pair in ("pt", "pg", "tg"):
            self.register_buffer(f"ema_cross_{pair}", torch.zeros(dim, dim))
        self.register_buffer("step_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("solved", torch.zeros((), dtype=torch.bool))

    @staticmethod
    def _cosines(zp: torch.Tensor, zt: torch.Tensor, zg: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [F.cosine_similarity(zp, zt, dim=-1, eps=1e-8), F.cosine_similarity(zp, zg, dim=-1, eps=1e-8), F.cosine_similarity(zt, zg, dim=-1, eps=1e-8)],
            dim=-1,
        )

    @torch.no_grad()
    def _update_ema(self, h_p: torch.Tensor, h_t: torch.Tensor, h_g: torch.Tensor) -> None:
        """EMA of means, covariances and cross-covariances."""
        decay = K.GCCA_EMA_DECAY
        views = {"p": h_p.detach().float(), "t": h_t.detach().float(), "g": h_g.detach().float()}
        if not all(torch.isfinite(x).all() for x in views.values()):
            return
        means = {n: x.mean(dim=0) for n, x in views.items()}
        centered = {n: x - means[n] for n, x in views.items()}
        for n in self._VIEWS:
            cov = centered[n].t() @ centered[n] / max(1, views[n].shape[0])
            getattr(self, f"ema_mean_{n}").mul_(decay).add_(means[n], alpha=1.0 - decay)
            getattr(self, f"ema_cov_{n}").mul_(decay).add_(cov, alpha=1.0 - decay)
        for pair, left, right in (("pt", "p", "t"), ("pg", "p", "g"), ("tg", "t", "g")):
            cross = centered[left].t() @ centered[right] / max(1, views[left].shape[0])
            getattr(self, f"ema_cross_{pair}").mul_(decay).add_(cross, alpha=1.0 - decay)

    @torch.no_grad()
    def _solve(self) -> None:
        """Block MAXVAR-GCCA eigenproblem on the EMA statistics."""
        with torch.autocast(device_type=self.ema_cov_p.device.type, enabled=False):
            eps = K.GCCA_EPS
            eye = torch.eye(self.d_model, device=self.ema_cov_p.device, dtype=torch.float32)
            whiteners = {}
            for n in self._VIEWS:
                cov = getattr(self, f"ema_cov_{n}").float() + eps * eye
                cov = 0.5 * (cov + cov.t())
                evals, evecs = torch.linalg.eigh(cov)
                whiteners[n] = evecs @ torch.diag(evals.clamp_min(eps).rsqrt()) @ evecs.t()
            d = self.d_model
            cross = torch.zeros(3 * d, 3 * d, device=eye.device, dtype=torch.float32)
            offsets = {"p": 0, "t": d, "g": 2 * d}
            for pair, left, right in (("pt", "p", "t"), ("pg", "p", "g"), ("tg", "t", "g")):
                value = getattr(self, f"ema_cross_{pair}").float()
                ls, rs = slice(offsets[left], offsets[left] + d), slice(offsets[right], offsets[right] + d)
                cross[ls, rs] = value
                cross[rs, ls] = value.t()
            block_w = torch.block_diag(whiteners["p"], whiteners["t"], whiteners["g"])
            maxvar = block_w @ cross @ block_w
            maxvar = 0.5 * (maxvar + maxvar.t())
            _, evecs = torch.linalg.eigh(maxvar)
            top = evecs[:, -self.k:]
            for n in self._VIEWS:
                getattr(self, f"proj_{n}").copy_(whiteners[n] @ top[offsets[n]: offsets[n] + d])
            self.solved.fill_(True)

    def forward(self, h_p: torch.Tensor, h_t: torch.Tensor, h_g: torch.Tensor) -> torch.Tensor:
        """(B, D) x 3 -> (B, 3)., plus the EMA update and solve in training."""
        h_t = h_t.to(device=h_p.device, dtype=h_p.dtype)
        h_g = h_g.to(device=h_p.device, dtype=h_p.dtype)
        if self.training:
            self._update_ema(h_p, h_t, h_g)
            self.step_count.add_(1)
            step = int(self.step_count.item())
            if step >= K.GCCA_WARMUP_STEPS and step % K.GCCA_REFRESH_EVERY == 0:
                try:
                    self._solve()
                except Exception as exc:  # noqa: BLE001 - an ill-conditioned solve falls back to raw cosine
                    self.solved.fill_(False)
                    logger.warning("GCCA solve failed, falling back to raw cosine: %s", exc)
        if not bool(self.solved.item()):
            return self._cosines(self.norm(h_p), self.norm(h_t), self.norm(h_g))
        with torch.no_grad():
            project = lambda h, n: (h - getattr(self, f"ema_mean_{n}").to(h)) @ getattr(self, f"proj_{n}").to(h)  # noqa: E731
            return self._cosines(project(h_p, "p"), project(h_t, "t"), project(h_g, "g"))
