"""Pairwise cross-modal attention experts and the MoE router (Sections 2.4-2.5).

Experts: Price-Text, Price-Graph, Text-Graph. Each expert attends from the anchor to the context
and from the context to the anchor (each with a "difference" token), blends the two directions
with a learned gate, and adds the result to the mean of the two inputs.

Router: logits l = W_g u + b_g over u = [h_P; h_T; h_G; quality; h_P; agreement; regime],
gating weights g_k = softmax(l / tau)_k with tau = ROUTER_TEMPERATURE / ROUTER_LOGIT_SCALE = 1.8
(Eq. 14). An expert whose modality is unavailable for a sample (e.g. no tweet) is masked out.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pacmoe import constants as K

EXPERT_NAMES = ("price_text", "price_graph", "text_graph")


class PairwiseExpert(nn.Module):
    """Bidirectional cross-modal attention between an anchor and a context summary vector."""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.attn_anchor_to_context = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_context_to_anchor = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.delta_token_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.delta_proj_fwd = nn.Linear(d_model, d_model)
        self.delta_proj_bwd = nn.Linear(d_model, d_model)
        self.blend_gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(K.EXPERT_RESIDUAL_SCALE, dtype=torch.float32))
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

    def forward(self, anchor: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """(B, D), (B, D) -> (B, D). Q = anchor, K/V = [context, delta] (and the reverse)."""
        delta_ac = self.delta_token_proj(context - anchor)
        delta_ca = self.delta_token_proj(anchor - context)
        kv_fwd = torch.stack([context, delta_ac], dim=1)
        fwd, _ = self.attn_anchor_to_context(anchor.unsqueeze(1), kv_fwd, kv_fwd, need_weights=False)
        fwd = self.delta_proj_fwd(self.dropout(fwd.squeeze(1)))
        kv_bwd = torch.stack([anchor, delta_ca], dim=1)
        bwd, _ = self.attn_context_to_anchor(context.unsqueeze(1), kv_bwd, kv_bwd, need_weights=False)
        bwd = self.delta_proj_bwd(self.dropout(bwd.squeeze(1)))
        gate = self.blend_gate(torch.cat([fwd, bwd], dim=-1))
        blended = gate * fwd + (1.0 - gate) * bwd
        scale = self.residual_scale.to(device=anchor.device, dtype=anchor.dtype)
        return self.out(self.norm(0.5 * (anchor + context) + scale * blended))


class PairwiseExpertMoE(nn.Module):
    """Three pairwise experts + MoE router (softmax gate with adaptive top-k)."""

    def __init__(self, d_model: int, nhead: int, dropout: float, disabled_experts=(), uniform_router: bool = False) -> None:
        super().__init__()
        self.price_text_expert = PairwiseExpert(d_model, nhead, dropout)
        self.price_graph_expert = PairwiseExpert(d_model, nhead, dropout)
        self.text_graph_expert = PairwiseExpert(d_model, nhead, dropout)
        router_in = 3 * d_model + 3 + d_model + 3 + K.ROUTER_REGIME_DIM
        self.router = nn.Sequential(nn.Linear(router_in, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, len(EXPERT_NAMES)))
        self.out_norm = nn.LayerNorm(d_model)
        disabled = torch.tensor([1.0 if name in set(disabled_experts) else 0.0 for name in EXPERT_NAMES])
        self.register_buffer("disabled_mask", disabled, persistent=False)
        self.uniform_router = bool(uniform_router)

    # -------------------------- router weights --------------------------
    def _apply_disabled(self, logits: torch.Tensor) -> torch.Tensor:
        disabled = self.disabled_mask.to(device=logits.device) > 0.5
        if bool(disabled.any().item()):
            logits = logits.masked_fill(disabled.unsqueeze(0), -1e4)
        return logits

    def _topk_softmax(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """softmax over the top-k logits only."""
        k = min(max(1, int(k)), logits.shape[-1])
        if k >= logits.shape[-1]:
            return F.softmax(logits / K.ROUTER_TEMPERATURE, dim=-1)
        top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
        sparse = torch.full_like(logits, -1e4).scatter_(1, top_idx, top_vals)
        return F.softmax(sparse / K.ROUTER_TEMPERATURE, dim=-1)

    def router_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """Gating weights actually used for fusion.

        The trained implementation had an adaptive top-k rule: if the gap between the two largest
        dense weights is >= ADAPTIVE_TOPK_MARGIN_HIGH (0.3) only the top expert is used, if it is
        >= ADAPTIVE_TOPK_MARGIN (0.1) the top two, otherwise all three. The selected weights are
        renormalized by a softmax over the kept logits.
        """
        if self.uniform_router:
            active = (self.disabled_mask.to(device=logits.device, dtype=logits.dtype) <= 0.5).to(logits.dtype)
            if float(active.sum().item()) <= 0.0:
                active = torch.ones_like(active)
            return (active / active.sum()).unsqueeze(0).expand(logits.shape[0], -1).contiguous()
        dense = F.softmax(logits / K.ROUTER_TEMPERATURE, dim=-1)
        top2 = torch.topk(dense, k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        level = torch.full_like(margin, 2.0)
        level = torch.where(margin >= K.ADAPTIVE_TOPK_MARGIN, torch.ones_like(level), level)
        level = torch.where(margin >= K.ADAPTIVE_TOPK_MARGIN_HIGH, torch.zeros_like(level), level)
        weights = torch.where((level == 0).unsqueeze(-1), self._topk_softmax(logits, 1), self._topk_softmax(logits, 2))
        return torch.where((level == 2).unsqueeze(-1), self._topk_softmax(logits, 3), weights)

    # ----------------------------- forward ------------------------------
    def forward(
        self,
        h_p: torch.Tensor,
        h_t: torch.Tensor,
        h_g: torch.Tensor,
        quality: torch.Tensor,
        agreement: torch.Tensor,
        expert_availability: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """(B, D) x 3, quality (B, 3), agreement (B, 3), availability (B, 3) -> outputs dict."""
        experts = [self.price_text_expert(h_p, h_t), self.price_graph_expert(h_p, h_g), self.text_graph_expert(h_t, h_g)]
        expert_stack = torch.stack(experts, dim=1)  # (B, 3, D)
        regime = torch.zeros(h_p.shape[0], K.ROUTER_REGIME_DIM, device=h_p.device, dtype=h_p.dtype)
        # The router sees the price vector twice: the second copy is the "scale summary" slot,
        # which falls back to the price vector when no multi-scale summary is available.
        router_input = torch.cat([h_p, h_t, h_g, quality, h_p, agreement, regime], dim=-1)
        base_logits = self.router(router_input) * K.ROUTER_LOGIT_SCALE

        # Quality prior: + s * log(max(sqrt(q_i q_j), floor)) for the modality pair of each expert.
        q = quality.float().clamp(0.0, 1.0)
        pair_scores = torch.sqrt(torch.stack([q[:, 0] * q[:, 1], q[:, 0] * q[:, 2], q[:, 1] * q[:, 2]], dim=-1).clamp_min(0.0))
        pair_scores = pair_scores.clamp_min(K.QUALITY_PRIOR_FLOOR).to(device=base_logits.device, dtype=base_logits.dtype)
        prior_bias = (K.QUALITY_PRIOR_SCALE * pair_scores.clamp_min(1e-8).log()).to(dtype=base_logits.dtype)

        availability = None
        if expert_availability is not None:
            availability = (expert_availability.to(device=base_logits.device, dtype=base_logits.dtype) > 0.5).to(base_logits.dtype)
            empty = availability.sum(dim=1, keepdim=True) <= 0
            if bool(empty.any().item()):
                availability = availability.clone()
                availability[empty.expand_as(availability)] = 1.0
        logits = self._apply_disabled(base_logits + prior_bias)
        if availability is not None:
            logits = logits.masked_fill(availability <= 0.5, -1e4)

        dense_probs = F.softmax(logits / K.ROUTER_TEMPERATURE, dim=-1)
        weights = self.router_weights(logits)
        fused = self.out_norm(torch.sum(expert_stack * weights.unsqueeze(-1), dim=1))
        return {
            "fused": fused,
            "weights": weights,
            "dense_probs": dense_probs,
            "logits": logits,
            "expert_stack": expert_stack,
            "quality": quality,
            "availability": availability if availability is not None else torch.ones_like(dense_probs),
        }


# ------------------------------------------------------------------------------------------
# Router regularizers (implementation details; weights in constants.ROUTER_PRIOR_WEIGHTS)
# ------------------------------------------------------------------------------------------
def router_prior_losses(out: Dict[str, torch.Tensor], disabled_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Entropy floor, margin cap, expert diversity, z-loss, temporal smoothing and quality-pattern
    terms computed from the dense router probabilities."""
    probs = out["dense_probs"].float()
    active = int((disabled_mask.detach().float() < 0.5).sum().item())
    dense = probs.clamp_min(1e-8)
    entropy = -(dense * dense.log()).sum(dim=-1)
    target = K.ROUTER_ENTROPY_TARGET_RATIO * math.log(float(max(1, active)))
    entropy_prior = F.relu(entropy.new_tensor(target) - entropy).pow(2).mean()
    top2 = torch.topk(probs, k=2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    margin_prior = F.relu(margin - margin.new_tensor(K.ROUTER_MARGIN_TARGET)).pow(2).mean()
    expert_normed = F.normalize(out["expert_stack"].float(), dim=-1, eps=1e-8)
    cosine = torch.einsum("bed,bfd->bef", expert_normed, expert_normed)
    offdiag = ~torch.eye(cosine.shape[-1], dtype=torch.bool, device=cosine.device)
    diversity_prior = cosine[:, offdiag].pow(2).mean()
    z_prior = torch.logsumexp(out["logits"].float(), dim=-1).pow(2).mean()
    smoothing = probs.new_zeros(())
    if probs.shape[0] >= 2:
        left, right = dense[:-1], dense[1:]
        smoothing = (0.5 * ((left * (left.log() - right.log())).sum(-1) + (right * (right.log() - left.log())).sum(-1))).mean()
    consistency = separation = probs.new_zeros(())
    quality = out["quality"]
    if quality.shape[0] >= 2:
        pattern = quality.detach().float().clamp(0.0, 1.0) >= K.QUALITY_PATTERN_THRESHOLD
        same = (pattern[:, None, :] == pattern[None, :, :]).all(dim=-1)
        upper = torch.triu(torch.ones(same.shape, dtype=torch.bool, device=same.device), diagonal=1)
        dist_sq = (probs[:, None, :] - probs[None, :, :]).pow(2).sum(dim=-1)
        if bool((same & upper).any().item()):
            consistency = dist_sq[same & upper].mean()
        if bool((~same & upper).any().item()):
            dist = dist_sq[~same & upper].clamp_min(1e-12).sqrt()
            separation = F.relu(dist.new_tensor(K.QUALITY_PATTERN_MARGIN) - dist).pow(2).mean()
    return {
        "entropy": entropy_prior,
        "margin": margin_prior,
        "diversity": diversity_prior,
        "z": z_prior,
        "smoothing": smoothing,
        "pattern_consistency": consistency,
        "pattern_separation": separation,
    }


def load_balance_loss(dense_probs: torch.Tensor, disabled_mask: torch.Tensor) -> torch.Tensor:
    """Dense + straight-through top-1 load-balancing loss."""
    probs = dense_probs.float().clamp_min(1e-8)
    active = (~(disabled_mask.to(device=probs.device) > 0.5)).to(dtype=probs.dtype)
    count = active.sum().clamp_min(1.0)
    uniform = active / count
    dense_term = count * (probs.mean(dim=0) - uniform).pow(2).sum()
    hard = F.one_hot(probs.argmax(dim=-1), num_classes=probs.shape[-1]).to(dtype=probs.dtype)
    straight = hard - probs.detach() + probs
    top1_term = count * (straight.mean(dim=0) - uniform).pow(2).sum()
    return dense_term + top1_term
