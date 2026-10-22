"""PaCMoE: Pairwise Cross-Modal Attention Fusion with Mixture-of-Experts.

Forward order (identical to the implementation that produced the paper's results):
  1. Price:  window-z-scored OHLCV -> OHLCVPatchTST -> h_P; indicators -> IndicatorPatchTST -> CCA residual
             alignment of h_P with the indicator view -> + ticker embedding.
  2. Text:   day-level text features -> TextTemporalEncoder -> h_T (learned vector if no tweet).
  3. Graph:  node features [h_P; h_T] on the daily correlation graph -> R-GCN -> h_G -> CCA view
             adapter.
  4. Fusion: pre-norm MLP per modality -> three pairwise experts -> MoE router -> z.
  5. Heads:  direction (2 logits), standardized return, volatility, ranking score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pacmoe import constants as K
from pacmoe.nn.cca import BatchCCAAdapter, MultiViewCCAAdapter
from pacmoe.nn.conditioning import GCCACondition, MissingModalityEmbedding, PreNormMLPBlock, QualityGate, TickerEmbedding
from pacmoe.nn.encoders import GraphEncoder, OHLCVPatchTST, IndicatorPatchTST, TextTemporalEncoder, window_zscore
from pacmoe.nn.fusion import EXPERT_NAMES, PairwiseExpertMoE


@dataclass
class ModelSpec:
    """Values stated in the paper (Table 2) plus the switches that select a reported setup."""

    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    text_feature_dim: int = 825
    cca_stat_mode: str = "batch"          # "batch" (main) | "running" (ablation, walk_forward)
    graph_readout: str = "mean"           # "mean" (main) | "node" (ablation, walk_forward)
    graph_view_exclude_self: bool = False  # False (main) | True (ablation, walk_forward)
    num_classes: int = 2
    disabled_experts: tuple = ()
    uniform_router: bool = False
    price_only: bool = False              # ablation: text and graph branches removed
    ranking: bool = True                  # ranking head (setups "main" and "walk_forward")


class PaCMoE(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        d = spec.hidden_dim
        self.spec = spec
        # Price
        self.ohlcv_encoder = OHLCVPatchTST(out_dim=d)
        self.tech_encoder = IndicatorPatchTST(out_dim=K.TECH_D_MODEL)
        self.tech_proj = nn.Linear(K.TECH_D_MODEL, d)
        self.price_tech_cca = BatchCCAAdapter(d, K.CCA_RESIDUAL_WEIGHT, spec.cca_stat_mode, input_norm=True, output_norm=True)
        self.ticker_embedding = TickerEmbedding(K.NUM_TICKERS, d)
        # Text
        self.text_encoder = TextTemporalEncoder(spec.text_feature_dim, d, spec.num_heads, spec.dropout)
        # Missing-modality vectors
        self.missing_price = MissingModalityEmbedding(d)
        self.missing_text = MissingModalityEmbedding(d)
        self.missing_graph = MissingModalityEmbedding(d)
        # Graph
        self.graph_encoder = GraphEncoder(2 * d, d, spec.graph_readout)
        self.graph_view_cca = MultiViewCCAAdapter(d, K.CCA_RESIDUAL_WEIGHT, spec.cca_stat_mode, spec.graph_view_exclude_self)
        # Fusion
        self.pre_fusion_price = PreNormMLPBlock(d, spec.dropout)
        self.pre_fusion_text = PreNormMLPBlock(d, spec.dropout)
        self.pre_fusion_graph = PreNormMLPBlock(d, spec.dropout)
        self.quality_gate = QualityGate(d)
        self.gcca = GCCACondition(d)
        self.fusion = PairwiseExpertMoE(d, spec.num_heads, spec.dropout, spec.disabled_experts, spec.uniform_router)
        # Heads
        self.head_cls = nn.Linear(d, spec.num_classes)
        self.head_ret = nn.Linear(d, 1)
        self.head_vol = nn.Linear(d, 1)
        self.head_rank = nn.Linear(d, 1) if spec.ranking else None
        # Auxiliary heads used only by training objectives
        # The price-only ablation of the original code had no graph branch head (unused there).
        branch_modalities = ("price", "text") if spec.price_only else ("price", "text", "graph")
        self.branch_heads = nn.ModuleDict({m: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, spec.num_classes)) for m in branch_modalities})
        self.expert_heads = nn.ModuleDict({e: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, spec.num_classes)) for e in EXPERT_NAMES})

    # --------------------------------------------------------------------------------------
    @staticmethod
    def _pair_availability(avail: torch.Tensor) -> torch.Tensor:
        """Expert availability from modality availability (B, 3) -> (B, 3)."""
        a = (avail[:, :3] > 0.5).to(avail.dtype)
        p, t, g = a[:, 0], a[:, 1], a[:, 2]
        out = torch.stack([p * t, p * g, t * g], dim=1)
        empty = out.sum(dim=1, keepdim=True) <= 0
        if bool(empty.any().item()):
            fallback = torch.stack([torch.clamp(p + t, max=1.0), torch.clamp(p + g, max=1.0), torch.clamp(t + g, max=1.0)], dim=1)
            fb_empty = fallback.sum(dim=1, keepdim=True) <= 0
            if bool(fb_empty.any().item()):
                fallback = fallback.clone()
                fallback[fb_empty.expand_as(fallback)] = 1.0
            out = torch.where(empty.expand_as(out), fallback, out)
        return out

    def _graph_node_features(self, h_ts, h_text, edge_index, edge_type, symbol_ids):
        """Average the sample vectors of each node; keep edges between nodes present in the batch.

        Returns node features (N, 2D), edges and edge types restricted to active nodes."""
        feats = torch.cat([h_ts, h_text], dim=-1)
        sid = symbol_ids.to(device=feats.device, dtype=torch.long)
        num_nodes = max(int(feats.size(0)), int(sid.max().item()) + 1 if sid.numel() else 0)
        if edge_index.numel() > 0:
            num_nodes = max(num_nodes, int(edge_index.max().item()) + 1)
        total = torch.zeros(num_nodes, feats.size(1), device=feats.device, dtype=feats.dtype)
        count = torch.zeros(num_nodes, 1, device=feats.device, dtype=feats.dtype)
        total.index_add_(0, sid, feats)
        count.index_add_(0, sid, torch.ones(sid.size(0), 1, device=feats.device, dtype=feats.dtype))
        nodes = total / count.clamp_min(1.0)
        valid = (edge_index[0] >= 0) & (edge_index[1] >= 0) & (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        active = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        active[sid.to(edge_index.device)] = True
        in_batch = torch.zeros_like(valid)
        if valid.any():
            in_batch[valid] = active[edge_index[0, valid]] & active[edge_index[1, valid]]
        keep = valid & in_batch
        return nodes, edge_index[:, keep], edge_type[keep], sid

    # --------------------------------------------------------------------------------------
    def forward(
        self,
        x_ohlcv: torch.Tensor,
        x_tech: torch.Tensor,
        text_features: torch.Tensor,
        text_day_mask: torch.Tensor,
        has_text: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        symbol_ids: torch.Tensor,
        ticker_ids: torch.Tensor,
        skip_text: bool = False,
        skip_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Batch tensors -> dict with logits_cls (B, 2), pred_ret (B, 1), logits_vol (B, 1),
        logit_rank (B,), router weights and intermediate representations.

        skip_text / skip_graph remove a branch (used by the training curriculum and the price-only
        ablation): its representation is zero and its experts are masked out.
        """
        skip_text = skip_text or self.spec.price_only
        skip_graph = skip_graph or self.spec.price_only
        text_mask = torch.zeros(x_ohlcv.shape[0], dtype=torch.bool, device=x_ohlcv.device) if skip_text else has_text.to(device=x_ohlcv.device, dtype=torch.bool)

        # 1. Price
        h_ts = self.ohlcv_encoder(window_zscore(x_ohlcv))
        # 2. Text
        if skip_text:
            h_text = torch.zeros(x_ohlcv.shape[0], self.spec.hidden_dim, device=x_ohlcv.device, dtype=h_ts.dtype)
        else:
            h_text, _ = self.text_encoder(text_features.to(dtype=h_ts.dtype), text_day_mask)
        # 1b. Indicators and the price-indicator CCA alignment
        h_tech = self.tech_encoder(x_tech)
        h_ts = self.price_tech_cca(h_ts, self.tech_proj(h_tech.to(device=h_ts.device, dtype=h_ts.dtype)), anchor=h_ts)
        h_ts = self.ticker_embedding(h_ts, ticker_ids)

        batch = h_ts.size(0)
        avail = torch.zeros(batch, 3, device=h_ts.device, dtype=h_ts.dtype)
        avail[:, 0] = 1.0
        avail[:, 1] = text_mask.to(dtype=h_ts.dtype)
        graph_available = (not skip_graph) and edge_index.numel() > 0
        avail[:, 2] = 1.0 if graph_available else 0.0
        h_ts = self.missing_price(h_ts, avail[:, 0])
        h_text = self.missing_text(h_text, avail[:, 1])

        # 3. Graph
        nodes, ei, et, sid = self._graph_node_features(h_ts, h_text, edge_index, edge_type, symbol_ids)
        if skip_graph:
            h_graph = torch.zeros(nodes.size(0), self.spec.hidden_dim, device=nodes.device)
        else:
            h_graph = self.graph_encoder(nodes, ei.to(dtype=torch.long), et.to(dtype=torch.long))
        h_graph = h_graph[sid]
        h_graph = self.graph_view_cca([h_graph, h_graph, None, None, None], anchor=h_graph)
        h_graph = self.missing_graph(h_graph, avail[:, 2])

        # 4. Fusion
        h_p = self.pre_fusion_price(h_ts)
        h_t = self.pre_fusion_text(h_text)
        h_g = self.pre_fusion_graph(h_graph)
        if skip_text:
            h_t = torch.zeros_like(h_t)
        if skip_graph:
            h_g = torch.zeros_like(h_g)
        quality_base = torch.stack(
            [torch.ones(batch, device=h_p.device, dtype=h_p.dtype), text_mask.to(device=h_p.device, dtype=h_p.dtype).reshape(-1),
             torch.full((batch,), 1.0 if graph_available else 0.0, device=h_p.device, dtype=h_p.dtype)],
            dim=-1,
        )
        quality = self.quality_gate(h_p, h_t, h_g, quality_base)
        agreement = self.gcca(h_p, h_t, h_g)
        fx = self.fusion(h_p, h_t, h_g, quality, agreement, self._pair_availability(avail))
        z = fx["fused"]

        # 5. Heads
        out = {
            "logits_cls": self.head_cls(z),
            "pred_ret": self.head_ret(z),
            "logits_vol": self.head_vol(z),
            "logit_rank": self.head_rank(z).squeeze(-1) if self.head_rank is not None else None,
            "router_weights": fx["weights"],
            "h_ts": h_ts,
            "h_text": h_text,
            "h_graph": h_graph,
            "h_fused": z,
            "modality_availability": avail,
            "fusion": fx,
        }
        branch_inputs = {"price": h_ts, "text": h_text, "graph": h_graph}
        out["branch_logits"] = {m: head(branch_inputs[m]) for m, head in self.branch_heads.items()}
        out["branch_availability"] = {
            "price": torch.ones(batch, dtype=torch.bool, device=h_ts.device),
            "text": text_mask,
            "graph": torch.full((batch,), graph_available, dtype=torch.bool, device=h_ts.device),
        }
        if self.training:
            out["expert_logits"] = torch.stack([self.expert_heads[e](fx["expert_stack"][:, i, :]) for i, e in enumerate(EXPERT_NAMES)], dim=1)
        return out
