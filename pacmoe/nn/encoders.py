"""
Modality encoders of PaCMoE (Section 2.2 of the paper).

Price:  PatchTST (Nie et al., 2023) for the OHLCV window and for the technical indicators; the
        indicator encoder is channel-independent and uses three patch scales.
Text:   day-level text features (FinGPT/FinBERT/CCA, see preprocessing) -> causal Transformer
        with a temporal relevance gate.
Graph:  daily rolling-correlation graph -> single-relation R-GCN (one RGCNConv layer).

All encoders are trained from scratch; the pretrained text models are used only in preprocessing.
Tensor shapes use B = batch, T = look-back days, C = channels, D = hidden dim.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from pacmoe import constants as K


def _pre_norm_encoder(d_model: int, nhead: int, dim_feedforward: int, dropout: float, num_layers: int) -> nn.TransformerEncoder:
    """Pre-norm GELU Transformer encoder used by every sequence encoder. to build."""
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


# --------------------------------------------------------------------------------------
# Price: OHLCV (PatchTST)
# --------------------------------------------------------------------------------------
class _PatchEmbedding(nn.Module):
    """Split (B, T, C) into overlapping patches and project them to d_model."""

    def __init__(self, in_channels: int, patch_len: int, stride: int, d_model: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(in_channels * patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, P, d_model)."""
        batch, _, _ = x.shape
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, P, C, patch_len)
        num_patches = patches.shape[1]
        patches = patches.permute(0, 1, 3, 2).contiguous().view(batch, num_patches, -1)
        return self.norm(self.proj(patches))


class OHLCVPatchTST(nn.Module):
    """PatchTST for the OHLCV window: patches span the five channels; trained from scratch."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        d_model = K.OHLCV_D_MODEL
        self.patch_emb = _PatchEmbedding(K.OHLCV_CHANNELS, K.OHLCV_PATCH_LEN, K.OHLCV_STRIDE, d_model)
        self.pos_emb = nn.Embedding(K.OHLCV_MAX_PATCHES, d_model)
        self.transformer = _pre_norm_encoder(d_model, K.OHLCV_NHEAD, 4 * d_model, K.OHLCV_DROPOUT, K.OHLCV_LAYERS)
        self.out_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim))
        nn.init.trunc_normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x_ohlcv: torch.Tensor) -> torch.Tensor:
        """(B, T, 5) window-normalized OHLCV -> (B, out_dim)."""
        tokens = self.patch_emb(x_ohlcv)
        positions = torch.arange(tokens.shape[1], device=x_ohlcv.device)
        tokens = tokens + self.pos_emb(positions).unsqueeze(0)
        hidden = self.transformer(tokens)
        return self.out_proj(hidden.mean(dim=1))


def window_zscore(x: torch.Tensor) -> torch.Tensor:
    """Z-score each channel within its own look-back window (no future data)."""
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True).clamp(min=1e-6)
    return (x - mean) / std


# --------------------------------------------------------------------------------------
# Price: technical indicators 
# --------------------------------------------------------------------------------------
def _num_patches(seq_len: int, patch_size: int, stride: int) -> int:
    """Number of patches after zero-padding short series to one patch."""
    effective_len = max(int(seq_len), int(patch_size))
    return 1 + max(0, (effective_len - int(patch_size)) // int(stride))


class MultiScalePatchEmbedding(nn.Module):
    """Patch tokens of a univariate series at several temporal scales."""

    def __init__(self, d_model: int, scales: Sequence[Tuple[int, int]], dropout: float) -> None:
        super().__init__()
        self.scales = tuple(scales)
        self.patch_projections = nn.ModuleList([nn.Linear(size, d_model) for size, _ in self.scales])
        self.patch_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in self.scales])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """(N, T, 1) -> list of (N, P_s, d_model)."""
        series = x.squeeze(-1)
        tokens_per_scale = []
        for (size, stride), proj, norm in zip(self.scales, self.patch_projections, self.patch_norms):
            padded = series if series.shape[1] >= size else F.pad(series, (0, size - series.shape[1]))
            patches = padded.unfold(dimension=1, size=size, step=stride)
            tokens_per_scale.append(self.dropout(norm(proj(patches))))
        return tokens_per_scale


class PatchTSTScaleEncoder(nn.Module):
    """PatchTST Transformer over the patch tokens of one scale, with learnable positions."""

    def __init__(self, patch_size: int, stride: int, d_model: int, num_layers: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        max_tokens = _num_patches(K.TECH_MAX_SEQ_LEN, patch_size, stride)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
        self.input_norm = nn.LayerNorm(d_model)
        self.transformer = _pre_norm_encoder(d_model, n_heads, 4 * d_model, dropout, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, P, d_model) -> (N, P, d_model)."""
        pos = self.pos_embedding[:, : x.shape[1], :].to(device=x.device, dtype=x.dtype)
        x = self.dropout(self.input_norm(x + pos))
        return self.output_norm(self.transformer(x))


class _QueryAttentionPool(nn.Module):
    """Learnable-query attention pooling followed by a residual FFN (shared by two stages)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """(N, S, d_model) -> (N, d_model)."""
        query = self.query.expand(tokens.shape[0], -1, -1)
        attn_out, _ = self.attn(query, tokens, tokens, need_weights=False)
        pooled = self.attn_norm(query + self.attn_dropout(attn_out))
        pooled = self.ffn_norm(pooled + self.ffn_dropout(self.ffn(pooled)))
        return pooled.squeeze(1)


class IndicatorPatchTST(nn.Module):
    """PatchTST for the technical indicators: each indicator is encoded independently (channel
    independence) at three patch scales; scales and then indicators are pooled with attention."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        d_model, layers, heads, dropout = K.TECH_D_MODEL, K.TECH_LAYERS, K.TECH_NHEAD, K.TECH_DROPOUT
        self.d_model = d_model
        self.patch_embedding = MultiScalePatchEmbedding(d_model, K.TECH_SCALES, dropout)
        self.scale_encoders = nn.ModuleList(
            [PatchTSTScaleEncoder(size, stride, d_model, layers, heads, dropout) for size, stride in K.TECH_SCALES]
        )
        self.scale_fusion = _QueryAttentionPool(d_model, heads, dropout)
        self.channel_pool = _QueryAttentionPool(d_model, heads, dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(out_dim)
        assert out_dim == d_model, "the paper configuration uses out_dim == d_model (identity projection)"

    def forward(self, x_tech: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, d_model)."""
        batch, seq_len, channels = x_tech.shape
        series = x_tech.transpose(1, 2).contiguous().reshape(batch * channels, seq_len, 1)
        encoded = [enc(tok) for enc, tok in zip(self.scale_encoders, self.patch_embedding(series))]
        pooled_scales = torch.stack([tokens.mean(dim=1) for tokens in encoded], dim=1)
        per_channel = self.scale_fusion(pooled_scales).reshape(batch, channels, self.d_model)
        return self.output_norm(self.output_dropout(self.channel_pool(per_channel)))


# --------------------------------------------------------------------------------------
# Text: day-level features over the last TEXT_WINDOW_DAYS days
# --------------------------------------------------------------------------------------
class TextTemporalEncoder(nn.Module):
    """Causal Transformer over daily text features + temporal relevance gate (TRG) + TanhNorm."""

    def __init__(self, in_dim: int, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.max_len = K.TEXT_WINDOW_DAYS
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos_embed = nn.Embedding(self.max_len, d_model)
        self.encoder = _pre_norm_encoder(d_model, nhead, 4 * d_model, dropout, K.TEXT_LAYERS)
        self.seq_norm = nn.LayerNorm(d_model)
        self.summary_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._s_r = nn.Parameter(torch.zeros(1))
        self.relevance_head = nn.Linear(d_model, 1)
        nn.init.constant_(self.relevance_head.bias, 1.0)

    def _relevance_weights(self, h_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """alpha_t = max(g_t - (1 - r), 0)^2, normalized over valid days."""
        g = torch.sigmoid(self.relevance_head(h_seq)).squeeze(-1) * mask.float()
        r = torch.sigmoid(self._s_r).clamp(min=1e-4)
        alpha = torch.clamp(g - (1.0 - r), min=0.0).pow(2)
        alpha_sum = alpha.sum(dim=1, keepdim=True)
        uniform = mask.float() / mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = alpha / alpha_sum.clamp_min(1e-9)
        return torch.where((alpha_sum < 1e-9).expand_as(weights), uniform, weights)

    @staticmethod
    def _tanh_norm(h: torch.Tensor) -> torch.Tensor:
        """Rescale h to norm tanh(||h||)."""
        norms = h.norm(p=2, dim=-1, keepdim=True)
        scaled = (torch.tanh(norms) / norms.clamp(min=1e-8)) * h
        return torch.where(norms < 1e-8, h, scaled)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, T, F), (B, T) bool -> h_text (B, D), h_text_seq (B, T, D)."""
        dtype = features.dtype
        seq_len = features.shape[1]
        mask = mask.to(dtype=torch.bool)
        h = self.in_proj(features.to(dtype=torch.float32))
        positions = torch.arange(seq_len, device=features.device).clamp(max=self.max_len - 1)
        h = self.dropout(h + self.pos_embed(positions).unsqueeze(0))
        key_padding_mask = ~mask
        all_padded = key_padding_mask.all(dim=1)
        if all_padded.any():
            # A row without any text day keeps day 0 attendable so that attention stays defined.
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padded, 0] = False
        causal = torch.triu(torch.ones(seq_len, seq_len, device=features.device, dtype=torch.bool), diagonal=1)
        encoded = self.encoder(h, mask=causal, src_key_padding_mask=key_padding_mask)
        h_seq = self.seq_norm(encoded.to(dtype=dtype)) * mask.unsqueeze(-1).to(dtype=dtype)
        weights = self._relevance_weights(h_seq, mask)
        h_agg = self._tanh_norm((h_seq * weights.unsqueeze(-1)).sum(dim=1)).to(dtype=dtype)
        return self.summary_norm(h_agg), h_seq


# --------------------------------------------------------------------------------------
# Graph: single-relation R-GCN
# --------------------------------------------------------------------------------------
class GraphEncoder(nn.Module):
    """One RGCNConv layer (Schlichtkrull et al., 2018) + LayerNorm.

    readout = "mean": the node outputs are averaged over all nodes of the batch graph and the
    same vector is given to every sample (setup "main"). readout = "node": each sample takes the
    output of its own node (setups "ablation" and "walk_forward").
    """

    def __init__(self, in_dim: int, out_dim: int, readout: str) -> None:
        super().__init__()
        self.conv = RGCNConv(in_dim, out_dim, num_relations=K.GRAPH_NUM_RELATIONS)
        self.norm = nn.LayerNorm(out_dim)
        self.readout = readout

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """(N, in_dim), (2, E), (E,) -> (N, out_dim)."""
        h = self.norm(self.conv(node_features, edge_index, edge_type))
        if self.readout == "node":
            return h
        return h.mean(dim=0, keepdim=True).expand(node_features.shape[0], -1)
