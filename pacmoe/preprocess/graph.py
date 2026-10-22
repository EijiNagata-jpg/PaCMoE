"""Daily correlation graphs (Section 2.2.3).

For each sample date t: daily log returns of the dividend-adjusted close prices up to t (the full
price history in the StockNet files, which starts before 2014), Pearson correlations over the last
60, 126 and 252 trading days, averaged; each stock is connected to the 10 stocks with the largest
absolute averaged correlation, the edges are symmetrized and weighted by the (signed) correlation.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

WINDOWS = (60, 126, 252)
TOP_K = 10


def log_returns_up_to(close: Dict[str, pd.Series], symbols: List[str], date: pd.Timestamp) -> torch.Tensor:
    """[T, N] daily log returns of the close prices on or before `date` (forward filled)."""
    frame = pd.DataFrame({s: close[s].loc[close[s].index <= date].replace([np.inf, -np.inf], np.nan) if s in close else pd.Series(dtype=float) for s in symbols}).sort_index()
    frame = frame.reindex(columns=symbols)
    if frame.empty:
        return torch.zeros(0, len(symbols), dtype=torch.float32)
    log_prices = np.log(frame.ffill().clip(lower=1e-12))
    returns = log_prices.diff().replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
    return torch.tensor(returns.values, dtype=torch.float32)


def rolling_correlation(x: torch.Tensor, window: int) -> torch.Tensor:
    """Correlation over the last `window` rows, from running sums (as in the original code)."""
    steps, nodes = x.shape
    if nodes == 0 or steps < 2:
        return torch.zeros(nodes, nodes, dtype=x.dtype)
    window = max(2, min(int(window), int(steps)))
    sum_x = x[:window].sum(dim=0)
    sum_xx = x[:window].T @ x[:window]
    for t in range(window, steps):
        sum_x = sum_x + x[t] - x[t - window]
        sum_xx = sum_xx + torch.outer(x[t], x[t]) - torch.outer(x[t - window], x[t - window])
    cov = (sum_xx - torch.outer(sum_x, sum_x) / float(window)) / float(window - 1)
    inv_std = torch.rsqrt(cov.diag().clamp_min(1e-8))
    return torch.nan_to_num(cov * inv_std.unsqueeze(0) * inv_std.unsqueeze(1), nan=0.0, posinf=0.0, neginf=0.0)


def top_k_edges(corr: torch.Tensor, k: int = TOP_K) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep each row's k largest |corr|, symmetrize (larger |value| wins), return COO edges."""
    n = corr.shape[0]
    if n <= 1:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)
    prunable = corr.clone()
    prunable.fill_diagonal_(0.0)
    abs_matrix = prunable.abs()
    vals, idx = torch.topk(abs_matrix, k=min(max(int(k), 1), n - 1), dim=1)
    keep = torch.zeros_like(abs_matrix).scatter_(1, idx, vals)
    kept = torch.where(keep > 0, prunable, torch.zeros_like(prunable))
    kept = torch.where(kept.abs() >= kept.T.abs(), kept, kept.T)
    kept.fill_diagonal_(0.0)
    mask = kept != 0
    coords = mask.nonzero(as_tuple=False)
    return coords.T.to(dtype=torch.long), kept[mask]


def daily_graphs(close: Dict[str, pd.Series], symbols: List[str], dates: Iterable[str],
                 windows: Iterable[int] = WINDOWS, top_k: int = TOP_K) -> Dict[str, Dict[str, torch.Tensor]]:
    """{date: {"edge_index", "edge_weight", "edge_type"}} for every sample date."""
    out = {}
    for date in tqdm(sorted({str(d) for d in dates}), desc="daily graphs", unit="day"):
        x = log_returns_up_to(close, symbols, pd.Timestamp(date))
        corr = torch.stack([rolling_correlation(x, w) for w in windows], dim=0).mean(dim=0)
        edge_index, edge_weight = top_k_edges(corr, top_k)
        out[date] = {"edge_index": edge_index, "edge_weight": edge_weight, "edge_type": torch.zeros(edge_index.shape[1], dtype=torch.long)}
    return out
