"""
Preprocessed bundle loading and date-grouped batching.

A bundle directory (output of preprocessing) contains
  samples_{train,val,test}.pt  per-sample tensors (see Bundle below),
  graph_snapshots.pt           {date: {"edge_index", "edge_weight", "edge_type"}} built from data up to the date,
  metadata.json                symbol order of the graph nodes, feature names, ...

Batches never split a trading day: the samples of a day form one chunk, and consecutive chunks are
packed into a batch while the batch has at most `batch_size` samples. The graphs of the days in a
batch are stacked block-diagonally (node ids shifted by day).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Sampler

from pacmoe import constants as K

# Return-target keys stored in the bundle (Section 2.6.2).
RETURN_TARGET_KEYS = {"vol_normalized": "y_ret_h1_vol_normalized", "raw": "y_ret_h1"}


def _text_windows(raw, raw_mask=None):
    """List of (L_i, F) day-feature windows -> padded (N, L_max, F) and mask (N, L_max)."""
    if isinstance(raw, torch.Tensor):
        windows = raw.detach().cpu().float().contiguous()
        if isinstance(raw_mask, torch.Tensor) and tuple(raw_mask.shape) == tuple(windows.shape[:2]):
            return windows, raw_mask.detach().cpu().bool().contiguous()
        return windows, (windows.abs().sum(dim=-1) > 1e-6).contiguous()
    tensors = []
    feature_dim, max_len = 0, 0
    for item in raw:
        w = (item if isinstance(item, torch.Tensor) else torch.as_tensor(item, dtype=torch.float32)).detach().cpu().float()
        if w.ndim == 1:
            w = w.unsqueeze(0)
        if w.numel() > 0 and feature_dim == 0:
            feature_dim = int(w.shape[-1])
        max_len = max(max_len, int(w.shape[0]))
        tensors.append(w)
    windows = torch.zeros(len(tensors), max_len, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(tensors), max_len, dtype=torch.bool)
    for i, w in enumerate(tensors):
        if w.shape[0] > 0:
            windows[i, : w.shape[0]] = w
            mask[i, : w.shape[0]] = True
    return windows.contiguous(), mask.contiguous()


class Bundle:
    """One split of a preprocessed bundle held in memory."""

    def __init__(self, bundle_dir: str | Path, split: str, return_target: str = "vol_normalized", graph_snapshots_path: str | None = None) -> None:
        self.dir = Path(bundle_dir)
        meta = json.loads((self.dir / "metadata.json").read_text(encoding="utf-8"))
        payload = torch.load(self.dir / f"samples_{split}.pt", map_location="cpu", weights_only=False)
        self.x_ts = payload["x_ts"].float()                      # (N, T, 6 + indicators)
        self.x_ohlcv = self.x_ts[..., :5]                         # (N, T, 5) normalized OHLCV columns of x_ts
        self.text_features, self.text_day_mask = _text_windows(payload.get("x_text_day_feature_window", []), payload.get("x_text_day_mask"))
        self.has_text = torch.as_tensor(payload["has_text"]).bool().reshape(-1)
        self.y_cls = torch.as_tensor(payload["y_cls"]).long()
        self.y_ret = torch.as_tensor(payload[RETURN_TARGET_KEYS[return_target]]).float()
        self.y_vol = torch.as_tensor(payload["y_vol"]).float()
        self.ret_decimal = torch.as_tensor(payload["y_ret_h1"]).float() / 100.0  # next-day return (decimal)
        self.symbols: List[str] = [str(s) for s in payload["symbols"]]
        self.dates: List[str] = [str(d) for d in payload["dates"]]
        graph_symbols = list(meta.get("graph_symbols") or meta.get("symbols") or [])
        self.symbol_to_node = {s: i for i, s in enumerate(graph_symbols)}
        self.node_ids = torch.tensor([self.symbol_to_node.get(s, 0) for s in self.symbols], dtype=torch.long)
        # graph_snapshots_path replaces the daily graphs (e.g. graphs built from DCC-GARCH correlations).
        snapshots = torch.load(graph_snapshots_path or (self.dir / "graph_snapshots.pt"), map_location="cpu", weights_only=False)
        self.graphs: Dict[str, Dict[str, torch.Tensor]] = {}
        for date, g in snapshots.items():
            edge_type = g.get("edge_type")
            self.graphs[str(date)] = {
                "edge_index": g["edge_index"].detach().cpu().long(),
                "edge_weight": g["edge_weight"].detach().cpu().float(),
                "edge_type": edge_type.detach().cpu().long() if isinstance(edge_type, torch.Tensor) else torch.zeros(g["edge_index"].shape[1], dtype=torch.long),
            }

    def __len__(self) -> int:
        return len(self.dates)

    def collate(self, indices: Sequence[int]) -> Dict[str, torch.Tensor]:
        """Batch of sample indices -> model inputs and targets."""
        idx = torch.as_tensor(list(indices), dtype=torch.long)
        x_ts = torch.nan_to_num(self.x_ts[idx])
        batch = {
            "x_ohlcv": torch.nan_to_num(self.x_ohlcv[idx]),
            "x_tech": x_ts[..., K.PRICE_FEATURE_COLUMNS:].contiguous(),
            "text_features": torch.nan_to_num(self.text_features[idx]),
            "text_day_mask": self.text_day_mask[idx],
            "has_text": self.has_text[idx],
            "y_cls": self.y_cls[idx],
            "y_ret": torch.nan_to_num(self.y_ret[idx]),
            "y_vol": self.y_vol[idx],
            "ret_decimal": self.ret_decimal[idx],
            "ticker_ids": self.node_ids[idx],
            "dates": [self.dates[i] for i in indices],
            "symbols": [self.symbols[i] for i in indices],
        }
        batch.update(self._graph_batch(list(indices)))
        return batch

    def _graph_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Single-day batch: the day's graph. Multi-day batch: block-diagonal stack."""
        days: List[str] = []
        group_of: Dict[str, int] = {}
        groups: List[int] = []
        for i in indices:
            d = self.dates[i]
            if d not in group_of:
                group_of[d] = len(days)
                days.append(d)
            groups.append(group_of[d])
        if len(days) == 1:
            g = self.graphs[days[0]]
            return {"edge_index": g["edge_index"], "edge_type": g["edge_type"], "symbol_ids": self.node_ids[indices].clone()}
        node_counts = []
        for gi, d in enumerate(days):
            ids = [int(self.node_ids[i]) + 1 for i, grp in zip(indices, groups) if grp == gi]
            ei = self.graphs[d]["edge_index"]
            ids.append(int(ei.max().item()) + 1 if ei.numel() else 0)
            node_counts.append(max(ids))
        per_day = max(max(node_counts), 1)
        edge_index = torch.cat([self.graphs[d]["edge_index"] + gi * per_day for gi, d in enumerate(days)], dim=1)
        edge_type = torch.cat([self.graphs[d]["edge_type"] for d in days], dim=0)
        symbol_ids = torch.tensor([grp * per_day + int(self.node_ids[i]) for i, grp in zip(indices, groups)], dtype=torch.long)
        return {"edge_index": edge_index, "edge_type": edge_type, "symbol_ids": symbol_ids}


def apply_market_mean(batch: Dict, keys=("x_ohlcv", "x_tech")) -> Dict:
    """
    Replace each sample's price inputs by the cross-sectional mean of its trading day.

    This is how the reported models received the price modality: every stock of a day gets the
    same (market-average) OHLCV and indicator windows; stock identity enters through the ticker
    embedding, the text and the graph.
    """
    rows: Dict[str, List[int]] = {}
    for row, date in enumerate(batch["dates"]):
        rows.setdefault(str(date), []).append(row)
    out = dict(batch)
    for key in keys:
        x = batch[key]
        shared = x.clone()
        for idx in rows.values():
            index = torch.tensor(idx, device=x.device, dtype=torch.long)
            mean = x.index_select(0, index).mean(dim=0, keepdim=True)
            shared.index_copy_(0, index, mean.expand(len(idx), *mean.shape[1:]))
        out[key] = shared
    return out


class DateBatchSampler(Sampler[List[int]]):
    """Pack whole trading days into batches of at most `batch_size` samples.

    Training: samples within a day and the order of batches are shuffled with
    random.Random(seed + epoch_index) (epoch_index starts at 0).
    """

    def __init__(self, dates: Sequence[str], batch_size: int, shuffle: bool, seed: int) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._iteration = 0
        grouped: Dict[str, List[int]] = defaultdict(list)
        self._ordered: List[str] = []
        for i, d in enumerate(dates):
            if d not in grouped:
                self._ordered.append(d)
            grouped[d].append(i)
        self._by_date = dict(grouped)

    def _chunks(self, rng: Optional[random.Random]) -> List[List[int]]:
        chunks = []
        for d in self._ordered:
            indices = list(self._by_date[d])
            if self.shuffle and rng is not None:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                chunks.append(indices[start: start + self.batch_size])
        return chunks

    def _pack(self, chunks: List[List[int]]) -> List[List[int]]:
        batches, current = [], []
        for chunk in chunks:
            if current and len(current) + len(chunk) > self.batch_size:
                batches.append(current)
                current = []
            if len(chunk) >= self.batch_size:
                if current:
                    batches.append(current)
                    current = []
                batches.append(list(chunk))
                continue
            current.extend(chunk)
        if current:
            batches.append(current)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self._iteration = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._iteration)
        self._iteration += 1
        batches = self._pack(self._chunks(rng))
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return len(self._pack(self._chunks(None)))
