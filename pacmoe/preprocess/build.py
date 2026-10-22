"""Write a preprocessed bundle (the format read by pacmoe.data.Bundle).

Steps (Section 2.1-2.3): dividend-adjusted prices -> technical indicators -> per-stock normalization
fitted on the training period -> 60-day windows; labels and targets; daily correlation graphs;
text day-feature windows from the cached FinGPT / FinBERT outputs, and the text CCA (256) vectors.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from pacmoe.preprocess.features import PriceFeatureTransform, compute_indicators
from pacmoe.preprocess.graph import daily_graphs
from pacmoe.preprocess.labels import labels
from pacmoe.preprocess.prices import load_prices
from pacmoe.preprocess.text import FINBERT_DIM, day_features, load_sentence_table, sentence_embeddings, text_windows
from pacmoe.preprocess.text_cca import TextCCA

SPLITS = ("train", "val", "test")
# Implementation detail not stated in the paper: the walk-forward bundles were written from a cache of
# the indicator tables shared by the folds, stored in float32; normalization and labels were then
# computed on the float32 tables. The other bundles used the float64 tables.
FLOAT32_TABLES = {"main": False, "ablation": False, "walk_forward": True}


@dataclass
class PreprocessConfig:
    price_dir: str
    text_dir: str               # text_inputs_<split>.json and finbert_sentences_<split>.pt
    out_dir: str
    symbols: List[str]          # graph node order
    split_dates: Dict[str, Tuple[str, str]]  # split -> [start, end)
    window: int = 60
    up_threshold: float = 0.55
    down_threshold: float = -0.55
    vol_threshold: float = 1.0
    vol_target: str = "fixed"   # "fixed" | "scaled" | "abs_price_diff"
    use_neutral: bool = False
    graph_windows: Tuple[int, ...] = (60, 126, 252)
    graph_top_k: int = 10
    setup: str = "main"         # preprocessing preset: "main" | "ablation" | "walk_forward"
    cca_blas_threads: int = 8   # BLAS threads of the text CCA fit (numerical environment, not a model setting)


def _in(index: pd.DatetimeIndex, start: str, end: str):
    return (index >= pd.Timestamp(start)) & (index < pd.Timestamp(end))


def _samples(cfg: PreprocessConfig, feats: Dict[str, pd.DataFrame], targets: Dict[str, pd.DataFrame], split: str):
    """(symbol, date) of the labelled days of a split whose 60-day window is complete."""
    start, end = cfg.split_dates[split]
    rows = []
    for sym in cfg.symbols:
        if sym not in feats or sym not in targets:
            continue
        pos = {d: i for i, d in enumerate(feats[sym].index)}
        lab = targets[sym]
        for date in lab.index[_in(lab.index, start, end)]:
            if date in pos and pos[date] >= cfg.window - 1:
                rows.append((sym, date))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def build_bundle(cfg: PreprocessConfig) -> None:
    """Write samples_<split>.pt, graph_snapshots.pt and metadata.json to cfg.out_dir."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prices = load_prices(cfg.price_dir, cfg.symbols)
    frames = {s: compute_indicators(df) for s, df in tqdm(prices.items(), desc="technical indicators", unit="stock")}
    if FLOAT32_TABLES[cfg.setup]:
        frames = {s: f.astype(np.float32) for s, f in frames.items()}
    tr_start, tr_end = cfg.split_dates["train"]
    transform = PriceFeatureTransform().fit({s: f.loc[_in(f.index, tr_start, tr_end)] for s, f in frames.items()})
    feats = transform.transform(frames)
    targets = {s: labels(f["close"].astype(float), cfg.up_threshold, cfg.down_threshold, cfg.vol_threshold,
                         use_neutral=cfg.use_neutral, vol_target=cfg.vol_target) for s, f in frames.items()}
    payloads = {}
    for split in SPLITS:
        samples = _samples(cfg, feats, targets, split)
        keys = [(s, d.strftime("%Y-%m-%d")) for s, d in samples]
        x_ts = torch.stack([torch.tensor(feats[s].iloc[feats[s].index.get_loc(d) - cfg.window + 1: feats[s].index.get_loc(d) + 1].values,
                                         dtype=torch.float32)
                            for s, d in tqdm(samples, desc=f"{split}: price windows", unit="sample")])
        lab = [targets[s].loc[d] for s, d in samples]
        text = {(r["symbol"], r["date"]): r for r in json.loads((Path(cfg.text_dir) / f"text_inputs_{split}.json").read_text(encoding="utf-8"))}
        empty = {"tweets": [], "structures": []}
        tweets = [text.get(k, empty)["tweets"] for k in keys]
        structs = [text.get(k, empty)["structures"] for k in keys]
        sentences = sentence_embeddings(tweets, structs, load_sentence_table(str(Path(cfg.text_dir) / f"finbert_sentences_{split}.pt")))
        day = day_features(tqdm(keys, desc=f"{split}: text day features", unit="sample"), sentences, structs)
        payloads[split] = ({
            "x_ts": x_ts,
            "x_text_day_feature_window": text_windows(keys, day),
            "has_text": torch.tensor([len(t) > 0 for t in tweets]),
            "y_cls": torch.tensor([int(r["y_cls"]) for r in lab]),
            "y_vol": torch.tensor([int(r["y_vol"]) for r in lab]),
            "y_ret_h1": torch.tensor([float(r["y_ret_h1"]) for r in lab], dtype=torch.float32),
            "y_ret_h1_vol_normalized": torch.tensor([float(r["y_ret_h1_vol_normalized"]) for r in lab], dtype=torch.float32),
            "symbols": [k[0] for k in keys],
            "dates": [k[1] for k in keys],
        }, day, sentences)
    # Text CCA (Section 2.2.2), fitted on the training split: pooled FinBERT vs structured view.
    _, day_tr, _ = payloads["train"]
    print("text CCA: fitting on the training split ...", flush=True)
    # sklearn's iterative CCA and randomized PCA round differently with different BLAS thread counts.
    with threadpool_limits(limits=cfg.cca_blas_threads, user_api="blas"):
        cca = TextCCA(256).fit(day_tr[:, :FINBERT_DIM], day_tr[:, FINBERT_DIM:].numpy())
        for payload, day, sentences in payloads.values():
            payload["x_text_embed"] = cca.transform(day[:, :FINBERT_DIM])
            payload["x_text_embed_seq"] = cca.transform_sequences(sentences)
    all_dates = []
    for split, (payload, _, _) in payloads.items():
        torch.save(payload, out / f"samples_{split}.pt")
        all_dates.extend(payload["dates"])
        print(f"{split}: {len(payload['dates'])} samples written", flush=True)
    close = {s: f["close"].astype(float) for s, f in frames.items()}  # the (float32 for walk-forward) table
    torch.save(daily_graphs(close, cfg.symbols, all_dates, windows=cfg.graph_windows, top_k=cfg.graph_top_k), out / "graph_snapshots.pt")
    split_meta = {}
    for split, (start, end) in cfg.split_dates.items():
        split_meta[f"{split}_start"], split_meta[f"{split}_end"] = start, end
    meta = {"graph_symbols": cfg.symbols, "ts_feature_columns": transform.columns, "window_len": cfg.window, "split": split_meta}
    (out / "metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
