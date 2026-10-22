"""Compare the clean price features (60-day windows) with an existing preprocessed bundle.

Usage: python scripts/check_price_features.py <stocknet price/raw dir> <bundle dir> [split ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pacmoe.preprocess.features import PriceFeatureTransform, compute_indicators  # noqa: E402
from pacmoe.preprocess.prices import load_prices  # noqa: E402

price_dir, bundle = sys.argv[1], Path(sys.argv[2])
splits = sys.argv[3:] or ["test"]
meta = json.loads((bundle / "metadata.json").read_text())
t0 = time.time()
prices = load_prices(price_dir, meta["graph_symbols"])
frames = {s: compute_indicators(df) for s, df in prices.items()}
print("indicators", round(time.time() - t0, 1), "s", flush=True)
sp = meta["split"]
train = {s: f.loc[(f.index >= sp["train_start"]) & (f.index < sp["train_end"])] for s, f in frames.items()}
tf = PriceFeatureTransform().fit(train)
feats = tf.transform(frames)
print("columns equal:", tf.columns == meta["ts_feature_columns"], len(tf.columns), flush=True)
for split in splits:
    d = torch.load(bundle / f"samples_{split}.pt", map_location="cpu", weights_only=False)
    worst = 0.0
    worst_col = None
    n = len(d["dates"])
    for i in range(n):
        s, t = str(d["symbols"][i]), pd.Timestamp(str(d["dates"][i]))
        f = feats[s]
        end = f.index.get_loc(t)
        win = torch.tensor(f.iloc[end - 59: end + 1].values, dtype=torch.float32)
        diff = (win - d["x_ts"][i]).abs()
        m = float(diff.max())
        if m > worst:
            worst = m
            worst_col = tf.columns[int(diff.max(dim=0).values.argmax())]
    print(f"{split}: n={n} max|x_ts diff|={worst:.3e} (column {worst_col})", flush=True)
