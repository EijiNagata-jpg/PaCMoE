"""Section 3.5: DCC-GARCH in its intended role (forecasting correlations) against the rolling correlations
used for PaCMoE's graph and constant conditional correlations (CCC).

- 85 stocks with complete histories from 2013-01-03; log returns of the adjusted close (LOCF).
- Forecasts made with data up to the close of day t: DCC-GARCH R_{t+1|t} (composite likelihood, contiguous
  pairs, estimated before the validation period), rolling correlation (mean of 60/126/252 days), CCC (S).
- Main metric: realized volatility of the global minimum-variance portfolio (H = D R D, D from the same
  GARCH sigma_{t+1|t}). Auxiliary: Frobenius distance to the realized correlation of days t+1..t+20 and
  overlap of the ten most correlated stocks. Day-block bootstrap (2000, seed 0).

usage: python -m analysis.dcc_corr_eval --bundle data/preprocessed/main
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from analysis.common import PANEL_START, complete_symbols, locf_prices, log_return_panel
from analysis.dcc_garch import fit_dcc

WINDOWS = (60, 126, 252)
K = 10
H_REAL = 20


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", default="data/preprocessed/main", help="split dates and graph symbols are read from it")
    ap.add_argument("--price-dir", default="data/stocknet-dataset/price/raw")
    ap.add_argument("--out", default="outputs/econometrics/dcc_corr_eval.json")
    args = ap.parse_args(argv)
    RAW = args.price_dir
    meta = json.load(open(f"{args.bundle}/metadata.json"))
    split = meta["split"]
    tr1, te0, te1 = (pd.Timestamp(split[k]) for k in ("train_end", "test_start", "test_end"))
    syms_all = list(meta["graph_symbols"])
    P = pd.DataFrame({s: pd.read_csv(f"{RAW}/{s}.csv", parse_dates=["Date"]).set_index("Date")["Adj Close"]
                      for s in syms_all}).sort_index()
    # Forecasts use data up to the end of the test period only; the realized correlations use the next 20 days.
    Rp = log_return_panel(locf_prices(P.loc[P.index < te1]))
    start = pd.Timestamp(PANEL_START)
    syms = complete_symbols(Rp, start, te1)
    R = Rp.loc[(Rp.index >= start) & (Rp.index < te1), syms]
    Rfull = log_return_panel(locf_prices(P[syms]))
    Rfull = Rfull.loc[Rfull.index >= start]
    est = np.asarray(R.index < tr1)
    m = fit_dcc(R.loc[est].to_numpy(), mean="constant", method="composite", pairs="contiguous")
    h_next, R_next = m.one_step_forecasts(R.to_numpy())  # row i = forecast for day i+1 made with data up to day i
    test_idx = [i for i, d in enumerate(R.index) if te0 <= d < te1]
    # The last test day (next day outside the test period) is included: realized returns come from Rfull.
    n = len(syms)
    one = np.ones(n)

    def roll_corr(i):
        Cs = [np.corrcoef(R.iloc[max(0, i - w + 1):i + 1].to_numpy().T) for w in WINDOWS]
        return np.mean(Cs, axis=0)

    full_pos = {d: j for j, d in enumerate(Rfull.index)}
    rows = []
    for i in test_idx:
        d = R.index[i]
        j = full_pos[d]
        if j + 1 >= len(Rfull):
            continue
        r_next = Rfull.iloc[j + 1].to_numpy()
        real = Rfull.iloc[j + 1:j + 1 + H_REAL].to_numpy()
        C_real = np.corrcoef(real.T)
        sd = np.sqrt(h_next[i])
        fc = {"dcc": R_next[i], "rolling": roll_corr(i), "ccc": m.S / np.sqrt(np.outer(np.diag(m.S), np.diag(m.S)))}
        rec = {"date": str(d.date())}
        off = ~np.eye(n, dtype=bool)
        top_real = np.argsort(-np.where(off, np.abs(C_real), -np.inf), axis=1)[:, :K]
        for name, C in fc.items():
            H = np.outer(sd, sd) * C
            w = np.linalg.solve(H + 1e-10 * np.eye(n), one)
            w = w / w.sum()
            rec[f"{name}_sq"] = float((w @ r_next) ** 2)
            rec[f"{name}_frob"] = float(np.sqrt(np.sum((C - C_real)[off] ** 2)))
            top = np.argsort(-np.where(off, np.abs(C), -np.inf), axis=1)[:, :K]
            rec[f"{name}_overlap"] = float(np.mean([len(set(top[a]) & set(top_real[a])) / K for a in range(n)]))
        rows.append(rec)
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)

    def boot(fn, B=2000):
        idx = np.arange(len(df))
        vals = [fn(df.iloc[rng.choice(idx, len(idx), replace=True)]) for _ in range(B)]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    res = {"n_symbols": n, "n_days": int(len(df)), "dcc_a": float(m.a), "dcc_b": float(m.b), "models": {}, "diffs": {}}
    for name in ("dcc", "rolling", "ccc"):
        res["models"][name] = {
            "gmv_ann_vol": float(math.sqrt(df[f"{name}_sq"].mean() * 252)),
            "gmv_ann_vol_ci": [math.sqrt(v * 252) for v in boot(lambda x: x[f"{name}_sq"].mean())],
            "frob_mean": float(df[f"{name}_frob"].mean()),
            "top10_overlap": float(df[f"{name}_overlap"].mean()),
        }
    for a, b in (("dcc", "rolling"), ("dcc", "ccc"), ("rolling", "ccc")):
        res["diffs"][f"{a}_minus_{b}"] = {
            "gmv_var_ratio": float(df[f"{a}_sq"].mean() / df[f"{b}_sq"].mean()),
            "gmv_var_ratio_ci": boot(lambda x: x[f"{a}_sq"].mean() / x[f"{b}_sq"].mean()),
            "frob": float((df[f"{a}_frob"] - df[f"{b}_frob"]).mean()),
            "frob_ci": boot(lambda x: (x[f"{a}_frob"] - x[f"{b}_frob"]).mean()),
            "overlap": float((df[f"{a}_overlap"] - df[f"{b}_overlap"]).mean()),
            "overlap_ci": boot(lambda x: (x[f"{a}_overlap"] - x[f"{b}_overlap"]).mean()),
        }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
