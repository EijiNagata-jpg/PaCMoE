"""Shared helpers of the econometric comparisons (Section 3.5) and the transaction-cost backtest."""
from __future__ import annotations

import glob
import math
import os
import zipfile

import numpy as np
import pandas as pd

PANEL_START = "2013-01-03"  # first day on which the 85 stocks with complete histories (incl. ABBV) are observed
N_BOOT = 2000                # day-block bootstrap replications


# ---------------------------------------------------------------- prices and factors
def load_prices(price_dir: str, symbols) -> dict:
    """{symbol: DataFrame indexed by 'YYYY-MM-DD'} with lower-case columns (adj_close, close, open...)."""
    px = {}
    for s in symbols:
        df = pd.read_csv(os.path.join(price_dir, f"{s}.csv"))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        px[s] = df.set_index("date").sort_index()
    return px


def adj_close_panel(price_dir: str, symbols) -> pd.DataFrame:
    """Adjusted close prices (rows: dates, columns: symbols)."""
    return pd.DataFrame({s: pd.read_csv(f"{price_dir}/{s}.csv", parse_dates=["Date"]).set_index("Date")["Adj Close"]
                         for s in symbols}).sort_index()


def locf_prices(P: pd.DataFrame) -> pd.DataFrame:
    """Last observation carried forward (leading gaps stay missing)."""
    return P.sort_index().ffill()


def log_return_panel(P: pd.DataFrame) -> pd.DataFrame:
    """r_t = 100 log(P_t / P_{t-1}); the first day is dropped."""
    return (100.0 * np.log(P / P.shift(1))).iloc[1:]


def complete_symbols(R: pd.DataFrame, start, end) -> list:
    """Symbols without missing returns in [start, end), sorted."""
    w = R.loc[(R.index >= pd.Timestamp(start)) & (R.index < pd.Timestamp(end))]
    return sorted(c for c in R.columns if not w[c].isna().any())


def read_ff(path: str) -> pd.DataFrame:
    """Daily Fama-French factors from the Kenneth French data library zip (decimal returns)."""
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    start = next(i for i, l in enumerate(raw) if l.strip()[:8].isdigit())
    rows = []
    header = [c.strip() for c in raw[start - 1].split(",")]
    for l in raw[start:]:
        parts = [p.strip() for p in l.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            break
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date"] + header[1:])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    return df.set_index("date").astype(float) / 100.0


def returns_table(px: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """For each (date t, symbol): close(t)->close(t+1) (adjusted) and open(t+1)->close(t+1) simple returns."""
    out = []
    for s, g in rows.groupby("symbol"):
        d = px[s]
        dates = d.index.to_numpy()
        adj = d["adj_close"].to_numpy(float)
        close = d["close"].to_numpy(float)
        opn = d["open"].to_numpy(float)
        pos = {x: i for i, x in enumerate(dates)}
        for t in g["date"]:
            i = pos.get(t)
            if i is None or i + 1 >= len(dates):
                out.append((t, s, np.nan, np.nan)); continue
            out.append((t, s, adj[i + 1] / adj[i] - 1.0, close[i + 1] / opn[i + 1] - 1.0))
    return pd.DataFrame(out, columns=["date", "symbol", "r_cc", "r_oc"])


def load_predictions(pred_dir: str) -> list:
    """Prediction files written by scripts/dump_predictions.py: one DataFrame per run, indexed by (date, symbol)."""
    frames = []
    for f in sorted(glob.glob(os.path.join(pred_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        df = pd.DataFrame({"date": z["test_dates"], "symbol": z["test_symbols"], "y": z["test_labels"], "p": z["test_probs"],
                           "ret": z["test_pred_ret"], "vol": 1 / (1 + np.exp(-z["test_logit_vol"]))})
        if df.duplicated(["date", "symbol"]).any():
            raise SystemExit(f"{f}: duplicated (date, symbol)")
        frames.append(df.set_index(["date", "symbol"]).sort_index())
    return frames


# ---------------------------------------------------------------- statistics
def mcc(p, y) -> float:
    q = (p >= 0.5).astype(int)
    tp = int(((q == 1) & (y == 1)).sum()); tn = int(((q == 0) & (y == 0)).sum())
    fp = int(((q == 1) & (y == 0)).sum()); fn = int(((q == 0) & (y == 1)).sum())
    d = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return 0.0 if d == 0 else (tp * tn - fp * fn) / d


def auc(s, y) -> float:
    """ROC-AUC from ranks (ties averaged)."""
    s = np.asarray(s, float); y = np.asarray(y).astype(int)
    r = pd.Series(s).rank().to_numpy()
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def boot_days(days, fn, rng, n=N_BOOT):
    """95% percentile interval of fn(resampled days); days are resampled with replacement."""
    u = np.array(sorted(days))
    vals = [fn(rng.choice(u, len(u), replace=True)) for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
