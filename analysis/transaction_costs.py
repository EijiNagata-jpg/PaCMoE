"""Transaction-cost backtest of the direction predictions (response to Reviewer 2).

Predictions for every stock of every test day (neutral days included, as a trader would need them).
Rules: L/S-rank (long the top 20%, short the bottom 20% of each day) and Dir (long p >= 0.5, short the
others); execution close(t)->close(t+1) and open(t+1)->close(t+1); one-way costs of 0/5/10/20 bps on the
traded amount; break-even cost and deflated Sharpe ratio (4 trials = 2 rules x 2 executions).

usage: python -m analysis.transaction_costs --pred-dir outputs/predictions/main_all_days
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from analysis.common import load_prices, returns_table

COSTS_BPS = [0, 5, 10, 20]


def load_runs(pred_dir: str) -> dict:
    """{run: DataFrame(p, y) indexed by (date, symbol)} from scripts/dump_predictions.py outputs."""
    out = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        df = pd.DataFrame({"date": z["test_dates"], "symbol": z["test_symbols"], "p": z["test_probs"], "y": z["test_labels"]})
        if df.duplicated(["date", "symbol"]).any():
            raise SystemExit(f"{f}: duplicated (date, symbol)")
        out[os.path.basename(f)[:-4]] = df.set_index(["date", "symbol"]).sort_index()
    return out


def weights_for_day(p: np.ndarray, rule: str) -> np.ndarray:
    n = len(p)
    w = np.zeros(n)
    if rule == "ls_rank":
        k = max(1, int(round(0.2 * n)))
        order = np.argsort(p, kind="stable")
        w[order[-k:]] = 0.5 / k
        w[order[:k]] = -0.5 / k
    elif rule == "dir":
        long = p >= 0.5
        nl, ns = int(long.sum()), int((~long).sum())
        # gross exposure 1 (sum of |long| and |short|), equal weights: a bet on the market direction
        w[long] = 1.0 / n
        w[~long] = -1.0 / n
    else:
        raise ValueError(rule)
    return w


def backtest(df: pd.DataFrame, rule: str, ret_col: str) -> dict:
    """df: date, symbol, p, r_cc, r_oc. Daily gross P&L and one-way turnover."""
    days = sorted(df["date"].unique())
    prev = {}
    gross, turnover = [], []
    for t in days:
        g = df[df["date"] == t]
        g = g[np.isfinite(g[ret_col])]
        w = weights_for_day(g["p"].to_numpy(), rule)
        cur = dict(zip(g["symbol"], w))
        if ret_col == "r_oc":
            # open at the open and close at the same day's close: both legs are traded every day
            to = 2.0 * np.abs(w).sum()
        else:
            # held from close to next close and rebalanced into the next day's positions
            keys = set(cur) | set(prev)
            to = sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
        prev = cur if ret_col == "r_cc" else {}
        gross.append(float((w * g[ret_col].to_numpy()).sum()))
        turnover.append(float(to))
    if ret_col == "r_cc":
        turnover[-1] += sum(abs(v) for v in prev.values())  # close the last positions
    return {"days": days, "gross": np.array(gross), "turnover": np.array(turnover)}


def stats(gross, turnover, cost_bps):
    net = gross - turnover * cost_bps / 1e4
    mu, sd = net.mean(), net.std(ddof=1)
    sr_d = mu / sd if sd > 0 else float("nan")
    return {"ann_return": float(mu * 252), "ann_vol": float(sd * math.sqrt(252)), "sharpe_ann": float(sr_d * math.sqrt(252)),
            "sharpe_daily": float(sr_d), "cum_return": float(np.prod(1 + net) - 1), "mean_turnover": float(turnover.mean()),
            "skew": float(pd.Series(net).skew()), "kurt": float(pd.Series(net).kurt() + 3.0), "n_days": int(len(net))}


def breakeven_bps(gross, turnover):
    t = turnover.sum()
    return float(gross.sum() / t * 1e4) if t > 0 else float("nan")


def dsr(sr_daily, n_obs, skew, kurt, sr_trials_daily):
    """Deflated Sharpe ratio (Bailey and Lopez de Prado 2014); daily (not annualized) Sharpe ratios."""
    from scipy.stats import norm

    n = len(sr_trials_daily)
    var = float(np.var(sr_trials_daily, ddof=1)) if n > 1 else 0.0
    g = 0.5772156649
    sr0 = math.sqrt(var) * ((1 - g) * norm.ppf(1 - 1.0 / n) + g * norm.ppf(1 - 1.0 / (n * math.e)))
    den = math.sqrt(max(1e-12, 1 - skew * sr_daily + (kurt - 1) / 4.0 * sr_daily ** 2))
    return {"sr0_daily": sr0, "dsr": float(norm.cdf((sr_daily - sr0) * math.sqrt(n_obs - 1) / den)), "var_sr_trials": var}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--price-dir", default="data/stocknet-dataset/price/raw")
    ap.add_argument("--out", default="outputs/econometrics/transaction_costs.json")
    a = ap.parse_args(argv)
    neutral = load_runs(a.pred_dir)
    res = {}
    any_run = next(iter(neutral.values())).reset_index()
    px = load_prices(a.price_dir, sorted(any_run["symbol"].unique()))
    rt = returns_table(px, any_run[["date", "symbol"]])
    bt = {}
    for r, d in neutral.items():
        df = d.reset_index().merge(rt, on=["date", "symbol"], how="left")
        for rule in ("ls_rank", "dir"):
            for ex, col in (("cc", "r_cc"), ("oc", "r_oc")):
                out = backtest(df, rule, col)
                bt.setdefault((rule, ex), []).append(out)
    d1 = {}
    for (rule, ex), outs in bt.items():
        key = f"{rule}_{ex}"
        d1[key] = {"breakeven_bps_mean": float(np.mean([breakeven_bps(o["gross"], o["turnover"]) for o in outs])),
                   "breakeven_bps_sd": float(np.std([breakeven_bps(o["gross"], o["turnover"]) for o in outs], ddof=1))}
        for c in COSTS_BPS:
            ss = [stats(o["gross"], o["turnover"], c) for o in outs]
            d1[key][f"{c}bps"] = {k: {"mean": float(np.mean([s[k] for s in ss])), "sd": float(np.std([s[k] for s in ss], ddof=1))}
                                  for k in ss[0]}
    # Deflated Sharpe ratio: 4 trials; one series from the run-averaged daily P&L.
    for c in (0, 10):
        series = {}
        for (rule, ex), outs in bt.items():
            g = np.mean([o["gross"] for o in outs], 0); t = np.mean([o["turnover"] for o in outs], 0)
            series[f"{rule}_{ex}"] = stats(g, t, c)
        srs = [s["sharpe_daily"] for s in series.values()]
        for k, s in series.items():
            d1[k][f"dsr_{c}bps"] = {"sharpe_ann_runavg": s["sharpe_ann"], **dsr(s["sharpe_daily"], s["n_days"], s["skew"], s["kurt"], srs)}
    res["d1"] = d1
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps(d1, indent=1, default=float)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
