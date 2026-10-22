"""Section 3.5: PaCMoE's auxiliary heads against econometric models of expected returns and volatility.

- PaCMoE: return head, direction probability and volatility head (mean over the ten main runs).
- Expected returns: CAPM, Fama-French 3 and 5 factors (beta from a 60-day rolling OLS up to day t,
  lambda = mean daily factor returns of the training period); statistical factor model (3 principal
  components estimated on the training period), forecast = -(sum of the last 5 days' residuals).
- Volatility: GARCH(1,1) per stock (maximum likelihood on the training period, sigma_{t+1|t}) and the
  20-day rolling standard deviation.
- Metrics: mean daily Spearman IC; AUC for |r_{t+1}| >= 1%; day-block bootstrap (2000, seed 0).

usage: python -m analysis.econ_axes --pred-dir outputs/predictions/main
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from analysis.common import auc as _auc
from analysis.common import boot_days, load_predictions, load_prices, read_ff, returns_table

TRAIN_START, TRAIN_END = "2014-01-01", "2015-08-01"
WIN = 60


def load_heads(pred_dir):
    """Mean of the ten runs' outputs per (date, symbol)."""
    frames = load_predictions(pred_dir)
    base = frames[0][["y"]].copy()
    for k in ("p", "ret", "vol"):
        base[f"pacmoe_{k}"] = np.mean([fr[k].to_numpy() for fr in frames], axis=0)
    return base.reset_index(), len(frames)


def auc(score, y):
    return _auc(np.asarray(score, float), np.asarray(y, int))


def returns_panel(px):
    """Daily simple returns of the adjusted close (rows: dates, columns: symbols)."""
    return pd.DataFrame({s: d["adj_close"].pct_change() for s, d in px.items()}).sort_index()


def rolling_beta_expected(R, F, lam, dates_needed):
    """beta'lambda with beta from the OLS of excess returns on the factors over the WIN days up to t."""
    out = {}
    Fv = F.to_numpy()
    idx = {d: i for i, d in enumerate(R.index)}
    for s in R.columns:
        r = R[s].to_numpy()
        for t in dates_needed:
            i = idx.get(t)
            if i is None or i + 1 < WIN:
                continue
            y = r[i - WIN + 1:i + 1]; X = Fv[i - WIN + 1:i + 1]
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            if ok.sum() < WIN * 0.8:
                continue
            Xc = np.column_stack([np.ones(ok.sum()), X[ok]])
            b, *_ = np.linalg.lstsq(Xc, y[ok], rcond=None)
            out[(t, s)] = float(b[1:] @ lam)
    return pd.Series(out)


def pca_residual_reversal(R, dates_needed, k=3, core=None):
    """-(sum of the residuals of the last 5 days) of the OLS on PCA factors (estimated on train) over the WIN days up to t."""
    tr = R.loc[(R.index >= TRAIN_START) & (R.index < TRAIN_END), core].dropna()
    mu, sd = tr.mean(), tr.std()
    Z = ((tr - mu) / sd).to_numpy()
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    V = Vt[:k].T
    Zall = ((R[core] - mu) / sd)
    F = pd.DataFrame(Zall.fillna(0).to_numpy() @ V, index=R.index)
    idx = {d: i for i, d in enumerate(R.index)}
    Fv = F.to_numpy(); out = {}
    for s in R.columns:
        r = R[s].to_numpy()
        for t in dates_needed:
            i = idx.get(t)
            if i is None or i + 1 < WIN:
                continue
            y = r[i - WIN + 1:i + 1]; X = Fv[i - WIN + 1:i + 1]
            ok = np.isfinite(y)
            if ok.sum() < WIN * 0.8 or not np.isfinite(y[-5:]).all():
                continue
            Xc = np.column_stack([np.ones(len(y)), X])
            b, *_ = np.linalg.lstsq(Xc[ok], y[ok], rcond=None)
            resid = y[-5:] - Xc[-5:] @ b
            out[(t, s)] = float(-resid.sum())
    return pd.Series(out)


def garch_fit(r):
    """GARCH(1,1) maximum likelihood (zero mean); r: daily returns in %."""
    r = r[np.isfinite(r)]
    v0 = r.var()

    def nll(th):
        w, a, b = th
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e10
        h = np.empty_like(r); h[0] = v0
        for t in range(1, len(r)):
            h[t] = w + a * r[t - 1] ** 2 + b * h[t - 1]
        return 0.5 * np.sum(np.log(h) + r ** 2 / h)

    best = None
    for a0, b0 in ((0.05, 0.9), (0.1, 0.8), (0.03, 0.95)):
        res = minimize(nll, [v0 * (1 - a0 - b0), a0, b0], method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, v0


def garch_sigma(R, dates_needed):
    out = {}
    idx = {d: i for i, d in enumerate(R.index)}
    params = {}
    for s in R.columns:
        r = (R[s] * 100).to_numpy()
        train = (R.index >= TRAIN_START) & (R.index < TRAIN_END)
        (w, a, b), v0 = garch_fit(r[train])
        params[s] = (float(w), float(a), float(b))
        first = np.flatnonzero(np.isfinite(r))[0]
        h = v0
        hs = np.full(len(r), np.nan)
        for t in range(first, len(r)):
            rt = r[t] if np.isfinite(r[t]) else 0.0
            h = w + a * rt ** 2 + b * h  # sigma^2_{t+1|t} from returns up to day t
            hs[t] = h
        for t in dates_needed:
            i = idx.get(t)
            if i is not None and np.isfinite(hs[i]):
                out[(t, s)] = float(math.sqrt(hs[i]))
    return pd.Series(out), params


def daily_ic(df, col, target="r"):
    ics = {}
    for t, g in df.groupby("date"):
        if g[col].nunique() > 1 and len(g) >= 5:
            ics[t] = spearmanr(g[col], g[target]).correlation
    return pd.Series(ics)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True, help="output of scripts/dump_predictions.py for the ten main runs")
    ap.add_argument("--price-dir", default="data/stocknet-dataset/price/raw")
    ap.add_argument("--ff-zip", default="data/fama_french/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    ap.add_argument("--out", default="outputs/econometrics/econ_axes.json")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    df, n_runs = load_heads(a.pred_dir)
    syms = sorted(df["symbol"].unique())
    px = load_prices(a.price_dir, syms)
    rt = returns_table(px, df[["date", "symbol"]])
    df = df.merge(rt[["date", "symbol", "r_cc"]], on=["date", "symbol"]).rename(columns={"r_cc": "r"})
    df["absr"] = df["r"].abs(); df["big"] = (df["absr"] >= 0.01).astype(int)
    dates = sorted(df["date"].unique())

    R = returns_panel(px)
    ff5 = read_ff(a.ff_zip).reindex(R.index)
    excess = R.sub(ff5["RF"], axis=0)
    tr = (R.index >= TRAIN_START) & (R.index < TRAIN_END)
    models = {}
    for name, cols in (("capm", ["Mkt-RF"]), ("ff3", ["Mkt-RF", "SMB", "HML"]), ("ff5", ["Mkt-RF", "SMB", "HML", "RMW", "CMA"])):
        lam = ff5.loc[tr, cols].mean().to_numpy()
        models[name] = rolling_beta_expected(excess, ff5[cols], lam, dates)
    core = [s for s in R.columns if R.loc[tr, s].notna().all()]
    models["pca_rev"] = pca_residual_reversal(R, dates, core=core)
    gsig, gparams = garch_sigma(R, dates)
    models["garch"] = gsig
    rolls = {}
    for s in R.columns:
        sd = R[s].rolling(20, min_periods=20).std()
        for t in dates:
            if t in sd.index and np.isfinite(sd.loc[t]):
                rolls[(t, s)] = float(sd.loc[t])
    models["rollsd"] = pd.Series(rolls)
    for k, s in models.items():
        s.index = pd.MultiIndex.from_tuples(s.index, names=["date", "symbol"])
        df = df.merge(s.rename(k).reset_index(), on=["date", "symbol"], how="left")

    res = {"n_runs": n_runs, "n_rows": int(len(df)), "n_days": len(dates), "garch_params_median": {
        k: float(np.median([v[i] for v in gparams.values()])) for i, k in enumerate(("omega", "alpha", "beta"))}}
    # ---- axis 1: expected returns (common rows)
    cols1 = ["pacmoe_ret", "pacmoe_p", "capm", "ff3", "ff5", "pca_rev"]
    a1 = df.dropna(subset=cols1 + ["r"]).copy()
    res["axis1_rows"] = int(len(a1))
    ics = {c: daily_ic(a1, c) for c in cols1}
    axis1 = {}
    for c in cols1:
        s = ics[c]
        top_bot = []
        for t, g in a1.groupby("date"):
            q = g[c].rank(pct=True)
            top_bot.append(g.loc[q > 0.8, "r"].mean() - g.loc[q <= 0.2, "r"].mean())
        axis1[c] = {"ic_mean": float(s.mean()), "ic_ci": boot_days(s.index, lambda d: s.reindex(d).mean(), rng),
                    "ic_positive_days": float((s > 0).mean()),
                    "sign_hit": float((np.sign(a1[c]) == np.sign(a1["r"])).mean()),
                    "top_minus_bottom_bps": float(np.nanmean(top_bot) * 1e4)}
    for c in ("capm", "ff3", "ff5", "pca_rev"):
        dd = ics["pacmoe_ret"] - ics[c]
        axis1[f"diff_pacmoe_ret_minus_{c}"] = {"mean": float(dd.mean()), "ci": boot_days(dd.index, lambda d: dd.reindex(d).mean(), rng)}
    res["axis1"] = axis1
    # ---- axis 2: volatility
    cols2 = ["pacmoe_vol", "garch", "rollsd"]
    a2 = df.dropna(subset=cols2 + ["absr"]).copy()
    res["axis2_rows"] = int(len(a2))
    axis2 = {}
    days2 = a2["date"].to_numpy()
    by_day = {d: np.flatnonzero(days2 == d) for d in np.unique(days2)}

    def rows_of(ds):
        return np.concatenate([by_day[d] for d in ds])
    for c in cols2:
        x, yb, ya = a2[c].to_numpy(), a2["big"].to_numpy(), a2["absr"].to_numpy()
        axis2[c] = {"auc_big": auc(x, yb), "auc_ci": boot_days(list(by_day), lambda d: auc(x[rows_of(d)], yb[rows_of(d)]), rng),
                    "spearman_absr": float(spearmanr(x, ya).correlation)}
    x0 = a2["pacmoe_vol"].to_numpy(); yb = a2["big"].to_numpy()
    for c in ("garch", "rollsd"):
        x1 = a2[c].to_numpy()
        axis2[f"diff_pacmoe_minus_{c}"] = {"mean": auc(x0, yb) - auc(x1, yb),
                                          "ci": boot_days(list(by_day), lambda d: auc(x0[rows_of(d)], yb[rows_of(d)]) - auc(x1[rows_of(d)], yb[rows_of(d)]), rng)}
    res["axis2"] = axis2
    res["big_rate"] = float(a2["big"].mean())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
