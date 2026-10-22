"""Direction-prediction baselines from econometric models (Section 3.5; response to Reviewer 2).

Logistic regressions of the next-day direction (same labels and samples as PaCMoE), C chosen by
validation AUC, test MCC at threshold 0.5 with day-block bootstrap intervals.
- M0: lagged own and market returns and own 5-day volatility.
- factor: F1 CAPM (equal-weighted market), F2 sector (leave-one-out sector mean minus market),
  F3 statistical factors (3 principal components estimated on the training period), optionally
  Fama-French (FF3, FF5, FF5M); features: factor lags, 60-day rolling betas, residual lags and volatility.
- dcc: M1 DCC-GARCH features (sigma, average correlation, correlation and beta to the market, market
  sigma; one-step forecasts with parameters estimated on the training period), M2 = M0 + M1.

usage: python -m analysis.direction_baselines factor --bundle data/preprocessed/main --all-symbols
       python -m analysis.direction_baselines dcc --bundle data/preprocessed/main --all-symbols
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analysis.common import PANEL_START, complete_symbols, locf_prices, log_return_panel
from analysis.dcc_garch import dcc_recursion, fit_dcc

FEATURES = ["log_sigma", "avg_corr", "corr_mkt", "beta_mkt", "log_sigma_mkt"]


def _load(bundle: str, split: str):
    """Close-return column of the 60-day window, dates, symbols and labels of a bundle split."""
    import torch

    p = torch.load(f"{bundle}/samples_{split}.pt", map_location="cpu", weights_only=False)
    k = p["x_ts"][..., :5].numpy().astype(np.float64)  # [N, 60, 5]: normalized open, high, low, close, volume
    close_ret = k[:, :, 3]
    dates = np.asarray(p["dates"])
    y = p["y_cls"].numpy().astype(np.int64)
    return close_ret, dates, np.asarray(p["symbols"]), y


def load_ff_csv(path) -> pd.DataFrame:
    """Daily CSV of the Kenneth French data library (percent)."""
    lines = open(path, encoding="latin-1").read().splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.startswith(","))
    cols = [c.strip() for c in lines[head].split(",")[1:]]
    rows, idx = [], []
    for ln in lines[head + 1:]:
        parts = [x.strip() for x in ln.split(",")]
        if len(parts[0]) != 8 or not parts[0].isdigit():
            break  # end of the daily table
        idx.append(pd.Timestamp(parts[0]))
        rows.append([float(x) for x in parts[1:]])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=cols)


def build_features(close_ret: np.ndarray, dates: np.ndarray, lags: int = 5):
    own = close_ret[:, -lags:][:, ::-1]  
    n = len(dates)
    mkt = np.zeros((n, lags))
    for u in np.unique(dates):
        m = dates == u
        mkt[m] = own[m].mean(axis=0, keepdims=True)  
    vol = own.std(axis=1, keepdims=True)
    return np.concatenate([own, mkt, vol], axis=1)


def _auc(p, y):
    r = np.empty(len(p))
    r[np.argsort(p)] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _mcc(p, y, thr=0.5):
    q = (p >= thr).astype(int)
    tp = ((q == 1) & (y == 1)).sum()
    tn = ((q == 0) & (y == 0)).sum()
    fp = ((q == 1) & (y == 0)).sum()
    fn = ((q == 0) & (y == 1)).sum()
    d = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return 0.0 if d == 0 else float((tp * tn - fp * fn) / d)


def _boot(p, y, d, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    U = np.unique(d)
    idx = {u: np.where(d == u)[0] for u in U}
    a = np.zeros(B)
    m = np.zeros(B)
    for b in range(B):
        ii = np.concatenate([idx[u] for u in rng.choice(U, len(U))])
        a[b] = _auc(p[ii], y[ii])
        m[b] = _mcc(p[ii], y[ii])
    return np.percentile(a, [2.5, 97.5]), np.percentile(m, [2.5, 97.5])


def load_sector_map(path: str) -> dict[str, str]:
    df = pd.read_csv(path, sep="\t")
    return {str(s).strip().lstrip("$"): str(sec).strip() for sec, s in zip(df["Sector"], df["Symbol"])}


def market_factor(R: pd.DataFrame) -> pd.Series:
    return R.mean(axis=1)


def sector_factor_loo(R: pd.DataFrame, sectors: dict[str, str], core=None) -> pd.DataFrame:
    core = list(R.columns) if core is None else list(core)
    mkt = market_factor(R[core])
    out = {}
    for s in R.columns:
        peers = [c for c in core if c != s and sectors[c] == sectors[s]]
        out[s] = R[peers].mean(axis=1) - mkt
    return pd.DataFrame(out, index=R.index)


def pca_factors(R: pd.DataFrame, train_mask: np.ndarray, k: int = 3):
    X = R.to_numpy(dtype=np.float64)
    mean = X[train_mask].mean(axis=0)
    std = X[train_mask].std(axis=0, ddof=1)
    Z = (X[train_mask] - mean) / std
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    loadings = Vt[:k].T
    for j in range(k):  
        if loadings[:, j].sum() < 0:
            loadings[:, j] *= -1
    f = ((X - mean) / std) @ loadings
    return pd.DataFrame(f, index=R.index, columns=[f"f{j}" for j in range(k)]), {"loadings": loadings, "mean": mean, "std": std}


def rolling_beta_residuals(y: pd.Series, F: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    yv = y.to_numpy(dtype=np.float64)
    Fv = F.to_numpy(dtype=np.float64)
    T, k = Fv.shape
    cols = ["alpha"] + [f"beta_{c}" for c in F.columns] + ["resid", "resid_vol"]
    out = np.full((T, len(cols)), np.nan)
    X_all = np.column_stack([np.ones(T), Fv])
    for t in range(window, T):
        X = X_all[t - window:t]
        yy = yv[t - window:t]
        if np.isnan(X).any() or np.isnan(yy).any() or np.isnan(yv[t]) or np.isnan(Fv[t]).any():
            continue
        coef = np.linalg.lstsq(X, yy, rcond=None)[0]
        e = yy - X @ coef
        out[t, :k + 1] = coef
        out[t, k + 1] = yv[t] - X_all[t] @ coef
        out[t, k + 2] = e.std(ddof=k + 1)
    return pd.DataFrame(out, index=y.index, columns=cols)


def factor_features(R: pd.DataFrame, kind: str, sectors: dict[str, str], train_mask: np.ndarray,
                    window: int = 60, lags: int = 5, k: int = 3, core=None,
                    external: pd.DataFrame | None = None, rf: pd.Series | None = None) -> pd.DataFrame:
    Rc = R if core is None else R[list(core)]
    mkt = market_factor(Rc)
    if kind == "capm":
        common = pd.DataFrame({"f0": mkt})
    elif kind == "pca":
        common, _ = pca_factors(Rc, train_mask, k=k)
    elif kind == "sector":
        sec = sector_factor_loo(R, sectors, core=core)
        common = None
    elif kind == "external":
        common = external.reindex(R.index)
    else:
        raise ValueError(kind)
    frames = []
    for s in R.columns:
        F = common if common is not None else pd.DataFrame({"f0": mkt, "f1": sec[s]})
        y = R[s] if rf is None else R[s] - rf.reindex(R.index)
        rb = rolling_beta_residuals(y, F, window=window)
        feats = {}
        for c in F.columns:
            for L in range(lags):
                feats[f"{c}_lag{L}"] = F[c].shift(L)
        for c in F.columns:
            feats[f"beta_{c}"] = rb[f"beta_{c}"]
        for L in range(lags):
            feats[f"resid_lag{L}"] = rb["resid"].shift(L)
        feats["resid_vol"] = rb["resid_vol"]
        df = pd.DataFrame(feats, index=R.index)
        df["symbol"] = s
        frames.append(df)
    out = pd.concat(frames)
    out.index.name = "date"
    return out.set_index("symbol", append=True).sort_index()


def common_start(R: pd.DataFrame, start):
    w = R.loc[R.index >= pd.Timestamp(start)]
    return w.index[w.notna().all(axis=1).to_numpy()][0]


def dcc_feature_frame(h: np.ndarray, R: np.ndarray, dates, symbols) -> pd.DataFrame:
    T, n = h.shape
    sd = np.sqrt(h)
    H = R * sd[:, :, None] * sd[:, None, :]
    w = np.full(n, 1.0 / n)
    cov_im = H @ w                              # (T, n)
    var_m = np.einsum("tij,i,j->t", H, w, w)    # (T,)
    feats = {
        "log_sigma": 0.5 * np.log(h),
        "avg_corr": (R.sum(axis=2) - 1.0) / (n - 1),
        "corr_mkt": cov_im / np.sqrt(h * var_m[:, None]),
        "beta_mkt": cov_im / var_m[:, None],
        "log_sigma_mkt": np.repeat(0.5 * np.log(var_m)[:, None], n, axis=1),
    }
    idx = pd.MultiIndex.from_product([pd.DatetimeIndex(dates), list(symbols)], names=["date", "symbol"])
    return pd.DataFrame({k: v.reshape(-1) for k, v in feats.items()}, index=idx)[FEATURES]


def _paired_diff_boot(metric, p_a, p_b, y, days, B, seed):
    rng = np.random.default_rng(seed)
    U = np.unique(days)
    idx = {u: np.where(days == u)[0] for u in U}
    diff = metric(p_a, y) - metric(p_b, y)
    d = np.zeros(B)
    for b in range(B):
        ii = np.concatenate([idx[u] for u in rng.choice(U, len(U))])
        d[b] = metric(p_a[ii], y[ii]) - metric(p_b[ii], y[ii])
    return float(diff), [float(v) for v in np.percentile(d, [2.5, 97.5])]


def paired_auc_diff_boot(p_a, p_b, y, days, B=2000, seed=0):
    return _paired_diff_boot(_auc, p_a, p_b, y, days, B, seed)


def paired_mcc_diff_boot(p_a, p_b, y, days, B=2000, seed=0):
    return _paired_diff_boot(_mcc, p_a, p_b, y, days, B, seed)


def _correlation_loglik_terms(eps: np.ndarray, R: np.ndarray) -> np.ndarray:
    _, logdet = np.linalg.slogdet(R)
    quad = np.einsum("ti,ti->t", eps, np.linalg.solve(R, eps[:, :, None])[:, :, 0])
    return -0.5 * (logdet + quad - np.einsum("ti,ti->t", eps, eps))


def factor_main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--raw-dir", default="data/stocknet-dataset/price/raw")
    ap.add_argument("--sector-table", default="data/stocknet-dataset/etc/StockTable")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--all-symbols", action="store_true",
                    help="also late-listed stocks (factors from the stocks with complete histories; no imputation)")
    ap.add_argument("--ff-dir", default=None, help="directory of the extracted Fama-French daily CSV files (models FF3 / FF5 / FF5M)")
    ap.add_argument("--models", default="F1,F2,F3", help="factor models (F1,F2,F3,FF3,FF5,FF5M); M0 is always fitted")
    a = ap.parse_args(argv)

    meta = json.load(open(f"{a.bundle}/metadata.json"))["split"]
    tr0, tr1, te1 = (pd.Timestamp(meta[k]) for k in ("train_start", "train_end", "test_end"))
    samples = {sp: _load(a.bundle, sp) for sp in ("train", "val", "test")}
    all_syms = sorted(set().union(*(set(s[2]) for s in samples.values())))
    P = pd.DataFrame({s: pd.read_csv(f"{a.raw_dir}/{s}.csv", parse_dates=["Date"]).set_index("Date")["Adj Close"]
                      for s in all_syms}).sort_index()
    Rp = log_return_panel(locf_prices(P.loc[P.index < te1]))
    core = complete_symbols(Rp, tr0, te1)
    syms = all_syms if a.all_symbols else core
    R = Rp.loc[(Rp.index >= pd.Timestamp(PANEL_START)) & (Rp.index < te1), syms]
    assert not R[core].isna().any().any()
    sectors = load_sector_map(a.sector_table)
    train_mask = np.asarray((R.index >= tr0) & (R.index < tr1))

    models = a.models.split(",")
    feats = {name: factor_features(R, kind, sectors, train_mask, window=a.window, k=a.k,
                                   core=core if a.all_symbols else None)
             for name, kind in (("F1", "capm"), ("F2", "sector"), ("F3", "pca")) if name in models}
    if any(m.startswith("FF") for m in models):
        ff3 = load_ff_csv(f"{a.ff_dir}/F-F_Research_Data_Factors_daily.csv")
        ff5 = load_ff_csv(f"{a.ff_dir}/F-F_Research_Data_5_Factors_2x3_daily.csv")
        mom = load_ff_csv(f"{a.ff_dir}/F-F_Momentum_Factor_daily.csv")
        mom.columns = ["Mom"]
        spec = {"FF3": (ff3[["Mkt-RF", "SMB", "HML"]], ff3["RF"]),
                "FF5": (ff5[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]], ff5["RF"]),
                "FF5M": (ff5[["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].join(mom, how="inner"), ff5["RF"])}
        out_ff = {}
        for m in models:
            if m in spec:
                F, rf = spec[m]
                miss = R.index.difference(F.index)
                out_ff[m] = {"panel_days_missing_in_ff": [str(x.date()) for x in miss]}
                feats[m] = factor_features(R, "external", sectors, train_mask, window=a.window, external=F, rf=rf)
    out = {"symbols": len(syms), "core_symbols": len(core), "all_symbols": a.all_symbols, "panel_start": PANEL_START,
           "window": a.window, "k": a.k, "feature_columns": {n: list(f.columns) for n, f in feats.items()}}
    if any(m.startswith("FF") for m in models):
        out["ff"] = out_ff

    S = {}
    n_drop_feat = {}
    for sp, (cr, d, sy, y) in samples.items():
        keep = np.isin(sy, syms)
        idx = pd.MultiIndex.from_arrays([pd.to_datetime([str(x)[:10] for x in d[keep]]), sy[keep]])
        Xs = {n: f.reindex(idx) for n, f in feats.items()}
        ok = np.logical_and.reduce([~X.isna().any(axis=1).to_numpy() for X in Xs.values()])
        if sp != "train":
            assert ok.all(), f"{sp}: missing features for {int((~ok).sum())} samples"
        
        n_drop_feat[sp] = int((~ok).sum())
        S[sp] = {"M0": build_features(cr[keep], d[keep])[ok], "y": y[keep][ok], "d": d[keep][ok], "s": sy[keep][ok]}
        for n, X in Xs.items():
            S[sp][n] = X.to_numpy()[ok]
    out["n_dropped_symbols"] = {sp: int((~np.isin(samples[sp][2], syms)).sum()) for sp in samples}
    out["n_dropped_no_features"] = n_drop_feat

    preds = {}
    for name in ["M0"] + [m for m in models if m in feats]:
        sc = StandardScaler().fit(S["train"][name])
        X = {sp: sc.transform(S[sp][name]) for sp in S}
        best = None
        for C in (0.001, 0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=C, max_iter=5000).fit(X["train"], S["train"]["y"])
            v = _auc(clf.predict_proba(X["val"])[:, 1], S["val"]["y"])
            if best is None or v > best[0]:
                best = (v, C, clf)
        _, C, clf = best
        res = {"C": C}
        for sp in ("val", "test"):
            p = clf.predict_proba(X[sp])[:, 1]
            y, d = S[sp]["y"], S[sp]["d"]
            aci, mci = _boot(p, y, d)
            res[sp] = {"mcc": _mcc(p, y), "mcc_ci95": mci.tolist(), "acc": float(((p >= 0.5) == y).mean()),
                       "auc": _auc(p, y), "auc_ci95": aci.tolist(), "n": int(len(y)), "up_rate": float(y.mean()),
                       "pred_up_rate": float((p >= 0.5).mean())}
            preds[(name, sp)] = p
        out[name] = res
        print(name, json.dumps(res), flush=True)

    for sp in ("val", "test"):
        for name in [m for m in models if m in feats]:
            dm, cm = paired_mcc_diff_boot(preds[(name, sp)], preds[("M0", sp)], S[sp]["y"], S[sp]["d"])
            da, ca = paired_auc_diff_boot(preds[(name, sp)], preds[("M0", sp)], S[sp]["y"], S[sp]["d"])
            out[f"diff_{name}_minus_M0_{sp}"] = {"mcc": dm, "mcc_ci95": cm, "auc": da, "auc_ci95": ca}
            print(sp, name, "- M0  MCC", dm, cm, " AUC", da, ca, flush=True)

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0


def dcc_main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--raw-dir", default="data/stocknet-dataset/price/raw")
    ap.add_argument("--out", default=None)
    ap.add_argument("--all-symbols", action="store_true",
                    help="also late-listed stocks, estimated from the first day on which all stocks are observed (no imputation)")
    a = ap.parse_args(argv)

    meta = json.load(open(f"{a.bundle}/metadata.json"))["split"]
    tr0, tr1, te1 = (pd.Timestamp(meta[k]) for k in ("train_start", "train_end", "test_end"))

    samples = {sp: _load(a.bundle, sp) for sp in ("train", "val", "test")}
    all_syms = sorted(set().union(*(set(s[2]) for s in samples.values())))

    P = pd.DataFrame({s: pd.read_csv(f"{a.raw_dir}/{s}.csv", parse_dates=["Date"]).set_index("Date")["Adj Close"]
                      for s in all_syms}).sort_index()
    P = P.loc[P.index < te1]
    Rp = log_return_panel(locf_prices(P))
    est0 = tr0
    if a.all_symbols:
        est0 = common_start(Rp[all_syms], tr0)
    syms = complete_symbols(Rp, est0, te1)
    excluded = sorted(set(all_syms) - set(syms))
    Rp = Rp.loc[(Rp.index >= est0) & (Rp.index < te1), syms]
    assert not Rp.isna().any().any()
    is_train = Rp.index < tr1
    print(f"symbols={len(syms)} excluded={excluded} days={len(Rp)} train_days={int(is_train.sum())}", flush=True)

    
    model = fit_dcc(Rp.loc[is_train].to_numpy(), mean="constant")
    h_next, R_next = model.one_step_forecasts(Rp.to_numpy())
    feat = dcc_feature_frame(h_next, R_next, Rp.index, syms)

    
    eps = model.standardized_residuals(Rp.to_numpy())
    _, R_dcc = dcc_recursion(eps, model.a, model.b, model.S, Q0=model.Q0)
    _, R_ccc = dcc_recursion(eps, 0.0, 0.0, model.S)
    lc_dcc = _correlation_loglik_terms(eps, R_dcc[:-1])
    lc_ccc = _correlation_loglik_terms(eps, R_ccc[:-1])
    va1 = pd.Timestamp(meta["val_end"])
    seg = {"train": is_train, "val": (Rp.index >= tr1) & (Rp.index < va1), "test": Rp.index >= va1}
    out = {
        "symbols": len(syms), "excluded": excluded, "estimation_start": str(est0.date()), "days": len(Rp), "train_days": int(is_train.sum()),
        "dcc": {"a": model.a, "b": model.b, "a_plus_b": model.a + model.b,
                "garch_alpha_median": float(np.median(model.kappa)), "garch_beta_median": float(np.median(model.lam)),
                "garch_persistence_min_med_max": np.quantile(model.kappa + model.lam, [0, .5, 1]).round(4).tolist(),
                "S_offdiag_mean": float((model.S.sum() - len(syms)) / (len(syms) * (len(syms) - 1)))},
        "lc_dcc_minus_ccc": {k: float(lc_dcc[m].sum() - lc_ccc[m].sum()) for k, m in seg.items()},
        "lc_days": {k: int(m.sum()) for k, m in seg.items()},
    }
    print(json.dumps(out), flush=True)

    
    S = {}
    for sp, (cr, d, sy, y) in samples.items():
        keep = np.isin(sy, syms)
        dt = pd.to_datetime([str(x)[:10] for x in d[keep]])
        f = feat.reindex(pd.MultiIndex.from_arrays([dt, sy[keep]]))
        ok = ~f.isna().any(axis=1).to_numpy()
        if sp != "train":
            assert ok.all(), f"{sp}: missing DCC features"
        
        f = f.to_numpy()[ok]
        x0 = build_features(cr[keep], d[keep])[ok]
        S[sp] = {"M0": x0, "M1": f, "M2": np.concatenate([x0, f], axis=1),
                 "y": y[keep][ok], "d": d[keep][ok], "s": sy[keep][ok], "n_dropped": int((~keep).sum()),
                 "n_dropped_before_estimation": int((~ok).sum())}
    out["n_dropped"] = {sp: S[sp]["n_dropped"] for sp in S}
    out["n_dropped_before_estimation"] = {sp: S[sp]["n_dropped_before_estimation"] for sp in S}

    preds = {}
    for name in ("M0", "M1", "M2"):
        sc = StandardScaler().fit(S["train"][name])
        X = {sp: sc.transform(S[sp][name]) for sp in S}
        best = None
        for C in (0.001, 0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=C, max_iter=5000).fit(X["train"], S["train"]["y"])
            v = _auc(clf.predict_proba(X["val"])[:, 1], S["val"]["y"])
            if best is None or v > best[0]:
                best = (v, C, clf)
        v_auc, C, clf = best
        res = {"C": C}
        for sp in ("val", "test"):
            p = clf.predict_proba(X[sp])[:, 1]
            y, d = S[sp]["y"], S[sp]["d"]
            aci, mci = _boot(p, y, d)
            res[sp] = {"auc": _auc(p, y), "mcc": _mcc(p, y), "auc_ci95": aci.tolist(), "mcc_ci95": mci.tolist(),
                       "n": int(len(y)), "up_rate": float(y.mean())}
            preds[(name, sp)] = p
        out[name] = res
        print(name, json.dumps(res), flush=True)

    for sp in ("val", "test"):
        diff, ci = paired_auc_diff_boot(preds[("M2", sp)], preds[("M0", sp)], S[sp]["y"], S[sp]["d"])
        out[f"auc_diff_M2_minus_M0_{sp}"] = {"diff": diff, "ci95": ci}
        print(sp, "M2-M0 AUC", diff, ci, flush=True)
        
        for x, z in (("M2", "M0"), ("M1", "M0")):
            dm, cm = paired_mcc_diff_boot(preds[(x, sp)], preds[(z, sp)], S[sp]["y"], S[sp]["d"])
            out[f"mcc_diff_{x}_minus_{z}_{sp}"] = {"diff": dm, "ci95": cm}
            print(sp, f"{x}-{z} MCC", dm, cm, flush=True)

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("factor", "dcc"):
        raise SystemExit("usage: python -m analysis.direction_baselines {factor,dcc} --bundle <dir> [...]")
    return factor_main(argv[1:]) if argv[0] == "factor" else dcc_main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
