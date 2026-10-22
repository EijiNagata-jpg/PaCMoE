"""Labels and regression/volatility targets (Sections 2.1 and 2.6).

For sample date t (the last day of the input window):
  r_t     = 100 * (P_{t+1} - P_t) / P_t on dividend-adjusted closes, clipped to [-5, 5]
  y_cls   = 1 if r_t >= up_threshold, 0 if r_t <= down_threshold; other days are removed
            (setup "main"; with use_neutral the three classes 0 / 1 / 2 = down / up / neutral)
  y_ret   = clip(r_t / max(sigma_t, 0.25), -5, 5); sigma_t = std of the daily returns (%) over the
            20 days up to t (at least 5 observations)
  y_vol   = 1 if |r_t| >= vol_threshold (target "fixed"); |r_t| >= max(sigma_t, 0.25) with the
            unclipped r_t ("scaled");
            |P_{t+1} - P_t| >= vol_threshold (setup "ablation")
"""
from __future__ import annotations

import pandas as pd

VOL_WINDOW = 20
VOL_MIN_PERIODS = 5
VOL_FLOOR = 0.25
CLIP = 5.0


def return_targets(close: pd.Series) -> pd.DataFrame:
    """Per-date next-day return (%), its volatility-normalized version and sigma_t."""
    close = close.astype(float)
    raw = (close.shift(-1) - close) / close * 100.0
    hist = close.pct_change() * 100.0
    sigma = hist.rolling(window=VOL_WINDOW, min_periods=VOL_MIN_PERIODS).std()
    denom = sigma.abs().clip(lower=VOL_FLOOR).fillna(VOL_FLOOR)
    return pd.DataFrame({
        "y_ret_h1": raw.clip(-CLIP, CLIP),
        "y_ret_h1_vol_normalized": (raw / denom).clip(-CLIP, CLIP),
        "sigma": sigma,
        "r_unclipped": raw,
        "price_diff": close.shift(-1) - close,
    })


def labels(close: pd.Series, up: float, down: float, vol_threshold: float = 1.0, use_neutral: bool = False,
           vol_target: str = "fixed") -> pd.DataFrame:
    """Labelled rows (neutral rows dropped unless use_neutral)."""
    frame = return_targets(close)
    r = frame["y_ret_h1"]
    frame = frame.loc[r.notna()].copy()
    r = frame["y_ret_h1"]
    up_mask, down_mask = r >= up, r <= down
    if use_neutral:
        frame["y_cls"] = 2
        frame.loc[up_mask, "y_cls"] = 1
        frame.loc[down_mask, "y_cls"] = 0
    else:
        frame = frame.loc[up_mask | down_mask].copy()
        frame["y_cls"] = (frame["y_ret_h1"] >= up).astype(int)
    if vol_target == "scaled":
        scale = frame["sigma"].abs().clip(lower=VOL_FLOOR).fillna(VOL_FLOOR)
        frame["y_vol"] = (frame["r_unclipped"].abs() >= scale).astype(int)
    elif vol_target == "abs_price_diff":
        frame["y_vol"] = (frame["price_diff"].abs() >= vol_threshold).astype(int)
    else:
        frame["y_vol"] = (frame["y_ret_h1"].abs() >= vol_threshold).astype(int)
    return frame
