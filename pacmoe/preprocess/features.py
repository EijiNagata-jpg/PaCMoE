"""Price features of the 60-day input windows (Section 2.2.1): 6 price columns + 116 indicators.

1. Indicators are computed on each stock's full dividend-adjusted history (`compute_indicators`).
2. Normalization statistics are fitted on the training-period rows of each stock only:
   open/high/low/close/amount -> daily log return; volume -> z-score of log1p(volume);
   indicators -> z-score (11 slowly varying indicators are differenced first).
3. Three derived features (volatility acceleration, 59-day OBV change, MFI x OBV) and the
   fractionally differenced log close (FFD; the order d is chosen per stock on the training rows by
   an ADF test) are appended.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from pacmoe.preprocess.ta.frac_diff import fracDiff_FFD, find_min_d
from pacmoe.preprocess.ta.indicator_kernels import compute_cci_mad, compute_obv
from pacmoe.preprocess.ta.technical_indicators import apo, compute_extended_indicators, natr, t3, tema, trima, willr

PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
AUTOCORR_DIFF_FEATURES = ["vwap_cumulative", "ema_signal", "sma_signal", "sma_20", "ema_20", "vwap_window",
                          "ichimoku_span_a", "ichimoku_span_b", "ad_signal", "ichimoku_base", "ichimoku_conversion"]
EXTENDED_INDICATORS = {
    "ab_ratio": {"period": 14}, "adxr": {"period": 14}, "aroon": {"period": 14}, "binary_wave": {"sma_period": 20},
    "chaikin_oscillator": {"long_period": 10, "short_period": 3}, "dema": {"period": 10}, "deviation_rate": {"period": 20},
    "dpo": {"period": 20}, "emv": {"period": 14}, "enabled": True, "energy": {"period": 14},
    "envelope": {"period": 20, "rate": 0.05}, "ht_dcperiod": {"enabled": False}, "lri": {"period": 14}, "lrs": {"period": 14},
    "mao": {"long_period": 26, "short_period": 12}, "mass_index": {"ema_period": 9, "sum_period": 25},
    "normalize_by_close": True, "oscp": {"long_period": 26, "short_period": 12}, "oscv": {"long_period": 26, "short_period": 12},
    "parabolic_sar": {"af_max": 0.2, "af_start": 0.02}, "psychological_line": {"period": 12}, "r_squared": {"period": 14},
    "rci": {"period": 9}, "sigma": {"period": 20}, "sonar": {"period": 10}, "standard_error": {"period": 14},
    "stddev": {"period": 20}, "trix": {"period": 14}, "tsf": {"period": 14}, "tsf_oscillator": {"long_period": 20, "short_period": 5},
    "tsi": {"long_period": 25, "short_period": 13}, "vhf": {"period": 28}, "volume_macd": {"long_period": 26, "short_period": 12},
    "volume_ratio": {"period": 26}, "volume_rsi": {"period": 14}, "vroc": {"period": 14}, "safe_division": False,
}
FFD_THRESHOLD = 1e-5
FFD_P_VALUE = 0.05
NEW_FEATURES = ("vol_acceleration", "obv_trend_60", "mfi_x_obv")


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append the technical indicators to an OHLCV(+amount) frame. per indicator."""
    df = df.copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    ema12, ema26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema12 - ema26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    df["sma_20"] = close.rolling(20).mean()
    df["sma_signal"] = close.rolling(50).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_signal"] = close.ewm(span=50, adjust=False).mean()
    sma20, std20 = close.rolling(20).mean(), close.rolling(20).std()
    df["bb_upper"], df["bb_lower"] = sma20 + 2 * std20, sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    df["bb_pct_b"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)
    alpha = 1.0 / 14
    df["tr"] = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    df["atr"] = df["tr"].ewm(alpha=alpha, adjust=False).mean()
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=alpha, adjust=False).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=alpha, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-9))
    tp = (high + low + close) / 3
    mad = tp.rolling(20).apply(compute_cci_mad, raw=True)
    df["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * mad + 1e-9)
    df["momentum"] = close - close.shift(10)
    df["momentum_signal"] = df["momentum"].ewm(span=9, adjust=False).mean()
    df["roc"] = close.pct_change(10) * 100
    df["roc_signal"] = df["roc"].ewm(span=9, adjust=False).mean()
    df["obv"] = compute_obv(close.to_numpy(dtype=np.float64), volume.to_numpy(dtype=np.float64))
    df["vwap_window"] = (tp * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    df["vwap_cumulative"] = (tp * volume).cumsum() / (volume.cumsum() + 1e-9)
    money_flow = tp * volume
    pos_mf = pd.Series(np.where(tp > tp.shift(1), money_flow, 0), index=df.index)
    neg_mf = pd.Series(np.where(tp < tp.shift(1), money_flow, 0), index=df.index)
    df["mfi"] = 100 - 100 / (1 + pos_mf.rolling(14).sum() / (neg_mf.rolling(14).sum() + 1e-9))
    clv = ((close - low) - (high - close)) / (high - low + 1e-9)
    df["ad"] = (clv * volume).cumsum()
    df["ad_signal"] = df["ad"].ewm(span=9, adjust=False).mean()
    rsi14 = df["rsi"]
    min_rsi, max_rsi = rsi14.rolling(14).min(), rsi14.rolling(14).max()
    df["stoch_rsi"] = (rsi14 - min_rsi) / (max_rsi - min_rsi + 1e-9)
    df["stoch_rsi_signal"] = df["stoch_rsi"].rolling(3).mean()
    lowest, highest = low.rolling(14).min(), high.rolling(14).max()
    df["stoch_k"] = (close - lowest) / (highest - lowest + 1e-9) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["stoch_d_signal"] = df["stoch_d"].ewm(span=3, adjust=False).mean()
    plus_dm = pd.Series(np.where((high.diff() > -low.diff()) & (high.diff() > 0), high.diff(), 0), index=df.index)
    minus_dm = pd.Series(np.where((-low.diff() > high.diff()) & (-low.diff() > 0), -low.diff(), 0), index=df.index)
    tr14 = df["tr"].ewm(span=14, adjust=False).mean()
    df["dmi_plus"] = plus_dm.ewm(span=14, adjust=False).mean() / (tr14 + 1e-9) * 100
    df["dmi_minus"] = minus_dm.ewm(span=14, adjust=False).mean() / (tr14 + 1e-9) * 100
    dx = (df["dmi_plus"] - df["dmi_minus"]).abs() / (df["dmi_plus"] + df["dmi_minus"] + 1e-9) * 100
    df["adx"] = dx.ewm(span=14, adjust=False).mean()
    df["ichimoku_conversion"] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    df["ichimoku_base"] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    df["ichimoku_span_a"] = ((df["ichimoku_conversion"] + df["ichimoku_base"]) / 2).shift(26)
    df["ichimoku_span_b"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    df["natr"] = natr(high, low, close, period=14)
    df["willr"] = willr(high, low, close, period=14, safe=False)
    denom = close.clip(lower=1e-10)
    df["tema"] = tema(close, 10) / denom
    df["trima"] = trima(close, 10) / denom
    df["t3"] = t3(close, 5, 0.7) / denom
    df["apo"] = apo(close, 12, 26) / denom
    log_ret = np.log(close.clip(lower=1e-10) / close.shift(1).clip(lower=1e-10))
    for p in (5, 10, 20, 60):
        df[f"ret_{p}d"] = close.pct_change(p)
    for p in (5, 20):
        df[f"realized_vol_{p}"] = log_ret.rolling(p).std()
    df["skew_20"] = log_ret.rolling(20).skew()
    df["kurt_20"] = log_ret.rolling(20).kurt()
    lr = log_ret.fillna(0.0).to_numpy(dtype=np.float64)
    dd = np.full(len(lr), np.nan, dtype=np.float64)
    for i in range(19, len(lr)):
        cs = np.cumsum(lr[i - 19: i + 1])
        dd[i] = float(np.min(cs - np.maximum.accumulate(cs)))
    df["max_drawdown_20"] = pd.Series(dd, index=df.index)
    hi52, lo52 = high.rolling(252).max().clip(lower=1e-10), low.rolling(252).min().clip(lower=1e-10)
    df["pct_from_52w_high"] = (close - hi52) / hi52
    df["pct_from_52w_low"] = (close - lo52) / lo52
    df["hl_ratio"] = (high - low) / close.clip(lower=1e-10)
    df["overnight_gap"] = np.log(df["open"].clip(lower=1e-10) / close.shift(1).clip(lower=1e-10))
    df["volume_ratio_20"] = volume / volume.rolling(20).mean().clip(lower=1e-10)
    df["amihud_illiquidity"] = log_ret.abs() / volume.clip(lower=1.0)
    clv_mf = ((close - low) - (high - close)) / (high - low + 1e-9) * volume
    df["chaikin_mf"] = clv_mf.rolling(21).sum() / volume.rolling(21).sum().clip(lower=1e-10)
    dow = df.index.dayofweek.astype(float)
    df["day_of_week_sin"] = np.sin(2.0 * np.pi * dow / 5.0)
    df["day_of_week_cos"] = np.cos(2.0 * np.pi * dow / 5.0)
    return compute_extended_indicators(df, EXTENDED_INDICATORS)


class PriceFeatureTransform:
    """Per-stock normalization fitted on the training rows (no information from later periods)."""

    def __init__(self) -> None:
        self.stats: Dict[str, Dict[str, float]] = {}
        self.ffd_order: Dict[str, float] = {}
        self.columns: List[str] = []

    @staticmethod
    def _normalized(df: pd.DataFrame, stats: Dict[str, float], normalize_new: bool) -> pd.DataFrame:
        out = df.copy()
        for col in out.columns:
            if col in PRICE_COLUMNS and col != "volume":
                prices = out[col].replace(0.0, np.nan)
                out[col] = np.log(prices / prices.shift(1))
            elif col == "volume":
                std = stats.get("volume_std", 1.0)
                out[col] = (np.log1p(out[col].clip(lower=0)) - stats.get("volume_mean", 0.0)) / (1.0 if std == 0 or np.isnan(std) else std)
            else:
                if col in AUTOCORR_DIFF_FEATURES:
                    out[col] = out[col].diff()
                std = stats.get(f"{col}_std", 1.0)
                out[col] = (out[col] - stats.get(f"{col}_mean", 0.0)) / (1.0 if std == 0 or np.isnan(std) else std)
        out = out.ffill().fillna(0.0)
        new = {}
        tr = out["tr"].copy()
        va = tr.rolling(10).mean() - tr.rolling(10).mean().shift(50)
        ot = out["obv"].diff(59)
        for name, series in (("vol_acceleration", va), ("obv_trend_60", ot)):
            if normalize_new:
                std = stats.get(f"{name}_std", 1.0)
                new[name] = ((series - stats.get(f"{name}_mean", 0.0)) / (1.0 if std == 0 or np.isnan(std) else std)).fillna(0.0)
            else:
                new[name] = series
        new["mfi_x_obv"] = out["mfi"] * out["obv"]
        return pd.concat([out.drop(columns=list(new), errors="ignore"), pd.DataFrame(new, index=out.index)], axis=1).copy()

    def fit(self, train_frames: Dict[str, pd.DataFrame]) -> "PriceFeatureTransform":
        """Means / standard deviations (ddof=0) per stock and column on the training rows."""
        for sym, df in train_frames.items():
            if len(df) == 0:
                continue
            st: Dict[str, float] = {}
            for col in df.columns:
                s = df[col].astype(float)
                if col in PRICE_COLUMNS and col != "volume":
                    continue
                if col == "volume":
                    s = np.log1p(s.clip(lower=0))
                elif col in AUTOCORR_DIFF_FEATURES:
                    s = s.diff()
                st[f"{col}_mean"], st[f"{col}_std"] = float(s.mean()), float(s.std(ddof=0))
            self.stats[sym] = st
            log_price = np.log(df["close"].astype(float).ffill().dropna().clip(lower=1e-10))
            self.ffd_order[sym] = find_min_d(log_price, ds=None, thres=FFD_THRESHOLD, p_value_threshold=FFD_P_VALUE)
            derived = self._normalized(df, st, normalize_new=False)
            for name in ("vol_acceleration", "obv_trend_60"):
                s = pd.to_numeric(derived[name], errors="coerce")
                st[f"{name}_mean"], st[f"{name}_std"] = float(s.mean()), float(s.std(ddof=0))
        first = next(iter(train_frames.values()))
        base = [c for c in first.columns]
        self.columns = [c for c in base if c not in NEW_FEATURES] + list(NEW_FEATURES) + ["close_ffd"]
        return self

    def transform(self, frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Normalized features with columns `self.columns`."""
        out = {}
        for sym, df in frames.items():
            if sym not in self.stats or len(df) == 0:
                continue
            norm = self._normalized(df, self.stats[sym], normalize_new=True)
            log_price = np.log(df["close"].astype(float).ffill().clip(lower=1e-10))
            ffd = fracDiff_FFD(log_price, self.ffd_order[sym], thres=FFD_THRESHOLD).reindex(norm.index).fillna(0.0)
            norm = pd.concat([norm.drop(columns=["close_ffd"], errors="ignore"), ffd.to_frame(name="close_ffd")], axis=1)
            out[sym] = norm.reindex(columns=self.columns, fill_value=0.0)
        return out
