"""Technical indicators used as price features (the extended set of the original preprocessing).

The column list is `pacmoe.preprocess.features.EXTENDED_INDICATORS`; every indicator is computed from
data up to the current day only.
"""
from __future__ import annotations
import logging
from typing import Dict, Optional
from pacmoe.preprocess.ta.safe_math import price_tol, safe_div
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range = max(H-L, |H-Cprev|, |L-Cprev|)。"""
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int=14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(span=period, adjust=False).mean()

def natr(high: pd.Series, low: pd.Series, close: pd.Series, period: int=14) -> pd.Series:
    a = atr(high, low, close, period=period)
    return a / close.clip(lower=1e-10) * 100.0

def willr(high: pd.Series, low: pd.Series, close: pd.Series, period: int=14, safe: bool=False) -> pd.Series:
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    if safe:
        return safe_div(close - highest_high, highest_high - lowest_low, tol=price_tol(close), positive_only=True) * 100.0
    return (close - highest_high) / (highest_high - lowest_low + 1e-12) * 100.0

def ema(series: pd.Series, period: int=10) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def tema(series: pd.Series, period: int=10) -> pd.Series:
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3

def trima(series: pd.Series, period: int=10) -> pd.Series:
    first_period = period // 2 + 1
    second_period = period - first_period + 1
    sma1 = series.rolling(first_period).mean()
    return sma1.rolling(second_period).mean()

def t3(series: pd.Series, period: int=5, v_factor: float=0.7) -> pd.Series:

    def gd(x: pd.Series) -> pd.Series:
        e1 = ema(x, period)
        e2 = ema(e1, period)
        return (1 + v_factor) * e1 - v_factor * e2
    return gd(gd(gd(series)))

def apo(series: pd.Series, fast_period: int=12, slow_period: int=26) -> pd.Series:
    return ema(series, fast_period) - ema(series, slow_period)

def dema(series: pd.Series, period: int=10) -> pd.Series:
    ema1 = ema(series, period)
    ema2 = ema(ema1, period)
    return 2.0 * ema1 - ema2

def _linreg_coeffs(y: np.ndarray) -> tuple:
    n = len(y)
    x = np.arange(1, n + 1, dtype=np.float64)
    sx = x.sum()
    sy = y.sum()
    sxy = (x * y).sum()
    sxx = (x * x).sum()
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return (0.0, float(y[-1]))
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return (slope, intercept)

def lri(series: pd.Series, period: int=14) -> pd.Series:

    def _val(y: np.ndarray) -> float:
        s, i = _linreg_coeffs(y)
        return i + s * len(y)
    return series.rolling(period).apply(_val, raw=True)

def lrs(series: pd.Series, period: int=14) -> pd.Series:

    def _slope(y: np.ndarray) -> float:
        return _linreg_coeffs(y)[0]
    return series.rolling(period).apply(_slope, raw=True)

def tsf(series: pd.Series, period: int=14) -> pd.Series:

    def _tsf_val(y: np.ndarray) -> float:
        s, i = _linreg_coeffs(y)
        return i + s * (len(y) + 1)
    return series.rolling(period).apply(_tsf_val, raw=True)

def tsf_oscillator(series: pd.Series, short_period: int=5, long_period: int=20) -> pd.Series:
    return tsf(series, short_period) - tsf(series, long_period)

def r_squared(series: pd.Series, period: int=14) -> pd.Series:

    def _rsq(y: np.ndarray) -> float:
        n = len(y)
        x = np.arange(1, n + 1, dtype=np.float64)
        sx = x.sum()
        sy = y.sum()
        sxy = (x * y).sum()
        sxx = (x * x).sum()
        syy = (y * y).sum()
        n_inv = 1.0 / n
        q1 = sxy - sx * sy * n_inv
        q2 = sxx - sx * sx * n_inv
        q3 = syy - sy * sy * n_inv
        denom = q2 * q3
        return 0.0 if abs(denom) < 1e-12 else q1 * q1 / denom
    return series.rolling(period).apply(_rsq, raw=True)

def standard_error(series: pd.Series, period: int=14) -> pd.Series:

    def _se(y: np.ndarray) -> float:
        n = len(y)
        if n < 3:
            return 0.0
        x = np.arange(1, n + 1, dtype=np.float64)
        sx = x.sum()
        sy = y.sum()
        sxy = (x * y).sum()
        sxx = (x * x).sum()
        syy = (y * y).sum()
        a_v = n * syy - sy * sy
        b_v = n * sxy - sx * sy
        c_v = n * sxx - sx * sx
        nn = 1.0 / (n * (n - 2))
        inner = nn * (a_v - (b_v * b_v / c_v if abs(c_v) > 1e-12 else 0.0))
        return float(np.sqrt(max(inner, 0.0)))
    return series.rolling(period).apply(_se, raw=True)

def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    ph = high.shift(1)
    pl = low.shift(1)
    pc = close.shift(1)
    pp = (ph + pl + pc) / 3.0
    r1 = 2.0 * pp - pl
    s1 = 2.0 * pp - ph
    r2 = pp + (ph - pl)
    s2 = pp - (ph - pl)
    return pd.DataFrame({'pivot': pp, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2}, index=close.index)

def envelope(series: pd.Series, period: int=20, rate: float=0.05) -> pd.DataFrame:
    sma_n = series.rolling(period).mean()
    return pd.DataFrame({'envelope_upper': sma_n * (1.0 + rate), 'envelope_lower': sma_n * (1.0 - rate)}, index=series.index)

def parabolic_sar(high: pd.Series, low: pd.Series, af_start: float=0.02, af_max: float=0.2) -> pd.Series:
    hi = high.to_numpy(dtype=np.float64)
    lo = low.to_numpy(dtype=np.float64)
    n = len(hi)
    sar_out = np.empty(n, dtype=np.float64)
    sar_out[0] = lo[0]
    bull = True
    ep = hi[0]
    af = af_start
    for i in range(1, n):
        prev_sar = sar_out[i - 1]
        if bull:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = min(new_sar, lo[i - 1], lo[max(0, i - 2)])
            if lo[i] < new_sar:
                bull = False
                new_sar = ep
                ep = lo[i]
                af = af_start
            elif hi[i] > ep:
                ep = hi[i]
                af = min(af + af_start, af_max)
        else:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = max(new_sar, hi[i - 1], hi[max(0, i - 2)])
            if hi[i] > new_sar:
                bull = True
                new_sar = ep
                ep = hi[i]
                af = af_start
            elif lo[i] < ep:
                ep = lo[i]
                af = min(af + af_start, af_max)
        sar_out[i] = new_sar
    return pd.Series(sar_out, index=high.index)

def td_range_projections(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    po = open_.shift(1)
    ph = high.shift(1)
    pl = low.shift(1)
    pc = close.shift(1)
    is_bull = pc > po
    is_bear = pc < po
    x = pd.Series(np.where(is_bear.to_numpy(), (po + ph + pc + pl) / 2.0, np.where(is_bull.to_numpy(), (po + ph + pc + ph) / 2.0, (po + ph + pc + pc) / 2.0)), index=close.index)
    return pd.DataFrame({'td_upper': x - pl, 'td_lower': x - ph}, index=close.index)

def mao(series: pd.Series, short_period: int=12, long_period: int=26) -> pd.Series:
    return series.rolling(long_period).mean() - series.rolling(short_period).mean()

def oscp(series: pd.Series, short_period: int=12, long_period: int=26) -> pd.Series:
    short_ema = ema(series, short_period)
    long_ema = ema(series, long_period)
    return (short_ema - long_ema) / short_ema.clip(lower=1e-10) * 100.0

def sonar(series: pd.Series, period: int=10) -> pd.Series:
    sma_n = series.rolling(period).mean()
    return (sma_n - sma_n.shift(1)) / sma_n.shift(1).clip(lower=1e-10) * 100.0

def trix(series: pd.Series, period: int=14) -> pd.Series:
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return (e3 - e3.shift(1)) / e3.shift(1).clip(lower=1e-10) * 100.0

def tsi(series: pd.Series, long_period: int=25, short_period: int=13) -> pd.Series:
    delta = series.diff()
    smooth = delta.ewm(span=long_period, adjust=False).mean().ewm(span=short_period, adjust=False).mean()
    smooth_abs = delta.abs().ewm(span=long_period, adjust=False).mean().ewm(span=short_period, adjust=False).mean()
    return smooth / smooth_abs.clip(lower=1e-12) * 100.0

def deviation_rate(series: pd.Series, period: int=20) -> pd.Series:
    sma_n = series.rolling(period).mean()
    return (series - sma_n) / sma_n.clip(lower=1e-10) * 100.0

def ab_ratio(high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series, period: int=14, safe: bool=False) -> pd.DataFrame:
    a_num = (high - open_).rolling(period).sum()
    a_den = (open_ - low).rolling(period).sum()
    b_num = (high - close.shift(1)).rolling(period).sum()
    b_den = (close.shift(1) - low).rolling(period).sum()
    if safe:
        tol = price_tol(close)
        return pd.DataFrame({'ab_ratio_a': safe_div(a_num, a_den, tol=tol, positive_only=True) * 100.0, 'ab_ratio_b': safe_div(b_num, b_den, tol=tol, positive_only=True) * 100.0}, index=close.index)
    return pd.DataFrame({'ab_ratio_a': a_num / a_den.clip(lower=1e-09) * 100.0, 'ab_ratio_b': b_num / b_den.clip(lower=1e-09) * 100.0}, index=close.index)

def adxr(high: pd.Series, low: pd.Series, close: pd.Series, period: int=14) -> pd.Series:
    tr_v = true_range(high, low, close)
    atr_v = tr_v.ewm(span=period, adjust=False).mean()
    up_move = high.diff()
    dn_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=high.index)
    plus_di = plus_dm.ewm(span=period, adjust=False).mean() / atr_v.clip(lower=1e-10) * 100.0
    minus_di = minus_dm.ewm(span=period, adjust=False).mean() / atr_v.clip(lower=1e-10) * 100.0
    dx_sum = plus_di + minus_di
    dx = (plus_di - minus_di).abs() / dx_sum.clip(lower=1e-10) * 100.0
    adx_v = dx.ewm(span=period, adjust=False).mean()
    return (adx_v + adx_v.shift(period)) / 2.0

def energy(high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series, period: int=14, safe: bool=False) -> pd.DataFrame:
    if safe:
        tol = price_tol(close)
        e1 = safe_div((high - open_).rolling(period).sum(), (open_ - low).rolling(period).sum(), tol=tol, positive_only=True) * 100.0
        e2 = safe_div((high - close).rolling(period).sum(), (close - low).rolling(period).sum(), tol=tol, positive_only=True) * 100.0
        return pd.DataFrame({'energy1': e1, 'energy2': e2}, index=close.index)
    e1 = (high - open_).rolling(period).sum() / (open_ - low + 1e-09).rolling(period).sum() * 100.0
    e2 = (high - close).rolling(period).sum() / (close - low + 1e-09).rolling(period).sum() * 100.0
    return pd.DataFrame({'energy1': e1, 'energy2': e2}, index=close.index)

def mass_index(high: pd.Series, low: pd.Series, ema_period: int=9, sum_period: int=25) -> pd.Series:
    hl = high - low
    e1 = hl.ewm(span=ema_period, adjust=False).mean()
    e2 = e1.ewm(span=ema_period, adjust=False).mean()
    return (e1 / e2.clip(lower=1e-12)).rolling(sum_period).sum()

def rci(series: pd.Series, period: int=9) -> pd.Series:

    def _rci(y: np.ndarray) -> float:
        n = len(y)
        date_rank = np.arange(n, 0, -1, dtype=np.float64)
        temp = np.argsort(-y)
        price_rank = np.empty(n, dtype=np.float64)
        price_rank[temp] = np.arange(1, n + 1, dtype=np.float64)
        d = date_rank - price_rank
        return float((1.0 - 6.0 * np.sum(d * d) / (n * (n * n - 1))) * 100.0)
    return series.rolling(period).apply(_rci, raw=True)

def sigma(series: pd.Series, period: int=20) -> pd.Series:
    sma_n = series.rolling(period).mean()
    std_n = series.rolling(period).std(ddof=0)
    return (series - sma_n) / std_n.clip(lower=1e-12)

def stddev(series: pd.Series, period: int=20) -> pd.Series:
    return series.rolling(period).std(ddof=0)

def vhf(series: pd.Series, period: int=28) -> pd.Series:
    highest = series.rolling(period).max()
    lowest = series.rolling(period).min()
    abs_sum = series.diff().abs().rolling(period).sum()
    return (highest - lowest) / abs_sum.clip(lower=1e-12)

def chaikin_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, short_period: int=3, long_period: int=10, safe: bool=False) -> pd.Series:
    """Chaikin Oscillator = EMA(A/D Line, short) - EMA(A/D Line, long)。"""
    if safe:
        clv = safe_div(close - low - (high - close), high - low, tol=price_tol(close), positive_only=True).fillna(0.0)
    else:
        clv = (close - low - (high - close)) / (high - low + 1e-09)
    ad_line = (clv * volume).cumsum()
    return ad_line.ewm(span=short_period, adjust=False).mean() - ad_line.ewm(span=long_period, adjust=False).mean()

def emv(high: pd.Series, low: pd.Series, volume: pd.Series, period: int=14, safe: bool=False) -> pd.Series:
    mid_move = (high + low) / 2.0 - (high.shift(1) + low.shift(1)) / 2.0
    if safe:
        per_day = safe_div(mid_move * (high - low), volume / 100.0, positive_only=True)
        return per_day.rolling(period).mean()
    box_ratio = volume / 100.0 / (high - low + 1e-09)
    return (mid_move / box_ratio.clip(lower=1e-12)).rolling(period).mean()

def nvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    pct = close.pct_change().fillna(0.0).to_numpy(dtype=np.float64)
    vol = volume.to_numpy(dtype=np.float64)
    result = np.empty(len(pct), dtype=np.float64)
    result[0] = 1000.0
    for i in range(1, len(pct)):
        if vol[i] < vol[i - 1]:
            result[i] = result[i - 1] + pct[i] * 100.0
        else:
            result[i] = result[i - 1]
    return pd.Series(result, index=close.index)

def pvi(close: pd.Series, volume: pd.Series) -> pd.Series:
    pct = close.pct_change().fillna(0.0).to_numpy(dtype=np.float64)
    vol = volume.to_numpy(dtype=np.float64)
    result = np.empty(len(pct), dtype=np.float64)
    result[0] = 1000.0
    for i in range(1, len(pct)):
        if vol[i] > vol[i - 1]:
            result[i] = result[i - 1] + pct[i] * 100.0
        else:
            result[i] = result[i - 1]
    return pd.Series(result, index=close.index)

def pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (close.pct_change().fillna(0.0) * volume).cumsum()

def volume_macd(close: pd.Series, volume: pd.Series, short_period: int=12, long_period: int=26, safe: bool=False) -> pd.Series:
    cv = close * volume
    if safe:
        short_vwap = safe_div(cv.rolling(short_period).sum(), volume.rolling(short_period).sum(), positive_only=True)
        long_vwap = safe_div(cv.rolling(long_period).sum(), volume.rolling(long_period).sum(), positive_only=True)
        return short_vwap - long_vwap
    short_vwap = cv.rolling(short_period).sum() / volume.rolling(short_period).sum().clip(lower=1e-10)
    long_vwap = cv.rolling(long_period).sum() / volume.rolling(long_period).sum().clip(lower=1e-10)
    return short_vwap - long_vwap

def volume_rsi(close: pd.Series, volume: pd.Series, period: int=14) -> pd.Series:
    diff = close.diff()
    up_vol = pd.Series(np.where(diff > 0, volume, 0.0), index=close.index)
    dn_vol = pd.Series(np.where(diff < 0, volume, 0.0), index=close.index)
    avg_up = up_vol.rolling(period).mean()
    avg_dn = dn_vol.rolling(period).mean()
    rs = avg_up / avg_dn.clip(lower=1e-12)
    return 100.0 - 100.0 / (1.0 + rs)

def volume_ratio(close: pd.Series, volume: pd.Series, period: int=26, safe: bool=False) -> pd.Series:
    diff = close.diff()
    up_v = pd.Series(np.where(diff > 0, volume, 0.0), index=close.index)
    dn_v = pd.Series(np.where(diff < 0, volume, 0.0), index=close.index)
    eq_v = pd.Series(np.where(diff == 0, volume, 0.0), index=close.index)
    num = up_v.rolling(period).sum() + eq_v.rolling(period).sum() * 0.5
    den = dn_v.rolling(period).sum() + eq_v.rolling(period).sum() * 0.5
    if safe:
        return safe_div(num, den, positive_only=True) * 100.0
    return num / den.clip(lower=1e-12) * 100.0

def vroc(volume: pd.Series, period: int=14, safe: bool=False) -> pd.Series:
    """VROC（Volume Rate of Change）= (V - V[N]) / V[N] × 100。"""
    if safe:
        return safe_div(volume - volume.shift(period), volume.shift(period), positive_only=True) * 100.0
    return (volume - volume.shift(period)) / volume.shift(period).clip(lower=1e-10) * 100.0

def oscv(volume: pd.Series, short_period: int=12, long_period: int=26, safe: bool=False) -> pd.Series:
    short_ma = volume.rolling(short_period).mean()
    long_ma = volume.rolling(long_period).mean()
    if safe:
        return safe_div(short_ma - long_ma, long_ma, positive_only=True) * 100.0
    return (short_ma - long_ma) / long_ma.clip(lower=1e-10) * 100.0

def dpo(series: pd.Series, period: int=20) -> pd.Series:
    shift = int(period / 2) + 1
    return series - series.rolling(period).mean().shift(shift)

def psychological_line(close: pd.Series, period: int=12) -> pd.Series:
    return (close.diff() > 0).astype(float).rolling(period).sum() / period * 100.0

def swing_index(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    pc = close.shift(1)
    po = open_.shift(1)
    r1 = (high - pc).abs()
    r2 = (low - pc).abs()
    r3 = high - low
    t = (pc - po).abs()
    r1_np = r1.to_numpy()
    r2_np = r2.to_numpy()
    r3_np = r3.to_numpy()
    t_np = t.to_numpy()
    max_r = np.maximum(r1_np, np.maximum(r2_np, r3_np))
    r_np = np.where(r1_np == max_r, r1_np - 0.5 * r2_np + 0.25 * t_np, np.where(r2_np == max_r, r2_np - 0.5 * r1_np + 0.25 * t_np, r3_np + 0.25 * t_np))
    r_s = pd.Series(r_np, index=close.index)
    numerator = close - pc + 0.5 * (close - open_) + 0.25 * (pc - po)
    k = pd.concat([(high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    m = pc.clip(lower=1e-10) * 0.05
    f_val = numerator / r_s.clip(lower=1e-12)
    return 50.0 * f_val * (k / m.clip(lower=1e-12))

def asi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return swing_index(open_, high, low, close).cumsum()

def binary_wave(close: pd.Series, macd_hist: pd.Series, roc_series: pd.Series, stoch_k: pd.Series, sma_period: int=20) -> pd.Series:
    sma_sig = np.sign((close - close.rolling(sma_period).mean()).fillna(0.0))
    macd_sig = np.sign(macd_hist.fillna(0.0))
    roc_sig = np.sign(roc_series.fillna(0.0))
    stoch_sig = pd.Series(np.where(stoch_k.fillna(50.0) > 50.0, 1.0, -1.0), index=close.index)
    return sma_sig + macd_sig + roc_sig + stoch_sig

def aroon(high: pd.Series, low: pd.Series, period: int=14) -> pd.DataFrame:

    def _up(x: np.ndarray) -> float:
        n = len(x) - 1
        days_since = n - int(np.argmax(x))
        return (n - days_since) / n * 100.0

    def _down(x: np.ndarray) -> float:
        n = len(x) - 1
        days_since = n - int(np.argmin(x))
        return (n - days_since) / n * 100.0
    aroon_up = high.rolling(period + 1).apply(_up, raw=True)
    aroon_dn = low.rolling(period + 1).apply(_down, raw=True)
    return pd.DataFrame({'aroon_up': aroon_up, 'aroon_down': aroon_dn}, index=high.index)

def aroon_oscillator(high: pd.Series, low: pd.Series, period: int=14) -> pd.Series:
    """AROONOSC = Aroon Up - Aroon Down ∈ [-100, +100]。"""
    arr = aroon(high, low, period)
    return arr['aroon_up'] - arr['aroon_down']

def ht_dcperiod(close: pd.Series) -> pd.Series:
    close_np = close.to_numpy(dtype=np.float64)
    n = len(close_np)
    warm = 32
    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(warm, n):
        win = close_np[i - warm + 1:i + 1]
        sma4 = np.convolve(win, np.ones(4) / 4.0, mode='same')
        det = win - sma4
        std = det.std()
        if std < 1e-12:
            result[i] = 8.0
            continue
        det_norm = det / std
        best_p = 8.0
        best_c = -np.inf
        m = len(det_norm)
        for lag in range(2, min(32, m - 2)):
            c = float(np.dot(det_norm[lag:], det_norm[:m - lag]) / (m - lag))
            if c > best_c:
                best_c = c
                best_p = float(lag * 2)
        result[i] = max(6.0, min(50.0, best_p))
    return pd.Series(result, index=close.index)

def compute_extended_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if not bool(cfg.get('enabled', False)):
        return df
    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        logger.warning('compute_extended_indicators: missing columns %s', required - set(df.columns))
        return df
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    lo = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)
    norm = bool(cfg.get('normalize_by_close', True))
    denom = c.clip(lower=1e-10)
    safe = bool(cfg.get('safe_division', False))

    def _n(s: pd.Series) -> pd.Series:
        return s / denom if norm else s
    new_columns: Dict[str, pd.Series] = {}
    _p = int((cfg.get('dema', {}) or {}).get('period', 10))
    new_columns['dema'] = _n(dema(c, _p))
    _p = int((cfg.get('lri', {}) or {}).get('period', 14))
    new_columns['lri'] = _n(lri(c, _p))
    _p = int((cfg.get('lrs', {}) or {}).get('period', 14))
    new_columns['lrs'] = lrs(c, _p)
    _p = int((cfg.get('tsf', {}) or {}).get('period', 14))
    new_columns['tsf'] = _n(tsf(c, _p))
    _ps = int((cfg.get('tsf_oscillator', {}) or {}).get('short_period', 5))
    _pl = int((cfg.get('tsf_oscillator', {}) or {}).get('long_period', 20))
    new_columns['tsf_oscillator'] = _n(tsf_oscillator(c, _ps, _pl))
    _p = int((cfg.get('r_squared', {}) or {}).get('period', 14))
    new_columns['r_squared'] = r_squared(c, _p)
    _p = int((cfg.get('standard_error', {}) or {}).get('period', 14))
    new_columns['standard_error'] = standard_error(c, _p)
    _pv = pivot_points(h, lo, c)
    for _col in _pv.columns:
        new_columns[f'pivot_{_col}'] = _n(_pv[_col])
    _ep = int((cfg.get('envelope', {}) or {}).get('period', 20))
    _er = float((cfg.get('envelope', {}) or {}).get('rate', 0.05))
    _env = envelope(c, _ep, _er)
    new_columns['envelope_upper'] = _n(_env['envelope_upper'])
    new_columns['envelope_lower'] = _n(_env['envelope_lower'])
    _af_s = float((cfg.get('parabolic_sar', {}) or {}).get('af_start', 0.02))
    _af_m = float((cfg.get('parabolic_sar', {}) or {}).get('af_max', 0.2))
    new_columns['parabolic_sar'] = _n(parabolic_sar(h, lo, _af_s, _af_m))
    _td = td_range_projections(o, h, lo, c)
    new_columns['td_upper'] = _n(_td['td_upper'])
    new_columns['td_lower'] = _n(_td['td_lower'])
    _ms = int((cfg.get('mao', {}) or {}).get('short_period', 12))
    _ml = int((cfg.get('mao', {}) or {}).get('long_period', 26))
    new_columns['mao'] = _n(mao(c, _ms, _ml))
    _ms = int((cfg.get('oscp', {}) or {}).get('short_period', 12))
    _ml = int((cfg.get('oscp', {}) or {}).get('long_period', 26))
    new_columns['oscp'] = oscp(c, _ms, _ml)
    _p = int((cfg.get('sonar', {}) or {}).get('period', 10))
    new_columns['sonar'] = sonar(c, _p)
    _p = int((cfg.get('trix', {}) or {}).get('period', 14))
    new_columns['trix'] = trix(c, _p)
    _pl2 = int((cfg.get('tsi', {}) or {}).get('long_period', 25))
    _ps2 = int((cfg.get('tsi', {}) or {}).get('short_period', 13))
    new_columns['tsi'] = tsi(c, _pl2, _ps2)
    _p = int((cfg.get('deviation_rate', {}) or {}).get('period', 20))
    new_columns['deviation_rate'] = deviation_rate(c, _p)
    _p = int((cfg.get('ab_ratio', {}) or {}).get('period', 14))
    _ab = ab_ratio(h, lo, c, o, _p, safe=safe)
    new_columns['ab_ratio_a'] = _ab['ab_ratio_a']
    new_columns['ab_ratio_b'] = _ab['ab_ratio_b']
    _p = int((cfg.get('adxr', {}) or {}).get('period', 14))
    new_columns['adxr'] = adxr(h, lo, c, _p)
    _p = int((cfg.get('energy', {}) or {}).get('period', 14))
    _en = energy(h, lo, c, o, _p, safe=safe)
    new_columns['energy1'] = _en['energy1']
    new_columns['energy2'] = _en['energy2']
    _ep2 = int((cfg.get('mass_index', {}) or {}).get('ema_period', 9))
    _sp2 = int((cfg.get('mass_index', {}) or {}).get('sum_period', 25))
    new_columns['mass_index'] = mass_index(h, lo, _ep2, _sp2)
    _p = int((cfg.get('rci', {}) or {}).get('period', 9))
    new_columns['rci'] = rci(c, _p)
    _p = int((cfg.get('sigma', {}) or {}).get('period', 20))
    new_columns['sigma'] = sigma(c, _p)
    _p = int((cfg.get('stddev', {}) or {}).get('period', 20))
    new_columns['stddev'] = stddev(c, _p)
    _p = int((cfg.get('vhf', {}) or {}).get('period', 28))
    new_columns['vhf'] = vhf(c, _p)
    _cs = int((cfg.get('chaikin_oscillator', {}) or {}).get('short_period', 3))
    _cl = int((cfg.get('chaikin_oscillator', {}) or {}).get('long_period', 10))
    new_columns['chaikin_oscillator'] = chaikin_oscillator(h, lo, c, v, _cs, _cl, safe=safe)
    _p = int((cfg.get('emv', {}) or {}).get('period', 14))
    new_columns['emv'] = emv(h, lo, v, _p, safe=safe)
    new_columns['nvi'] = nvi(c, v)
    new_columns['pvi'] = pvi(c, v)
    new_columns['pvt'] = pvt(c, v)
    _vs = int((cfg.get('volume_macd', {}) or {}).get('short_period', 12))
    _vl = int((cfg.get('volume_macd', {}) or {}).get('long_period', 26))
    new_columns['volume_macd'] = volume_macd(c, v, _vs, _vl, safe=safe)
    _p = int((cfg.get('volume_rsi', {}) or {}).get('period', 14))
    new_columns['volume_rsi'] = volume_rsi(c, v, _p)
    _p = int((cfg.get('volume_ratio', {}) or {}).get('period', 26))
    new_columns['volume_ratio_ext'] = volume_ratio(c, v, _p, safe=safe)
    _p = int((cfg.get('vroc', {}) or {}).get('period', 14))
    new_columns['vroc'] = vroc(v, _p, safe=safe)
    _vs2 = int((cfg.get('oscv', {}) or {}).get('short_period', 12))
    _vl2 = int((cfg.get('oscv', {}) or {}).get('long_period', 26))
    new_columns['oscv'] = oscv(v, _vs2, _vl2, safe=safe)
    _p = int((cfg.get('dpo', {}) or {}).get('period', 20))
    new_columns['dpo'] = dpo(c, _p)
    _p = int((cfg.get('psychological_line', {}) or {}).get('period', 12))
    new_columns['psychological_line'] = psychological_line(c, _p)
    new_columns['swing_index'] = swing_index(o, h, lo, c)
    new_columns['asi'] = asi(o, h, lo, c)
    if {'macd_hist', 'roc', 'stoch_k'}.issubset(df.columns):
        _p = int((cfg.get('binary_wave', {}) or {}).get('sma_period', 20))
        new_columns['binary_wave'] = binary_wave(c, df['macd_hist'], df['roc'], df['stoch_k'], _p)
    _p = int((cfg.get('aroon', {}) or {}).get('period', 14))
    _arr = aroon(h, lo, _p)
    new_columns['aroon_up'] = _arr['aroon_up']
    new_columns['aroon_down'] = _arr['aroon_down']
    new_columns['aroonosc'] = aroon_oscillator(h, lo, _p)
    if bool((cfg.get('ht_dcperiod', {}) or {}).get('enabled', False)):
        new_columns['ht_dcperiod'] = ht_dcperiod(c)
    return pd.concat([df.drop(columns=list(new_columns), errors='ignore'), pd.DataFrame(new_columns, index=df.index)], axis=1).copy()
