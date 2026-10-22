"""Fixed-width-window fractional differencing (Lopez de Prado, Advances in Financial Machine
Learning, Chapter 5): the smallest order d whose differenced log price passes the ADF test.
"""
from __future__ import annotations
import warnings
from typing import Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd

def getWeights_FFD(d: float, thres: float=1e-05) -> np.ndarray:
    if d == 0.0:
        return np.array([1.0])
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] / k * (d - k + 1)
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])

def fracDiff_FFD(series: Union[pd.Series, pd.DataFrame], d: float, thres: float=1e-05) -> Union[pd.Series, pd.DataFrame]:
    is_series = isinstance(series, pd.Series)
    if is_series:
        name = series.name
        df = series.to_frame(name=name if name is not None else 'value')
    else:
        df = series.copy()
    w = getWeights_FFD(d, thres)
    width = len(w) - 1
    result: Dict[str, pd.Series] = {}
    for col in df.columns:
        col_series = df[col].ffill().astype(float)
        n = len(col_series)
        out = pd.Series(np.nan, index=col_series.index, dtype=float)
        if n <= width:
            result[col] = out
            continue
        values = col_series.values
        for iloc1 in range(width, n):
            segment = values[iloc1 - width:iloc1 + 1]
            out.iloc[iloc1] = float(np.dot(w, segment))
        result[col] = out
    result_df = pd.concat(result, axis=1)
    result_df.columns = list(df.columns)
    if is_series:
        out_series = result_df.iloc[:, 0]
        out_series.name = name
        return out_series
    return result_df

def plot_min_ffd(series: Union[pd.Series, pd.DataFrame], ds: Optional[Sequence[float]]=None, thres: float=1e-05, p_value_threshold: float=0.05) -> pd.DataFrame:
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as e:
        raise ImportError('plot_min_ffd requires statsmodels. Install with: pip install statsmodels') from e
    if isinstance(series, pd.DataFrame):
        col0 = series.columns[0]
        s = series[col0].ffill().dropna().astype(float)
    else:
        s = series.ffill().dropna().astype(float)
    if ds is None:
        ds = list(np.linspace(0.0, 1.0, 11))
    rows = []
    for d in ds:
        fd = fracDiff_FFD(s, d, thres=thres).dropna()
        if len(fd) < 10:
            continue
        fd_vals = fd.values.astype(float)
        with warnings.catch_warnings():
            # statsmodels 0.15 announces a future change of the return type; the tuple is still returned.
            warnings.simplefilter('ignore', FutureWarning)
            adf_res = adfuller(fd_vals, maxlag=1, regression='c', autolag=None)
        adf_stat = float(adf_res[0])
        p_val = float(adf_res[1])
        lags = int(adf_res[2])
        n_obs = int(adf_res[3])
        crit_95 = float(adf_res[4].get('5%', np.nan))
        corr = float(s.corr(fd, method='spearman')) if len(fd) > 2 else np.nan
        rows.append({'d': d, 'adfStat': adf_stat, 'pVal': p_val, 'lags': lags, 'nObs': n_obs, '95%': crit_95, 'corr': corr})
    if not rows:
        return pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95%', 'corr'], index=pd.Index([], name='d'))
    result = pd.DataFrame(rows).set_index('d')
    return result

def find_min_d(series: Union[pd.Series, pd.DataFrame], ds: Optional[Sequence[float]]=None, thres: float=1e-05, p_value_threshold: float=0.05) -> float:
    result_df = plot_min_ffd(series, ds=ds, thres=thres, p_value_threshold=p_value_threshold)
    if result_df.empty:
        return 1.0
    passed = result_df[result_df['pVal'] < p_value_threshold]
    if passed.empty:
        return 1.0
    return float(passed.index.min())
