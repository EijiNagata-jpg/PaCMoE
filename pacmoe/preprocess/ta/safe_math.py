"""Division used by the indicators: a zero denominator gives NaN."""
from __future__ import annotations
from typing import Union
import pandas as pd
Number = Union[float, int]

def safe_div(num: pd.Series, den: pd.Series, *, tol: Union[Number, pd.Series]=0.0, positive_only: bool=False) -> pd.Series:
    den = den.astype(float)
    bad = den <= tol if positive_only else den.abs() <= tol
    return num.astype(float) / den.where(~bad)

def price_tol(close: pd.Series, rel: float=1e-06) -> pd.Series:
    return close.astype(float).abs() * rel
