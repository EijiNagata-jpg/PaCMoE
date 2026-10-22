"""StockNet price files: loading and dividend adjustment (Section 2.1)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_prices(price_dir: str | Path, symbols: List[str] | None = None) -> Dict[str, pd.DataFrame]:
    """Read price/raw/<SYMBOL>.csv; multiply open/high/low/close by Adj Close / Close (dividend
    adjustment; the raw Close is already split-adjusted). Volume is not adjusted."""
    price_dir = Path(price_dir)
    names = symbols or sorted(p.stem for p in price_dir.glob("*.csv"))
    out: Dict[str, pd.DataFrame] = {}
    for sym in names:
        path = price_dir / f"{sym}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df.set_index(pd.to_datetime(df["Date"])).drop(columns=["Date"])
        close = df["Close"].astype(float)
        factor = (df["Adj Close"].astype(float) / close.replace(0.0, pd.NA)).fillna(1.0)
        for col in ("Open", "High", "Low", "Close"):
            df[col] = df[col].astype(float) * factor
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df["amount"] = df["close"] * df["volume"]
        if len(df) > 0:
            out[sym] = df
    return out
