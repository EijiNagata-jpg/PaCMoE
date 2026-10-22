"""Daily graphs from DCC-GARCH correlation forecasts (Section 3.5, "retrained with the DCC graph").

For each sample date t: the one-step DCC-GARCH forecast R_{t+1|t} (data up to the close of day t; DCC
estimated before the validation period), each stock connected to its 10 stocks with the largest |corr|.
Stocks without a complete return history from 2013-01-03 (AGFS, BABA, GMRE) get no edges.

usage: python -m analysis.dcc_graph --bundle data/preprocessed/main --out data/graph_overrides/g3_dcc.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from analysis.common import PANEL_START, complete_symbols, locf_prices, log_return_panel
from analysis.dcc_garch import fit_dcc


def topk_graph_from_corr(C: np.ndarray, k: int = 10, node_mask: np.ndarray | None = None) -> dict:
    """Undirected COO graph from a correlation matrix: each stock's k largest |corr| (same rule as the daily graphs)."""
    A = np.array(C, dtype=np.float64, copy=True)
    n = A.shape[0]
    if node_mask is not None:
        A[~node_mask, :] = 0.0
        A[:, ~node_mask] = 0.0
    np.fill_diagonal(A, 0.0)
    A = np.nan_to_num(A, nan=0.0)
    absA = np.abs(A)
    k_eff = min(max(int(k), 1), n - 1)
    idx = np.argsort(-absA, axis=1, kind="stable")[:, :k_eff]
    keep = np.zeros_like(A)
    rows = np.repeat(np.arange(n), k_eff)
    keep[rows, idx.ravel()] = A[rows, idx.ravel()]
    take_self = np.abs(keep) >= np.abs(keep.T)
    kept = np.where(take_self, keep, keep.T)
    src, dst = np.nonzero(kept)
    return {"edge_index": torch.tensor(np.vstack([src, dst]), dtype=torch.long),
            "edge_weight": torch.tensor(kept[src, dst], dtype=torch.float32),
            "edge_type": torch.zeros(len(src), dtype=torch.long)}


def embed_corr(C_sub: np.ndarray, sub_symbols, graph_symbols) -> tuple[np.ndarray, np.ndarray]:
    """Embed the correlation matrix of a subset into the matrix of all graph nodes (graph_symbols order)."""
    pos = {s: i for i, s in enumerate(graph_symbols)}
    n = len(graph_symbols)
    C = np.zeros((n, n))
    mask = np.zeros(n, dtype=bool)
    ix = np.array([pos[s] for s in sub_symbols])
    C[np.ix_(ix, ix)] = C_sub
    mask[ix] = True
    return C, mask


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--raw-dir", default="data/stocknet-dataset/price/raw")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    bundle_snaps = torch.load(f"{a.bundle}/graph_snapshots.pt", map_location="cpu", weights_only=False)
    dates = sorted(bundle_snaps)
    meta = json.load(open(f"{a.bundle}/metadata.json"))
    graph_symbols = list(meta["graph_symbols"])
    info = {"kind": "dcc", "bundle": a.bundle, "n_dates": len(dates), "k": a.k}

    if True:
        split = meta["split"]
        tr0, tr1, te1 = (pd.Timestamp(split[k]) for k in ("train_start", "train_end", "test_end"))
        P = pd.DataFrame({s: pd.read_csv(f"{a.raw_dir}/{s}.csv", parse_dates=["Date"]).set_index("Date")["Adj Close"]
                          for s in graph_symbols}).sort_index()
        Rp = log_return_panel(locf_prices(P.loc[P.index < te1]))
        start = pd.Timestamp(PANEL_START)
        syms = complete_symbols(Rp, start, te1)
        R = Rp.loc[(Rp.index >= start) & (Rp.index < te1), syms]
        assert not R.isna().any().any()
        info["symbols_used"] = len(syms)
        info["symbols_without_edges"] = sorted(set(graph_symbols) - set(syms))
        # Estimated on [2013-01-03, train_end) by composite likelihood with contiguous pairs.
        est = np.asarray(R.index < tr1)
        m = fit_dcc(R.loc[est].to_numpy(), mean="constant", method="composite", pairs="contiguous")
        _, Rn = m.one_step_forecasts(R.to_numpy())   # row t = R_{t+1|t} from data up to the close of day t
        corr = {R.index[i]: Rn[i] for i in range(len(R))}
        info.update({"dcc_a": m.a, "dcc_b": m.b})
        snaps = {}
        for d in dates:
            ts = pd.Timestamp(d)
            if ts not in corr:
                raise KeyError(f"no correlation for {d}")
            C, mask = embed_corr(corr[ts], syms, graph_symbols)
            snaps[d] = topk_graph_from_corr(C, k=a.k, node_mask=mask)
    torch.save(snaps, a.out)
    edges = [int(v["edge_index"].shape[1]) for v in snaps.values()]
    info.update({"edges_min_med_max": [min(edges), int(np.median(edges)), max(edges)], "out": a.out})
    print(json.dumps(info, default=str))
    with open(a.out + ".json", "w") as fh:
        json.dump(info, fh, indent=1, default=str)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
