"""Compare the clean daily correlation graphs with a bundle's graph_snapshots.pt (all dates).

usage: python scripts/check_graphs.py <stocknet price/raw dir> <bundle dir>
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pacmoe.preprocess.graph import daily_graphs  # noqa: E402
from pacmoe.preprocess.prices import load_prices  # noqa: E402


def _as_dict(g):
    """{(src, dst): weight} of one snapshot."""
    return {(int(a), int(b)): w for (a, b), w in zip(g["edge_index"].T.tolist(), g["edge_weight"].tolist())}


def main() -> int:
    price_dir, bundle = sys.argv[1], Path(sys.argv[2])
    meta = json.loads((bundle / "metadata.json").read_text())
    symbols = meta["graph_symbols"]
    ref = torch.load(bundle / "graph_snapshots.pt", map_location="cpu", weights_only=False)
    close = {s: df["close"] for s, df in load_prices(price_dir, symbols).items()}
    t0 = time.time()
    mine = daily_graphs(close, symbols, ref.keys())
    exact_edges = exact_all = 0
    worst = 0.0
    for date, g in ref.items():
        a, b = _as_dict(mine[date]), _as_dict(g)
        same_edges = set(a) == set(b)
        exact_edges += same_edges
        if same_edges:
            d = max((abs(a[k] - b[k]) for k in a), default=0.0)
            worst = max(worst, d)
            exact_all += d == 0.0
    print(json.dumps({"dates": len(ref), "same_edge_sets": exact_edges, "bit_exact_weights": exact_all,
                      "max_abs_weight_diff": worst, "seconds": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
