"""Compare a bundle written by preprocess.py with an original bundle (the keys the model reads).

usage: python scripts/compare_bundles.py <new bundle> <original bundle>
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pacmoe.data import Bundle  # noqa: E402


def _max_diff(a, b) -> float:
    return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0


def main() -> int:
    new, old = sys.argv[1], sys.argv[2]
    report = {}
    for split in ("train", "val", "test"):
        for target in ("vol_normalized", "raw"):
            a, b = Bundle(new, split, return_target=target), Bundle(old, split, return_target=target)
            if target == "raw":
                report[f"{split}.y_ret_raw"] = _max_diff(a.y_ret, b.y_ret)
                continue
            same = a.symbols == b.symbols and a.dates == b.dates
            r = {"n": [len(a), len(b)], "same_samples": same}
            if same:
                r.update({
                    "x_ts": _max_diff(a.x_ts, b.x_ts), "x_ohlcv": _max_diff(a.x_ohlcv, b.x_ohlcv),
                    "text_features": _max_diff(a.text_features, b.text_features),
                    "text_day_mask_equal": bool(torch.equal(a.text_day_mask, b.text_day_mask)),
                    "has_text_equal": bool(torch.equal(a.has_text, b.has_text)),
                    "y_cls_equal": bool(torch.equal(a.y_cls, b.y_cls)), "y_vol_equal": bool(torch.equal(a.y_vol, b.y_vol)),
                    "y_ret_vol_normalized": _max_diff(a.y_ret, b.y_ret), "node_ids_equal": bool(torch.equal(a.node_ids, b.node_ids)),
                })
                pa = torch.load(Path(new) / f"samples_{split}.pt", map_location="cpu", weights_only=False)
                pb = torch.load(Path(old) / f"samples_{split}.pt", map_location="cpu", weights_only=False)
                if "x_text_embed" in pa and "x_text_embed" in pb:
                    r["x_text_embed_cca"] = _max_diff(pa["x_text_embed"], pb["x_text_embed"])
                    r["x_text_embed_seq_equal"] = all(torch.equal(x.float(), y.float()) for x, y in zip(pa["x_text_embed_seq"], pb["x_text_embed_seq"]))
                dates = sorted(set(a.dates))
                r["graphs_equal"] = all(torch.equal(a.graphs[d]["edge_index"], b.graphs[d]["edge_index"]) and torch.equal(a.graphs[d]["edge_weight"], b.graphs[d]["edge_weight"]) for d in dates)
            report[split] = r
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
