"""
Build a preprocessed bundle from the StockNet prices and the released text caches.

usage: python preprocess.py --config configs/preprocess_main.toml
"""
import argparse
import tomllib
from pathlib import Path

from pacmoe.preprocess.build import PreprocessConfig, build_bundle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--price-dir", default=None, help="overrides [data] price_dir")
    ap.add_argument("--text-dir", default=None, help="overrides [data] text_dir")
    ap.add_argument("--out-dir", default=None, help="overrides [data] out_dir")
    a = ap.parse_args()
    raw = tomllib.loads(Path(a.config).read_text(encoding="utf-8"))
    data = raw["data"]
    price_dir = a.price_dir or data["price_dir"]
    text_dir, out_dir = a.text_dir or data["text_dir"], a.out_dir or data["out_dir"]
    # Walk-forward: one bundle per fold, written to <out_dir>/<fold name> from <text_dir>/<fold name>.
    folds = raw.get("fold") or [{"name": "", **raw["split"]}]
    for fold in folds:
        cfg = _config(raw, price_dir, str(Path(text_dir) / fold["name"]), str(Path(out_dir) / fold["name"]),
                      {k: tuple(fold[k]) for k in ("train", "val", "test")})
        cfg.cca_blas_threads = int(fold.get("cca_blas_threads", cfg.cca_blas_threads))
        build_bundle(cfg)
    return 0


def _config(raw, price_dir, text_dir, out_dir, split_dates) -> PreprocessConfig:
    data, lab, graph = raw["data"], raw["labels"], raw["graph"]
    return PreprocessConfig(
        price_dir=price_dir,
        text_dir=text_dir,
        out_dir=out_dir,
        symbols=sorted(p.stem for p in Path(price_dir).glob("*.csv")),
        split_dates=split_dates,
        window=int(data["window"]),
        up_threshold=float(lab["up_threshold"]),
        down_threshold=float(lab["down_threshold"]),
        vol_threshold=float(lab["vol_threshold"]),
        vol_target=str(lab["vol_target"]),
        use_neutral=bool(lab.get("use_neutral", False)),
        graph_windows=tuple(int(w) for w in graph["windows"]),
        graph_top_k=int(graph["top_k"]),
        setup=str(data.get("setup", "main")),
        cca_blas_threads=int(data.get("cca_blas_threads", 8)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
