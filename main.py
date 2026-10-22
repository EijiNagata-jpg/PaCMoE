"""PaCMoE command line.

  python main.py train    --config configs/main.toml [--seed 2025] [--output-dir outputs/main_2025]
  python main.py evaluate --config configs/main.toml --checkpoint outputs/main_2025/best.pt [--split test]
"""
from __future__ import annotations

import argparse
import json
import logging

import torch

from pacmoe.config import load_config
from pacmoe.data import Bundle
from pacmoe.evaluate import binary_metrics, configure_backends, multiclass_metrics, predict
from pacmoe.model import PaCMoE


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("train", "evaluate"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bundle-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--graph-snapshots", default=None, help="daily graphs replacing those of the bundle (DCC-GARCH)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--deterministic", action="store_true", help="deterministic CUDA kernels (evaluate)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    overrides = {k: v for k, v in {"seed": a.seed, "bundle_dir": a.bundle_dir, "output_dir": a.output_dir, "graph_snapshots": a.graph_snapshots}.items() if v is not None}
    cfg = load_config(a.config, overrides)

    if cfg.setup == "walk_forward" and a.command == "train":
        from pacmoe.walk_forward import run_walk_forward
        print(json.dumps(run_walk_forward(cfg), indent=1))
        return 0
    if a.command == "train":
        from pacmoe.train import Trainer
        result = Trainer(cfg).fit()
        print(json.dumps(result["best"], indent=1))
        return 0

    configure_backends(deterministic=a.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = Bundle(cfg.bundle_dir, a.split, cfg.details["return_target"], graph_snapshots_path=cfg.graph_snapshots)
    model = PaCMoE(cfg.model_spec(int(bundle.text_features.shape[-1])))
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=False)["model_state"])
    model.to(device)
    res = predict(model, bundle, cfg.batch_size, device)
    if res["probs_all"].shape[1] > 2:
        metrics = multiclass_metrics(res["probs_all"], res["labels"])
    else:
        metrics = binary_metrics(res["probs"], res["labels"], cfg.threshold)
    print(json.dumps(metrics, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
