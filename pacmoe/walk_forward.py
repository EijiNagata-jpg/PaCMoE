"""
Rolling walk-forward evaluation (Section 3.4, Table 6).

Each fold has its own preprocessed bundle (statistics fitted on the fold's training data only; the
periods are set in configs/preprocess_walk_forward.toml). For every fold the model is trained
for `epochs` epochs and the epoch with the highest validation MCC within `select_epochs` is reported.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from pacmoe.config import Config
from pacmoe.train import Trainer


def fold_bundle_dirs(cfg: Config) -> List[Path]:
    """Fold bundles: [walk_forward] fold_bundle_dirs, or the fold_* subdirectories of bundle_dir."""
    wf = cfg.walk_forward
    if wf.get("fold_bundle_dirs"):
        return [Path(p) for p in wf["fold_bundle_dirs"]]
    return sorted(p for p in Path(cfg.bundle_dir).iterdir() if p.is_dir() and p.name.startswith("fold_"))


def run_walk_forward(cfg: Config) -> Dict[str, object]:
    first, last = cfg.walk_forward.get("select_epochs", [10, 15])
    folds = []
    for index, bundle in enumerate(fold_bundle_dirs(cfg)):
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg.bundle_dir = str(bundle)
        fold_cfg.output_dir = str(Path(cfg.output_dir) / bundle.name)
        chosen = Trainer(fold_cfg).fit(select_epochs=(int(first), int(last)))["best"]
        folds.append({"fold": index, "bundle": bundle.name, "epoch": chosen["epoch"], "val": chosen["val"], "test": chosen["test"]})
    summary = {
        "folds": folds,
        "mean_test_mcc": float(np.mean([f["test"]["mcc"] for f in folds])) if folds else float("nan"),
        "mean_test_accuracy": float(np.mean([f["test"]["accuracy"] for f in folds])) if folds else float("nan"),
    }
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "walk_forward_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary
