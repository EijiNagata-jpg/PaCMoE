"""TOML configuration.

Only values stated in the paper are configurable. `[experiment] setup` selects one of the three
configurations used in the paper; their differences (not described in the paper) are fixed here.

  setup = "main"          Tables 3 and 5, Section 3.5 (econometric comparison), response tables
                          (scaled volatility target, transaction costs)
  setup = "ablation"      Table 4, Section 3.6 (three classes), Figures 2-4
  setup = "walk_forward"  Table 6 (rolling walk-forward evaluation)
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pacmoe.model import ModelSpec

SETUPS = ("main", "ablation", "walk_forward")
ABLATION_VARIANTS = ("full", "router_uniform", "only_pt", "only_pg", "only_tg", "price_only", "cls_only", "three_class")

# Differences between the setups (implementation details, not stated in the paper).
_SETUP_DETAILS: Dict[str, Dict[str, Any]] = {
    "main": {"cca_stat_mode": "batch", "graph_readout": "mean", "graph_view_exclude_self": False,
             "return_target": "vol_normalized", "branch_aux": True, "ranking": True},
    "ablation": {"cca_stat_mode": "running", "graph_readout": "node", "graph_view_exclude_self": True,
                 "return_target": "raw", "branch_aux": False, "ranking": False},
    "walk_forward": {"cca_stat_mode": "running", "graph_readout": "node", "graph_view_exclude_self": True,
                     "return_target": "vol_normalized", "branch_aux": True, "ranking": True},
}
_ONLY_EXPERT = {"only_pt": ("price_graph", "text_graph"), "only_pg": ("price_text", "text_graph"), "only_tg": ("price_text", "price_graph")}


def setup_details(setup: str) -> Dict[str, Any]:
    if setup not in _SETUP_DETAILS:
        raise ValueError(f"[experiment] setup must be one of {SETUPS}, got {setup!r}")
    return dict(_SETUP_DETAILS[setup])


def model_spec_for(setup: str, variant: str = "full", *, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1,
                   text_feature_dim: int = 825) -> ModelSpec:
    """ModelSpec of a setup (and of an ablation variant)."""
    d = setup_details(setup)
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"[ablation] variant must be one of {ABLATION_VARIANTS}, got {variant!r}")
    if variant != "full" and setup != "ablation":
        raise ValueError("[ablation] variant other than 'full' requires setup = 'ablation'")
    return ModelSpec(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        text_feature_dim=text_feature_dim,
        cca_stat_mode=d["cca_stat_mode"],
        graph_readout=d["graph_readout"],
        graph_view_exclude_self=d["graph_view_exclude_self"],
        num_classes=3 if variant == "three_class" else 2,
        disabled_experts=_ONLY_EXPERT.get(variant, ()),
        uniform_router=variant in ("router_uniform", "only_pt", "only_pg", "only_tg"),
        price_only=variant == "price_only",
        ranking=bool(d["ranking"]),
    )


@dataclass
class Config:
    setup: str = "main"
    seed: int = 2025
    bundle_dir: str = ""
    output_dir: str = "outputs"
    # [model] (Table 2)
    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    # [loss] (Section 2.6.4)
    loss_weights: List[float] = field(default_factory=lambda: [1.0, 0.05, 0.05])
    huber_delta: float = 1.0
    # [training] (Table 2)
    learning_rate: float = 8e-5
    batch_size: int = 128
    epochs: int = 15
    # [evaluation] (Section 2.1)
    threshold: float = 0.5
    # [ablation]
    variant: str = "full"
    # [experiment] graph_snapshots: replaces the daily graphs of the bundle (DCC-GARCH graphs, Section 3.5)
    graph_snapshots: Optional[str] = None
    # [walk_forward] (Section 3.4)
    walk_forward: Dict[str, Any] = field(default_factory=dict)

    @property
    def details(self) -> Dict[str, Any]:
        d = setup_details(self.setup)
        d["router_z_prior_off"] = self.variant in _ONLY_EXPERT  # single-expert ablations trained without the z-loss
        return d

    def model_spec(self, text_feature_dim: int) -> ModelSpec:
        return model_spec_for(self.setup, self.variant, hidden_dim=self.hidden_dim, num_heads=self.num_heads,
                              dropout=self.dropout, text_feature_dim=text_feature_dim)


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Read a TOML file into Config."""
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    exp, model, loss, train = raw.get("experiment", {}), raw.get("model", {}), raw.get("loss", {}), raw.get("training", {})
    cfg = Config(
        setup=str(exp.get("setup", "main")),
        seed=int(exp.get("seed", 2025)),
        bundle_dir=str(exp.get("bundle_dir", "")),
        output_dir=str(exp.get("output_dir", "outputs")),
        hidden_dim=int(model.get("hidden_dim", 256)),
        num_heads=int(model.get("num_heads", 4)),
        dropout=float(model.get("dropout", 0.1)),
        loss_weights=[float(v) for v in loss.get("weights", [1.0, 0.05, 0.05])],
        huber_delta=float(loss.get("huber_delta", 1.0)),
        learning_rate=float(train.get("learning_rate", 8e-5)),
        batch_size=int(train.get("batch_size", 128)),
        epochs=int(train.get("epochs", 15)),
        threshold=float(raw.get("evaluation", {}).get("threshold", 0.5)),
        variant=str(raw.get("ablation", {}).get("variant", "full")),
        walk_forward=dict(raw.get("walk_forward", {})),
        graph_snapshots=exp.get("graph_snapshots"),
    )
    for key, value in (overrides or {}).items():
        setattr(cfg, key, value)
    setup_details(cfg.setup)
    if cfg.variant == "cls_only":
        cfg.loss_weights = [1.0, 0.0, 0.0]
    return cfg
