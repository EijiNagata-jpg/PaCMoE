"""
Training loop (Section 2.7): AdamW, 15 epochs, batch size 128; the checkpoint with the highest
validation MCC (threshold 0.5) is used for testing. The test split is never used for selection.
"""
from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from pacmoe import constants as K
from pacmoe.config import Config
from pacmoe.data import Bundle, DateBatchSampler, apply_market_mean
from pacmoe.evaluate import MODEL_INPUT_KEYS, autocast, binary_metrics, configure_backends, predict, to_device
from pacmoe.losses import balanced_class_weights, total_loss
from pacmoe.model import PaCMoE

logger = logging.getLogger("pacmoe.train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(epoch_index: int, epochs: int) -> float:
    """LR multiplier for the epoch with 0-based index `epoch_index`."""
    if epoch_index < K.WARMUP_EPOCHS:
        return float(epoch_index + 1) / max(1, K.WARMUP_EPOCHS)
    progress = float(epoch_index - K.WARMUP_EPOCHS) / max(1, epochs - K.WARMUP_EPOCHS)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


class Trainer:
    def __init__(self, cfg: Config, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        configure_backends()
        set_seed(cfg.seed)
        rt = cfg.details["return_target"]
        self.train = Bundle(cfg.bundle_dir, "train", rt, graph_snapshots_path=cfg.graph_snapshots)
        self.val = Bundle(cfg.bundle_dir, "val", rt, graph_snapshots_path=cfg.graph_snapshots)
        self.test = Bundle(cfg.bundle_dir, "test", rt, graph_snapshots_path=cfg.graph_snapshots)
        self.model = PaCMoE(cfg.model_spec(int(self.train.text_features.shape[-1]))).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=K.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda e: warmup_cosine(e, cfg.epochs))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        counts = [int((self.train.y_cls == c).sum()) for c in range(self.model.spec.num_classes)]
        self.class_weight = balanced_class_weights(counts).to(self.device)
        self.sampler = DateBatchSampler(self.train.dates, cfg.batch_size, shuffle=True, seed=cfg.seed)
        graph_prefixes = ("graph_encoder.", "pre_fusion_graph.")
        self.graph_params = [p for n, p in self.model.named_parameters() if n.startswith(graph_prefixes)]
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _step(self, indices: List[int], epoch: int, step_optimizer: bool = True) -> Dict[str, float]:
        """One optimization step on a batch of sample indices."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = to_device(apply_market_mean(self.train.collate(indices)), self.device)
        with autocast(self.device):
            out = self.model(**{k: batch[k] for k in MODEL_INPUT_KEYS})
            losses = total_loss(out, batch, weights=self.cfg.loss_weights, class_weight=self.class_weight,
                                huber_delta=self.cfg.huber_delta, epoch=epoch, details=self.cfg.details,
                                disabled_mask=self.model.fusion.disabled_mask)
        loss = losses["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        self.scaler.scale(loss).backward()
        if step_optimizer:
            self.scaler.unscale_(self.optimizer)
            for p in self.model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad.zero_()
            graph_grads = [p for p in self.graph_params if p.grad is not None]
            if graph_grads:
                torch.nn.utils.clip_grad_norm_(graph_grads, K.GRAPH_GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), K.GRAD_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return {k: float(v.detach().item()) for k, v in losses.items()}

    def evaluate(self, bundle: Bundle) -> Dict[str, float]:
        res = predict(self.model, bundle, self.cfg.batch_size, self.device)
        return binary_metrics(res["probs"], res["labels"], self.cfg.threshold)

    def fit(self, select_epochs: Optional[Tuple[int, int]] = None) -> Dict[str, object]:
        """
        Train for cfg.epochs epochs; the epoch with the highest validation MCC (within `select_epochs`
        if given, earlier epoch on ties) is saved and evaluated once on the test split.
        """
        self.sampler.set_epoch(1)
        self._step(next(iter(self.sampler)), epoch=1, step_optimizer=False)
        history, best = [], {"val_mcc": -float("inf")}
        for epoch in range(1, self.cfg.epochs + 1):
            self.sampler.set_epoch(epoch)
            losses = [self._step(idx, epoch)
                      for idx in tqdm(self.sampler, desc=f"epoch {epoch}/{self.cfg.epochs}", unit="batch", leave=False)]
            self.scheduler.step()
            val = self.evaluate(self.val)
            row = {"epoch": epoch, "train_loss": float(np.mean([l["total"] for l in losses])), "val": val,
                   "lr": self.optimizer.param_groups[0]["lr"]}
            history.append(row)
            logger.info("epoch %d | loss %.4f | val acc %.4f mcc %+.4f", epoch, row["train_loss"], val["accuracy"], val["mcc"])
            eligible = epoch >= K.CHECKPOINT_MIN_EPOCH and (select_epochs is None or select_epochs[0] <= epoch <= select_epochs[1])
            if eligible and val["mcc"] > best["val_mcc"]:
                best = {"epoch": epoch, "val_mcc": val["mcc"], "val": val}
                torch.save({"model_state": self.model.state_dict(), "epoch": epoch, "val_mcc": val["mcc"]}, self.out_dir / "best.pt")
        # The test split is evaluated once, with the checkpoint selected on the validation split.
        state = torch.load(self.out_dir / "best.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        best["test"] = self.evaluate(self.test)
        logger.info("selected epoch %d | test acc %.4f mcc %+.4f", best["epoch"], best["test"]["accuracy"], best["test"]["mcc"])
        result = {"best": best, "history": history}
        (self.out_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        return result
