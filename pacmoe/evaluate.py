"""Inference and metrics (accuracy and MCC at the fixed threshold 0.5, Section 2.1)."""
from __future__ import annotations

import math
import os
from typing import Dict, List

import numpy as np
import torch

from pacmoe import constants as K
from pacmoe.data import Bundle, DateBatchSampler, apply_market_mean

MODEL_INPUT_KEYS = ("x_ohlcv", "x_tech", "text_features", "text_day_mask", "has_text", "edge_index", "edge_type", "symbol_ids", "ticker_ids")


def autocast(device: torch.device):
    """bfloat16 mixed precision on CUDA, used for training forward passes (evaluation runs in float32)."""
    return torch.autocast(device_type=device.type, dtype=K.AMP_DTYPE, enabled=device.type == "cuda")


def configure_backends(deterministic: bool = False) -> None:
    """
    GPU numerics of the original runs: TF32 matmul, cuDNN autotuning, flash/memory-efficient SDP.

    deterministic=True keeps TF32 but forces deterministic kernels (used to check bit-level
    agreement with the original code; the original runs themselves were not deterministic).
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)


def to_device(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(model: torch.nn.Module, bundle: Bundle, batch_size: int, device: torch.device, keep_router: bool = False) -> Dict[str, np.ndarray]:
    """
    Probability of "up", return-head and volatility-head outputs for every sample, in the order of the
    date-grouped batches. forwards.
    """
    model.eval()
    probs: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    dates: List[str] = []
    symbols: List[str] = []
    router: List[np.ndarray] = []
    pred_ret: List[np.ndarray] = []
    logit_vol: List[np.ndarray] = []
    for indices in DateBatchSampler(bundle.dates, batch_size, shuffle=False, seed=0):
        batch = to_device(apply_market_mean(bundle.collate(indices)), device)
        out = model(**{k: batch[k] for k in MODEL_INPUT_KEYS})
        p_all = torch.softmax(out["logits_cls"].float(), dim=-1)
        probs_all.append(p_all.cpu().numpy())
        probs.append(p_all[:, 1].cpu().numpy())
        labels.append(batch["y_cls"].cpu().numpy())
        pred_ret.append(out["pred_ret"].float().reshape(-1).cpu().numpy())
        logit_vol.append(out["logits_vol"].float().reshape(-1).cpu().numpy())
        dates += batch["dates"]
        symbols += batch["symbols"]
        if keep_router:
            router.append(out["router_weights"].float().cpu().numpy())
    result = {"probs": np.concatenate(probs), "probs_all": np.concatenate(probs_all), "labels": np.concatenate(labels),
              "pred_ret": np.concatenate(pred_ret), "logit_vol": np.concatenate(logit_vol),
              "dates": np.asarray(dates), "symbols": np.asarray(symbols)}
    if keep_router:
        result["router_weights"] = np.concatenate(router)
    return result


def multiclass_metrics(probs_all: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Accuracy and multiclass MCC (Gorodkin, 2004) of argmax predictions."""
    pred = np.asarray(probs_all).argmax(axis=1)
    y = np.asarray(labels).astype(int)
    k = int(max(pred.max(), y.max()) + 1)
    cm = np.zeros((k, k), dtype=np.float64)
    np.add.at(cm, (y, pred), 1.0)
    t, p, c, s = cm.sum(axis=1), cm.sum(axis=0), np.trace(cm), cm.sum()
    denom = math.sqrt((s * s - (p * p).sum()) * (s * s - (t * t).sum()))
    return {"accuracy": float((pred == y).mean()), "mcc": 0.0 if denom == 0 else float((c * s - (t * p).sum()) / denom), "n": int(len(y))}


def binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Accuracy and MCC of the predictions p >= threshold."""
    pred = (np.asarray(probs) >= threshold).astype(int)
    y = np.asarray(labels).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    denom = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {"accuracy": float((pred == y).mean()), "mcc": 0.0 if denom == 0 else (tp * tn - fp * fn) / denom, "n": int(len(y))}
