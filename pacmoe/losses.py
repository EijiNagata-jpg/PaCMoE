"""Training objective.

Stated in the paper (Section 2.6): L = w_cls * CE(direction) + w_ret * Huber(standardized return)
                                     + w_vol * BCE(volatility),  weights = [1.0, 0.05, 0.05].
Additional terms of the implementation (weights in constants.py):
  - router regularizers and load balancing,
  - branch auxiliary classification (one linear head per modality representation),
  - modality negative learning (price branch as teacher for the text and graph branches),
  - routing auxiliary loss (one classification head per pairwise expert; from epoch 3),
  - RankNet ranking loss on next-day returns.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from pacmoe import constants as K
from pacmoe.nn.fusion import load_balance_loss, router_prior_losses


def balanced_class_weights(counts) -> torch.Tensor:
    """total / (n_classes * count_c)."""
    c = torch.tensor(counts, dtype=torch.float32)
    return c.sum() / (float(len(c)) * c.clamp(min=1.0))


def _mean(loss: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    w = sample_weight.to(device=loss.device, dtype=loss.dtype).view(-1)
    return (loss.view(-1) * w).sum() / w.sum().clamp_min(1e-12)


def task_losses(out, batch, weights, class_weight: Optional[torch.Tensor], huber_delta: float) -> Dict[str, torch.Tensor]:
    """The three task losses of Section 2.6 and their weighted sum."""
    sw = torch.ones(batch["y_cls"].shape[0], device=out["logits_cls"].device)
    cw = class_weight.to(out["logits_cls"].device) if class_weight is not None else None
    l_cls = _mean(F.cross_entropy(out["logits_cls"], batch["y_cls"].long(), weight=cw, label_smoothing=K.CLS_LABEL_SMOOTHING, reduction="none"), sw)
    l_ret = _mean(F.huber_loss(out["pred_ret"].squeeze(-1), batch["y_ret"].float(), delta=huber_delta, reduction="none"), sw)
    pos_w = torch.tensor([K.VOL_POS_WEIGHT], device=out["logits_vol"].device)
    l_vol = _mean(F.binary_cross_entropy_with_logits(out["logits_vol"].squeeze(-1), batch["y_vol"].float(), pos_weight=pos_w, reduction="none"), sw)
    w1, w2, w3 = weights
    return {"loss": w1 * l_cls + w2 * l_ret + w3 * l_vol, "loss_cls": l_cls, "loss_ret": l_ret, "loss_vol": l_vol}


def branch_aux_loss(out, y_cls: torch.Tensor) -> torch.Tensor:
    """sum_m w * w_m * CE(head_m(h_m), y) over samples where modality m is available."""
    total = torch.zeros((), device=y_cls.device, dtype=torch.float32)
    for name in ("price", "text", "graph"):
        if name not in out["branch_logits"]:
            continue
        mask = out["branch_availability"][name].to(device=y_cls.device, dtype=torch.bool)
        if not bool(mask.any().item()):
            continue
        loss = F.cross_entropy(out["branch_logits"][name][mask], y_cls[mask].long(), reduction="mean")
        total = total + loss * (K.BRANCH_AUX_WEIGHT * K.BRANCH_AUX_MODALITY_WEIGHTS[name])
    return total


def negative_learning_loss(out, y_cls: torch.Tensor) -> torch.Tensor:
    """Where the (detached) price branch is confident, the text and graph branches are penalized for
    giving the class the price branch rules out more probability than the price branch plus a margin."""
    teacher = F.softmax(out["branch_logits"]["price"].detach(), dim=-1)
    conf, _ = teacher.max(dim=-1)
    neg_idx = teacher.argmin(dim=-1)
    teacher_neg = teacher.gather(1, neg_idx.unsqueeze(1)).squeeze(1)
    active = conf >= K.NEGATIVE_LEARNING_CONFIDENCE
    t_weight = ((conf - K.NEGATIVE_LEARNING_CONFIDENCE) / max(1.0 - K.NEGATIVE_LEARNING_CONFIDENCE, 1e-6)).clamp(0.0, 1.0)
    total = torch.zeros((), device=y_cls.device, dtype=torch.float32)
    for name in ("text", "graph"):
        if name not in out["branch_logits"]:
            continue
        student = F.softmax(out["branch_logits"][name], dim=-1)
        student_neg = student.gather(1, neg_idx.unsqueeze(1)).squeeze(1)
        sample_loss = F.relu(student_neg - (teacher_neg + K.NEGATIVE_LEARNING_MARGIN).clamp(0.0, 1.0))
        mask = active & out["branch_availability"][name].to(device=y_cls.device, dtype=torch.bool)
        if not bool(mask.any().item()):
            continue
        w = t_weight * mask.float()
        total = total + (sample_loss * w).sum() / w.sum().clamp_min(1.0) * K.NEGATIVE_LEARNING_WEIGHT
    return total


def routing_aux_loss(out, y_cls: torch.Tensor) -> torch.Tensor:
    """Mean CE of the per-expert heads over available experts (router KL term has weight 0)."""
    logits = out["expert_logits"]  # (B, E, C)
    avail = out["fusion"]["availability"].to(dtype=torch.bool)
    b, e, c = logits.shape
    ce = F.cross_entropy(logits.reshape(b * e, c), y_cls[:, None].expand(b, e).reshape(-1).long(), reduction="none").reshape(b, e)
    if not bool(avail.any().item()):
        return torch.zeros((), device=y_cls.device)
    return ce[avail].mean() * K.ROUTING_AUX_EXPERT_WEIGHT


def ranknet_loss(scores: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """RankNet over all pairs of the batch with different returns."""
    scores, returns = scores.float().reshape(-1), returns.float().reshape(-1)
    diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    label_diff = returns.unsqueeze(1) - returns.unsqueeze(0)
    target = 0.5 * (1.0 + torch.sign(label_diff))
    mask = label_diff.abs() > 1e-6
    if int(mask.sum().item()) == 0:
        return scores.new_zeros(())
    return F.binary_cross_entropy_with_logits(diff[mask], target[mask], reduction="mean")


def total_loss(out, batch, *, weights, class_weight, huber_delta: float, epoch: int, details: Dict, disabled_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Full training objective of one batch."""
    parts = task_losses(out, batch, weights, class_weight, huber_delta)
    y = batch["y_cls"]
    fx = out["fusion"]
    priors = router_prior_losses(fx, disabled_mask)
    aux = K.LOAD_BALANCE_WEIGHT * load_balance_loss(fx["dense_probs"], disabled_mask)
    prior_weights = dict(K.ROUTER_PRIOR_WEIGHTS)
    if details.get("router_z_prior_off"):
        prior_weights["z"] = 0.0
    aux = aux + sum(prior_weights[k] * v for k, v in priors.items())
    aux = aux + negative_learning_loss(out, y)
    if details["branch_aux"]:
        aux = aux + branch_aux_loss(out, y)
    if epoch >= K.ROUTING_AUX_START_EPOCH and "expert_logits" in out:
        aux = aux + routing_aux_loss(out, y)
    total = parts["loss"] + aux
    if details["ranking"] and out["logit_rank"] is not None:
        total = total + K.RANKING_LOSS_WEIGHT * ranknet_loss(out["logit_rank"], batch["ret_decimal"])
    parts["total"] = total
    parts["aux"] = aux
    return parts
