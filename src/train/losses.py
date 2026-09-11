"""Loss functions.

BCE alone handles the interior channel fine (buildings cover ~20-25% of these
tiles, which is not badly imbalanced), but the boundary channel is only a few
percent positive, where BCE happily converges to predicting all-zeros. Soft
Dice is scale-invariant with respect to class frequency, so combining the two
keeps the boundary head honest.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                   eps: float = 1.0) -> torch.Tensor:
    """Per-image, per-channel soft Dice, averaged. Expects raw logits."""
    probs = torch.sigmoid(logits)
    dims = (2, 3)
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


class SegLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        lc = cfg.train.loss
        self.w_interior = float(lc.interior_weight)
        self.w_boundary = float(lc.boundary_weight)
        self.w_bce = float(lc.bce_weight)
        self.w_dice = float(lc.dice_weight)
        pw = lc.get("boundary_pos_weight", 1.0)
        self.register_buffer("boundary_pos_weight", torch.tensor(float(pw)))

    def _head(self, logits, target, pos_weight=None):
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight
        )
        dice = soft_dice_loss(logits, target)
        return self.w_bce * bce + self.w_dice * dice, bce.detach(), dice.detach()

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        li, bce_i, dice_i = self._head(logits[:, 0:1], target[:, 0:1])
        lb, bce_b, dice_b = self._head(
            logits[:, 1:2], target[:, 1:2], self.boundary_pos_weight
        )
        total = self.w_interior * li + self.w_boundary * lb
        parts = {
            "loss": float(total.detach()),
            "bce_int": float(bce_i),
            "dice_int": float(dice_i),
            "bce_bnd": float(bce_b),
            "dice_bnd": float(dice_b),
        }
        return total, parts


# --------------------------------------------------------------------------- #
# v2: Focal Tversky
# --------------------------------------------------------------------------- #
def tversky_index(logits: torch.Tensor, target: torch.Tensor,
                  alpha: float, beta: float, eps: float = 1.0) -> torch.Tensor:
    """Tversky index per image and channel.

    Dice weights false positives and false negatives equally. Tversky lets you
    tilt: with beta > alpha a missed building costs more than a spurious one.
    v1 measured precision 0.877 against recall 0.771 on unseen regions, so the
    errors that matter are misses, and this is the knob that addresses them
    directly rather than hoping a threshold change papers over it.
    """
    probs = torch.sigmoid(logits)
    dims = (2, 3)
    tp = (probs * target).sum(dims)
    fp = (probs * (1 - target)).sum(dims)
    fn = ((1 - probs) * target).sum(dims)
    return (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def focal_tversky_loss(logits: torch.Tensor, target: torch.Tensor,
                       alpha: float, beta: float, gamma: float) -> torch.Tensor:
    """(1 - Tversky) ** (1/gamma). gamma > 1 focuses on the hard examples."""
    ti = tversky_index(logits, target, alpha, beta)
    return torch.pow(torch.clamp(1.0 - ti, min=1e-7), 1.0 / max(gamma, 1e-6)).mean()


class SegLossV2(nn.Module):
    """Dispatches between the v1 BCE+Dice loss and Focal Tversky."""

    def __init__(self, cfg):
        super().__init__()
        lc = cfg["train"]["loss"]
        self.kind = lc.get("type", "bce_dice")
        self.w_interior = float(lc["interior_weight"])
        self.w_boundary = float(lc["boundary_weight"])
        self.w_bce = float(lc["bce_weight"])
        self.w_dice = float(lc["dice_weight"])
        self.alpha = float(lc.get("tversky_alpha", 0.3))
        self.beta = float(lc.get("tversky_beta", 0.7))
        self.gamma = float(lc.get("tversky_gamma", 1.0))
        pw = float(lc.get("boundary_pos_weight", 1.0))
        self.register_buffer("boundary_pos_weight", torch.tensor(pw))

    def _head(self, logits, target, pos_weight=None):
        bce = F.binary_cross_entropy_with_logits(logits, target,
                                                 pos_weight=pos_weight)
        if self.kind == "focal_tversky":
            region = focal_tversky_loss(logits, target, self.alpha,
                                        self.beta, self.gamma)
        elif self.kind == "bce_dice":
            region = soft_dice_loss(logits, target)
        else:
            raise ValueError(f"unknown loss type: {self.kind}")
        return self.w_bce * bce + self.w_dice * region, bce.detach(), region.detach()

    def forward(self, logits, target):
        li, bce_i, reg_i = self._head(logits[:, 0:1], target[:, 0:1])
        lb, bce_b, reg_b = self._head(logits[:, 1:2], target[:, 1:2],
                                      self.boundary_pos_weight)
        total = self.w_interior * li + self.w_boundary * lb
        return total, {
            "loss": float(total.detach()),
            "bce_int": float(bce_i), "reg_int": float(reg_i),
            "bce_bnd": float(bce_b), "reg_bnd": float(reg_b),
        }
