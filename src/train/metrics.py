"""Pixel-level metrics.

Accumulated as raw counts over the entire validation set and reduced once at
the end, rather than averaged per batch. Per-batch averaging silently inflates
IoU, because batches that happen to contain almost no buildings score near 1.0
and drag the mean up. Dataset-level accumulation is the number you can compare
against published benchmarks.
"""
from __future__ import annotations

from typing import Dict

import torch


class BinaryCounter:
    """Streaming TP / FP / FN accumulator for one output channel."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = (torch.sigmoid(logits) > self.threshold)
        gt = target > 0.5
        self.tp += int((pred & gt).sum())
        self.fp += int((pred & ~gt).sum())
        self.fn += int((~pred & gt).sum())
        self.tn += int((~pred & ~gt).sum())

    def compute(self) -> Dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-9
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        return {
            "iou": tp / (tp + fp + fn + eps),
            "f1": 2 * precision * recall / (precision + recall + eps),
            "precision": precision,
            "recall": recall,
            "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        }


class SegMetrics:
    """Both output channels at once."""

    def __init__(self, thr_interior: float = 0.5, thr_boundary: float = 0.5):
        self.interior = BinaryCounter(thr_interior)
        self.boundary = BinaryCounter(thr_boundary)

    def reset(self) -> None:
        self.interior.reset()
        self.boundary.reset()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        self.interior.update(logits[:, 0], target[:, 0])
        self.boundary.update(logits[:, 1], target[:, 1])

    def compute(self) -> Dict[str, float]:
        out = {f"int_{k}": v for k, v in self.interior.compute().items()}
        out.update({f"bnd_{k}": v for k, v in self.boundary.compute().items()})
        return out
