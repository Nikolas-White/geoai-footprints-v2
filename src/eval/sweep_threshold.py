"""Calibrate the decision threshold on already-saved probability rasters.

The default 0.5 threshold is an arbitrary inheritance from binary
classification, and out-of-domain it is usually wrong. On the unseen regions
the model runs precision-heavy (0.86 precision against 0.77 recall), which
means 0.5 is throwing away real buildings the model was genuinely unsure about
because they look nothing like Austin or Vienna.

Implementation note: the sweep costs one pass over each raster, not one pass
per threshold. We histogram the predicted probability byte separately over
positive and negative ground-truth pixels; every threshold's TP/FP/FN then
falls out of the cumulative sums. 256 thresholds for the price of one.

    python -m src.eval.sweep_threshold --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from src.common.config import (experiment_settings, load_config,
                               resolve)

EPS = 1e-9


def scene_histograms(prob_path: Path, gt_path: Path) -> Dict[str, np.ndarray]:
    with rasterio.open(prob_path) as src:
        prob = src.read(1)                    # uint8 0-255
    with rasterio.open(gt_path) as src:
        gt = src.read(1) > 127
    pos = np.bincount(prob[gt].ravel(), minlength=256).astype(np.int64)
    neg = np.bincount(prob[~gt].ravel(), minlength=256).astype(np.int64)
    return {"pos": pos, "neg": neg}


def curve_from_hist(pos: np.ndarray, neg: np.ndarray) -> pd.DataFrame:
    """For threshold t: predict positive where prob >= t."""
    # suffix sums: pixels with value >= t
    pos_ge = np.cumsum(pos[::-1])[::-1]
    neg_ge = np.cumsum(neg[::-1])[::-1]
    total_pos = pos.sum()

    t = np.arange(256)
    tp = pos_ge.astype(np.float64)
    fp = neg_ge.astype(np.float64)
    fn = total_pos - tp

    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    return pd.DataFrame({
        "threshold": t / 255.0,
        "iou": tp / (tp + fp + fn + EPS),
        "f1": 2 * prec * rec / (prec + rec + EPS),
        "precision": prec,
        "recall": rec,
    })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    outputs = resolve(cfg.paths.outputs_dir)
    # Tiles and scene manifests live under the experiment mode, because
    # ablation and final use different strides and different city subsets.
    processed = resolve(cfg["paths"]["processed_dir"]) / experiment_settings(cfg)["mode"]
    pred_dir = outputs / "predictions"
    eval_dir = outputs / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    scenes = pd.read_csv(processed / "test_scenes.csv").to_dict("records")
    have = {p.stem.replace("_prob", "") for p in pred_dir.glob("*_prob.tif")}
    scenes = [s for s in scenes if s["scene"] in have]
    if args.limit:
        scenes = scenes[: args.limit]
    if not scenes:
        print("[FAIL] no probability rasters found.")
        return 1

    acc: Dict[str, Dict[str, np.ndarray]] = {}
    for s in tqdm(scenes, desc="histogramming", ncols=90):
        h = scene_histograms(pred_dir / f"{s['scene']}_prob.tif", Path(s["gt_path"]))
        for key in ("ALL", s["city"]):
            if key not in acc:
                acc[key] = {"pos": np.zeros(256, np.int64),
                            "neg": np.zeros(256, np.int64)}
            acc[key]["pos"] += h["pos"]
            acc[key]["neg"] += h["neg"]

    rows = []
    print("\n" + "=" * 78)
    print("THRESHOLD CALIBRATION")
    print("=" * 78)
    for group, h in acc.items():
        curve = curve_from_hist(h["pos"], h["neg"])
        curve["group"] = group
        rows.append(curve)

        best_iou = curve.loc[curve["iou"].idxmax()]
        best_f1 = curve.loc[curve["f1"].idxmax()]
        at_half = curve.iloc[128]

        print(f"\n{group}")
        print(f"  at 0.500 (current) : IoU {at_half.iou:.4f}  F1 {at_half.f1:.4f}  "
              f"P {at_half.precision:.4f}  R {at_half.recall:.4f}")
        print(f"  best IoU  @ {best_iou.threshold:.3f} : IoU {best_iou.iou:.4f}  "
              f"F1 {best_iou.f1:.4f}  P {best_iou.precision:.4f}  R {best_iou.recall:.4f}")
        print(f"  best F1   @ {best_f1.threshold:.3f} : IoU {best_f1.iou:.4f}  "
              f"F1 {best_f1.f1:.4f}  P {best_f1.precision:.4f}  R {best_f1.recall:.4f}")
        print(f"  gain from recalibration: {best_iou.iou - at_half.iou:+.4f} IoU")

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(eval_dir / "threshold_sweep.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for group in acc:
            c = df[df.group == group]
            ax[0].plot(c.threshold, c.iou, label=group)
            ax[1].plot(c.recall, c.precision, label=group)
        ax[0].axvline(0.5, color="k", ls=":", lw=1, label="default 0.5")
        ax[0].set_xlabel("decision threshold")
        ax[0].set_ylabel("IoU")
        ax[0].set_title("IoU vs threshold (unseen regions)")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        ax[1].set_xlabel("recall")
        ax[1].set_ylabel("precision")
        ax[1].set_title("Precision-recall")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(eval_dir / "threshold_sweep.png", dpi=140)
        print(f"\nPlot: {eval_dir / 'threshold_sweep.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"(plot skipped: {exc})")

    print(f"Curves: {eval_dir / 'threshold_sweep.csv'}")
    print("\nIf the optimum is far from 0.500, set inference.threshold_interior")
    print("in configs/default.yaml and re-run vectorize + evaluate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
