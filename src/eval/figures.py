"""Qualitative figures.

Metrics tables persuade people who already believe you. A hiring engineer
scrolling a README looks at the pictures first, and the picture that matters
here is raw contours beside regularized polygons over the same rooftops: the
IoU is nearly identical and one of them is obviously the usable one.

    python -m src.eval.figures --config configs/default.yaml --n 6

Writes outputs/figures/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio

from src.common.config import (experiment_settings, load_config,
                               resolve)

CROP = 900          # pixels; 900 * 0.3 m = 270 m across
MIN_BUILDINGS = 12  # a crop with three sheds in it demonstrates nothing


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def world_to_pixel(poly, transform, r0: int, c0: int):
    """Map polygon coordinates into crop-local pixel space."""
    inv = ~transform
    rings = []
    for ring in [poly.exterior] + list(poly.interiors):
        xy = np.asarray(ring.coords)
        cols, rows = inv * (xy[:, 0], xy[:, 1])
        rings.append(np.column_stack([np.asarray(cols) - c0,
                                      np.asarray(rows) - r0]))
    return rings


def pick_crop(gt_path: Path, size: int) -> Optional[Tuple[int, int]]:
    """Find a window with a decent building density, cheaply."""
    with rasterio.open(gt_path) as src:
        h, w = src.height, src.width
        # coarse overview read instead of the full 25 MP raster
        factor = 10
        small = src.read(
            1, out_shape=(h // factor, w // factor),
            resampling=rasterio.enums.Resampling.average,
        )
    ss = size // factor
    best, best_val = None, -1.0
    for r in range(0, small.shape[0] - ss, max(1, ss // 2)):
        for c in range(0, small.shape[1] - ss, max(1, ss // 2)):
            v = float(small[r : r + ss, c : c + ss].mean())
            if v > best_val:
                best_val, best = v, (r * factor, c * factor)
    if best_val <= 0:
        return None
    return best


def draw_panel(ax, rgb, polys, transform, r0, c0, color, title):
    ax.imshow(rgb)
    n = 0
    for poly in polys:
        if poly is None or poly.geom_type != "Polygon":
            continue
        for ring in world_to_pixel(poly, transform, r0, c0):
            if ring[:, 0].max() < 0 or ring[:, 1].max() < 0:
                continue
            if ring[:, 0].min() > rgb.shape[1] or ring[:, 1].min() > rgb.shape[0]:
                continue
            ax.plot(ring[:, 0], ring[:, 1], color=color, lw=1.1)
        n += 1
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)
    ax.axis("off")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--crop", type=int, default=CROP)
    args = ap.parse_args()

    import geopandas as gpd

    plt = _mpl()
    cfg = load_config(args.config)
    outputs = resolve(cfg.paths.outputs_dir)
    # Tiles and scene manifests live under the experiment mode, because
    # ablation and final use different strides and different city subsets.
    processed = resolve(cfg["paths"]["processed_dir"]) / experiment_settings(cfg)["mode"]
    fig_dir = outputs / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    scenes = pd.read_csv(processed / "test_scenes.csv").to_dict("records")
    have = {p.stem for p in (outputs / "vectors" / "regularized").glob("*.gpkg")}
    scenes = [s for s in scenes if s["scene"] in have]
    if not scenes:
        print("[FAIL] no vectors found; run src.vector.vectorize first.")
        return 1

    # alternate between the two unseen regions so the figure shows both
    by_city = {}
    for s in scenes:
        by_city.setdefault(s["city"], []).append(s)
    chosen, i = [], 0
    while len(chosen) < args.n and any(by_city.values()):
        for city in sorted(by_city):
            if by_city[city] and len(chosen) < args.n:
                chosen.append(by_city[city].pop(i % max(1, len(by_city[city]))))
        i += 3

    made = 0
    for s in chosen:
        scene = s["scene"]
        gt_path = Path(s["gt_path"])
        img_path = Path(s["image_path"])
        if not img_path.exists():
            continue

        crop = pick_crop(gt_path, args.crop)
        if crop is None:
            continue
        r0, c0 = crop

        with rasterio.open(img_path) as src:
            win = rasterio.windows.Window(c0, r0, args.crop, args.crop)
            rgb = np.transpose(src.read([1, 2, 3], window=win), (1, 2, 0))
            transform = src.transform

        panels = []
        for name, color, title in [
            ("ground_truth", "#00e5ff", "Ground truth (raster-traced)"),
            ("raw", "#ff5252", "Predicted, no regularization"),
            ("regularized", "#7CFF6B", "Predicted, regularized"),
        ]:
            gpkg = outputs / "vectors" / name / f"{scene}.gpkg"
            if not gpkg.exists():
                continue
            g = gpd.read_file(gpkg, layer="buildings")
            panels.append((list(g.geometry), color, title))

        if len(panels) < 2:
            continue

        fig, axes = plt.subplots(1, len(panels) + 1,
                                 figsize=(4.2 * (len(panels) + 1), 4.6))
        axes[0].imshow(rgb)
        axes[0].set_title(f"{scene}  ({s['city']})", fontsize=10)
        axes[0].axis("off")
        counts = []
        for ax, (polys, color, title) in zip(axes[1:], panels):
            counts.append(draw_panel(ax, rgb, polys, transform, r0, c0, color, title))
        fig.suptitle(
            f"{args.crop * 0.3:.0f} m across at 0.3 m/px - region never seen in training",
            fontsize=11, y=0.98,
        )
        fig.tight_layout()
        out = fig_dir / f"compare_{scene}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")
        made += 1

    # training curves
    hist = outputs / "run_resnet50" / "history.csv"
    if hist.exists():
        h = pd.read_csv(hist)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(h.epoch, h.train_loss, label="train")
        ax[0].plot(h.epoch, h.val_loss, label="val")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
        ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
        ax[1].plot(h.epoch, h.train_int_iou, label="train IoU")
        ax[1].plot(h.epoch, h.val_int_iou, label="val IoU")
        ax[1].plot(h.epoch, h.val_bnd_iou, label="val boundary IoU", ls="--")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("IoU")
        ax[1].set_title("Segmentation quality"); ax[1].legend(); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "training_curves.png", dpi=140)
        plt.close(fig)
        print(f"wrote {fig_dir / 'training_curves.png'}")

    print(f"\n{made} comparison figures in {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
