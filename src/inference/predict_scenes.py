"""Sliding-window inference across full 5000x5000 scenes.

Two details matter here and both are easy to get wrong:

1. **Seam blending.** Naive tiling leaves visible grid artefacts where windows
   meet, because a pixel at the very edge of a tile has no surrounding context
   and the model is least confident there. We weight each window with a 2D
   raised-cosine and accumulate weighted sums, so predictions near a window's
   centre dominate and seams disappear.

2. **Georeferencing.** Output rasters inherit the source CRS and affine
   transform, so downstream polygons land in real-world coordinates
   (UTM metres) rather than pixel space. Without this the vectorizer cannot
   apply metric thresholds and the GeoPackage is useless in QGIS.

Usage:

    python -m src.inference.predict_scenes --config configs/default.yaml \
        --checkpoint outputs/run_resnet50/best.pth --limit 4

    # all 72 test scenes with 8-fold test-time augmentation
    python -m src.inference.predict_scenes --config configs/default.yaml \
        --checkpoint outputs/run_resnet50/best.pth --tta

Writes outputs/predictions/<scene>_prob.tif: a 2-band uint8 GeoTIFF where
band 1 is interior probability and band 2 is boundary probability, both
scaled 0-255 and DEFLATE compressed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
from tqdm import tqdm

from src.common.config import (apply_overrides, experiment_settings,
                               load_config, resolve)
from src.data.augment import build_input
from src.models.build import build_model


# --------------------------------------------------------------------------- #
def cosine_window(size: int, overlap: int) -> np.ndarray:
    """2D raised-cosine taper. Flat in the middle, falls to ~0 at the edges."""
    ramp = np.ones(size, dtype=np.float32)
    if overlap > 0:
        t = np.linspace(0, np.pi, overlap * 2, dtype=np.float32)
        taper = (1.0 - np.cos(t)) / 2.0
        ramp[:overlap] = taper[:overlap]
        ramp[-overlap:] = taper[overlap:]
    w = np.outer(ramp, ramp).astype(np.float32)
    return np.maximum(w, 1e-3)  # never exactly zero, or edge pixels divide by 0


def window_starts(total: int, tile: int, stride: int) -> List[int]:
    if total <= tile:
        return [0]
    pos = list(range(0, total - tile + 1, stride))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


# --------------------------------------------------------------------------- #
TTA_OPS = [
    (0, False), (1, False), (2, False), (3, False),
    (0, True), (1, True), (2, True), (3, True),
]


def _apply_tta(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if flip:
        x = torch.flip(x, dims=[3])
    if k:
        x = torch.rot90(x, k, dims=[2, 3])
    return x


def _undo_tta(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if k:
        x = torch.rot90(x, -k, dims=[2, 3])
    if flip:
        x = torch.flip(x, dims=[3])
    return x


@torch.no_grad()
def infer_batch(model, batch: torch.Tensor, amp: bool, tta: bool) -> torch.Tensor:
    ops = TTA_OPS if tta else [(0, False)]
    acc = None
    for k, flip in ops:
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(_apply_tta(batch, k, flip))
        out = _undo_tta(out.float(), k, flip)
        prob = torch.sigmoid(out)
        acc = prob if acc is None else acc + prob
    return acc / len(ops)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_scene(
    model, img_path: Path, device, cfg, tta: bool = False
) -> Tuple[np.ndarray, dict]:
    """Return a (2, H, W) float32 probability array plus the rasterio profile."""
    tile = cfg["inference"]["tile_size"]
    overlap = cfg["inference"]["overlap"]
    stride = tile - overlap
    bs = cfg["inference"]["batch_size"]
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    norm = cfg["preprocess"]["normalization"]
    extra = list(cfg["preprocess"].get("extra_channels") or [])

    with rasterio.open(img_path) as src:
        image = np.transpose(src.read([1, 2, 3]), (1, 2, 0))  # HWC RGB uint8
        profile = src.profile.copy()

    h, w = image.shape[:2]
    prob_sum = np.zeros((2, h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    win = cosine_window(tile, overlap // 2)

    coords = [(r, c) for r in window_starts(h, tile, stride)
              for c in window_starts(w, tile, stride)]

    buf: List[np.ndarray] = []
    buf_rc: List[Tuple[int, int]] = []

    def flush():
        if not buf:
            return
        batch = torch.from_numpy(np.stack(buf)).to(device, non_blocking=True)
        probs = infer_batch(model, batch, amp, tta).cpu().numpy()
        for (r, c), p in zip(buf_rc, probs):
            prob_sum[:, r : r + tile, c : c + tile] += p * win
            weight_sum[r : r + tile, c : c + tile] += win
        buf.clear()
        buf_rc.clear()

    for r, c in coords:
        chip = image[r : r + tile, c : c + tile]
        buf.append(build_input(chip, norm, extra))
        buf_rc.append((r, c))
        if len(buf) == bs:
            flush()
    flush()

    prob = prob_sum / np.maximum(weight_sum, 1e-6)[None]
    return np.clip(prob, 0.0, 1.0), profile


def write_prob_raster(prob: np.ndarray, profile: dict, out_path: Path) -> None:
    """uint8 + DEFLATE: ~20x smaller than float32 and lossless enough at 1/255
    resolution, which is far finer than any threshold we apply downstream."""
    prof = profile.copy()
    prof.update(
        dtype="uint8", count=2, compress="deflate", predictor=2,
        tiled=True, blockxsize=512, blockysize=512, nodata=None,
    )
    prof.pop("photometric", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write((prob[0] * 255).round().astype(np.uint8), 1)
        dst.write((prob[1] * 255).round().astype(np.uint8), 2)
        dst.set_band_description(1, "interior_probability")
        dst.set_band_description(2, "boundary_probability")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default="outputs/run_resnet50/best.pth")
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="test = unseen regions; val = held-out scenes of train cities")
    ap.add_argument("--limit", type=int, default=0, help="only first N scenes")
    ap.add_argument("--scenes", default=None,
                    help="comma-separated scene names, e.g. kitsap1,tyrol-w3")
    ap.add_argument("--tta", action="store_true",
                    help="8-fold flip/rotate test-time augmentation (8x slower)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[FAIL] checkpoint not found: {ckpt_path}")
        return 1
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    print(f"Loaded {ckpt_path.name} (epoch {ckpt['epoch'] + 1}, "
          f"val IoU {ckpt.get('best_iou', float('nan')):.4f})")

    # Tiles and scene manifests live under the experiment mode, because
    # ablation and final use different strides and different city subsets.
    processed = resolve(cfg["paths"]["processed_dir"]) / experiment_settings(cfg)["mode"]
    if args.split == "test":
        scenes = pd.read_csv(processed / "test_scenes.csv").to_dict("records")
    else:
        inria = resolve(cfg.paths.inria_dir)
        val_ids = set(cfg.data.val_image_ids)
        scenes = [
            {"scene": f"{city}{i}",
             "city": city,
             "image_path": str(inria / "images" / f"{city}{i}.tif"),
             "gt_path": str(inria / "gt" / f"{city}{i}.tif")}
            for city in cfg.data.train_cities for i in sorted(val_ids)
        ]

    if args.scenes:
        keep = {s.strip() for s in args.scenes.split(",")}
        scenes = [s for s in scenes if s["scene"] in keep]
    if args.limit:
        scenes = scenes[: args.limit]

    out_dir = resolve(cfg.paths.outputs_dir) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scenes: {len(scenes)}   TTA: {args.tta}   -> {out_dir}\n")

    t0 = time.time()
    for s in tqdm(scenes, desc="scenes", ncols=90):
        prob, profile = predict_scene(
            model, Path(s["image_path"]), device, cfg, tta=args.tta
        )
        write_prob_raster(prob, profile, out_dir / f"{s['scene']}_prob.tif")
    dt = time.time() - t0

    print(f"\nDone in {dt / 60:.1f} min ({dt / max(1, len(scenes)):.1f} s/scene)")
    print(f"Probability rasters written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
