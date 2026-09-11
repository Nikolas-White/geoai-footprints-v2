"""Score a checkpoint on the out-of-domain validation region.

This is the number ablations are chosen by. It must NOT be the test regions:
selecting techniques on Kitsap and Tyrol would make every honest claim in the
report false. Instead a training-set city (Vienna by default) is withheld
entirely, which makes it a legitimate unseen region for these runs.

    python -m src.experiments.oodval --config configs/default.yaml \
        --checkpoint outputs/run_A_baseline/best.pth --tag A_baseline

Appends one row to outputs/experiments/ablation_results.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from tqdm import tqdm

from src.common.config import (apply_overrides, experiment_settings,
                               load_config, resolve)
from src.inference.predict_scenes import predict_scene
from src.models.build import build_model

EPS = 1e-9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True, help="row label in the results table")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.overrides)
    exp = experiment_settings(cfg)
    data_dir = resolve(cfg["paths"]["processed_dir"]) / exp["mode"]

    scenes_csv = data_dir / "oodval_scenes.csv"
    if not scenes_csv.exists():
        print(f"[FAIL] {scenes_csv} not found. Re-run make_tiles in "
              f"experiment.mode=ablation.")
        return 1
    scenes = pd.read_csv(scenes_csv).to_dict("records")
    if args.limit:
        scenes = scenes[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(resolve(args.checkpoint), map_location=device,
                      weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    thr = cfg["inference"]["threshold_interior"]
    tp = fp = fn = 0
    t0 = time.time()
    for s in tqdm(scenes, desc=f"oodval[{args.tag}]", ncols=90):
        prob, _ = predict_scene(model, Path(s["image_path"]), device, cfg,
                                tta=args.tta)
        pred = prob[0] >= thr
        with rasterio.open(s["gt_path"]) as src:
            gt = src.read(1) > 127
        tp += int((pred & gt).sum())
        fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum())

    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    row = {
        "tag": args.tag,
        "oodval_cities": ",".join(exp.get("oodval_cities") or []),
        "scenes": len(scenes),
        "iou": round(tp / (tp + fp + fn + EPS), 4),
        "f1": round(2 * prec * rec / (prec + rec + EPS), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "normalization": cfg["preprocess"]["normalization"],
        "extra_channels": ",".join(cfg["preprocess"].get("extra_channels") or []),
        "loss": cfg["train"]["loss"]["type"],
        "scale_range": str(cfg["augment"]["scale_range"]),
        "fda": cfg["augment"]["fda_mode"],
        "encoder": cfg["train"]["encoder"],
        "arch": cfg["train"].get("arch", "unet"),
        "epochs": cfg["train"]["epochs"],
        "in_domain_val_iou": round(float(ckpt.get("best_iou", float("nan"))), 4),
        "tta": args.tta,
        "minutes": round((time.time() - t0) / 60.0, 1),
    }

    out_dir = resolve(cfg["paths"]["outputs_dir"]) / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ablation_results.csv"
    df = pd.DataFrame([row])
    if path.exists():
        prev = pd.read_csv(path)
        prev = prev[prev.tag != args.tag]          # re-running replaces
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(path, index=False)

    print(f"\n{args.tag}:  OOD IoU {row['iou']:.4f}   F1 {row['f1']:.4f}   "
          f"P {row['precision']:.4f}   R {row['recall']:.4f}"
          f"   (in-domain val {row['in_domain_val_iou']:.4f})")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
