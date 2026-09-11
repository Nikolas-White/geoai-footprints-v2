"""Train the dual-head U-Net.

Usage (from the repository root, venv active):

    # 2-minute smoke test on a few hundred tiles
    python -m src.train.train --config configs/default.yaml --smoke

    # full run
    python -m src.train.train --config configs/default.yaml

    # resume after an interruption
    python -m src.train.train --config configs/default.yaml --resume

Outputs land in outputs/run_<name>/:
    best.pth      checkpoint with the highest validation interior IoU
    last.pth      most recent epoch (for resuming)
    history.csv   per-epoch losses and metrics
    config.yaml   exact config this run used
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.config import (apply_overrides, experiment_settings,
                               load_config, resolve, set_seed)
from src.data.dataset import TileDatasetV2
from src.models.build import build_model, count_parameters
from src.train.losses import SegLossV2
from src.train.metrics import SegMetrics


# --------------------------------------------------------------------------- #
def make_loaders(cfg, smoke: bool, workers: Optional[int], data_dir):
    nw = cfg["train"]["num_workers"] if workers is None else workers
    limit = 400 if smoke else None

    train_ds = TileDatasetV2(data_dir / "manifest_train.csv", data_dir,
                             train=True, cfg=cfg, seed=cfg["project"]["seed"],
                             limit=limit)
    val_ds = TileDatasetV2(data_dir / "manifest_val.csv", data_dir,
                           train=False, cfg=cfg, seed=cfg["project"]["seed"],
                           limit=(limit // 2 if limit else None))

    common = dict(num_workers=nw, pin_memory=True,
                  persistent_workers=nw > 0,
                  prefetch_factor=4 if nw > 0 else None)
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                          shuffle=True, drop_last=True, **common)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, drop_last=False, **common)
    return train_ds, val_ds, train_dl, val_dl


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


# --------------------------------------------------------------------------- #
def run_epoch(
    model, loader, criterion, optimizer, scaler, scheduler_fn,
    device, epoch, epochs, train: bool, global_step: int, amp: bool,
):
    model.train(train)
    metrics = SegMetrics()
    running: Dict[str, float] = {}
    n = 0

    desc = f"{'train' if train else 'val  '} {epoch + 1}/{epochs}"
    bar = tqdm(loader, desc=desc, ncols=110, leave=False)

    for x, y in bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
                loss, parts = criterion(logits, y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            lr_scale = scheduler_fn(global_step)
            for g in optimizer.param_groups:
                g["lr"] = g["initial_lr"] * lr_scale

        metrics.update(logits.detach().float(), y)
        n += 1
        for k, v in parts.items():
            running[k] = running.get(k, 0.0) + v
        bar.set_postfix(
            loss=f"{running['loss'] / n:.4f}",
            iou=f"{metrics.interior.compute()['iou']:.3f}",
        )

    out = {k: v / max(1, n) for k, v in running.items()}
    out.update(metrics.compute())
    bar.close()
    return out, global_step


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--name", default=None, help="run name; defaults to encoder")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset, 2 epochs -- verifies the loop runs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="override any config key, e.g. --set preprocess.normalization=per_tile")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args.overrides)
    exp = experiment_settings(cfg)
    cfg["train"]["epochs"] = exp["epochs"]
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.smoke:
        cfg["train"]["epochs"] = 2

    set_seed(cfg['project']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = resolve(cfg["paths"]["processed_dir"]) / exp["mode"]
    name = args.name or ("smoke" if args.smoke else cfg["train"]["encoder"])
    run_dir = resolve(cfg['paths']['outputs_dir']) / f"run_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(resolve(args.config), run_dir / "config.yaml")

    train_ds, val_ds, train_dl, val_dl = make_loaders(cfg, args.smoke,
                                                      args.workers, data_dir)
    model = build_model(cfg).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    criterion = SegLossV2(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    epochs = cfg["train"]["epochs"]
    steps_per_epoch = max(1, len(train_dl))
    total_steps = epochs * steps_per_epoch
    warmup = min(500, max(50, steps_per_epoch // 2))
    sched = lambda s: cosine_with_warmup(s, total_steps, warmup)  # noqa: E731

    start_epoch, global_step, best_iou = 0, 0, -1.0
    last_ckpt = run_dir / "last.pth"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        global_step = ck["global_step"]
        best_iou = ck.get("best_iou", -1.0)
        print(f"Resumed from epoch {start_epoch}, best IoU so far {best_iou:.4f}")

    print("=" * 78)
    print(f"run          : {run_dir}")
    print(f"device       : {device}  amp={amp}")
    print(f"mode         : {exp['mode']}  train={exp['train_cities']} "
          f"oodval={exp['oodval_cities']}")
    print(f"arch         : {cfg['train'].get('arch','unet')} / {cfg['train']['encoder']}")
    print(f"preprocess   : norm={cfg['preprocess']['normalization']} "
          f"extra={cfg['preprocess'].get('extra_channels') or []}  "
          f"in_ch={train_ds.in_channels}")
    print(f"loss         : {cfg['train']['loss']['type']}")
    print(f"augment      : scale={cfg['augment']['scale_range']} "
          f"fda={cfg['augment']['fda_mode']}")
    print(f"params       : {count_parameters(model) / 1e6:.1f} M")
    print(f"train tiles  : {len(train_ds)}   val tiles: {len(val_ds)}")
    print(f"batch size   : {cfg['train']['batch_size']}   epochs: {epochs}")
    print(f"steps/epoch  : {steps_per_epoch}   total: {total_steps}")
    print("=" * 78)

    hist_path = run_dir / "history.csv"
    if not args.resume or not hist_path.exists():
        hist_path.write_text("", encoding="utf-8")
    history_rows = []

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        tr, global_step = run_epoch(
            model, train_dl, criterion, optimizer, scaler, sched,
            device, epoch, epochs, True, global_step, amp,
        )
        va, _ = run_epoch(
            model, val_dl, criterion, optimizer, scaler, sched,
            device, epoch, epochs, False, global_step, amp,
        )
        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch + 1:3d}/{epochs}  "
            f"{dt:5.0f}s  lr {cur_lr:.2e}  |  "
            f"train loss {tr['loss']:.4f} IoU {tr['int_iou']:.4f}  |  "
            f"val loss {va['loss']:.4f} IoU {va['int_iou']:.4f} "
            f"F1 {va['int_f1']:.4f}  bndIoU {va['bnd_iou']:.4f}"
        )

        row = {"epoch": epoch + 1, "seconds": round(dt, 1), "lr": cur_lr}
        row.update({f"train_{k}": v for k, v in tr.items()})
        row.update({f"val_{k}": v for k, v in va.items()})
        history_rows.append(row)
        with open(hist_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(history_rows[0].keys()))
            w.writeheader()
            w.writerows(history_rows)

        improved = va["int_iou"] > best_iou
        if improved:
            best_iou = va["int_iou"]

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_iou": best_iou,
            "cfg": json.loads(json.dumps(cfg)),
        }
        torch.save(state, last_ckpt)
        if improved:
            torch.save(state, run_dir / "best.pth")
            print(f"           new best val IoU {best_iou:.4f} -> best.pth")

    print("=" * 78)
    print(f"Done. Best validation interior IoU: {best_iou:.4f}")
    print(f"Checkpoint: {run_dir / 'best.pth'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
