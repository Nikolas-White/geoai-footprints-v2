"""Drive the whole ablation ladder unattended.

Each rung trains a model and scores it on the out-of-domain validation region,
appending a row to outputs/experiments/ablation_results.csv. Runs are resumable:
a rung whose result row already exists is skipped, so if the machine reboots at
3 a.m. you just start it again.

    python -m src.experiments.run_ablations --config configs/default.yaml
    python -m src.experiments.run_ablations --config configs/default.yaml --dry-run
    python -m src.experiments.run_ablations --config configs/default.yaml --only D_tversky

IMPORTANT: every rung is scored on a withheld TRAINING city, never on the test
regions. Techniques are chosen here; the test regions are evaluated once, at
the very end, with the winning configuration.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from src.common.config import load_config, resolve

# Each rung is cumulative on the previous winner, except where noted. Keeping
# them cumulative answers "does this help on top of what I already have?",
# which is the question that matters, rather than "does this help in isolation?"
LADDER = [
    ("A_baseline", [], "v1 settings, 2-city control"),
    ("B_pertile", ["preprocess.normalization=per_tile"],
     "per-tile standardization: cancels regional colour/brightness offsets"),
    ("C_exg", ["preprocess.extra_channels=[exg]"],
     "Excess-Green channel: explicit foliage signal for canopy failures"),
    ("D_tversky", ["train.loss.type=focal_tversky"],
     "Focal Tversky: penalises misses harder than false alarms"),
    ("E_scale", ["augment.scale_range=[0.5, 2.0]"],
     "wide scale jitter: small outbuildings the training cities lack"),
    ("F_fda", ["augment.fda_mode=intra"],
     "Fourier style transfer between training cities"),
    ("G_segformer", ["train.arch=segformer", "train.encoder=mit_b2",
                     "train.decoder_attention=null"],
     "transformer encoder: usually more robust out of domain"),
]


# Confirmation ladder. The main ladder is cumulative, so a rung that HURTS
# contaminates every rung after it. E_scale cost -0.0217, which means F and G
# were both measured on top of a handicap. These runs remove the known-bad and
# known-useless changes and re-measure, so the final configuration is chosen
# from a clean comparison rather than an accident of ordering.
CONFIRM = [
    ("H_no_scale",
     ["preprocess.normalization=per_tile",
      "preprocess.extra_channels=[exg]",
      "train.loss.type=focal_tversky",
      "augment.fda_mode=intra",
      "train.arch=segformer", "train.encoder=mit_b2",
      "train.decoder_attention=null"],
     "G without the harmful wide scale jitter"),
    ("I_lean",
     ["preprocess.normalization=per_tile",
      "train.loss.type=focal_tversky",
      "augment.fda_mode=intra",
      "train.arch=segformer", "train.encoder=mit_b2",
      "train.decoder_attention=null"],
     "H without ExG, which measured as no better than nothing"),
    ("J_no_fda",
     ["preprocess.normalization=per_tile",
      "train.loss.type=focal_tversky",
      "train.arch=segformer", "train.encoder=mit_b2",
      "train.decoder_attention=null"],
     "I without FDA, to see whether FDA earns its place off E's back"),
]


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="run just this rung")
    ap.add_argument("--from-rung", default=None, help="start at this rung")
    ap.add_argument("--force", action="store_true", help="ignore existing results")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--ladder", default="main", choices=["main", "confirm"],
                    help="main = the 7 cumulative rungs; confirm = clean re-tests")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results = resolve(cfg["paths"]["outputs_dir"]) / "experiments" / "ablation_results.csv"
    done = set()
    if results.exists() and not args.force:
        done = set(pd.read_csv(results)["tag"].tolist())

    LADDER_ACTIVE = LADDER if args.ladder == "main" else CONFIRM
    cumulative_mode = args.ladder == "main"

    ladder = LADDER_ACTIVE
    if args.from_rung:
        names = [t for t, _, _ in LADDER_ACTIVE]
        if args.from_rung not in names:
            print(f"[FAIL] unknown rung {args.from_rung}")
            return 1
        ladder = LADDER_ACTIVE[names.index(args.from_rung):]
    if args.only:
        ladder = [r for r in ladder if r[0] == args.only]
        if not ladder:
            print(f"[FAIL] unknown rung {args.only}")
            return 1

    cumulative: list[str] = []
    print("=" * 78)
    title = ("ABLATION LADDER" if cumulative_mode else "CONFIRMATION RUNS")
    print(f"{title}  (scored on the withheld training city, never on test)")
    print("=" * 78)
    for tag, overrides, why in LADDER_ACTIVE:
        if cumulative_mode:
            cumulative.extend(overrides)
        mark = "skip" if tag in done else "run "
        if args.only and tag != args.only:
            mark = "    "
        print(f"  [{mark}] {tag:14s} {why}")
        if overrides:
            print(f"          {' '.join(overrides)}")
    print()

    if args.dry_run:
        return 0

    cumulative = []
    t_start = time.time()
    for tag, overrides, why in LADDER_ACTIVE:
        if cumulative_mode:
            cumulative.extend(overrides)
        else:
            cumulative = list(overrides)   # each confirm run stands alone
        if args.only and tag != args.only:
            continue
        if not any(r[0] == tag for r in ladder):
            continue
        if tag in done:
            print(f"[skip] {tag} already has a result row")
            continue

        sets: list[str] = []
        for o in cumulative:
            sets += ["--set", o]

        rc = run([sys.executable, "-m", "src.train.train",
                  "--config", args.config, "--name", tag]
                 + (["--workers", str(args.workers)] if args.workers is not None else [])
                 + sets)
        if rc != 0:
            print(f"[FAIL] training failed for {tag} (exit {rc})")
            return rc

        ckpt = f"{cfg['paths']['outputs_dir']}/run_{tag}/best.pth"
        rc = run([sys.executable, "-m", "src.experiments.oodval",
                  "--config", args.config, "--checkpoint", ckpt, "--tag", tag]
                 + sets)
        if rc != 0:
            print(f"[FAIL] oodval failed for {tag} (exit {rc})")
            return rc

    if results.exists():
        df = pd.read_csv(results)
        order = [t for t, _, _ in LADDER] + [t for t, _, _ in CONFIRM]
        df["_o"] = df.tag.apply(lambda t: order.index(t) if t in order else 99)
        df = df.sort_values("_o").drop(columns="_o")
        print("\n" + "=" * 78)
        print("ABLATION RESULTS  (out-of-domain validation region)")
        print("=" * 78)
        print(df[["tag", "iou", "f1", "precision", "recall",
                  "in_domain_val_iou"]].to_string(index=False))
        base = df[df.tag == "A_baseline"]
        if len(base):
            b = float(base.iloc[0]["iou"])
            print(f"\nchange vs baseline ({b:.4f}):")
            for _, r in df.iterrows():
                if r.tag != "A_baseline":
                    print(f"  {r.tag:14s} {float(r['iou']) - b:+.4f}")
    print(f"\nTotal wall time: {(time.time() - t_start) / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
