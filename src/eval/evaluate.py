"""GIS-grade evaluation of predicted footprints.

Pixel IoU is the number everyone reports and the least informative one. A model
can score 0.80 while merging every terrace into one polygon and emitting 200
vertices per building, and it would be commercially worthless. This module
measures the things that actually decide whether the output is usable.

Metric groups
-------------
**Pixel** -- IoU, F1, precision, recall, accumulated as raw counts across the
whole split rather than averaged per scene (per-scene averaging inflates the
result, because sparse rural scenes score near-perfectly and drag the mean up).

**Boundary IoU** -- IoU restricted to a band of width ``d`` along each mask's
edge. Large buildings dominate plain IoU, so a model can look good while
placing every wall a metre off. Boundary IoU is insensitive to object size and
exposes exactly that error.

**Instance** -- precision / recall / F1 over matched *polygons* at several IoU
thresholds. This is what "did we find each building" actually means.

**Shape quality** -- vertices per building and corner orthogonality, versus
ground truth processed identically.

The merged-terrace caveat
-------------------------
INRIA ground truth is a binary raster. Touching row houses are a single
connected component in the labels, so a model that *correctly* separates six
terraced houses is scored as five false positives plus one poor match. That
penalises the right behaviour.

We therefore report instance metrics two ways:

  strict          classical one-to-one greedy IoU matching.
  merge-tolerant  a ground-truth polygon counts as found if the union of the
                  predictions lying inside it reaches the IoU threshold.

The gap between the two quantifies how much of the apparent error is really a
limitation of the benchmark's label format.

Usage:

    python -m src.eval.evaluate --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from tqdm import tqdm

from src.common.config import (experiment_settings, load_config,
                               resolve)
from src.vector.regularize import orthogonality_score

EPS = 1e-9


# --------------------------------------------------------------------------- #
# pixel-level
# --------------------------------------------------------------------------- #
def pixel_counts(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
    p, g = pred.astype(bool), gt.astype(bool)
    return {
        "tp": int((p & g).sum()),
        "fp": int((p & ~g).sum()),
        "fn": int((~p & g).sum()),
        "tn": int((~p & ~g).sum()),
    }


def boundary_band(mask: np.ndarray, d: int) -> np.ndarray:
    """The mask minus its own erosion: a band of width d hugging every edge."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    eroded = cv2.erode(mask.astype(np.uint8), k, borderValue=0)
    return (mask.astype(np.uint8) - eroded).astype(bool)


def boundary_counts(pred: np.ndarray, gt: np.ndarray, ds: List[int]) -> Dict[str, int]:
    """Boundary IoU at several band widths.

    A single width is not interpretable on its own: if the band is narrower
    than the model's typical wall displacement, the two bands barely intersect
    and the score collapses regardless of how good the result is. Reporting a
    sweep shows where the error actually sits.
    """
    out = {}
    for d in ds:
        pb, gb = boundary_band(pred, d), boundary_band(gt, d)
        out[f"b{d}_tp"] = int((pb & gb).sum())
        out[f"b{d}_fp"] = int((pb & ~gb).sum())
        out[f"b{d}_fn"] = int((~pb & gb).sum())
    return out


def reduce_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    prec = tp / (tp + fp + EPS)
    rec = tp / (tp + fn + EPS)
    return {
        "iou": tp / (tp + fp + fn + EPS),
        "f1": 2 * prec * rec / (prec + rec + EPS),
        "precision": prec,
        "recall": rec,
    }


# --------------------------------------------------------------------------- #
# instance-level
# --------------------------------------------------------------------------- #
def iou_pair(a: Polygon, b: Polygon) -> float:
    if not a.intersects(b):
        return 0.0
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    return inter / (a.area + b.area - inter + EPS)


def match_strict(
    preds: List[Polygon], gts: List[Polygon], thresholds: List[float]
) -> Dict[float, Dict[str, int]]:
    """Greedy one-to-one matching, highest IoU first."""
    out = {t: {"tp": 0, "fp": len(preds), "fn": len(gts)} for t in thresholds}
    if not preds or not gts:
        return out

    tree = STRtree(gts)
    candidates: List[Tuple[float, int, int]] = []
    for pi, p in enumerate(preds):
        for gi in tree.query(p):
            v = iou_pair(p, gts[int(gi)])
            if v > 0:
                candidates.append((v, pi, int(gi)))
    candidates.sort(reverse=True)

    for t in thresholds:
        used_p, used_g, tp = set(), set(), 0
        for v, pi, gi in candidates:
            if v < t:
                break
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi)
            used_g.add(gi)
            tp += 1
        out[t] = {"tp": tp, "fp": len(preds) - tp, "fn": len(gts) - tp}
    return out


def match_merge_tolerant(
    preds: List[Polygon], gts: List[Polygon], thresholds: List[float]
) -> Dict[float, Dict[str, int]]:
    """A ground-truth polygon is found if the union of predictions lying inside
    it reaches the IoU threshold. Removes the penalty for correctly splitting a
    terrace that the labels record as one blob."""
    out = {t: {"tp": 0, "fp": len(preds), "fn": len(gts)} for t in thresholds}
    if not preds or not gts:
        return out

    tree = STRtree(preds)
    groups: List[Tuple[int, List[int], float]] = []
    for gi, g in enumerate(gts):
        members = []
        for pi in tree.query(g):
            p = preds[int(pi)]
            inter = p.intersection(g).area if p.intersects(g) else 0.0
            # a prediction belongs to this GT if most of it lies inside
            if inter > 0.5 * p.area:
                members.append(int(pi))
        if not members:
            continue
        merged = shapely.union_all([preds[i] for i in members])
        groups.append((gi, members, iou_pair(merged, g)))

    for t in thresholds:
        used_p, used_g, tp = set(), set(), 0
        for gi, members, v in sorted(groups, key=lambda x: -x[2]):
            if v < t:
                break
            if gi in used_g or any(m in used_p for m in members):
                continue
            used_g.add(gi)
            used_p.update(members)
            tp += 1
        out[t] = {"tp": tp, "fp": len(preds) - len(used_p), "fn": len(gts) - tp}
    return out


def polis_distance(a: Polygon, b: Polygon) -> float:
    """Symmetric mean vertex-to-boundary distance, in CRS units (metres).

    Unlike IoU this is a *metric* in map units, so "our walls sit 0.7 m off"
    is a statement a surveyor can act on.
    """
    try:
        pa = np.asarray(a.exterior.coords[:-1])
        pb = np.asarray(b.exterior.coords[:-1])
        if len(pa) == 0 or len(pb) == 0:
            return float("nan")
        d1 = shapely.distance(shapely.points(pa), b.exterior).mean()
        d2 = shapely.distance(shapely.points(pb), a.exterior).mean()
        return float((d1 + d2) / 2.0)
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
def load_gpkg(path: Path) -> Optional[gpd.GeoDataFrame]:
    if not path.exists():
        return None
    try:
        g = gpd.read_file(path, layer="buildings")
    except Exception:
        return None
    return g if len(g) else g


def shape_stats(polys: List[Polygon], simplify_tol: float = 0.0) -> Dict[str, float]:
    if not polys:
        return {"n": 0, "vertices": float("nan"), "orthogonality": float("nan")}
    verts, orth = [], []
    for p in polys:
        if simplify_tol > 0:
            p = p.simplify(simplify_tol, preserve_topology=True)
            if p.is_empty or p.geom_type != "Polygon":
                continue
        verts.append(len(p.exterior.coords) - 1)
        orth.append(orthogonality_score(p))
    return {
        "n": len(verts),
        "vertices": float(np.mean(verts)) if verts else float("nan"),
        "orthogonality": float(np.mean(orth)) if orth else float("nan"),
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pred-vectors", default="regularized")
    ap.add_argument("--raw-vectors", default="raw")
    ap.add_argument("--gt-vectors", default="ground_truth")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-instance-scenes", type=int, default=24,
                    help="polygon matching is O(n^2)-ish; cap scenes used for it")
    args = ap.parse_args()

    cfg = load_config(args.config)
    outputs = resolve(cfg.paths.outputs_dir)
    # Tiles and scene manifests live under the experiment mode, because
    # ablation and final use different strides and different city subsets.
    processed = resolve(cfg["paths"]["processed_dir"]) / experiment_settings(cfg)["mode"]
    pred_dir = outputs / "predictions"
    vec = outputs / "vectors"
    eval_dir = outputs / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    scenes = pd.read_csv(processed / "test_scenes.csv").to_dict("records")
    have = {p.stem.replace("_prob", "") for p in pred_dir.glob("*_prob.tif")}
    scenes = [s for s in scenes if s["scene"] in have]
    if args.limit:
        scenes = scenes[: args.limit]
    if not scenes:
        print("[FAIL] no predictions found. Run src.inference.predict_scenes first.")
        return 1

    thr = cfg.inference.threshold_interior
    band_m = [1.0, 2.0, 3.0]
    band_px = [max(1, int(round(m / 0.3))) for m in band_m]
    iou_thresholds = list(cfg.evaluate.iou_thresholds)

    print(f"Scenes: {len(scenes)}   pixel threshold {thr}   "
          f"boundary bands {band_px} px (~{band_m} m)\n")

    # ---------------- pixel + boundary ---------------- #
    rows = []
    for s in tqdm(scenes, desc="pixel metrics", ncols=90):
        with rasterio.open(pred_dir / f"{s['scene']}_prob.tif") as src:
            pred = (src.read(1).astype(np.float32) / 255.0) >= thr
        with rasterio.open(s["gt_path"]) as src:
            gt = src.read(1) > 127
        r = {"scene": s["scene"], "city": s["city"]}
        r.update(pixel_counts(pred, gt))
        r.update(boundary_counts(pred, gt, band_px))
        rows.append(r)
    px = pd.DataFrame(rows)
    px.to_csv(eval_dir / "pixel_per_scene.csv", index=False)

    def summarize(df: pd.DataFrame, label: str) -> Dict:
        out = {"group": label, "scenes": len(df)}
        out.update({f"pixel_{k}": round(v, 4) for k, v in
                    reduce_counts(df.tp.sum(), df.fp.sum(), df.fn.sum()).items()})
        for d, m in zip(band_px, band_m):
            r = reduce_counts(df[f"b{d}_tp"].sum(), df[f"b{d}_fp"].sum(),
                              df[f"b{d}_fn"].sum())
            out[f"boundary_iou_{m:g}m"] = round(r["iou"], 4)
        return out

    pixel_summary = [summarize(px, "ALL (unseen regions)")]
    for city, grp in px.groupby("city"):
        pixel_summary.append(summarize(grp, city))
    ps = pd.DataFrame(pixel_summary)
    ps.to_csv(eval_dir / "pixel_summary.csv", index=False)

    print("\n" + "=" * 78)
    print("PIXEL & BOUNDARY METRICS (regions never seen in training)")
    print("=" * 78)
    cols = ["group", "scenes", "pixel_iou", "pixel_f1", "pixel_precision",
            "pixel_recall"] + [f"boundary_iou_{m:g}m" for m in band_m]
    print(ps[cols].to_string(index=False))

    # ---------------- instance + shape ---------------- #
    inst_scenes = scenes[: args.max_instance_scenes]
    strict_acc = {t: {"tp": 0, "fp": 0, "fn": 0} for t in iou_thresholds}
    merged_acc = {t: {"tp": 0, "fp": 0, "fn": 0} for t in iou_thresholds}
    polis_vals: List[float] = []
    shape_rows = []

    for s in tqdm(inst_scenes, desc="instance metrics", ncols=90):
        gp = load_gpkg(vec / args.pred_vectors / f"{s['scene']}.gpkg")
        gg = load_gpkg(vec / args.gt_vectors / f"{s['scene']}.gpkg")
        gr = load_gpkg(vec / args.raw_vectors / f"{s['scene']}.gpkg")
        if gp is None or gg is None:
            continue

        preds = [g for g in gp.geometry if g is not None and g.geom_type == "Polygon"]
        gts = [g for g in gg.geometry if g is not None and g.geom_type == "Polygon"]

        for t, c in match_strict(preds, gts, iou_thresholds).items():
            for k in ("tp", "fp", "fn"):
                strict_acc[t][k] += c[k]
        for t, c in match_merge_tolerant(preds, gts, iou_thresholds).items():
            for k in ("tp", "fp", "fn"):
                merged_acc[t][k] += c[k]

        # PoLiS over confidently matched pairs only
        if preds and gts:
            tree = STRtree(gts)
            for p in preds[:400]:
                best, bi = 0.0, None
                for gi in tree.query(p):
                    v = iou_pair(p, gts[int(gi)])
                    if v > best:
                        best, bi = v, int(gi)
                if bi is not None and best >= 0.5:
                    d = polis_distance(p, gts[bi])
                    if np.isfinite(d):
                        polis_vals.append(d)

        row = {"scene": s["scene"], "city": s["city"]}
        for name, polys, tol in [
            ("pred_reg", preds, 0.0),
            ("pred_raw", [g for g in gr.geometry] if gr is not None else [], 0.0),
            ("gt_traced", gts, 0.0),
            ("gt_simplified", gts, cfg.vectorize.simplify_tolerance_m),
        ]:
            st = shape_stats([p for p in polys if p is not None
                              and p.geom_type == "Polygon"], tol)
            row[f"{name}_n"] = st["n"]
            row[f"{name}_vertices"] = st["vertices"]
            row[f"{name}_orthogonality"] = st["orthogonality"]
        shape_rows.append(row)

    inst_rows = []
    for t in iou_thresholds:
        for label, acc in [("strict", strict_acc), ("merge-tolerant", merged_acc)]:
            c = acc[t]
            m = reduce_counts(c["tp"], c["fp"], c["fn"])
            inst_rows.append({
                "matching": label, "iou_threshold": t,
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
            })
    inst = pd.DataFrame(inst_rows)
    inst.to_csv(eval_dir / "instance_metrics.csv", index=False)

    print("\n" + "=" * 78)
    print(f"INSTANCE METRICS ({len(inst_scenes)} scenes, per-building polygon matching)")
    print("=" * 78)
    print(inst.to_string(index=False))

    sh = pd.DataFrame(shape_rows)
    sh.to_csv(eval_dir / "shape_per_scene.csv", index=False)
    if len(sh):
        print("\n" + "=" * 78)
        print("SHAPE QUALITY  (mean per building)")
        print("=" * 78)
        summary = []
        for name, nice in [("pred_raw", "predicted, no regularization"),
                           ("pred_reg", "predicted, regularized"),
                           ("gt_traced", "ground truth, raster-traced"),
                           ("gt_simplified", "ground truth, simplified")]:
            summary.append({
                "geometry": nice,
                "buildings": int(sh[f"{name}_n"].sum()),
                "vertices": round(float(sh[f"{name}_vertices"].mean()), 2),
                "orthogonality": round(float(sh[f"{name}_orthogonality"].mean()), 3),
            })
        sdf = pd.DataFrame(summary)
        sdf.to_csv(eval_dir / "shape_summary.csv", index=False)
        print(sdf.to_string(index=False))
        print("\nNote: ground truth traced straight from the label raster inherits a")
        print("pixel staircase, which is why its vertex count is high AND its")
        print("orthogonality is near 1.0 (every stair step is a perfect 90 degrees).")
        print("Orthogonality is only meaningful read together with vertex count.")

    if polis_vals:
        pv = np.array(polis_vals)
        print(f"\nPoLiS boundary displacement (matched pairs, n={len(pv)}):")
        print(f"  median {np.median(pv):.2f} m   mean {pv.mean():.2f} m   "
              f"90th pct {np.percentile(pv, 90):.2f} m")
        pd.DataFrame({"polis_m": pv}).to_csv(eval_dir / "polis.csv", index=False)

    print(f"\nAll metrics written to {eval_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
