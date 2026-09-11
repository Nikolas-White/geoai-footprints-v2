"""Turn probability rasters into clean building polygons.

Pipeline per scene:

    interior + boundary probability
        -> seeds        (confidently interior AND not on a wall)
        -> watershed    (flood from seeds, dammed by the boundary ridge)
        -> polygonize   (all instances at once, in map coordinates)
        -> filter       (drop specks below min_area_m2)
        -> simplify     (Douglas-Peucker, tolerance in metres)
        -> regularize   (snap to the building's dominant axis)
        -> repair       (valid, non-self-intersecting)
        -> GeoPackage

The watershed step is what the boundary head was trained for. Thresholding the
interior channel alone merges every terrace of row houses into one polygon,
which destroys per-building counts, areas and geocodes -- exactly the
attributes a customer buys.

Usage:

    python -m src.vector.vectorize --config configs/default.yaml
    python -m src.vector.vectorize --config configs/default.yaml --no-regularize
    python -m src.vector.vectorize --config configs/default.yaml --ground-truth
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy import ndimage as ndi
from shapely.geometry import shape as shapely_shape
from skimage.segmentation import watershed
from tqdm import tqdm

from src.common.config import (experiment_settings, load_config,
                               resolve)
from src.vector.regularize import orthogonality_score, regularize_polygon


# --------------------------------------------------------------------------- #
def instance_labels(
    interior: np.ndarray,
    boundary: np.ndarray,
    thr_interior: float,
    thr_boundary: float,
    min_seed_px: int = 16,
) -> np.ndarray:
    """Split a building mask into individual instances via marker watershed."""
    mask = interior >= thr_interior

    # A seed is a pixel we are confident is *inside* a building and *not* on a
    # wall. Raising the interior threshold shrinks each seed away from the
    # edges, which is what keeps neighbouring buildings' seeds disjoint.
    seeds = (interior >= max(thr_interior, 0.6)) & (boundary < thr_boundary)
    seeds = ndi.binary_erosion(seeds, np.ones((3, 3), bool), border_value=0)

    markers, n = ndi.label(seeds)
    if n == 0:
        return np.zeros_like(mask, dtype=np.int32)

    # Discard seeds too small to be a real structure; they fragment buildings.
    sizes = np.bincount(markers.ravel())
    too_small = np.flatnonzero(sizes < min_seed_px)
    if len(too_small):
        markers[np.isin(markers, too_small)] = 0
        markers, _ = ndi.label(markers > 0)

    # Flood from the seeds using boundary probability as the elevation surface:
    # basins grow outward and meet along the ridge the model drew on each wall.
    labels = watershed(boundary, markers=markers, mask=mask)
    return labels.astype(np.int32)


def polygonize(labels: np.ndarray, transform, crs) -> List[Dict]:
    """All instances to map-coordinate polygons in a single pass."""
    out = []
    for geom, value in features.shapes(
        labels, mask=labels > 0, transform=transform, connectivity=8
    ):
        if value == 0:
            continue
        poly = shapely_shape(geom)
        if poly.is_empty or poly.geom_type != "Polygon":
            continue
        out.append({"label": int(value), "geometry": poly})
    return out


# --------------------------------------------------------------------------- #
def clean_polygon(
    poly,
    min_area: float,
    simplify_tol: float,
    regularize: bool,
    angle_snap_deg: float,
    hole_min_area: float,
    min_edge_m: float,
) -> Optional[Dict]:
    """Simplify, regularize and repair one polygon. Returns None if rejected."""
    if poly.area < min_area:
        return None

    raw_vertices = len(poly.exterior.coords) - 1

    if simplify_tol > 0:
        poly = poly.simplify(simplify_tol, preserve_topology=True)
        if poly.is_empty or poly.geom_type != "Polygon":
            return None

    if regularize:
        reg = regularize_polygon(poly, angle_snap_deg, hole_min_area, min_edge_m)
        if reg is None:
            return None
        # Regularization should nudge a footprint, not reshape it. A large
        # area change means the dominant-axis estimate was wrong (curved or
        # genuinely oblique building), so we keep the unregularized version.
        if abs(reg.area - poly.area) / max(poly.area, 1e-9) < 0.25:
            poly = reg

    if poly.is_empty or poly.area < min_area or not poly.is_valid:
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.geom_type != "Polygon" or poly.area < min_area:
            return None

    return {
        "geometry": poly,
        "area_m2": round(float(poly.area), 2),
        "perimeter_m": round(float(poly.length), 2),
        "n_vertices": len(poly.exterior.coords) - 1,
        "raw_vertices": raw_vertices,
        "orthogonality": round(orthogonality_score(poly), 4),
    }


# --------------------------------------------------------------------------- #
def vectorize_scene(prob_path: Path, cfg, regularize: bool) -> gpd.GeoDataFrame:
    with rasterio.open(prob_path) as src:
        interior = src.read(1).astype(np.float32) / 255.0
        boundary = src.read(2).astype(np.float32) / 255.0
        transform, crs = src.transform, src.crs

    labels = instance_labels(
        interior, boundary,
        cfg.inference.threshold_interior,
        cfg.inference.threshold_boundary,
    )
    raw = polygonize(labels, transform, crs)

    rows = []
    for item in raw:
        cleaned = clean_polygon(
            item["geometry"],
            cfg.vectorize.min_area_m2,
            cfg.vectorize.simplify_tolerance_m,
            regularize,
            cfg.vectorize.angle_snap_deg,
            cfg.vectorize.hole_min_area_m2,
            cfg.vectorize.get("min_edge_m", 2.0),
        )
        if cleaned is not None:
            rows.append(cleaned)

    if not rows:
        return gpd.GeoDataFrame(
            {"area_m2": [], "perimeter_m": [], "n_vertices": [],
             "raw_vertices": [], "orthogonality": []},
            geometry=[], crs=crs,
        )
    return gpd.GeoDataFrame(rows, crs=crs)


def vectorize_ground_truth(gt_path: Path, cfg) -> gpd.GeoDataFrame:
    """Reference polygons from the label raster.

    Deliberately NOT simplified or regularized: this is the yardstick, so it
    must stay exactly as the annotators drew it.
    """
    with rasterio.open(gt_path) as src:
        mask = (src.read(1) > 127).astype(np.uint8)
        transform, crs = src.transform, src.crs

    labels, _ = ndi.label(mask, structure=np.ones((3, 3), int))
    rows = []
    for item in polygonize(labels.astype(np.int32), transform, crs):
        poly = item["geometry"]
        if poly.area < cfg.vectorize.min_area_m2:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
        rows.append({
            "geometry": poly,
            "area_m2": round(float(poly.area), 2),
            "perimeter_m": round(float(poly.length), 2),
            "n_vertices": len(poly.exterior.coords) - 1,
        })
    if not rows:
        return gpd.GeoDataFrame({"area_m2": [], "perimeter_m": [],
                                 "n_vertices": []}, geometry=[], crs=crs)
    return gpd.GeoDataFrame(rows, crs=crs)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-regularize", action="store_true",
                    help="skip orthogonal snapping (for the ablation table)")
    ap.add_argument("--ground-truth", action="store_true",
                    help="vectorize label rasters instead of predictions")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suffix", default=None, help="output subfolder name")
    args = ap.parse_args()

    cfg = load_config(args.config)
    outputs = resolve(cfg.paths.outputs_dir)
    # Tiles and scene manifests live under the experiment mode, because
    # ablation and final use different strides and different city subsets.
    processed = resolve(cfg["paths"]["processed_dir"]) / experiment_settings(cfg)["mode"]

    if args.ground_truth:
        scenes = pd.read_csv(processed / "test_scenes.csv").to_dict("records")
        pred_dir = outputs / "predictions"
        available = {p.stem.replace("_prob", "") for p in pred_dir.glob("*_prob.tif")}
        scenes = [s for s in scenes if s["scene"] in available]
        out_dir = outputs / "vectors" / (args.suffix or "ground_truth")
    else:
        pred_dir = outputs / "predictions"
        probs = sorted(pred_dir.glob("*_prob.tif"))
        if not probs:
            print(f"[FAIL] no probability rasters in {pred_dir}. "
                  "Run src.inference.predict_scenes first.")
            return 1
        scenes = [{"scene": p.stem.replace("_prob", ""), "prob_path": str(p)}
                  for p in probs]
        name = "regularized" if not args.no_regularize else "raw"
        out_dir = outputs / "vectors" / (args.suffix or name)

    if args.limit:
        scenes = scenes[: args.limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scenes: {len(scenes)}")
    print(f"Mode  : {'ground truth' if args.ground_truth else 'predictions'}"
          f"{'' if args.ground_truth else (', regularize=' + str(not args.no_regularize))}")
    print(f"Output: {out_dir}\n")

    summary, t0 = [], time.time()
    for s in tqdm(scenes, desc="vectorizing", ncols=90):
        if args.ground_truth:
            gdf = vectorize_ground_truth(Path(s["gt_path"]), cfg)
        else:
            gdf = vectorize_scene(Path(s["prob_path"]), cfg,
                                  regularize=not args.no_regularize)

        gdf["scene"] = s["scene"]
        gpkg = out_dir / f"{s['scene']}.gpkg"
        gdf.to_file(gpkg, layer="buildings", driver="GPKG")

        row = {"scene": s["scene"], "n_buildings": len(gdf)}
        if len(gdf):
            row["median_area_m2"] = float(gdf["area_m2"].median())
            row["mean_vertices"] = float(gdf["n_vertices"].mean())
            if "raw_vertices" in gdf:
                row["mean_raw_vertices"] = float(gdf["raw_vertices"].mean())
                row["mean_orthogonality"] = float(gdf["orthogonality"].mean())
        summary.append(row)

    df = pd.DataFrame(summary)
    df.to_csv(out_dir / "summary.csv", index=False)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    print(f"Total buildings: {int(df['n_buildings'].sum())}")
    if "mean_vertices" in df:
        print(f"Mean vertices per building : {df['mean_vertices'].mean():.1f}")
    if "mean_raw_vertices" in df:
        print(f"  before simplify/regularize: {df['mean_raw_vertices'].mean():.1f}")
        print(f"Mean orthogonality         : {df['mean_orthogonality'].mean():.3f}")
    print(f"\nGeoPackages in {out_dir} -- drag one into QGIS to inspect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
