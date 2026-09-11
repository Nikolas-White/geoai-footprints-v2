# GeoAI Building Footprints: Raster → Regularized Vector, with GIS-Grade Evaluation

An end-to-end pipeline that converts high-resolution aerial imagery into **clean, topologically
valid, orthogonally regularized building footprint polygons** — and then evaluates them the way a
GIS professional would, not the way a Kaggle leaderboard would.

## Why this project

Semantic segmentation of buildings is a solved problem. The hard, commercially relevant problems
sit on either side of the model:

1. **Generalization across geography.** A model trained on Austin and Vienna has to work in Kitsap
   County and the Austrian Tyrol without retraining. This repo enforces a strict geographic
   holdout so the headline numbers are honest.
2. **Vectorization quality.** Customers buy polygons, not probability rasters. A footprint with 340
   noisy vertices and 91.4° corners is unusable in a CAD/GIS workflow even if its pixel IoU is
   excellent. This repo measures over-noding and corner-angle distribution explicitly.
3. **Instance separation.** Touching row houses must come out as separate polygons.
4. **Change detection.** Comparing two epochs of imagery to classify new / demolished / modified
   structures.

## Pipeline

```
GeoTIFF imagery
      │
      ├─ [1] tiling & sampling            src/data/make_tiles.py
      │
      ├─ [2] U-Net (ResNet-34) with       src/models/
      │      dual heads: interior + boundary
      │
      ├─ [3] sliding-window inference     src/inference/
      │      over full 5000×5000 scenes
      │
      ├─ [4] vectorization                src/vector/
      │      watershed → contours → Douglas-Peucker
      │      → orthogonal regularization → topology repair
      │
      ├─ [5] evaluation                   src/eval/
      │      instance P/R/F1 @ IoU, Boundary IoU, PoLiS,
      │      vertex-count ratio, corner-angle histogram
      │
      └─ [6] change detection             src/change/
             epoch-to-epoch polygon matching
                    │
                    ▼
         GeoPackage (.gpkg) + report + web map
```

## Dataset

[INRIA Aerial Image Labeling Benchmark](https://project.inria.fr/aerialimagelabeling/) —
360 georeferenced 5000×5000 px GeoTIFFs at 0.3 m/px covering 810 km² across five regions.

Only the labelled `train/` portion is used (180 images, 36 per region). It is re-split:

| Split | Regions | Images | Purpose |
|---|---|---|---|
| train | Austin, Chicago, Vienna | 1–30 each (90) | model fitting |
| val | Austin, Chicago, Vienna | 31–36 each (18) | checkpoint selection |
| **test** | **Kitsap County, Tyrol-w** | **all 72** | **unseen-geography evaluation** |

The test regions are never seen in any form during training. This is the standard
generalization protocol for this benchmark and it is deliberately harsh: Kitsap is
low-density North American suburbia under heavy tree canopy, Tyrol-w is Alpine
village architecture. Neither resembles the training cities.

## Why regularization is the point

Measured on a controlled scene with known footprints, holding segmentation
quality fixed and varying only the vectorizer:

| pipeline | vertices / building | orthogonality |
|---|---|---|
| raw traced contours | 11.8 | 0.31 |
| **+ simplify + regularize** | **4.2** | **0.95** |
| hand-digitized reference | 4.0 | — |

Identical pixels, identical IoU. One output drops into a CAD or GIS workflow;
the other has to be redrawn by hand.

## A note on the benchmark's labels

INRIA ground truth is a **binary raster**, which has two consequences this repo
handles explicitly rather than quietly benefiting from:

1. Touching row houses form one connected component in the labels. A model that
   correctly separates a terrace is scored as several false positives. Instance
   metrics are therefore reported under both strict one-to-one matching and a
   merge-tolerant variant; the gap between them measures the benchmark's
   limitation rather than the model's.
2. Vectorising the labels reproduces a pixel staircase, so ground truth traces
   at ~69 vertices per building with near-perfect corner orthogonality (every
   stair step is exactly 90 degrees). Orthogonality is meaningless read alone
   and is always reported alongside vertex count.

## Status

- [x] Phase 0 — environment
- [x] Phase 1 — data acquisition & tiling
- [x] Phase 2 — model & training
- [x] Phase 3 — full-scene inference
- [x] Phase 4 — vectorization & regularization
- [x] Phase 5 — GIS-grade evaluation
- [ ] Phase 6 — change detection (not attempted; see REPORT.md limitations)
- [x] Phase 7 — report & figures
