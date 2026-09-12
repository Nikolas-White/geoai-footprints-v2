<h1 align="center">Building Footprints: Aerial Imagery → Usable Vectors</h1>

<p align="center">
  Turning 0.3 m aerial imagery into clean, topologically valid building polygons,
  and measuring what it costs to point the model at a region it has never seen.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-SegFormer%20MiT--B2-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/dataset-INRIA%20Aerial-4B8BBE" alt="INRIA">
  <img src="https://img.shields.io/badge/output-GeoPackage-2E7D32" alt="GeoPackage">
</p>

---

## Results on regions never seen in training

| | v1 | **v2** |
|---|---|---|
| **Pixel IoU, unseen regions** | 0.6957 | **0.7454** |
| Kitsap County | 0.6012 | **0.6639** |
| Tyrol-w | 0.7775 | **0.8186** |
| Instance F1 @ IoU 0.5 | 0.6019 | **0.6752** |
| Recall | 0.7712 | **0.8504** |
| **In-domain → out-of-domain gap** | **0.111** | **0.082** |

**37,639 building polygons** across 72 unseen scenes as valid GeoPackages, walls a
median **0.83 m** from truth. The gap closing by more than the in-domain score rose
is the result that matters: the changes improved *transfer*, not just capacity.

**[→ Full methodology, ablations and failure analysis (REPORT.md)](REPORT.md)**

---

## What it looks like

Left to right in every image: imagery, ground truth, raw model output,
regularized output. Every scene is from a region held out entirely from training.

### Tyrol-w: Alpine village cores and valley-floor industry

![tyrol-w1](outputs/figures/compare_tyrol-w1.png)
<p align="center"><em>Dense village core, ~250 buildings in a 270 m window.</em></p>

![tyrol-w24](outputs/figures/compare_tyrol-w24.png)
<p align="center"><em>The highest building density in the test set, courtyards and shared walls throughout.</em></p>

![tyrol-w20](outputs/figures/compare_tyrol-w20.png)
<p align="center"><em>Note the tennis courts. Large, flat, rectangular, high-contrast, and correctly rejected. A model that had learned "big rectangle" instead of "building" would claim them.</em></p>

![tyrol-w6](outputs/figures/compare_tyrol-w6.png)
<p align="center"><em>An institutional complex with wings at several orientations, the case that should break a regularizer assuming one dominant axis per building.</em></p>

![tyrol-w31](outputs/figures/compare_tyrol-w31.png)
<p align="center"><em>Isolated industrial building. Easy to detect, hard to delineate: nearly all the error here is boundary placement rather than detection.</em></p>

### Kitsap County: forested low-density suburbia

![kitsap31](outputs/figures/compare_kitsap31.png)
<p align="center"><em>Big-box retail beside housing. The large roof, including the stepped notch at its lower right, is followed accurately, single-axis buildings at this scale were a v1 weakness.</em></p>

![kitsap35](outputs/figures/compare_kitsap35.png)
<p align="center"><em>Suburban streets threaded through woodland: the hardest density-versus-occlusion mix in the region.</em></p>

![kitsap13](outputs/figures/compare_kitsap13.png)
<p align="center"><em>Houses scattered under conifer canopy. Most structures found; the smallest outbuildings are not.</em></p>

![kitsap20](outputs/figures/compare_kitsap20.png)
<p align="center"><em>Very low density, heavy shadow. Detection holds in the open and degrades under tree crowns.</em></p>

Each crop is the densest 270 m window in its scene, chosen automatically, which
biases this gallery toward busy areas. The failure figures in the report are not
selected that way.

---

## The two questions this is about

### 1. What does generalizing to new geography cost?

Most work on this benchmark splits tiles randomly, which lets a model train on
one half of a rooftop and be tested on the other. Entire **regions** are held out
here instead.

| split | regions | scenes |
|---|---|---|
| train | Austin, Chicago, Vienna | 90 |
| val | Austin, Chicago, Vienna | 18 |
| **test** | **Kitsap County, Tyrol-w** | **72** |

The answer is **0.082 IoU**, improved from 0.111 in v1 by the changes below.

### 2. Is the output a raster, or something a surveyor can use?

| geometry | vertices/building | orthogonality |
|---|---|---|
| raw traced contours | 11.08 | 0.143 |
| **+ simplify + regularize** | **6.56** | **0.643** |
| ground truth, simplified identically | 7.34 | 0.880 |

Identical pixels, identical IoU. One of these drops into a CAD workflow. The
other has to be redrawn by hand.

> Read the two columns **together**. Ground truth traced off the label raster
> inherits a pixel staircase, and every stair step is a perfect 90° corner, so it
> scores 1.000 orthogonality at 65 vertices per building while being the least
> usable geometry in the dataset. Orthogonality alone is a trap.

---

## How v2 improved on v1

Techniques were selected on a **withheld training city (Vienna)**, never on the
test regions, seven cumulative ablations plus three confirmation runs.

| change | effect on OOD IoU | verdict |
|---|---|---|
| SegFormer (MiT-B2) encoder | **+0.017** | clear win, and ~3× faster |
| Focal Tversky loss | **+0.009** | clear win; recall 0.886 → 0.919 |
| Per-tile standardization | **+0.005** | small win |
| Excess-Green 4th channel | ±0.005 | within noise |
| Fourier style transfer | ±0.004 | within noise |
| Wide scale jitter 0.5-2.0× | **−0.022** | **actively harmful** |

**The largest single gain came from deleting something.** Removing the wide scale
jitter, which I had predicted would help, took the best configuration from
0.7534 to 0.7603.

Threshold calibration also fixed itself as a side effect: in v1 the optimal
decision threshold was 0.25 overall and 0.18 for Kitsap, far below the 0.5 in
use. In v2 all regions peak flat across 0.40-0.55, so one fixed threshold now
transfers to regions you have no labels for.

![threshold sweep](outputs/eval/threshold_sweep.png)

---

## Pipeline

```
GeoTIFF (5000×5000 @ 0.3 m/px)
   │
   ├─ SegFormer MiT-B2, 24.7M params, two output heads:
   │    building interior  +  1.5 m band along every wall
   │
   ├─ sliding-window inference, cosine-blended seams, 8-fold TTA
   │
   ├─ boundary-aware watershed  →  separates touching row houses
   │
   ├─ Douglas-Peucker → orthogonal regularization → topology repair
   │
   └─ GeoPackage: valid geometry, source CRS, per-building attributes
```

```bash
python -m src.data.make_tiles           --config configs/final.yaml --clean
python -m src.train.train               --config configs/final.yaml --name final
python -m src.inference.predict_scenes  --config configs/final.yaml \
        --checkpoint outputs/run_final/best.pth --tta
python -m src.vector.vectorize          --config configs/final.yaml
python -m src.eval.evaluate             --config configs/final.yaml
python -m src.eval.figures              --config configs/final.yaml
```

Ablations: `python -m src.experiments.run_ablations`, results in
`outputs/experiments/ablation_results.csv`. Every path and hyperparameter lives
in `configs/`.

60 epochs on 30,797 tiles, ~9 h on an RTX 4080 SUPER. Inference runs at 6.9 s per
25 MP scene with 8-fold test-time augmentation.

---

<p align="center">
  <sub>Data: <a href="https://project.inria.fr/aerialimagelabeling/">INRIA Aerial Image Labeling Benchmark</a></sub>
</p>
