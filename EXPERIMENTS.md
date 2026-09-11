# v2 experiment protocol

## The rule

**Technique selection never sees the test regions.**

Ablations are scored on a *withheld training city* (Vienna). Kitsap and Tyrol are
evaluated exactly once, at the end, with the winning configuration. Choosing
techniques by watching the test score is test-set tuning, and it would make every
honest claim in the v1 report false.

| mode | trains on | scored on | purpose |
|---|---|---|---|
| `ablation` | Austin + Chicago | **Vienna** | choose techniques |
| `final` | Austin + Chicago + Vienna | **Kitsap + Tyrol**, once | report |

## Ladder

Cumulative: each rung keeps the previous overrides, answering "does this help on
top of what I already have?" rather than "does this help in isolation?"

| rung | change | targets which v1 failure |
|---|---|---|
| A | control, v1 settings | — |
| B | per-tile standardization | regional colour/brightness shift |
| C | + Excess-Green channel | tree canopy (Kitsap recall 0.68) |
| D | + Focal Tversky loss | recall generally (P 0.877 vs R 0.771) |
| E | + wide scale jitter 0.5–2.0× | missed small outbuildings |
| F | + Fourier style transfer | sensor / illumination shift |
| G | + SegFormer encoder | out-of-domain robustness |

If a rung makes things worse, drop it and carry the previous overrides forward.
A ladder where everything helps usually means the measurement is wrong.

## Running it

```bash
# 1. tile for the ablation protocol (2 cities, Vienna withheld)
python -m src.data.make_tiles --config configs/default.yaml --clean

# 2. the whole ladder, unattended and resumable
python -m src.experiments.run_ablations --config configs/default.yaml

# see the plan without running anything
python -m src.experiments.run_ablations --config configs/default.yaml --dry-run
# re-run one rung
python -m src.experiments.run_ablations --config configs/default.yaml --only D_tversky --force
```

Results accumulate in `outputs/experiments/ablation_results.csv`.

## Then the final run

```bash
python -m src.data.make_tiles --config configs/default.yaml --clean \
    --set experiment.mode=final
python -m src.train.train --config configs/default.yaml --name final \
    --set experiment.mode=final  <winning overrides>
```
