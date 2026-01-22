# Minecraft Aerodynamics Mod – Dataset Generator (Phase 1)

This repository contains a dataset generator for 3D incompressible flow using PhiFlow. It produces training samples on a 32×32×32 grid with randomized voxel obstacles and an inflow boundary condition.

## What it generates
Each sample is saved as a `.npy` file containing a pickled Python dict with:
- `obstacle_mask`: `(32, 32, 32)` uint8 mask (1 = obstacle)
- `velocity_t`: `(32, 32, 32, 3)` float32 velocity at time $t$
- `velocity_t1`: `(32, 32, 32, 3)` float32 velocity at time $t+1$
- `pressure_t`: `(32, 32, 32)` float32 pressure at time $t$
- `pressure_t1`: `(32, 32, 32)` float32 pressure at time $t+1$

Metadata is saved in `meta.json` alongside the samples.

## Setup
Install dependencies (CPU-only is fine):

```bash
pip install -r requirements.txt
```

## Generate data

```bash
python data_gen.py --num-samples 100 --grid-size 32 --out-dir data
```

Samples are stored in `data/grid_32/`.

## Train the surrogate (Phase 2)
The training loop uses **pushforward training** (unrolled loss). It consumes sequences of length $k$ by sliding a window over the generated `.npy` samples. Each unroll step compares the predicted $(u,v,w,p)$ to the next-step targets and sums the MSE across $k$ steps.

```bash
python train.py --data-dir data/grid_32 --unroll-steps 5 --epochs 20
```

Notes:
- The input expects channels `[mask, v_x, v_y, v_z, pressure]` and outputs `[v_x, v_y, v_z, pressure]`.
- The training loop now reads both $p_t$ and $p_{t+1}$ directly from each sample.

## Loading a sample

```python
import numpy as np

sample = np.load("data/grid_32/sample_00000.npy", allow_pickle=True).item()
print(sample.keys())
```

## Notes
- The generator enforces a cap on obstacle density to keep the solver stable.
- For higher variance, adjust `--min-obstacles`, `--max-obstacles`, and `--obstacle-density-cap`.
- If you see convergence issues, increase `--pressure-iterations` or reduce `--dt`.
