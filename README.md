# Aerodynamics4MC Core

Physics + ML backend for Minecraft airflow simulation.

This repository contains:
- dataset generation (`data_gen.py`)
- 3D U-Net surrogate model (`model.py`)
- long-horizon training (`train.py`)
- checkpoint selection and visualization (`eval_*.py`)
- real-time PhiFlow TCP backend (`server.py`)

It does **not** contain the Fabric mod code anymore. The mod is maintained in a separate repository (`fabric-mod`).

## Repository Scope

This repo is the "core backend" side:
- CFD data generation for fan-driven voxel flow
- neural surrogate training for long rollout prediction
- optional runtime CPU/GPU PhiFlow solver server

The Minecraft integration layer should consume this repo as:
- trained checkpoints (for neural inference), or
- TCP solver backend (`server.py`).

## Project Structure

- `data_gen.py`: generate rollout dataset (`sample_XXXXX.npy` + `meta.json`)
- `model.py`: `SteadyRolloutCFDUNet` (fan-conditioned 3D U-Net)
- `train.py`: pushforward/unrolled training with curriculum + teacher forcing schedule
- `eval_checkpoints.py`: evaluate many checkpoints and select best epoch
- `eval_pyvista.py`: rollout animation rendering (3D + slice)
- `server.py`: PhiFlow backend with semi-Lagrangian advection + projection + optional SGS
- `requirements.txt`: Python dependencies

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Format

Each `sample_XXXXX.npy` is a pickled dict (loaded via `np.load(..., allow_pickle=True).item()`).

Important fields:
- static geometry:
  - `obstacle_mask`: `(N,N,N)` bool
  - `fan_mask`: `(N,N,N)` float32
  - `fan_velocity`: `(N,N,N,3)` float32
- initial state:
  - `velocity_t`: `(N,N,N,3)` float32
  - `pressure_t`: `(N,N,N)` float32
- rollout targets:
  - `obstacle_mask_tn`: `(T,N,N,N)` bool
  - `velocity_tn`: `(T,N,N,N,3)` float32
  - `pressure_tn`: `(T,N,N,N)` float32
- source metadata:
  - `inflow_centers`, `inflow_axes`, `inflow_signs`, `inflow_params_list`, etc.

Model input channels (preferred 9ch):
- `[obstacle, fan_mask, fan_vx, fan_vy, fan_vz, vx, vy, vz, p]`

Model output channels (4ch):
- `[vx, vy, vz, p]`

## 1) Generate Dataset

Example (32^3):

```bash
python data_gen.py \
  --out-dir data_fixed \
  --num-samples 6000 \
  --grid-size 32 \
  --steps-per-sample 32 \
  --warmup-steps 4 \
  --inflow-speed 8.0 \
  --min-inflows 1 \
  --max-inflows 3 \
  --karman-ratio 0.15
```

Output directory:
- `data_fixed/grid_32/`

## 2) Train U-Net3D (Long Rollout)

Example start command:

```bash
python train.py \
  --data-dir data_fixed/grid_32 \
  --output-dir runs_mc_unet3d_long \
  --epochs 180 \
  --batch-size 4 \
  --lr 1e-3 \
  --weight-decay 1e-6 \
  --curriculum-start 2 \
  --unroll-steps 16 \
  --curriculum-increase-every 4 \
  --curriculum-increment 1 \
  --p-tf-start 1.0 \
  --p-tf-end 0.1 \
  --div-weight 1.0 \
  --energy-weight 0.01 \
  --source-weight 0.25 \
  --pressure-mean-weight 0.01 \
  --late-step-weight 1.0 \
  --model-in-channels 9 \
  --model-base-channels 48 \
  --model-depth 4 \
  --model-res-per-stage 2 \
  --model-source-inject-strength 0.1 \
  --checkpoint-interval 5 \
  --amp
```

Resume training:

```bash
python train.py --data-dir data_fixed/grid_32 --output-dir runs_mc_unet3d_long --resume
```

## 3) Select Best Checkpoint

```bash
python eval_checkpoints.py \
  --checkpoint-dir runs_mc_unet3d_long \
  --data-dir val_data \
  --steps 16 \
  --max-samples 1500 \
  --select-by hybrid \
  --top-k 5 \
  --output-csv outputs/checkpoint_eval.csv
```

This reports mean/final-step MSE for velocity and pressure, then ranks checkpoints.

## 4) Render Rollout

```bash
python eval_pyvista.py \
  --checkpoint-dir runs_mc_unet3d_long \
  --data-dir val_data \
  --sample-index 0 \
  --steps 32 \
  --format gif \
  --output-dir outputs
```

## 5) Run Real-Time PhiFlow TCP Backend

```bash
python server.py \
  --host 0.0.0.0 \
  --port 5001 \
  --grid-size 32 \
  --dt 1.0 \
  --diffusivity 1e-5 \
  --fan-strength 0.8 \
  --pressure-iterations 24 \
  --pressure-tolerance 3e-3 \
  --projection-interval 4 \
  --target-hz 2.0 \
  --sgs-model smagorinsky
```

Protocol:
- request: `uint32 length` + float32 `(N,N,N,C)` where `C=10` (thermal-source optional), `C=9` (standard), or `C=5` (legacy)
- response: `uint32 length` + float32 `(N,N,N,4)` (`vx,vy,vz,p`)

## 6) Run Native Solver Benchmarks (Windows + Prebuilt DLL)

If you download `aero_lbm.dll` from the separate `fabric-mod` repository artifacts
(`natives-windows-x86_64` or `natives-bundle`), you can run the native benchmark
directly from this repo with `eval_native_solver_experiments.py`.

The script accepts an explicit DLL path via `--library`, which is the recommended
way to run Windows experiments with a prebuilt artifact.

The `cylinder_crossflow_2d` diagnostic now uses a rectangular DFG/Schäfer-Turek-style setup:
- channel length/height ratio `2.2 / 0.41`
- cylinder center at `(0.2, 0.2)` with diameter `0.1`
- parabolic inlet profile
- top/bottom no-slip walls
- outflow convective boundary
- periodic extrusion along `z`
- `Re = 100`

For this case, `--diag-grid-size` now means the streamwise resolution `nx`, not a cubic `n^3`.
The harness automatically maps it to:

- `nx = --diag-grid-size`
- `ny ≈ round((0.41 / 2.2) * (nx - 1)) + 1`
- `nz = 8`

So, for example:

- `--diag-grid-size 160` becomes approximately `160 x 31 x 8`
- `--diag-grid-size 256` becomes approximately `256 x 48 x 8`
- `--diag-grid-size 320` becomes approximately `320 x 60 x 8`

This rectangular benchmark mode is the recommended path for published cylinder runs,
because it spends resolution on the actual channel instead of embedding the flow in an
expensive cubic box.

Smoke test (`auto`, short run, checks whether the runtime actually sees a GPU/OpenCL device):

```bash
python eval_native_solver_experiments.py \
  --library path/to/aero_lbm.dll \
  --skip-perf \
  --skip-report \
  --modes auto \
  --diag-cases cylinder_crossflow_2d \
  --diag-grid-size 160 \
  --diag-steps 512 \
  --output-dir outputs/native_cylinder_rect_auto_smoke
```

Print runtime/device summary:

```bash
python -c "import json; d=json.load(open('outputs/native_cylinder_rect_auto_smoke/native_solver_results.json', encoding='utf-8')); r=d['diag_results'][0]; print(r['runtime_info']); print(r['grid_shape']); print(r['extra_summary'])"
```

Main published-setup run (`256 x 48 x 8` approximately):

```bash
python eval_native_solver_experiments.py \
  --library path/to/aero_lbm.dll \
  --skip-perf \
  --skip-report \
  --modes auto \
  --diag-cases cylinder_crossflow_2d \
  --diag-grid-size 256 \
  --diag-steps 4096 \
  --output-dir outputs/native_cylinder_rect_auto_g256
```

Higher-resolution run (`320 x 60 x 8` approximately):

```bash
python eval_native_solver_experiments.py \
  --library path/to/aero_lbm.dll \
  --skip-perf \
  --skip-report \
  --modes auto \
  --diag-cases cylinder_crossflow_2d \
  --diag-grid-size 320 \
  --diag-steps 6144 \
  --output-dir outputs/native_cylinder_rect_auto_g320
```

Optional extra run (`384 x 72 x 8` approximately, only if your GPU budget allows):

```bash
python eval_native_solver_experiments.py \
  --library path/to/aero_lbm.dll \
  --skip-perf \
  --skip-report \
  --modes auto \
  --diag-cases cylinder_crossflow_2d \
  --diag-grid-size 384 \
  --diag-steps 8192 \
  --output-dir outputs/native_cylinder_rect_auto_g384
```

Short CPU sanity line:

```bash
python eval_native_solver_experiments.py \
  --library path/to/aero_lbm.dll \
  --skip-perf \
  --skip-report \
  --modes cpu \
  --diag-cases cylinder_crossflow_2d \
  --diag-grid-size 160 \
  --diag-steps 512 \
  --output-dir outputs/native_cylinder_rect_cpu_sanity
```

Print the key metrics from the generated JSON files:

```bash
python -c "import json; paths=['outputs/native_cylinder_rect_auto_g256/native_solver_results.json','outputs/native_cylinder_rect_auto_g320/native_solver_results.json','outputs/native_cylinder_rect_auto_g384/native_solver_results.json','outputs/native_cylinder_rect_cpu_sanity/native_solver_results.json']; \
for p in paths: \
 try: d=json.load(open(p, encoding='utf-8')); \
 except FileNotFoundError: continue; \
 r=d['diag_results'][0]; s=r['extra_summary']; \
 print('\\n==', p, '=='); \
 print('runtime_info =', r['runtime_info']); \
 print('grid_shape =', r['grid_shape']); \
 print('all_finite =', r['all_finite']); \
 print('final_divergence_rms =', r['final_divergence_rms']); \
 print('max_speed =', r['max_speed']); \
 print('cycle =', (s.get('cycle_start_step'), s.get('cycle_end_step'), s.get('cycle_period_steps'))); \
 print('Cd(mean/min/max/amp) =', (s.get('mean_drag_coefficient'), s.get('min_drag_coefficient'), s.get('max_drag_coefficient'), s.get('amp_drag_coefficient'))); \
 print('Cl(mean/min/max/amp) =', (s.get('mean_lift_coefficient'), s.get('min_lift_coefficient'), s.get('max_lift_coefficient'), s.get('amp_lift_coefficient'))); \
 print('dp(mean/min/max/amp) =', (s.get('mean_pressure_difference'), s.get('min_pressure_difference'), s.get('max_pressure_difference'), s.get('amp_pressure_difference'))); \
 print('St =', s.get('strouhal_number'))"
```

Notes:
- `auto` tries OpenCL first and falls back to CPU if no usable device is available.
- For Windows experiments, seeing a GPU/OpenCL device in `runtime_info` is the success criterion for acceleration.
- The native benchmark measures the standalone solver path, not full Minecraft runtime integration.
- The rectangular cylinder benchmark is now the preferred mode for published runs; the older cubic embedding should only be kept for historical comparison.

## Integration Note (Separate Mod Repo)

Use the separate Fabric mod repository to:
- capture Minecraft voxel windows and fan sources
- send packets to `server.py` or call native backend
- apply forces to entities and render debug particles/flow

Keep version compatibility aligned:
- channel schema (`10/9/5` input compatibility, `4` output)
- grid size
- velocity scaling and timestep assumptions

## Troubleshooting

- `No sample_*.npy found`:
  - confirm `--data-dir` points to the folder containing sample files.
- Loss plateaus around rollout expansion:
  - expected when curriculum increases `unroll_steps`; evaluate by checkpoint ranking, not only train loss.
- `Payload size ... does not match voxel count ...`:
  - mod and server `grid-size` or channel count mismatch.
- Low runtime Hz:
  - increase `--projection-interval`, reduce `--pressure-iterations`, or lower grid size.

## License

See `LICENSE`.
