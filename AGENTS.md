# Repository Guidelines

## Project Structure & Module Organization
- Core Python scripts live in the repo root: `data_gen.py` (dataset generation), `train.py` (training), `model.py` (model definition), and `eval_*.py` (evaluation/visualization).
- Data and artifacts are stored in top-level folders: `train_data/`, `val_data/`, `train_dataset/`, `outputs/`, and `runs*/` for checkpoints/logs.
- The Fabric client mod is in `fabric-mod/`.
- Media outputs (e.g., `grid32.mp4`, `cfd_sample_0_animation.html`) are checked in for reference.

## Build, Test, and Development Commands
- Install dependencies: `pip install -r requirements.txt`
- Generate samples: `python data_gen.py --num-samples 100 --grid-size 32 --out-dir data` (writes to `data/grid_32/`).
- Train a model: `python train.py --data-dir data/grid_32 --unroll-steps 5 --epochs 20`.
- Render rollouts: `python eval_pyvista.py --checkpoint-dir runs_128 --data-dir val_data --sample-index 0 --format gif` (outputs to `outputs/`).
- Run the Minecraft client mod: `cd fabric-mod && ./gradlew runClient`.

## Coding Style & Naming Conventions
- Use standard Python style: 4-space indentation, `snake_case` for functions/variables, `CapWords` for classes.
- Keep scripts CLI-friendly; prefer `argparse` flags over hardcoded constants.
- Use descriptive file names for datasets and outputs (e.g., `sample_00000.npy`, `rollout_3d_*.gif`).

## Testing Guidelines
- No automated test suite is present. Treat evaluation scripts as smoke tests.
- If you add tests, keep them in a new `tests/` directory and name them `test_*.py`.

## Commit & Pull Request Guidelines
- Commit history uses short, imperative, sentence-case summaries (e.g., “Fixed Bugs”, “Added some variation…”). Keep messages brief and action-oriented.
- PRs should include:
  - A short summary and rationale.
  - Commands run (e.g., training or evaluation flags).
  - Sample outputs or screenshots for visualization changes (GIF/MP4 links or files in `outputs/`).

## Configuration Notes
- Dataset samples are `.npy` files storing a pickled dict with velocity/pressure fields. Keep schema changes documented in `README.md` if you modify it.
