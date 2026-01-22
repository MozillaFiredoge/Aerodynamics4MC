#!/usr/bin/env python3
"""Dataset generation for 3D incompressible flow using PhiFlow.

Generates samples with:
- Input: obstacle mask (0/1), velocity at time T (u, v, w)
- Target: velocity at time T+1 (u, v, w), pressure at time T+1

Outputs are saved as .npy files per sample.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np

try:
    from phi.flow import (
        Box,
        CenteredGrid,
        Solve,
        StaggeredGrid,
        advect,
        fluid,
        math as phi_math,
        vec,
    )
    from phi.math import Diverged, NotConverged
    from phi.math import backend as phi_backend
    from phi.geom import Sphere
except Exception as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "PhiFlow is required. Install dependencies with `pip install -r requirements.txt`."
    ) from exc


@dataclass
class GenerationConfig:
    grid_size: int = 32
    num_samples: int = 100
    out_dir: str = "data"
    steps_per_sample: int = 1
    warmup_steps: int = 4
    dt: float = 0.5
    inflow_speed: float = 1.0
    max_obstacles: int = 6
    min_obstacles: int = 1
    obstacle_density_cap: float = 0.35
    pressure_iterations: int = 200
    pressure_tolerance: float = 1e-4
    seed: int | None = 7


OBSTACLE_TYPES = ("cube", "wall", "sphere")


def make_box(lower: Tuple[float, float, float], upper: Tuple[float, float, float]) -> Any:
    return Box(  # type: ignore[arg-type]
        x=(lower[0], upper[0]),
        y=(lower[1], upper[1]),
        z=(lower[2], upper[2]),
    )


def make_sphere(center: Tuple[float, float, float], radius: float) -> Any:
    return cast(
        Any,
        Sphere(  # type: ignore[call-arg]
            x=center[0],
            y=center[1],
            z=center[2],
            radius=radius,
        ),
    )


def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Generate PhiFlow CFD dataset.")
    parser.add_argument("--out-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--grid-size", type=int, default=32, help="Voxel grid size")
    parser.add_argument("--steps-per-sample", type=int, default=1, help="Time steps per sample")
    parser.add_argument("--warmup-steps", type=int, default=4, help="Warmup steps")
    parser.add_argument("--dt", type=float, default=0.5, help="Time step size")
    parser.add_argument("--inflow-speed", type=float, default=1.0, help="Inflow speed on -X face")
    parser.add_argument("--min-obstacles", type=int, default=1, help="Minimum obstacles")
    parser.add_argument("--max-obstacles", type=int, default=6, help="Maximum obstacles")
    parser.add_argument(
        "--obstacle-density-cap",
        type=float,
        default=0.35,
        help="Abort if obstacle volume exceeds this fraction",
    )
    parser.add_argument(
        "--pressure-iterations",
        type=int,
        default=200,
        help="Pressure solver max iterations",
    )
    parser.add_argument(
        "--pressure-tolerance",
        type=float,
        default=1e-4,
        help="Pressure solver tolerance",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()
    return GenerationConfig(
        grid_size=args.grid_size,
        num_samples=args.num_samples,
        out_dir=args.out_dir,
        steps_per_sample=args.steps_per_sample,
        warmup_steps=args.warmup_steps,
        dt=args.dt,
        inflow_speed=args.inflow_speed,
        min_obstacles=args.min_obstacles,
        max_obstacles=args.max_obstacles,
        obstacle_density_cap=args.obstacle_density_cap,
        pressure_iterations=args.pressure_iterations,
        pressure_tolerance=args.pressure_tolerance,
        seed=args.seed,
    )


def _random_int(low: int, high: int) -> int:
    return random.randint(low, high)


def _random_float(low: float, high: float) -> float:
    return random.uniform(low, high)


def generate_obstacle_mask(cfg: GenerationConfig) -> Tuple[np.ndarray, List[object]]:
    grid = cfg.grid_size
    mask = np.zeros((grid, grid, grid), dtype=np.uint8)
    obstacles: List[object] = []

    obstacle_count = _random_int(cfg.min_obstacles, cfg.max_obstacles)
    for _ in range(obstacle_count):
        obstacle_type = random.choice(OBSTACLE_TYPES)
        if obstacle_type == "cube":
            size = _random_int(3, max(4, grid // 5))
            x0 = _random_int(1, grid - size - 1)
            y0 = _random_int(1, grid - size - 1)
            z0 = _random_int(1, grid - size - 1)
            mask[x0 : x0 + size, y0 : y0 + size, z0 : z0 + size] = 1
            obstacles.append(
                make_box((x0, y0, z0), (x0 + size, y0 + size, z0 + size))
            )
        elif obstacle_type == "wall":
            thickness = _random_int(1, 2)
            orientation = random.choice(["x", "y", "z"])
            if orientation == "x":
                x0 = _random_int(4, grid - 5)
                mask[x0 : x0 + thickness, 2:-2, 2:-2] = 1
                obstacles.append(
                    make_box(
                        (x0, 2, 2),
                        (x0 + thickness, grid - 2, grid - 2),
                    )
                )
            elif orientation == "y":
                y0 = _random_int(4, grid - 5)
                mask[2:-2, y0 : y0 + thickness, 2:-2] = 1
                obstacles.append(
                    make_box(
                        (2, y0, 2),
                        (grid - 2, y0 + thickness, grid - 2),
                    )
                )
            else:
                z0 = _random_int(4, grid - 5)
                mask[2:-2, 2:-2, z0 : z0 + thickness] = 1
                obstacles.append(
                    make_box(
                        (2, 2, z0),
                        (grid - 2, grid - 2, z0 + thickness),
                    )
                )
        else:
            radius = _random_int(2, max(3, grid // 6))
            center = (
                _random_float(4, grid - 4),
                _random_float(4, grid - 4),
                _random_float(4, grid - 4),
            )
            xs = np.arange(grid).reshape(-1, 1, 1)
            ys = np.arange(grid).reshape(1, -1, 1)
            zs = np.arange(grid).reshape(1, 1, -1)
            dist = np.sqrt(
                (xs - center[0]) ** 2 + (ys - center[1]) ** 2 + (zs - center[2]) ** 2
            )
            mask[dist <= radius] = 1
            obstacles.append(make_sphere(center, float(radius)))

    occupied = mask.mean()
    if occupied > cfg.obstacle_density_cap:
        return generate_obstacle_mask(cfg)
    return mask, obstacles


def build_inflow_mask(cfg: GenerationConfig) -> np.ndarray:
    grid = cfg.grid_size
    inflow = np.zeros((grid, grid, grid), dtype=np.float32)
    inflow[0:2, :, :] = 1.0
    return inflow


def field_from_mask(mask: np.ndarray, bounds: Any) -> Any:
    tensor = phi_math.tensor(mask, phi_math.spatial("x,y,z"))
    return CenteredGrid(
        tensor,
        bounds=bounds,
        extrapolation=0.0,
        x=mask.shape[0],
        y=mask.shape[1],
        z=mask.shape[2],
    )


def build_velocity_field(cfg: GenerationConfig, bounds: Any) -> Any:
    return StaggeredGrid(vec(x=0.0, y=0.0, z=0.0), bounds=bounds, x=cfg.grid_size, y=cfg.grid_size, z=cfg.grid_size)


def apply_inflow(velocity: Any, inflow_mask: Any, inflow_speed: float) -> Any:
    inflow = StaggeredGrid(
        vec(x=inflow_speed, y=0.0, z=0.0),
        bounds=velocity.bounds,
        x=velocity.resolution.spatial["x"].size,
        y=velocity.resolution.spatial["y"].size,
        z=velocity.resolution.spatial["z"].size,
    )
    inflow_mask_staggered = inflow_mask.at(inflow)
    return velocity + inflow * inflow_mask_staggered


def step_flow(
    velocity: Any,
    obstacles: List[object],
    inflow_mask: Any,
    cfg: GenerationConfig,
) -> Tuple[Any, Any]:
    velocity = advect.semi_lagrangian(velocity, velocity, dt=cfg.dt)
    velocity = apply_inflow(velocity, inflow_mask, cfg.inflow_speed)
    solve = Solve(
        method="CG",
        abs_tol=cfg.pressure_tolerance,
        rel_tol=cfg.pressure_tolerance,
        max_iterations=cfg.pressure_iterations,
    )
    try:
        velocity, pressure = fluid.make_incompressible(velocity, obstacles=obstacles, solve=solve)
    except (NotConverged, Diverged):
        relaxed_tol = cfg.pressure_tolerance * 10
        fallback = Solve(
            method="CG",
            abs_tol=relaxed_tol,
            rel_tol=relaxed_tol,
            max_iterations=cfg.pressure_iterations * 3,
        )
        velocity, pressure = fluid.make_incompressible(velocity, obstacles=obstacles, solve=fallback)
    return velocity, pressure


def to_numpy_centered(field: Any) -> np.ndarray:
    return field.values.numpy("x,y,z")


def to_numpy_staggered(field: Any) -> np.ndarray:
    center = CenteredGrid(
        0.0,
        bounds=field.bounds,
        x=field.resolution.spatial["x"].size,
        y=field.resolution.spatial["y"].size,
        z=field.resolution.spatial["z"].size,
    )
    sampled = field.at(center)
    stacked = phi_math.stack(
        [sampled.values.vector["x"], sampled.values.vector["y"], sampled.values.vector["z"]],
        phi_math.channel("vector"),
    )
    return stacked.numpy("x,y,z,vector")


def save_sample(out_dir: Path, sample_idx: int, payload: Dict[str, np.ndarray]) -> None:
    out_path = out_dir / f"sample_{sample_idx:05d}.npy"
    np.save(out_path, np.array(payload, dtype=object), allow_pickle=True)


def save_metadata(out_dir: Path, cfg: GenerationConfig) -> None:
    meta = {**cfg.__dict__}
    with open(out_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def generate_dataset(cfg: GenerationConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = Box(x=cfg.grid_size, y=cfg.grid_size, z=cfg.grid_size)  # type: ignore[arg-type]
    inflow_mask_np = build_inflow_mask(cfg)
    inflow_mask = field_from_mask(inflow_mask_np, bounds)

    save_metadata(out_dir, cfg)

    sample_idx = 0
    attempts = 0
    max_attempts = cfg.num_samples * 5
    while sample_idx < cfg.num_samples and attempts < max_attempts:
        attempts += 1
        obstacle_mask, obstacles = generate_obstacle_mask(cfg)
        velocity = build_velocity_field(cfg, bounds)

        try:
            for _ in range(cfg.warmup_steps):
                velocity, _ = step_flow(velocity, obstacles, inflow_mask, cfg)

            velocity_t = velocity
            pressure_t = None
            if cfg.steps_per_sample > 0:
                _, pressure_t = step_flow(velocity_t, obstacles, inflow_mask, cfg)
            for _ in range(cfg.steps_per_sample):
                velocity, pressure = step_flow(velocity, obstacles, inflow_mask, cfg)
        except (NotConverged, Diverged):
            continue

        if pressure_t is None:
            pressure_t = pressure

        payload = {
            "obstacle_mask": obstacle_mask.astype(np.uint8),
            "velocity_t": to_numpy_staggered(velocity_t).astype(np.float32),
            "velocity_t1": to_numpy_staggered(velocity).astype(np.float32),
            "pressure_t": to_numpy_centered(pressure_t).astype(np.float32),
            "pressure_t1": to_numpy_centered(pressure).astype(np.float32),
        }

        if not all(np.isfinite(value).all() for value in payload.values()):
            continue

        save_sample(out_dir, sample_idx, payload)
        sample_idx += 1

    if sample_idx < cfg.num_samples:
        raise RuntimeError(
            f"Only generated {sample_idx} samples before hitting stability limits."
        )


def main() -> None:
    phi_backend.set_global_default_backend("torch")
    cfg = parse_args()
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
    out_dir = Path(cfg.out_dir) / f"grid_{cfg.grid_size}"
    generate_dataset(cfg, out_dir)
    print(f"Saved {cfg.num_samples} samples to {out_dir}")


if __name__ == "__main__":
    main()
