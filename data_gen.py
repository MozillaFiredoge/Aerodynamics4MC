from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast
import numpy as np
import torch  # Explicit import to ensure torch is available

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
        extrapolation,
        diffuse,
    )
    from phi.math import Diverged, NotConverged
    from phi.math import backend as phi_backend
    from phi.geom import Sphere
except Exception as exc:
    raise RuntimeError(
        "PhiFlow is required. Install dependencies with `pip install phiflow torch`."
    ) from exc


@dataclass
class GenerationConfig:
    grid_size: int = 32
    num_samples: int = 100
    out_dir: str = "data"
    steps_per_sample: int = 5
    warmup_steps: int = 320
    dt: float = 0.1
    inflow_speed: float = 1.0
    max_obstacles: int = 5
    min_obstacles: int = 1
    obstacle_density_cap: float = 0.35
    pressure_iterations: int = 100
    pressure_tolerance: float = 1e-5 # Stricter tolerance is fine now
    seed: int | None = 7

 
OBSTACLE_TYPES = ("cube", "wall", "sphere")


def make_box(lower: Tuple[float, float, float], upper: Tuple[float, float, float]) -> Any:
    return Box(
        x=(lower[0], upper[0]),
        y=(lower[1], upper[1]),
        z=(lower[2], upper[2]),
    )


def make_sphere(center: Tuple[float, float, float], radius: float) -> Any:
    return cast(
        Any,
        Sphere(
            x=center[0],
            y=center[1],
            z=center[2],
            radius=radius,
        ),
    )


def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Generate PhiFlow CFD dataset.")
    parser.add_argument("--out-dir", type=str, default="data_fixed", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--grid-size", type=int, default=32, help="Voxel grid size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    return GenerationConfig(
        grid_size=args.grid_size,
        num_samples=args.num_samples,
        out_dir=args.out_dir,
        seed=args.seed,
    )


def _random_int(low: int, high: int) -> int:
    return random.randint(low, high)


def _random_float(low: float, high: float) -> float:
    return random.uniform(low, high)


def generate_obstacle_mask(cfg: GenerationConfig) -> Tuple[np.ndarray, List[object]]:
    """
    Generates obstacles with safety margins to prevent blocking inflow/outflow.
    """
    grid = cfg.grid_size
    mask = np.zeros((grid, grid, grid), dtype=np.uint8)
    obstacles: List[object] = []

    # Safe margin
    margin_x = max(8, grid // 4)
    margin_yz = 2

    obstacle_count = _random_int(cfg.min_obstacles, cfg.max_obstacles)
    
    # Safety check for small grids
    if grid < 10:
        return mask, obstacles

    for _ in range(obstacle_count):
        obstacle_type = random.choice(OBSTACLE_TYPES)
        
        if obstacle_type == "cube":
            size = _random_int(3, max(4, grid // 4))
            # Restrict X placement
            x0 = _random_int(margin_x, grid - margin_x - size)
            y0 = _random_int(margin_yz, grid - margin_yz - size)
            z0 = _random_int(margin_yz, grid - margin_yz - size)
            
            mask[x0 : x0 + size, y0 : y0 + size, z0 : z0 + size] = 1
            obstacles.append(
                make_box((x0, y0, z0), (x0 + size, y0 + size, z0 + size))
            )
            
        elif obstacle_type == "wall":
            thickness = _random_int(1, 2)
            orientation = random.choice(["y", "z"]) # Avoid X-walls that block flow completely
            
            if orientation == "y":
                # Wall spanning X-Z plane? No, let's keep it simple.
                # Wall perpendicular to Y axis      
                y0 = _random_int(margin_yz, grid - margin_yz - thickness)
                # Ensure it doesn't span full X if possible, or allow it but flow goes around
                # Here we make partial walls
                x_start = _random_int(margin_x, grid // 2)
                x_len = _random_int(4, grid // 2)
                
                mask[x_start : x_start + x_len, y0 : y0 + thickness, 2:-2] = 1
                obstacles.append(
                    make_box(
                        (x_start, y0, 2),
                        (x_start + x_len, y0 + thickness, grid - 2),
                    )
                )
            else:
                # Wall perpendicular to Z axis
                z0 = _random_int(margin_yz, grid - margin_yz - thickness)
                x_start = _random_int(margin_x, grid // 2)
                x_len = _random_int(4, grid // 2)

                mask[x_start : x_start + x_len, 2:-2, z0 : z0 + thickness] = 1
                obstacles.append(
                    make_box(
                        (x_start, 2, z0),
                        (x_start + x_len, grid - 2, z0 + thickness),
                    )
                )
                
        else: # Sphere
            radius = _random_int(2, max(3, grid // 6))
            center = (
                _random_float(margin_x + radius, grid - margin_x - radius),
                _random_float(margin_yz + radius, grid - margin_yz - radius),
                _random_float(margin_yz + radius, grid - margin_yz - radius),
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
    # Define inflow region on the left face
    inflow[0:2, :, :] = 1.0 
    return inflow


def field_from_mask(mask: np.ndarray, bounds: Any) -> Any:
    tensor = phi_math.tensor(mask, phi_math.spatial("x,y,z"))
    return CenteredGrid(
        tensor,
        bounds=bounds,
        extrapolation=extrapolation.ZERO,
        x=mask.shape[0],
        y=mask.shape[1],
        z=mask.shape[2],
    )


def build_velocity_field(cfg: GenerationConfig, bounds: Any) -> Any:

    v_extrap = extrapolation.combine_sides(
        x=extrapolation.ZERO_GRADIENT,
        y=extrapolation.ZERO,
        z=extrapolation.ZERO
    )
    
    return StaggeredGrid(
        vec(x=cfg.inflow_speed, y=0.0, z=0.0), 
        bounds=bounds, 
        extrapolation=v_extrap,
        x=cfg.grid_size, 
        y=cfg.grid_size, 
        z=cfg.grid_size
    )


def apply_inflow(velocity: Any, inflow_mask: Any, inflow_speed: float) -> Any:
    # Use the grid's existing extrapolation for the new field
    inflow = StaggeredGrid(
        vec(x=inflow_speed, y=0.0, z=0.0),
        bounds=velocity.bounds,
        extrapolation=velocity.extrapolation,
        x=velocity.resolution.spatial["x"].size,
        y=velocity.resolution.spatial["y"].size,
        z=velocity.resolution.spatial["z"].size,
    )
    inflow_mask_staggered = inflow_mask.at(inflow)
    # Hard set the inflow velocity where mask is active
    return velocity * (1 - inflow_mask_staggered) + inflow * inflow_mask_staggered

# JIT Compile the physics step for massive speedup
@phi_math.jit_compile
def step_flow_jit(
    velocity: Any,
    pressure_guess: Any,
    obstacles: List[object],
    inflow_mask: Any,
    dt: float,
    inflow_speed: float,
    tolerance: float,
    max_iter: int
) -> Tuple[Any, Any]:
    
    # 1. Advect
    velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
    # 2. Diffuse
    velocity = diffuse.explicit(velocity, diffusivity=1e-3, dt=dt)
    # 3. Apply Inflow Boundary Condition (Dirichlet on velocity)
    velocity = apply_inflow(velocity, inflow_mask, inflow_speed)
    
    # 4. Projection (Make Incompressible)
    solve = Solve(
        method="CG",
        abs_tol=tolerance,
        rel_tol=tolerance,
        max_iterations=max_iter,
        rank_deficiency=0,
        x0=pressure_guess # Warm start pressure solver
    )
    
    velocity, pressure = fluid.make_incompressible(
        velocity, 
        obstacles=obstacles, 
        solve=solve,
    )
    
    return velocity, pressure


def to_numpy_pressure_centered(field: Any) -> np.ndarray:
    return field.values.numpy("x,y,z")


def to_numpy_velocity_centered(field: Any) -> np.ndarray:
    sampled = field.at_centers()
    stacked = phi_math.stack(
        [sampled.values.vector["x"], sampled.values.vector["y"], sampled.values.vector["z"]],
        phi_math.channel("vector"),
    )
    return stacked.numpy("x,y,z,vector")


def save_sample(out_dir: Path, sample_idx: int, payload: Dict[str, np.ndarray]) -> None:
    out_path = out_dir / f"sample_{sample_idx:05d}.npy"
    np.save(out_path, np.array(payload, dtype=object), allow_pickle=True)


def save_metadata(out_dir: Path, cfg: GenerationConfig) -> None:
    meta = {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')}
    with open(out_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def generate_dataset(cfg: GenerationConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = Box(x=cfg.grid_size, y=cfg.grid_size, z=cfg.grid_size)
    inflow_mask_np = build_inflow_mask(cfg)
    inflow_mask = field_from_mask(inflow_mask_np, bounds)

    save_metadata(out_dir, cfg)

    sample_idx = 0
    attempts = 0
    
    # Pre-define empty pressure for initial guess
    initial_pressure = CenteredGrid(
        0, bounds=bounds, extrapolation=extrapolation.ZERO, 
        x=cfg.grid_size, y=cfg.grid_size, z=cfg.grid_size
    )

    print(f"Starting generation with Open Boundary on +X. Target: {cfg.num_samples}")

    while sample_idx < cfg.num_samples:
        attempts += 1
        obstacle_mask, obstacles = generate_obstacle_mask(cfg)
        velocity = build_velocity_field(cfg, bounds)
        pressure = initial_pressure
        # Warmup
        try:
            for _ in range(cfg.warmup_steps):
                velocity, pressure = step_flow_jit(
                    velocity, pressure, obstacles, inflow_mask, 
                    cfg.dt, cfg.inflow_speed, cfg.pressure_tolerance, cfg.pressure_iterations
                )
            velocity_t = velocity
            pressure_t = pressure # Capturing pressure input is rare for simple datasets but requested
            div = fluid.divergence(velocity)
            mean_div = float(phi_math.mean(phi_math.abs(div.values)).numpy())
            momentum = phi_math.mean(velocity.values).numpy("vector")
            #energy = field.mean(field.l2_loss(velocity)).values
            print(f"Attempt {attempts}: Warmup complete, divergence = {mean_div}, total momentum = {momentum}. Generating sample...")
            # Step to Target
            for _ in range(cfg.steps_per_sample):
                velocity, pressure = step_flow_jit(
                    velocity, pressure, obstacles, inflow_mask, 
                    cfg.dt, cfg.inflow_speed, cfg.pressure_tolerance, cfg.pressure_iterations
                )
            div = fluid.divergence(velocity)
            mean_div = float(phi_math.mean(phi_math.abs(div.values)).numpy())
            momentum = phi_math.mean(velocity.values).numpy("vector")
            print(f"Attempt {attempts}: Step complete, divergence = {mean_div}, total momentum = {momentum}. Saving sample...")
        except (NotConverged, Diverged) as e:
            print(f"Attempt {attempts}: Solver failed ({e}). Retrying with new geometry...")
            continue
        except Exception as e:
            print(f"Attempt {attempts}: Unexpected error {e}")
            continue
        
        # Data Export
        # Note: We cast to float32 immediately to save space
        payload = {
            "obstacle_mask": obstacle_mask.astype(np.bool_),
            "velocity_t": to_numpy_velocity_centered(velocity_t).astype(np.float32),
            "velocity_t1": to_numpy_velocity_centered(velocity).astype(np.float32),
            "pressure_t": to_numpy_pressure_centered(pressure_t).astype(np.float32),
            "pressure_t1": to_numpy_pressure_centered(pressure).astype(np.float32),
        }
        # NaN check
        if any(np.isnan(v).any() for v in payload.values() if isinstance(v, np.ndarray)):
            print(f"Attempt {attempts}: NaN detected.")
            continue

        save_sample(out_dir, sample_idx, payload)
        print(f"Generated sample {sample_idx+1}/{cfg.num_samples}")
        sample_idx += 1


def main() -> None:
    # Use PyTorch or TensorFlow backend usually for JIT
    phi_backend.set_global_default_backend("torch")
    
    cfg = parse_args()
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    out_dir = Path(cfg.out_dir) / f"grid_{cfg.grid_size}"
    generate_dataset(cfg, out_dir)
    print(f"Done. Saved to {out_dir}")


if __name__ == "__main__":
    main()