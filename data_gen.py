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
    steps_per_sample: int = 20
    warmup_steps: int = random.randint(5, 20)
    dt: float = 0.1
    inflow_speed: float = 1.0
    max_obstacles: int = 5
    min_obstacles: int = 1
    obstacle_density_cap: float = 0.5
    pressure_iterations: int = 700
    pressure_tolerance: float = 1e-5 # Stricter tolerance is fine now
    seed: int | None = 7
    obstacle_move_range: int = 1
    obstacle_move_prob: float = 0.0
    obstacle_resize_prob: float = 0.0
    obstacle_add_prob: float = 0.0
    obstacle_remove_prob: float = 0.0
    voxel_add_prob: float = 0.5
    voxel_remove_prob: float = 0.5
    voxel_max_count: int = 64

 
OBSTACLE_TYPES = ("cube", "wall", "sphere")


@dataclass
class ObstacleSpec:
    kind: str
    params: Dict[str, Any]


def make_box(lower: Tuple[float, float, float], upper: Tuple[float, float, float]) -> Any:
    return Box(
        x=(lower[0], upper[0]), # type: ignore
        y=(lower[1], upper[1]), # type: ignore
        z=(lower[2], upper[2]), # type: ignore
    )


def make_sphere(center: Tuple[float, float, float], radius: float) -> Any:
    return cast(
        Any,
        Sphere(
            x=center[0], # type: ignore
            y=center[1], # type: ignore
            z=center[2], # type: ignore
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


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _margins(cfg: GenerationConfig) -> Tuple[int, int]:
    margin_x = max(8, cfg.grid_size // 4)
    margin_yz = 2
    return margin_x, margin_yz


def _random_obstacle_spec(cfg: GenerationConfig) -> ObstacleSpec:
    grid = cfg.grid_size
    margin_x, margin_yz = _margins(cfg)
    obstacle_type = random.choice(OBSTACLE_TYPES)
    if obstacle_type == "cube":
        size = _random_int(3, max(4, grid // 4))
        x0 = _random_int(margin_x, grid - margin_x - size)
        y0 = _random_int(margin_yz, grid - margin_yz - size)
        z0 = _random_int(margin_yz, grid - margin_yz - size)
        return ObstacleSpec("cube", {"x0": x0, "y0": y0, "z0": z0, "size": size})
    if obstacle_type == "wall":
        thickness = _random_int(1, 2)
        orientation = random.choice(["y", "z"])
        if orientation == "y":
            y0 = _random_int(margin_yz, grid - margin_yz - thickness)
            x_start = _random_int(margin_x, grid // 2)
            x_len = _random_int(4, grid // 2)
            return ObstacleSpec(
                "wall",
                {"orientation": "y", "y0": y0, "thickness": thickness, "x_start": x_start, "x_len": x_len},
            )
        z0 = _random_int(margin_yz, grid - margin_yz - thickness)
        x_start = _random_int(margin_x, grid // 2)
        x_len = _random_int(4, grid // 2)
        return ObstacleSpec(
            "wall",
            {"orientation": "z", "z0": z0, "thickness": thickness, "x_start": x_start, "x_len": x_len},
        )
    radius = _random_int(2, max(3, grid // 6))
    center = (
        _random_float(margin_x + radius, grid - margin_x - radius),
        _random_float(margin_yz + radius, grid - margin_yz - radius),
        _random_float(margin_yz + radius, grid - margin_yz - radius),
    )
    return ObstacleSpec("sphere", {"center": center, "radius": radius})


def _random_voxel_spec(cfg: GenerationConfig, occupied: np.ndarray) -> ObstacleSpec | None:
    margin_x, margin_yz = _margins(cfg)
    for _ in range(20):
        x = _random_int(margin_x, cfg.grid_size - margin_x - 1)
        y = _random_int(margin_yz, cfg.grid_size - margin_yz - 1)
        z = _random_int(margin_yz, cfg.grid_size - margin_yz - 1)
        if occupied[x, y, z] == 0:
            return ObstacleSpec("voxel", {"x": x, "y": y, "z": z})
    return None


def _rasterize_obstacles(cfg: GenerationConfig, specs: List[ObstacleSpec]) -> np.ndarray:
    grid = cfg.grid_size
    mask = np.zeros((grid, grid, grid), dtype=np.uint8)
    for spec in specs:
        if spec.kind == "cube":
            x0 = int(spec.params["x0"])
            y0 = int(spec.params["y0"])
            z0 = int(spec.params["z0"])
            size = int(spec.params["size"])
            mask[x0 : x0 + size, y0 : y0 + size, z0 : z0 + size] = 1
        elif spec.kind == "wall":
            orientation = spec.params["orientation"]
            thickness = int(spec.params["thickness"])
            x_start = int(spec.params["x_start"])
            x_len = int(spec.params["x_len"])
            if orientation == "y":
                y0 = int(spec.params["y0"])
                mask[x_start : x_start + x_len, y0 : y0 + thickness, 2:-2] = 1
            else:
                z0 = int(spec.params["z0"])
                mask[x_start : x_start + x_len, 2:-2, z0 : z0 + thickness] = 1
        elif spec.kind == "sphere":
            center = cast(Tuple[float, float, float], spec.params["center"])
            radius = float(spec.params["radius"])
            xs = np.arange(grid).reshape(-1, 1, 1)
            ys = np.arange(grid).reshape(1, -1, 1)
            zs = np.arange(grid).reshape(1, 1, -1)
            dist = np.sqrt(
                (xs - center[0]) ** 2 + (ys - center[1]) ** 2 + (zs - center[2]) ** 2
            )
            mask[dist <= radius] = 1
        elif spec.kind == "voxel":
            x = int(spec.params["x"])
            y = int(spec.params["y"])
            z = int(spec.params["z"])
            mask[x, y, z] = 1
    return mask


def _specs_to_phi(cfg: GenerationConfig, specs: List[ObstacleSpec]) -> List[object]:
    obstacles: List[object] = []
    for spec in specs:
        if spec.kind == "cube":
            x0 = float(spec.params["x0"])
            y0 = float(spec.params["y0"])
            z0 = float(spec.params["z0"])
            size = float(spec.params["size"])
            obstacles.append(make_box((x0, y0, z0), (x0 + size, y0 + size, z0 + size)))
        elif spec.kind == "wall":
            orientation = spec.params["orientation"]
            thickness = float(spec.params["thickness"])
            x_start = float(spec.params["x_start"])
            x_len = float(spec.params["x_len"])
            if orientation == "y":
                y0 = float(spec.params["y0"])
                obstacles.append(make_box((x_start, y0, 2), (x_start + x_len, y0 + thickness, cfg.grid_size - 2)))
            else:
                z0 = float(spec.params["z0"])
                obstacles.append(make_box((x_start, 2, z0), (x_start + x_len, cfg.grid_size - 2, z0 + thickness)))
        elif spec.kind == "sphere":
            center = cast(Tuple[float, float, float], spec.params["center"])
            radius = float(spec.params["radius"])
            obstacles.append(make_sphere(center, radius))
        elif spec.kind == "voxel":
            x = float(spec.params["x"])
            y = float(spec.params["y"])
            z = float(spec.params["z"])
            obstacles.append(make_box((x, y, z), (x + 1.0, y + 1.0, z + 1.0)))
    return obstacles


def _update_obstacles(cfg: GenerationConfig, specs: List[ObstacleSpec]) -> List[ObstacleSpec]:
    grid = cfg.grid_size
    margin_x, margin_yz = _margins(cfg)
    updated: List[ObstacleSpec] = []
    voxel_specs = [spec for spec in specs if spec.kind == "voxel"]
    macro_specs = [spec for spec in specs if spec.kind != "voxel"]

    for spec in macro_specs:
        if random.random() < cfg.obstacle_remove_prob:
            continue
        dx = _random_int(-cfg.obstacle_move_range, cfg.obstacle_move_range) if random.random() < cfg.obstacle_move_prob else 0
        dy = _random_int(-cfg.obstacle_move_range, cfg.obstacle_move_range) if random.random() < cfg.obstacle_move_prob else 0
        dz = _random_int(-cfg.obstacle_move_range, cfg.obstacle_move_range) if random.random() < cfg.obstacle_move_prob else 0
        if spec.kind == "cube":
            size = int(spec.params["size"])
            if random.random() < cfg.obstacle_resize_prob:
                size = _clamp(size + random.choice([-1, 1]), 2, max(3, grid // 3))
            x0 = _clamp(int(spec.params["x0"]) + dx, margin_x, grid - margin_x - size)
            y0 = _clamp(int(spec.params["y0"]) + dy, margin_yz, grid - margin_yz - size)
            z0 = _clamp(int(spec.params["z0"]) + dz, margin_yz, grid - margin_yz - size)
            updated.append(ObstacleSpec("cube", {"x0": x0, "y0": y0, "z0": z0, "size": size}))
        elif spec.kind == "wall":
            orientation = spec.params["orientation"]
            thickness = int(spec.params["thickness"])
            x_start = int(spec.params["x_start"])
            x_len = int(spec.params["x_len"])
            if random.random() < cfg.obstacle_resize_prob:
                thickness = _clamp(thickness + random.choice([-1, 1]), 1, 3)
                x_len = _clamp(x_len + random.choice([-2, 2]), 3, grid // 2)
            x_start = _clamp(x_start + dx, margin_x, grid - margin_x - x_len)
            if orientation == "y":
                y0 = _clamp(int(spec.params["y0"]) + dy, margin_yz, grid - margin_yz - thickness)
                updated.append(
                    ObstacleSpec(
                        "wall",
                        {"orientation": "y", "y0": y0, "thickness": thickness, "x_start": x_start, "x_len": x_len},
                    )
                )
            else:
                z0 = _clamp(int(spec.params["z0"]) + dz, margin_yz, grid - margin_yz - thickness)
                updated.append(
                    ObstacleSpec(
                        "wall",
                        {"orientation": "z", "z0": z0, "thickness": thickness, "x_start": x_start, "x_len": x_len},
                    )
                )
        else:
            center = list(cast(Tuple[float, float, float], spec.params["center"]))
            radius = float(spec.params["radius"])
            if random.random() < cfg.obstacle_resize_prob:
                radius = _clamp(int(radius + random.choice([-1, 1])), 2, max(3, grid // 5))
            center[0] = _clamp(int(center[0] + dx), margin_x + int(radius), grid - margin_x - int(radius))
            center[1] = _clamp(int(center[1] + dy), margin_yz + int(radius), grid - margin_yz - int(radius))
            center[2] = _clamp(int(center[2] + dz), margin_yz + int(radius), grid - margin_yz - int(radius))
            updated.append(ObstacleSpec("sphere", {"center": tuple(center), "radius": radius}))

    # Voxel-level mutations
    if voxel_specs and random.random() < cfg.voxel_remove_prob:
        voxel_specs.pop(random.randrange(len(voxel_specs)))

    current_mask = _rasterize_obstacles(cfg, updated + voxel_specs)
    if random.random() < cfg.voxel_add_prob and len(voxel_specs) < cfg.voxel_max_count:
        new_voxel = _random_voxel_spec(cfg, current_mask)
        if new_voxel is not None:
            voxel_specs.append(new_voxel)

    if random.random() < cfg.obstacle_add_prob:
        updated.append(_random_obstacle_spec(cfg))

    # Ensure density cap by trimming obstacles if needed
    while updated and _rasterize_obstacles(cfg, updated + voxel_specs).mean() > cfg.obstacle_density_cap:
        updated.pop(random.randrange(len(updated)))

    return updated + voxel_specs


def generate_obstacle_mask(cfg: GenerationConfig) -> Tuple[np.ndarray, List[ObstacleSpec]]:
    """
    Generates obstacles with safety margins to prevent blocking inflow/outflow.
    """
    grid = cfg.grid_size
    if grid < 10:
        return np.zeros((grid, grid, grid), dtype=np.uint8), []
    obstacle_count = _random_int(cfg.min_obstacles, cfg.max_obstacles)
    specs = [_random_obstacle_spec(cfg) for _ in range(obstacle_count)]
    mask = _rasterize_obstacles(cfg, specs)
    if mask.mean() > cfg.obstacle_density_cap:
        return generate_obstacle_mask(cfg)
    return mask, specs


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
        y=extrapolation.ZERO_GRADIENT,
        z=extrapolation.ZERO_GRADIENT
    )
    
    return StaggeredGrid(
        vec(x=cfg.inflow_speed, y=0.0, z=0.0), 
        bounds=bounds, 
        extrapolation=v_extrap,
        x=cfg.grid_size, 
        y=cfg.grid_size, 
        z=cfg.grid_size
    )


def apply_inflow(velocity: Any, inflow_mask: Any, inflow_speed: float, t: int) -> Any:
    u = inflow_speed * (0.5 + 0.5 * phi_math.sin(0.2 * t))
    # Use the grid's existing extrapolation for the new field
    inflow = StaggeredGrid(
        vec(x=u, y=0.2 * phi_math.sin(t), z=0.2 * phi_math.cos(t)),
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
#@phi_math.jit_compile
def step_flow_jit(
    velocity: Any,
    pressure_guess: Any,
    obstacles: List[object],
    inflow_mask: Any,
    t: int,
    dt: float,
    inflow_speed: float,
    tolerance: float,
    max_iter: int
) -> Tuple[Any, Any]:
    
    # 1. Advect
    velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
    # 2. Diffuse
    velocity = diffuse.explicit(velocity, diffusivity=1.8e-5, dt=dt)
    # 3. Apply Inflow Boundary Condition (Dirichlet on velocity)
    velocity = apply_inflow(velocity, inflow_mask, inflow_speed, t)
    
    # 4. Projection (Make Incompressible)
    solve = Solve(
        method="CG",
        abs_tol=tolerance,
        rel_tol=tolerance,
        max_iterations=max_iter,
        #rank_deficiency=0, # No nullspace for open boundaries
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
    bounds = Box(x=cfg.grid_size, y=cfg.grid_size, z=cfg.grid_size) # type: ignore
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
        obstacle_mask, obstacle_specs = generate_obstacle_mask(cfg)
        velocity = build_velocity_field(cfg, bounds)
        pressure = initial_pressure
        t = 0
        obstacle_mask_t = obstacle_mask
        # Warmup
        try:
            for _ in range(cfg.warmup_steps):
                obstacle_specs = _update_obstacles(cfg, obstacle_specs)
                obstacle_mask = _rasterize_obstacles(cfg, obstacle_specs)
                obstacles = _specs_to_phi(cfg, obstacle_specs)
                velocity, pressure = step_flow_jit(
                    velocity, pressure, obstacles, inflow_mask,
                    t, cfg.dt, cfg.inflow_speed, cfg.pressure_tolerance, cfg.pressure_iterations
                )
                t += 1
            velocity_t = velocity
            pressure_t = pressure # Capturing pressure input is rare for simple datasets but requested
            obstacle_mask_t = obstacle_mask
            div = fluid.divergence(velocity)
            mean_div = float(phi_math.mean(phi_math.abs(div.values)).numpy())
            momentum = phi_math.mean(velocity.values).numpy("vector")
            print(f"Attempt {attempts}: Warmup complete, divergence = {mean_div}, total momentum = {momentum}. Generating sample...")
            # Step to Target
            velocity_n = []
            pressure_n = []
            obstacle_mask_n = []
            for _ in range(cfg.steps_per_sample):
                obstacle_specs = _update_obstacles(cfg, obstacle_specs)
                obstacle_mask = _rasterize_obstacles(cfg, obstacle_specs)
                obstacles = _specs_to_phi(cfg, obstacle_specs)
                velocity, pressure = step_flow_jit(
                    velocity, pressure, obstacles, inflow_mask,
                    t, cfg.dt, cfg.inflow_speed, cfg.pressure_tolerance, cfg.pressure_iterations
                )
                velocity_n.append(to_numpy_velocity_centered(velocity).astype(np.float32))
                pressure_n.append(to_numpy_pressure_centered(pressure).astype(np.float32))
                obstacle_mask_n.append(obstacle_mask.astype(np.bool_))
                t += 1
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
            "obstacle_mask": obstacle_mask_t.astype(np.bool_),
            "obstacle_mask_tn": np.stack(obstacle_mask_n, axis=0),
            "velocity_t": to_numpy_velocity_centered(velocity_t).astype(np.float32),
            "velocity_tn": np.stack(velocity_n, axis=0),
            "pressure_t": to_numpy_pressure_centered(pressure_t).astype(np.float32),
            "pressure_tn": np.stack(pressure_n, axis=0),
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