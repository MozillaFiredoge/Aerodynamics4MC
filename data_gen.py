from __future__ import annotations

import argparse
import json
import math
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
        Obstacle,
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
    from phi.geom import SDFGrid, Sphere
except Exception as exc:
    raise RuntimeError(
        "PhiFlow is required. Install dependencies with `pip install phiflow torch`."
    ) from exc


@dataclass
class GenerationConfig:
    grid_size: int = 32
    num_samples: int = 100
    out_dir: str = "data"
    steps_per_sample: int = 32
    warmup_steps: int = 4
    dt: float = 1.0
    inflow_speed: float = 8.0
    inflow_direction_variation: float = 0.05
    inflow_direction_frequency: float = 0.15
    inflow_fan_strength: float = 0.8
    min_inflows: int = 1
    max_inflows: int = 3
    max_obstacles: int = 5
    min_obstacles: int = 1
    obstacle_density_cap: float = 0.5
    pressure_iterations: int = 300
    pressure_tolerance: float = 1e-4
    seed: int | None = 7
    obstacle_move_range: int = 1
    obstacle_move_prob: float = 0.0
    obstacle_resize_prob: float = 0.0
    obstacle_add_prob: float = 0.0
    obstacle_remove_prob: float = 0.0
    voxel_add_prob: float = 0.0
    voxel_remove_prob: float = 0.0
    voxel_max_count: int = 64
    karman_mode: bool = False
    karman_ratio: float = 0.0
    resume: bool = False

 
OBSTACLE_TYPES = ("cube", "wall", "sphere", "cylinder")
INFLOW_DIRECTIONS = (("x", 1), ("x", -1), ("y", 1), ("y", -1), ("z", 1), ("z", -1))


@dataclass
class ObstacleSpec:
    kind: str
    params: Dict[str, Any]


@dataclass
class InflowSpec:
    center: Tuple[int, int, int]
    axis: str
    sign: int
    mode: str
    shape: str
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


def make_cylinder(
    axis: str,
    start: float,
    end: float,
    center_a: float,
    center_b: float,
    radius: float,
) -> Any:
    if axis == "x":
        return make_box((start, center_a - radius, center_b - radius), (end, center_a + radius, center_b + radius))
    if axis == "y":
        return make_box((center_a - radius, start, center_b - radius), (center_a + radius, end, center_b + radius))
    return make_box((center_a - radius, center_b - radius, start), (center_a + radius, center_b + radius, end))


def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Generate PhiFlow CFD dataset.")
    parser.add_argument("--out-dir", type=str, default="data_fixed", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--grid-size", type=int, default=32, help="Voxel grid size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps-per-sample", type=int, default=32, help="Saved rollout steps per sample")
    parser.add_argument("--warmup-steps", type=int, default=4, help="Warmup steps before saving sample state")
    parser.add_argument("--inflow-speed", type=float, default=8.0, help="Target inflow speed magnitude")
    parser.add_argument("--min-inflows", type=int, default=1, help="Minimum number of fan sources per sample")
    parser.add_argument("--max-inflows", type=int, default=3, help="Maximum number of fan sources per sample")
    parser.add_argument(
        "--inflow-fan-strength",
        type=float,
        default=0.6,
        help="Relaxation strength for fan-style inflow forcing (0-1).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing samples")
    parser.add_argument("--karman-mode", action="store_true", help="Only generate Karman vortex street samples")
    parser.add_argument(
        "--karman-ratio",
        type=float,
        default=0.0,
        help="Fraction of samples to use Karman mode (0-1)",
    )
    args = parser.parse_args()
    min_inflows = max(1, args.min_inflows)
    max_inflows = max(min_inflows, args.max_inflows)
    return GenerationConfig(
        grid_size=args.grid_size,
        num_samples=args.num_samples,
        out_dir=args.out_dir,
        steps_per_sample=args.steps_per_sample,
        warmup_steps=args.warmup_steps,
        inflow_speed=args.inflow_speed,
        min_inflows=min_inflows,
        max_inflows=max_inflows,
        seed=args.seed,
        inflow_fan_strength=args.inflow_fan_strength,
        karman_mode=args.karman_mode,
        karman_ratio=args.karman_ratio,
        resume=args.resume,
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
    if obstacle_type == "cylinder":
        length = _random_int(4, max(6, grid // 2))
        axis = random.choice(["x", "y", "z"])
        if axis == "x":
            start = _random_int(margin_x, grid - margin_x - length)
            center_a = _random_float(margin_yz + radius, grid - margin_yz - radius)
            center_b = _random_float(margin_yz + radius, grid - margin_yz - radius)
        elif axis == "y":
            start = _random_int(margin_yz, grid - margin_yz - length)
            center_a = _random_float(margin_x + radius, grid - margin_x - radius)
            center_b = _random_float(margin_yz + radius, grid - margin_yz - radius)
        else:
            start = _random_int(margin_yz, grid - margin_yz - length)
            center_a = _random_float(margin_x + radius, grid - margin_x - radius)
            center_b = _random_float(margin_yz + radius, grid - margin_yz - radius)
        return ObstacleSpec(
            "cylinder",
            {
                "axis": axis,
                "start": start,
                "end": start + length,
                "center_a": center_a,
                "center_b": center_b,
                "radius": radius,
            },
        )
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


def _karman_obstacle_specs(cfg: GenerationConfig) -> List[ObstacleSpec]:
    grid = cfg.grid_size
    margin_x, margin_yz = _margins(cfg)
    radius = max(3, grid // 16)
    center_x = _clamp(grid // 3, margin_x + radius + 1, grid - margin_x - radius - 1)
    center_y = _clamp(grid // 2, margin_yz + radius + 1, grid - margin_yz - radius - 1)
    start = margin_yz
    end = grid - margin_yz
    return [
        ObstacleSpec(
            "cylinder",
            {
                "axis": "z",
                "start": start,
                "end": end,
                "center_a": center_x,
                "center_b": center_y,
                "radius": radius,
            },
        )
    ]


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
        elif spec.kind == "cylinder":
            axis = spec.params["axis"]
            start = int(spec.params["start"])
            end = int(spec.params["end"])
            center_a = float(spec.params["center_a"])
            center_b = float(spec.params["center_b"])
            radius = float(spec.params["radius"])
            coords = np.arange(grid).astype(np.float32)
            dist = np.sqrt((coords[:, None] - center_a) ** 2 + (coords[None, :] - center_b) ** 2)
            cylinder_mask = dist <= radius
            if axis == "x":
                mask[start:end, :, :] = np.where(cylinder_mask[None, :, :], 1, mask[start:end, :, :])
            elif axis == "y":
                mask[:, start:end, :] = np.where(cylinder_mask[:, None, :], 1, mask[:, start:end, :])
            else:
                mask[:, :, start:end] = np.where(cylinder_mask[:, :, None], 1, mask[:, :, start:end])
        elif spec.kind == "voxel":
            x = int(spec.params["x"])
            y = int(spec.params["y"])
            z = int(spec.params["z"])
            mask[x, y, z] = 1
    return mask


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
        elif spec.kind == "cylinder":
            axis = str(spec.params["axis"])
            start = int(spec.params["start"])
            end = int(spec.params["end"])
            length = max(2, end - start)
            center_a = float(spec.params["center_a"])
            center_b = float(spec.params["center_b"])
            radius = float(spec.params["radius"])
            if random.random() < cfg.obstacle_resize_prob:
                radius = _clamp(int(radius + random.choice([-1, 1])), 2, max(3, grid // 5))
                length = _clamp(length + random.choice([-2, 2]), 3, grid // 2)
            if axis == "x":
                start = _clamp(start + dx, margin_x, grid - margin_x - length)
                center_a = _clamp(int(center_a + dy), margin_yz + int(radius), grid - margin_yz - int(radius))
                center_b = _clamp(int(center_b + dz), margin_yz + int(radius), grid - margin_yz - int(radius))
            elif axis == "y":
                start = _clamp(start + dy, margin_yz, grid - margin_yz - length)
                center_a = _clamp(int(center_a + dx), margin_x + int(radius), grid - margin_x - int(radius))
                center_b = _clamp(int(center_b + dz), margin_yz + int(radius), grid - margin_yz - int(radius))
            else:
                start = _clamp(start + dz, margin_yz, grid - margin_yz - length)
                center_a = _clamp(int(center_a + dx), margin_x + int(radius), grid - margin_x - int(radius))
                center_b = _clamp(int(center_b + dy), margin_yz + int(radius), grid - margin_yz - int(radius))
            updated.append(
                ObstacleSpec(
                    "cylinder",
                    {
                        "axis": axis,
                        "start": start,
                        "end": start + length,
                        "center_a": center_a,
                        "center_b": center_b,
                        "radius": radius,
                    },
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


def _random_inflow_center(cfg: GenerationConfig, axis_name: str, sign: int, mode: str) -> Tuple[int, int, int]:
    grid = cfg.grid_size
    if mode == "boundary":
        coord = 1 if sign > 0 else grid - 2
        if axis_name == "x":
            return coord, _random_int(1, grid - 2), _random_int(1, grid - 2)
        if axis_name == "y":
            return _random_int(1, grid - 2), coord, _random_int(1, grid - 2)
        return _random_int(1, grid - 2), _random_int(1, grid - 2), coord
    return _random_int(2, grid - 3), _random_int(2, grid - 3), _random_int(2, grid - 3)


def random_inflow_spec(cfg: GenerationConfig) -> InflowSpec:
    grid = cfg.grid_size
    axis_name, sign = random.choice(INFLOW_DIRECTIONS)
    if random.random() < 0.7:
        mode = "boundary"
    else:
        mode = "interior"
    center = _random_inflow_center(cfg, axis_name, sign, mode)
    shape = random.choices(["rect", "disk", "scatter"], weights=[0.45, 0.35, 0.2], k=1)[0]
    if shape == "rect":
        if mode == "boundary":
            size_a = _random_int(3, max(4, grid // 4))
            size_b = _random_int(3, max(4, grid // 4))
            size = {"x": 1, "y": size_a, "z": size_b}
            if axis_name == "y":
                size = {"x": size_a, "y": 1, "z": size_b}
            elif axis_name == "z":
                size = {"x": size_a, "y": size_b, "z": 1}
        else:
            size = {
                "x": _random_int(3, max(4, grid // 4)),
                "y": _random_int(3, max(4, grid // 4)),
                "z": _random_int(3, max(4, grid // 4)),
            }
        params = {"size": size}
    elif shape == "disk":
        radius = _random_int(2, max(3, grid // 6))
        params = {"radius": radius}
    else:
        count = _random_int(6, 20)
        radius = _random_int(2, max(3, grid // 6))
        points: List[Tuple[int, int, int]] = []
        for _ in range(count):
            dx = _random_int(-radius, radius)
            dy = _random_int(-radius, radius)
            dz = _random_int(-radius, radius)
            x = _clamp(center[0] + dx, 1, grid - 2)
            y = _clamp(center[1] + dy, 1, grid - 2)
            z = _clamp(center[2] + dz, 1, grid - 2)
            if mode == "boundary":
                if axis_name == "x":
                    x = center[0]
                elif axis_name == "y":
                    y = center[1]
                else:
                    z = center[2]
            points.append((x, y, z))
        params = {"points": points}
    return InflowSpec(center=center, axis=axis_name, sign=sign, mode=mode, shape=shape, params=params)


def build_inflow_mask_for_spec(cfg: GenerationConfig, spec: InflowSpec) -> np.ndarray:
    grid = cfg.grid_size
    inflow = np.zeros((grid, grid, grid), dtype=np.float32)
    cx, cy, cz = spec.center
    if spec.shape == "rect":
        size = spec.params.get("size", {"x": 1, "y": 1, "z": 1})
        sx = int(size.get("x", 1))
        sy = int(size.get("y", 1))
        sz = int(size.get("z", 1))
        x0 = _clamp(cx - sx // 2, 0, grid - 1)
        y0 = _clamp(cy - sy // 2, 0, grid - 1)
        z0 = _clamp(cz - sz // 2, 0, grid - 1)
        x1 = _clamp(x0 + sx, 0, grid)
        y1 = _clamp(y0 + sy, 0, grid)
        z1 = _clamp(z0 + sz, 0, grid)
        inflow[x0:x1, y0:y1, z0:z1] = 1.0
    elif spec.shape == "disk":
        radius = int(spec.params.get("radius", 2))
        if spec.mode == "boundary":
            ys = np.arange(grid).reshape(1, -1)
            zs = np.arange(grid).reshape(-1, 1)
            dist = (ys - cy) ** 2 + (zs - cz) ** 2
            disk_mask = dist <= radius ** 2
            if spec.axis == "x":
                if 0 <= cx < grid:
                    inflow[cx, :, :] = np.where(disk_mask, 1.0, inflow[cx, :, :])
            elif spec.axis == "y":
                if 0 <= cy < grid:
                    inflow[:, cy, :] = np.where(disk_mask, 1.0, inflow[:, cy, :])
            else:
                if 0 <= cz < grid:
                    inflow[:, :, cz] = np.where(disk_mask, 1.0, inflow[:, :, cz])
        else:
            xs = np.arange(grid).reshape(-1, 1, 1)
            ys = np.arange(grid).reshape(1, -1, 1)
            zs = np.arange(grid).reshape(1, 1, -1)
            dist = (xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2
            inflow[dist <= radius ** 2] = 1.0
    else:
        points = cast(List[Tuple[int, int, int]], spec.params.get("points", []))
        for x, y, z in points:
            if 0 <= x < grid and 0 <= y < grid and 0 <= z < grid:
                inflow[x, y, z] = 1.0
    return inflow


def build_inflow_masks(cfg: GenerationConfig, specs: List[InflowSpec]) -> List[np.ndarray]:
    return [build_inflow_mask_for_spec(cfg, spec) for spec in specs]


def build_fan_condition_fields(
    cfg: GenerationConfig,
    inflow_specs: List[InflowSpec],
    inflow_masks: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    fan_mask = np.zeros((cfg.grid_size, cfg.grid_size, cfg.grid_size), dtype=np.float32)
    fan_velocity = np.zeros((cfg.grid_size, cfg.grid_size, cfg.grid_size, 3), dtype=np.float32)
    axis_map = {"x": 0, "y": 1, "z": 2}
    for spec, mask in zip(inflow_specs, inflow_masks):
        mask_f = mask.astype(np.float32)
        fan_mask = np.maximum(fan_mask, mask_f)
        axis_idx = axis_map[spec.axis]
        fan_velocity[..., axis_idx] += mask_f * (cfg.inflow_speed * spec.sign)
    return fan_mask, fan_velocity


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


def mask_to_tensor(mask: np.ndarray) -> Any:
    return phi_math.tensor(mask, phi_math.spatial("x,y,z"))


class SDFGridWithVector(SDFGrid):
    @property
    def vector(self):
        return self.bounds.vector # type: ignore


def obstacle_from_mask(mask: Any, bounds: Any) -> Obstacle:
    tensor = mask if not isinstance(mask, np.ndarray) else mask_to_tensor(mask)
    sdf = phi_math.where(tensor > 0.5, -1.0, 1.0)
    return Obstacle(SDFGridWithVector(sdf, bounds=bounds))


def build_velocity_field(cfg: GenerationConfig, bounds: Any, axis_name: str, sign: int) -> Any:

    v_extrap = extrapolation.combine_sides(
        x=extrapolation.ZERO_GRADIENT,
        y=extrapolation.ZERO_GRADIENT,
        z=extrapolation.ZERO_GRADIENT
    )
    components = {"x": 0.0, "y": 0.0, "z": 0.0}
    components[axis_name] = -cfg.inflow_speed * sign

    return StaggeredGrid(
        #vec(x=components["x"], y=components["y"], z=components["z"]), 
        bounds=bounds, 
        extrapolation=v_extrap,
        x=cfg.grid_size, 
        y=cfg.grid_size, 
        z=cfg.grid_size
    )


def apply_inflow(
    velocity: Any,
    inflow_mask: Any,
    inflow_speed: float,
    t: int,
    axis_name: str,
    sign: int,
    dir_variation: float,
    dir_frequency: float,
    dir_phase: float,
    fan_strength: float,
) -> Any:
    speed = inflow_speed #* (0.7 + 0.2 * phi_math.sin(0.2 * t))
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map[axis_name]
    tangential_axes = [idx for idx in range(3) if idx != axis_idx]
    # phase = dir_frequency * t + dir_phase
    # tangential_amp = speed * dir_variation
    components = [0.0, 0.0, 0.0]
    # Always blow inward normal from selected face
    components[axis_idx] = speed * sign
    components[tangential_axes[0]] = 0#tangential_amp * phi_math.sin(phase)
    components[tangential_axes[1]] = 0#tangential_amp * phi_math.cos(phase)
    inflow = StaggeredGrid(
        vec(x=components[0], y=components[1], z=components[2]),
        bounds=velocity.bounds,
        extrapolation=velocity.extrapolation,
        x=velocity.resolution.spatial["x"].size,
        y=velocity.resolution.spatial["y"].size,
        z=velocity.resolution.spatial["z"].size,
    )
    inflow_mask_staggered = inflow_mask.at(inflow)
    # Fan-style forcing: relax velocity toward inflow inside the mask
    fan_strength = float(max(0.0, min(1.0, fan_strength)))
    drive = inflow - velocity
    return velocity + inflow_mask_staggered * fan_strength * drive


def apply_inflows(
    velocity: Any,
    inflow_specs: List[InflowSpec],
    inflow_masks: List[Any],
    inflow_phases: List[float],
    inflow_speed: float,
    t: int,
    dir_variation: float,
    dir_frequency: float,
    fan_strength: float,
) -> Any:
    for spec, mask, phase in zip(inflow_specs, inflow_masks, inflow_phases):
        velocity = apply_inflow(
            velocity,
            mask,
            inflow_speed,
            t,
            spec.axis,
            spec.sign,
            dir_variation,
            dir_frequency,
            phase,
            fan_strength,
        )
    return velocity

def step_flow(
    velocity: Any,
    obstacles: Obstacle,
    inflow_mask: Any,
    t: int,
    dt: float,
    inflow_speed: float,
    axis_name: str,
    sign: int,
    dir_variation: float,
    dir_frequency: float,
    dir_phase: float,
    fan_strength: float,
    tolerance: float,
    max_iter: int,
) -> Tuple[Any, Any]:
    # 1. Advect
    velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
    # 2. Diffuse
    velocity = diffuse.explicit(velocity, diffusivity=1e-5, dt=dt)
    # 3. Apply Inflow Boundary Condition (Dirichlet on velocity)
    velocity = apply_inflow(
        velocity,
        inflow_mask,
        inflow_speed,
        t,
        axis_name,
        sign,
        dir_variation,
        dir_frequency,
        dir_phase,
        fan_strength,
    )
    
    # 4. Projection (Make Incompressible)
    solve = Solve(
        method="CG",
        abs_tol=tolerance,
        rel_tol=tolerance,
        max_iterations=max_iter,
    )
    
    velocity, pressure = fluid.make_incompressible(
        velocity,
        obstacles=obstacles,
        solve=solve,
    )
    
    return velocity, pressure


def step_flow_multi(
    velocity: Any,
    obstacles: Obstacle,
    inflow_specs: List[InflowSpec],
    inflow_masks: List[Any],
    inflow_phases: List[float],
    t: int,
    dt: float,
    inflow_speed: float,
    dir_variation: float,
    dir_frequency: float,
    fan_strength: float,
    tolerance: float,
    max_iter: int,
) -> Tuple[Any, Any]:
    velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
    velocity = diffuse.explicit(velocity, diffusivity=1e-5, dt=dt)
    velocity = apply_inflows(
        velocity,
        inflow_specs,
        inflow_masks,
        inflow_phases,
        inflow_speed,
        t,
        dir_variation,
        dir_frequency,
        fan_strength,
    )
    solve = Solve(
        method="CG",
        abs_tol=tolerance,
        rel_tol=tolerance,
        max_iterations=max_iter,
    )
    velocity, pressure = fluid.make_incompressible(
        velocity,
        obstacles=obstacles,
        solve=solve,
    )
    return velocity, pressure


def build_stepper(
    obstacles: Obstacle,
    inflow_mask: Any,
    axis_name: str,
    sign: int,
    dir_variation: float,
    dir_frequency: float,
    dir_phase: float,
    fan_strength: float,
    tolerance: float,
    max_iter: int,
) -> Any:
    @phi_math.jit_compile
    def _step(
        velocity: Any,
        t: int,
        dt: float,
        inflow_speed: float,
    ) -> Tuple[Any, Any]:
        return step_flow(
            velocity,
            obstacles,
            inflow_mask,
            t,
            dt,
            inflow_speed,
            axis_name,
            sign,
            dir_variation,
            dir_frequency,
            dir_phase,
            fan_strength,
            tolerance,
            max_iter,
        )

    return _step


def to_numpy_pressure_centered(field: Any) -> np.ndarray:
    return field.values.numpy("x,y,z")


def to_tensor_pressure_centered(field: Any) -> Any:
    return field.values


def to_numpy_velocity_centered(field: Any) -> np.ndarray:
    sampled = field.at_centers()
    stacked = phi_math.stack(
        [sampled.values.vector["x"], sampled.values.vector["y"], sampled.values.vector["z"]],
        phi_math.channel("vector"),
    )
    return stacked.numpy("x,y,z,vector")


def to_tensor_velocity_centered(field: Any) -> Any:
    sampled = field.at_centers()
    return phi_math.stack(
        [sampled.values.vector["x"], sampled.values.vector["y"], sampled.values.vector["z"]],
        phi_math.channel("vector"),
    )


def save_sample(out_dir: Path, sample_idx: int, payload: Dict[str, np.ndarray]) -> None:
    out_path = out_dir / f"sample_{sample_idx:05d}.npy"
    np.save(out_path, np.array(payload, dtype=object), allow_pickle=True)


def save_metadata(out_dir: Path, cfg: GenerationConfig) -> None:
    meta = {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')}
    with open(out_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def generate_dataset(cfg: GenerationConfig, out_dir: Path, resume: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bounds = Box(x=cfg.grid_size, y=cfg.grid_size, z=cfg.grid_size) # type: ignore
    save_metadata(out_dir, cfg)
    sample_idx = 0
    attempts = 0
    if resume:
        existing = sorted(out_dir.glob("sample_*.npy"))
        if existing:
            last_name = existing[-1].stem
            try:
                sample_idx = int(last_name.split("_")[-1]) + 1
            except ValueError:
                sample_idx = len(existing)
            attempts = sample_idx
            print(f"Resuming from sample index {sample_idx}")
    

    karman_ratio = min(max(cfg.karman_ratio, 0.0), 1.0)
    print(
        "Starting generation with randomized inflows (70% boundary, 30% interior). "
        f"Karman ratio={karman_ratio:.2f}, mode={cfg.karman_mode}"
    )

    while sample_idx < cfg.num_samples:
        attempts += 1
        use_karman = cfg.karman_mode or (karman_ratio > 0 and random.random() < karman_ratio)
        if use_karman:
            obstacle_specs = _karman_obstacle_specs(cfg)
            obstacle_mask = _rasterize_obstacles(cfg, obstacle_specs)
            center = cfg.grid_size // 2
            inflow_spec = InflowSpec(
                center=(1, center, center),
                axis="x",
                sign=1,
                mode="boundary",
                shape="disk",
                params={"radius": max(2, cfg.grid_size // 10)},
            )
            inflow_specs = [inflow_spec]
            inflow_phases = [0.0]
            dir_variation = 0.0
            dir_frequency = 0.0
        else:
            obstacle_mask, obstacle_specs = generate_obstacle_mask(cfg)
            inflow_specs = [random_inflow_spec(cfg) for _ in range(_random_int(cfg.min_inflows, cfg.max_inflows))]
            inflow_phases = [_random_float(0.0, 2.0 * math.pi) for _ in inflow_specs]
            dir_variation = cfg.inflow_direction_variation
            dir_frequency = cfg.inflow_direction_frequency
        if use_karman:
            dir_phase = inflow_phases[0]
        inflow_masks_np = build_inflow_masks(cfg, inflow_specs)
        fan_mask_np, fan_velocity_np = build_fan_condition_fields(cfg, inflow_specs, inflow_masks_np)
        inflow_masks = [field_from_mask(mask, bounds) for mask in inflow_masks_np]
        primary_inflow = inflow_specs[0]
        velocity = build_velocity_field(cfg, bounds, primary_inflow.axis, primary_inflow.sign)
        t = 0
        obstacle_mask_t = obstacle_mask
        obstacle_mask_tensor = mask_to_tensor(obstacle_mask.astype(np.float32))
        obstacles = obstacle_from_mask(obstacle_mask_tensor, bounds)
        stepper = None
        if use_karman:
            stepper = build_stepper(
                obstacles,
                inflow_masks[0],
                primary_inflow.axis,
                primary_inflow.sign,
                dir_variation,
                dir_frequency,
                inflow_phases[0],
                cfg.inflow_fan_strength,
                cfg.pressure_tolerance,
                cfg.pressure_iterations,
            )
        # Warmup
        try:
            for _ in range(cfg.warmup_steps):
                if not use_karman:
                    obstacle_specs = _update_obstacles(cfg, obstacle_specs)
                    obstacle_mask = _rasterize_obstacles(cfg, obstacle_specs)
                    obstacle_mask_tensor = mask_to_tensor(obstacle_mask.astype(np.float32))
                    obstacles = obstacle_from_mask(obstacle_mask_tensor, bounds)
                if stepper is None:
                    if use_karman:
                        velocity, pressure = step_flow(
                            velocity, obstacles, inflow_masks[0],
                            t,
                            cfg.dt,
                            cfg.inflow_speed,
                            primary_inflow.axis,
                            primary_inflow.sign,
                            dir_variation,
                            dir_frequency,
                            inflow_phases[0],
                            cfg.inflow_fan_strength,
                            cfg.pressure_tolerance,
                            cfg.pressure_iterations,
                        )
                    else:
                        velocity, pressure = step_flow_multi(
                            velocity,
                            obstacles,
                            inflow_specs,
                            inflow_masks,
                            inflow_phases,
                            t,
                            cfg.dt,
                            cfg.inflow_speed,
                            dir_variation,
                            dir_frequency,
                            cfg.inflow_fan_strength,
                            cfg.pressure_tolerance,
                            cfg.pressure_iterations,
                        )
                else:
                    velocity, pressure = stepper(velocity, t, cfg.dt, cfg.inflow_speed)
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
                if not use_karman:
                    obstacle_specs = _update_obstacles(cfg, obstacle_specs)
                    obstacle_mask = _rasterize_obstacles(cfg, obstacle_specs)
                    obstacle_mask_tensor = mask_to_tensor(obstacle_mask.astype(np.float32))
                    obstacles = obstacle_from_mask(obstacle_mask_tensor, bounds)
                if stepper is None:
                    if use_karman:
                        velocity, pressure = step_flow(
                            velocity, obstacles, inflow_masks[0],
                            t,
                            cfg.dt,
                            cfg.inflow_speed,
                            primary_inflow.axis,
                            primary_inflow.sign,
                            dir_variation,
                            dir_frequency,
                            inflow_phases[0],
                            cfg.inflow_fan_strength,
                            cfg.pressure_tolerance,
                            cfg.pressure_iterations,
                        )
                    else:
                        velocity, pressure = step_flow_multi(
                            velocity,
                            obstacles,
                            inflow_specs,
                            inflow_masks,
                            inflow_phases,
                            t,
                            cfg.dt,
                            cfg.inflow_speed,
                            dir_variation,
                            dir_frequency,
                            cfg.inflow_fan_strength,
                            cfg.pressure_tolerance,
                            cfg.pressure_iterations,
                        )
                else:
                    velocity, pressure = stepper(velocity, t, cfg.dt, cfg.inflow_speed)
                velocity_n.append(to_tensor_velocity_centered(velocity))
                pressure_n.append(to_tensor_pressure_centered(pressure))
                obstacle_mask_n.append(obstacle_mask_tensor)
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
        velocity_stack = phi_math.stack(velocity_n, phi_math.batch("t"))
        pressure_stack = phi_math.stack(pressure_n, phi_math.batch("t"))
        obstacle_stack = phi_math.stack(obstacle_mask_n, phi_math.batch("t"))
        inflow_centers = np.array([spec.center for spec in inflow_specs], dtype=np.int32)
        payload = {
            "obstacle_mask": obstacle_mask_t.astype(np.bool_),
            "obstacle_mask_tn": (obstacle_stack.numpy("t,x,y,z") > 0.5),
            "fan_mask": fan_mask_np.astype(np.float32),
            "fan_velocity": fan_velocity_np.astype(np.float32),
            "velocity_t": to_numpy_velocity_centered(velocity_t).astype(np.float32),
            "velocity_tn": velocity_stack.numpy("t,x,y,z,vector").astype(np.float32),
            "pressure_t": to_numpy_pressure_centered(pressure_t).astype(np.float32),
            "pressure_tn": pressure_stack.numpy("t,x,y,z").astype(np.float32),
            "inflow_point": inflow_centers[0],
            "inflow_center": inflow_centers[0],
            "inflow_axis": inflow_specs[0].axis,
            "inflow_sign": inflow_specs[0].sign,
            "inflow_mode": inflow_specs[0].mode,
            "inflow_shape": inflow_specs[0].shape,
            "inflow_params": inflow_specs[0].params,
            "inflow_centers": inflow_centers,
            "inflow_axes": [spec.axis for spec in inflow_specs],
            "inflow_signs": [spec.sign for spec in inflow_specs],
            "inflow_modes": [spec.mode for spec in inflow_specs],
            "inflow_shapes": [spec.shape for spec in inflow_specs],
            "inflow_params_list": [spec.params for spec in inflow_specs],
            "inflow_phases": inflow_phases,
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
    generate_dataset(cfg, out_dir, resume=cfg.resume)
    print(f"Done. Saved to {out_dir}")


if __name__ == "__main__":
    main()
