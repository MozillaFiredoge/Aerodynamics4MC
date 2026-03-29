#!/usr/bin/env python3
"""Paper-style numerical benchmark harness for the Aerodynamics4MC Python solver.

This script evaluates the PhiFlow-based solver on a small set of canonical,
deterministic scenarios and reports quantitative diagnostics such as:
- incompressibility residual (RMS divergence)
- obstacle impermeability leakage
- symmetry preservation under symmetric forcing
- long-rollout stability / finite-value checks
- kinetic-energy decay after fan shutdown

It is not a formal verification suite with closed-form CFD references, but it
is much closer to a paper-style numerical benchmark than an in-game smoke test.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import server as server_module


CASE_QUIESCENT = "quiescent_stability"
CASE_SYMMETRY = "dual_jet_symmetry"
CASE_OBSTACLE = "obstacle_impermeability"
CASE_DECAY = "fan_shutdown_decay"
ALL_CASES = (CASE_QUIESCENT, CASE_SYMMETRY, CASE_OBSTACLE, CASE_DECAY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the PhiFlow solver with paper-style numerical diagnostics.")
    parser.add_argument(
        "--cases",
        type=str,
        default="all",
        help=f"Comma-separated subset of cases or 'all'. Choices: {', '.join(ALL_CASES)}",
    )
    parser.add_argument(
        "--grid-sizes",
        type=str,
        default="32",
        help="Comma-separated grid sizes, e.g. '24,32,48'",
    )
    parser.add_argument("--steps", type=int, default=48, help="Total rollout steps per case")
    parser.add_argument(
        "--decay-fan-off-step",
        type=int,
        default=20,
        help="For fan_shutdown_decay: step index at which forcing is disabled",
    )
    parser.add_argument("--dt", type=float, default=1.0, help="Semi-Lagrangian timestep")
    parser.add_argument("--diffusivity", type=float, default=1e-5, help="Explicit diffusion coefficient")
    parser.add_argument(
        "--sgs-model",
        choices=["none", "smagorinsky"],
        default="smagorinsky",
        help="Subgrid closure model",
    )
    parser.add_argument("--sgs-cs", type=float, default=0.12, help="Smagorinsky constant")
    parser.add_argument("--sgs-filter-width", type=float, default=1.0, help="LES filter width")
    parser.add_argument("--sgs-nu-max", type=float, default=0.05, help="Maximum SGS viscosity")
    parser.add_argument("--fan-strength", type=float, default=0.8, help="Fan relaxation strength")
    parser.add_argument("--pressure-iterations", type=int, default=24, help="Max CG iterations for projection")
    parser.add_argument("--pressure-tolerance", type=float, default=3e-3, help="CG abs/rel tolerance")
    parser.add_argument(
        "--projection-interval",
        type=int,
        default=4,
        help="Pressure projection interval in steps (0 disables projection)",
    )
    parser.add_argument("--torch-backend", action="store_true", help="Use PhiFlow torch backend")
    parser.add_argument("--torch-device", choices=["cpu", "cuda"], default="cpu", help="Torch backend device")
    parser.add_argument(
        "--json-out",
        type=str,
        default="outputs/solver_benchmark.json",
        help="Path to save JSON report",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="",
        help="Optional directory for metric curves (.png per case/grid)",
    )
    return parser.parse_args()


def load_server_module() -> Any:
    import server as server_module

    return server_module


def parse_case_names(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(ALL_CASES)
    requested = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [name for name in requested if name not in ALL_CASES]
    if unknown:
        raise ValueError(f"Unknown case(s): {', '.join(unknown)}")
    if not requested:
        raise ValueError("No benchmark cases selected")
    return requested


def parse_grid_sizes(spec: str) -> list[int]:
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        grid = int(part)
        if grid < 8:
            raise ValueError("Grid sizes must be >= 8")
        values.append(grid)
    if not values:
        raise ValueError("No grid sizes selected")
    return values


def make_cfg(args: argparse.Namespace, grid_size: int, server_module: Any) -> Any:
    return server_module.ServerConfig(
        grid_size=grid_size,
        dt=args.dt,
        diffusivity=max(0.0, args.diffusivity),
        sgs_model=args.sgs_model,
        sgs_cs=max(0.0, args.sgs_cs),
        sgs_filter_width=max(1e-6, args.sgs_filter_width),
        sgs_nu_max=max(0.0, args.sgs_nu_max),
        fan_strength=max(0.0, min(1.0, args.fan_strength)),
        pressure_iterations=max(0, args.pressure_iterations),
        pressure_tolerance=max(1e-8, args.pressure_tolerance),
        projection_interval=max(0, args.projection_interval),
        max_projection_interval=max(0, args.projection_interval),
        target_hz=0.1,
        log_interval=0,
        use_torch_backend=args.torch_backend,
        torch_device=args.torch_device,
    )


def make_runtime(cfg: Any, server_module: Any) -> Any:
    return server_module.ClientRuntime(
        cfg=cfg,
        bounds=server_module.Box(x=cfg.grid_size, y=cfg.grid_size, z=cfg.grid_size),
        vel_extrapolation=server_module.extrapolation.combine_sides(
            x=server_module.extrapolation.ZERO_GRADIENT,
            y=server_module.extrapolation.ZERO_GRADIENT,
            z=server_module.extrapolation.ZERO_GRADIENT,
        ),
    )


def empty_packet(n: int) -> np.ndarray:
    return np.zeros((n, n, n, 9), dtype=np.float32)


def add_box_obstacle(packet: np.ndarray, lo: tuple[int, int, int], hi: tuple[int, int, int]) -> None:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    packet[x0:x1, y0:y1, z0:z1, 0] = 1.0


def add_fan_disk(
    packet: np.ndarray,
    center: tuple[int, int, int],
    axis: str,
    sign: int,
    radius: int,
    speed: float,
    thickness: int = 1,
) -> None:
    n = packet.shape[0]
    cx, cy, cz = center
    thickness = max(1, thickness)
    radius = max(1, radius)
    speed = float(speed) * (1.0 if sign >= 0 else -1.0)
    grid_x, grid_y, grid_z = np.indices((n, n, n))

    if axis == "x":
        mask = (np.abs(grid_x - cx) < thickness) & (((grid_y - cy) ** 2 + (grid_z - cz) ** 2) <= radius * radius)
        packet[mask, 1] = 1.0
        packet[mask, 2] = speed
    elif axis == "y":
        mask = (np.abs(grid_y - cy) < thickness) & (((grid_x - cx) ** 2 + (grid_z - cz) ** 2) <= radius * radius)
        packet[mask, 1] = 1.0
        packet[mask, 3] = speed
    elif axis == "z":
        mask = (np.abs(grid_z - cz) < thickness) & (((grid_x - cx) ** 2 + (grid_y - cy) ** 2) <= radius * radius)
        packet[mask, 1] = 1.0
        packet[mask, 4] = speed
    else:
        raise ValueError(f"Unsupported axis: {axis}")


def build_packet_for_case(case_name: str, n: int, step: int, args: argparse.Namespace) -> np.ndarray:
    packet = empty_packet(n)
    radius = max(2, n // 10)
    thickness = max(1, n // 32)

    if case_name == CASE_QUIESCENT:
        return packet

    if case_name == CASE_SYMMETRY:
        z_a = n // 3
        z_b = n - 1 - z_a
        center_a = (n // 6, n // 2, z_a)
        center_b = (n // 6, n // 2, z_b)
        add_fan_disk(packet, center_a, axis="x", sign=1, radius=radius, speed=1.0, thickness=thickness)
        add_fan_disk(packet, center_b, axis="x", sign=1, radius=radius, speed=1.0, thickness=thickness)
        return packet

    if case_name == CASE_OBSTACLE:
        add_fan_disk(packet, (n // 6, n // 2, n // 2), axis="x", sign=1, radius=radius, speed=1.0, thickness=thickness)
        half_x = max(2, n // 12)
        half_yz = max(3, n // 8)
        add_box_obstacle(
            packet,
            (n // 2 - half_x, n // 2 - half_yz, n // 2 - half_yz),
            (n // 2 + half_x, n // 2 + half_yz, n // 2 + half_yz),
        )
        return packet

    if case_name == CASE_DECAY:
        if step < args.decay_fan_off_step:
            add_fan_disk(packet, (n // 6, n // 2, n // 2), axis="x", sign=1, radius=radius, speed=1.0, thickness=thickness)
        return packet

    raise ValueError(f"Unsupported case: {case_name}")


def kinetic_energy(velocity: np.ndarray, obstacle_mask: np.ndarray) -> float:
    fluid = obstacle_mask <= 0.5
    if not np.any(fluid):
        return 0.0
    speed_sq = np.sum(velocity * velocity, axis=-1)
    return 0.5 * float(np.mean(speed_sq[fluid]))


def divergence_rms(velocity: np.ndarray, obstacle_mask: np.ndarray) -> float:
    fluid = obstacle_mask <= 0.5
    if velocity.shape[0] > 2:
        interior = np.zeros_like(fluid, dtype=bool)
        interior[1:-1, 1:-1, 1:-1] = True
        fluid &= interior
    if not np.any(fluid):
        return 0.0
    div = (
        np.gradient(velocity[..., 0], axis=0, edge_order=1)
        + np.gradient(velocity[..., 1], axis=1, edge_order=1)
        + np.gradient(velocity[..., 2], axis=2, edge_order=1)
    )
    return float(np.sqrt(np.mean(np.square(div[fluid]))))


def obstacle_flux_stats(velocity: np.ndarray, obstacle_mask: np.ndarray) -> tuple[float, float, float]:
    speed = np.linalg.norm(velocity, axis=-1)
    solid = obstacle_mask > 0.5
    inside_obstacle_speed = float(np.max(speed[solid])) if np.any(solid) else 0.0

    interface_values: list[np.ndarray] = []
    for axis in range(3):
        slicer_lo = [slice(None), slice(None), slice(None)]
        slicer_hi = [slice(None), slice(None), slice(None)]
        slicer_lo[axis] = slice(None, -1)
        slicer_hi[axis] = slice(1, None)
        solid_lo = solid[tuple(slicer_lo)]
        solid_hi = solid[tuple(slicer_hi)]
        fluid_lo = ~solid_lo
        fluid_hi = ~solid_hi

        vel_lo = np.abs(velocity[tuple(slicer_lo)][..., axis])
        vel_hi = np.abs(velocity[tuple(slicer_hi)][..., axis])

        mask_lo = fluid_lo & solid_hi
        mask_hi = solid_lo & fluid_hi
        if np.any(mask_lo):
            interface_values.append(vel_lo[mask_lo])
        if np.any(mask_hi):
            interface_values.append(vel_hi[mask_hi])

    if not interface_values:
        return inside_obstacle_speed, 0.0, 0.0

    stacked = np.concatenate(interface_values, axis=0)
    return inside_obstacle_speed, float(np.mean(stacked)), float(np.max(stacked))


def mirror_symmetry_error(velocity: np.ndarray, axis: int) -> float:
    mirrored = np.flip(velocity, axis=axis).copy()
    mirrored[..., axis] *= -1.0
    denom = float(np.sqrt(np.mean(np.square(velocity))) + 1e-8)
    diff = velocity - mirrored
    return float(np.sqrt(np.mean(np.square(diff))) / denom)


def summarize_history(case_name: str, grid_size: int, history: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    steps = [int(item["step"]) for item in history]
    step_times_ms = [float(item["step_time_ms"]) for item in history]
    energies = [float(item["kinetic_energy"]) for item in history]
    divergences = [float(item["divergence_rms"]) for item in history]
    max_speeds = [float(item["max_speed"]) for item in history]
    pressures = [float(item["max_abs_pressure"]) for item in history]
    finite_flags = [bool(item["all_finite"]) for item in history]

    summary: dict[str, Any] = {
        "case": case_name,
        "grid_size": grid_size,
        "steps": len(history),
        "mean_step_time_ms": float(np.mean(step_times_ms)) if step_times_ms else 0.0,
        "max_step_time_ms": float(np.max(step_times_ms)) if step_times_ms else 0.0,
        "mean_divergence_rms": float(np.mean(divergences)) if divergences else 0.0,
        "final_divergence_rms": divergences[-1] if divergences else 0.0,
        "mean_kinetic_energy": float(np.mean(energies)) if energies else 0.0,
        "final_kinetic_energy": energies[-1] if energies else 0.0,
        "max_speed": float(np.max(max_speeds)) if max_speeds else 0.0,
        "max_abs_pressure": float(np.max(pressures)) if pressures else 0.0,
        "all_steps_finite": all(finite_flags),
        "history": history,
    }

    if case_name == CASE_QUIESCENT:
        summary["max_residual_speed"] = float(np.max(max_speeds)) if max_speeds else 0.0

    if case_name == CASE_SYMMETRY:
        symmetry_errors = [float(item["symmetry_error"]) for item in history]
        summary["mean_symmetry_error"] = float(np.mean(symmetry_errors)) if symmetry_errors else 0.0
        summary["final_symmetry_error"] = symmetry_errors[-1] if symmetry_errors else 0.0

    if case_name == CASE_OBSTACLE:
        inside_speeds = [float(item["inside_obstacle_speed"]) for item in history]
        mean_fluxes = [float(item["mean_interface_normal_flux"]) for item in history]
        max_fluxes = [float(item["max_interface_normal_flux"]) for item in history]
        summary["max_inside_obstacle_speed"] = float(np.max(inside_speeds)) if inside_speeds else 0.0
        summary["mean_interface_normal_flux"] = float(np.mean(mean_fluxes)) if mean_fluxes else 0.0
        summary["max_interface_normal_flux"] = float(np.max(max_fluxes)) if max_fluxes else 0.0

    if case_name == CASE_DECAY:
        if 0 <= args.decay_fan_off_step < len(energies):
            start_energy = max(energies[min(args.decay_fan_off_step, len(energies) - 1)], 1e-12)
            end_energy = energies[-1] if energies else 0.0
            summary["energy_decay_ratio"] = float(end_energy / start_energy)
            summary["energy_at_fan_off"] = float(start_energy)
        else:
            summary["energy_decay_ratio"] = 1.0
            summary["energy_at_fan_off"] = energies[-1] if energies else 0.0

    return summary


def print_run_summary(summary: dict[str, Any]) -> None:
    line = (
        f"[{summary['case']:<23}] grid={summary['grid_size']:>3} "
        f"steps={summary['steps']:>3} "
        f"dt_mean={summary['mean_step_time_ms']:.2f}ms "
        f"div_final={summary['final_divergence_rms']:.4e} "
        f"ke_final={summary['final_kinetic_energy']:.4e} "
        f"vmax={summary['max_speed']:.4e} "
        f"finite={summary['all_steps_finite']}"
    )
    if "final_symmetry_error" in summary:
        line += f" sym={summary['final_symmetry_error']:.4e}"
    if "max_inside_obstacle_speed" in summary:
        line += f" leak={summary['max_inside_obstacle_speed']:.4e}"
    if "energy_decay_ratio" in summary:
        line += f" decay={summary['energy_decay_ratio']:.4e}"
    print(line)


def maybe_plot_history(summary: dict[str, Any], plot_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting
        print(f"Plot skipped for {summary['case']} grid={summary['grid_size']}: {exc}")
        return

    history = summary["history"]
    steps = [int(item["step"]) for item in history]
    divergence = [float(item["divergence_rms"]) for item in history]
    energy = [float(item["kinetic_energy"]) for item in history]
    max_speed = [float(item["max_speed"]) for item in history]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.suptitle(f"{summary['case']} | grid={summary['grid_size']}")

    axes[0].plot(steps, energy, color="tab:blue")
    axes[0].set_ylabel("Kinetic Energy")

    axes[1].plot(steps, divergence, color="tab:orange")
    axes[1].set_ylabel("Divergence RMS")
    axes[1].set_yscale("log")

    axes[2].plot(steps, max_speed, color="tab:green")
    axes[2].set_ylabel("Max Speed")
    axes[2].set_xlabel("Step")

    extra_key = None
    if summary["case"] == CASE_SYMMETRY:
        extra_key = "symmetry_error"
    elif summary["case"] == CASE_OBSTACLE:
        extra_key = "mean_interface_normal_flux"
    elif summary["case"] == CASE_DECAY:
        extra_key = "kinetic_energy"

    if extra_key is not None and extra_key != "kinetic_energy":
        extra_values = [float(item[extra_key]) for item in history]
        twin = axes[2].twinx()
        twin.plot(steps, extra_values, color="tab:red", alpha=0.65)
        twin.set_ylabel(extra_key.replace("_", " "))

    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"{summary['case']}_grid{summary['grid_size']}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_case(case_name: str, cfg: Any, args: argparse.Namespace, server_module: Any) -> dict[str, Any]:
    runtime = make_runtime(cfg, server_module)
    state_velocity = np.zeros((cfg.grid_size, cfg.grid_size, cfg.grid_size, 3), dtype=np.float32)
    state_pressure = np.zeros((cfg.grid_size, cfg.grid_size, cfg.grid_size), dtype=np.float32)
    history: list[dict[str, Any]] = []

    for step in range(args.steps):
        packet = build_packet_for_case(case_name, cfg.grid_size, step, args)
        packet[..., 5:8] = state_velocity
        packet[..., 8] = state_pressure

        start = time.perf_counter()
        output = server_module.solve_step(packet, runtime)
        step_time_ms = (time.perf_counter() - start) * 1000.0

        runtime.step_index += 1
        state_velocity = output[..., :3].astype(np.float32, copy=False)
        state_pressure = output[..., 3].astype(np.float32, copy=False)

        obstacle_mask = packet[..., 0]
        finite_mask = np.isfinite(output)
        velocity = output[..., :3]
        pressure = output[..., 3]
        speed = np.linalg.norm(velocity, axis=-1)

        metrics: dict[str, Any] = {
            "step": step,
            "step_time_ms": float(step_time_ms),
            "all_finite": bool(finite_mask.all()),
            "kinetic_energy": kinetic_energy(velocity, obstacle_mask),
            "divergence_rms": divergence_rms(velocity, obstacle_mask),
            "max_speed": float(np.max(speed)),
            "max_abs_pressure": float(np.max(np.abs(pressure))),
        }

        if case_name == CASE_SYMMETRY:
            metrics["symmetry_error"] = mirror_symmetry_error(velocity, axis=2)
        elif case_name == CASE_OBSTACLE:
            inside_speed, mean_flux, max_flux = obstacle_flux_stats(velocity, obstacle_mask)
            metrics["inside_obstacle_speed"] = inside_speed
            metrics["mean_interface_normal_flux"] = mean_flux
            metrics["max_interface_normal_flux"] = max_flux

        history.append(metrics)
        if not metrics["all_finite"]:
            break

    return summarize_history(case_name, cfg.grid_size, history, args)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def main() -> None:
    args = parse_args()
    cases = parse_case_names(args.cases)
    grid_sizes = parse_grid_sizes(args.grid_sizes)

    server_module = load_server_module()
    base_cfg = make_cfg(args, grid_sizes[0], server_module)
    server_module.setup_backend(base_cfg)

    summaries = []
    for grid_size in grid_sizes:
        cfg = replace(base_cfg, grid_size=grid_size)
        for case_name in cases:
            summary = run_case(case_name, cfg, args, server_module)
            summaries.append(summary)
            print_run_summary(summary)

    report = {
        "cases": cases,
        "grid_sizes": grid_sizes,
        "config": {
            "steps": args.steps,
            "decay_fan_off_step": args.decay_fan_off_step,
            "dt": args.dt,
            "diffusivity": args.diffusivity,
            "sgs_model": args.sgs_model,
            "sgs_cs": args.sgs_cs,
            "sgs_filter_width": args.sgs_filter_width,
            "sgs_nu_max": args.sgs_nu_max,
            "fan_strength": args.fan_strength,
            "pressure_iterations": args.pressure_iterations,
            "pressure_tolerance": args.pressure_tolerance,
            "projection_interval": args.projection_interval,
            "torch_backend": args.torch_backend,
            "torch_device": args.torch_device,
        },
        "results": summaries,
    }

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=json_default), encoding="utf-8")
        print(f"Saved JSON report to {path}")

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        for summary in summaries:
            maybe_plot_history(summary, plot_dir)
        print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
