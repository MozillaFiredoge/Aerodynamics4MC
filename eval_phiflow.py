#!/usr/bin/env python3
"""Evaluate a trained surrogate model using PhiFlow utilities and visualize autoregressive rollout.

Features:
- Loads a checkpointed UNet3D model.
- Runs an autoregressive rollout on 32^3 grid data.
- Computes per-step MSE vs. ground-truth sequence (if present).
- Uses PhiFlow to build grid fields and Matplotlib to render slices/animations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from matplotlib.axes import Axes

from model import SteadyRolloutCFDUNet, SteadyRolloutCFDUNetCfg


def require_phiflow():
    try:
        from phi.flow import Box, CenteredGrid, extrapolation, math as phi_math
    except Exception as exc:  # pragma: no cover - informational error
        raise RuntimeError(
            "PhiFlow is required. Install dependencies with `pip install phiflow torch`."
        ) from exc
    return Box, CenteredGrid, extrapolation, phi_math


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize surrogate rollout with PhiFlow.")
    parser.add_argument("--checkpoint", type=str, default="runs/checkpoint_020.pt", help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/grid_32", help="Directory with sample_*.npy files")
    parser.add_argument("--sample", type=str, default=None, help="Optional explicit sample path")
    parser.add_argument("--sample-index", type=int, default=0, help="Index of sample_XXXXX.npy if --sample not given")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    parser.add_argument("--steps", type=int, default=10, help="Number of rollout steps to visualize")
    parser.add_argument("--slice-axis", type=str, default="y", choices=["x", "y", "z"], help="Axis for 2D slice")
    parser.add_argument("--slice-index", type=int, default=None, help="Slice index (defaults to mid-plane)")
    parser.add_argument("--save", type=str, default=None, help="Output path (.gif or .mp4). If omitted, shows interactively.")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second for animation")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights if present in checkpoint")
    return parser.parse_args()


def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


@dataclass
class InflowSpec:
    center: Tuple[int, int, int]
    axis: str
    sign: int
    mode: str
    shape: str
    params: Dict[str, Any]


def load_metadata(data_dir: Path) -> Dict[str, Any]:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def build_inflow_mask_for_spec(grid: int, spec: InflowSpec) -> np.ndarray:
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
    elif spec.shape == "scatter":
        points = cast(List[Tuple[int, int, int]], spec.params.get("points", []))
        for x, y, z in points:
            if 0 <= x < grid and 0 <= y < grid and 0 <= z < grid:
                inflow[x, y, z] = 1.0
    else:
        if 0 <= cx < grid and 0 <= cy < grid and 0 <= cz < grid:
            inflow[cx, cy, cz] = 1.0
    return inflow


def parse_inflow_specs(sample: Dict[str, np.ndarray]) -> Tuple[List[InflowSpec], List[float]]:
    if "inflow_centers" in sample:
        centers = sample["inflow_centers"].tolist()
        axes = list(sample.get("inflow_axes", []))
        signs = list(sample.get("inflow_signs", []))
        modes = list(sample.get("inflow_modes", []))
        shapes = list(sample.get("inflow_shapes", []))
        params_list = list(sample.get("inflow_params_list", []))
        phases = list(sample.get("inflow_phases", []))
        specs: List[InflowSpec] = []
        for idx, center in enumerate(centers):
            axis = axes[idx] if idx < len(axes) else "x"
            sign = int(signs[idx]) if idx < len(signs) else 1
            mode = modes[idx] if idx < len(modes) else "boundary"
            shape = shapes[idx] if idx < len(shapes) else "point"
            params = params_list[idx] if idx < len(params_list) else {}
            specs.append(
                InflowSpec(
                    center=tuple(int(v) for v in center),
                    axis=axis,
                    sign=sign,
                    mode=mode,
                    shape=shape,
                    params=params,
                )
            )
        if not phases:
            phases = [0.0 for _ in specs]
        return specs, [float(p) for p in phases]

    center = sample.get("inflow_center") or sample.get("inflow_point")
    if center is None:
        return [], []
    spec = InflowSpec(
        center=tuple(int(v) for v in center),
        axis=str(sample.get("inflow_axis", "x")),
        sign=int(sample.get("inflow_sign", 1)),
        mode=str(sample.get("inflow_mode", "boundary")),
        shape=str(sample.get("inflow_shape", "point")),
        params=cast(Dict[str, Any], sample.get("inflow_params", {})),
    )
    return [spec], [0.0]


def apply_inflows(
    velocity: torch.Tensor,
    inflow_specs: List[InflowSpec],
    inflow_masks: List[torch.Tensor],
    inflow_phases: List[float],
    inflow_speed: float,
    t: int,
    dir_variation: float,
    dir_frequency: float,
    fan_strength: float,
) -> torch.Tensor:
    if not inflow_specs:
        return velocity
    speed = inflow_speed * (0.5 + 0.5 * math.sin(0.2 * t))
    axis_map = {"x": 0, "y": 1, "z": 2}
    for spec, mask, phase in zip(inflow_specs, inflow_masks, inflow_phases):
        axis_idx = axis_map.get(spec.axis, 0)
        tangential_axes = [idx for idx in range(3) if idx != axis_idx]
        tangential_amp = speed * dir_variation
        components = [0.0, 0.0, 0.0]
        components[axis_idx] = -speed * spec.sign
        theta = dir_frequency * t + phase
        components[tangential_axes[0]] = tangential_amp * math.sin(theta)
        components[tangential_axes[1]] = tangential_amp * math.cos(theta)
        inflow = torch.tensor(
            components,
            dtype=velocity.dtype,
            device=velocity.device,
        ).view(1, 3, 1, 1, 1)
        drive = inflow - velocity
        velocity = velocity + mask * fan_strength * drive
    return velocity


def load_sample(sample_path: Path) -> Dict[str, np.ndarray]:
    sample = np.load(sample_path, allow_pickle=True).item()
    required = ["obstacle_mask", "velocity_t", "pressure_t"]
    for key in required:
        if key not in sample:
            raise KeyError(f"Sample {sample_path} missing key: {key}")
    return sample


def select_sample_path(data_dir: Path, sample_path: Optional[str], sample_index: int) -> Path:
    if sample_path:
        return Path(sample_path)
    candidates = sorted(data_dir.glob("sample_*.npy"))
    if not candidates:
        raise FileNotFoundError(f"No samples found in {data_dir}")
    if sample_index >= len(candidates):
        raise IndexError(f"sample_index {sample_index} out of range (found {len(candidates)} samples)")
    return candidates[sample_index]


def prepare_tensors(sample: Dict[str, np.ndarray], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = torch.from_numpy(sample["obstacle_mask"].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    v0 = torch.from_numpy(sample["velocity_t"].astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    p0_np = sample["pressure_t"].astype(np.float32)
    if p0_np.ndim == 4:
        p0_np = p0_np[..., 0]
    p0 = torch.from_numpy(p0_np).unsqueeze(0).unsqueeze(0).to(device)
    return mask, v0, p0


def mask_for_step(mask_seq: Optional[np.ndarray], mask_static: np.ndarray, step: int, device: torch.device) -> torch.Tensor:
    if mask_seq is not None and step < mask_seq.shape[0]:
        mask_np = mask_seq[step].astype(np.float32)
    else:
        mask_np = mask_static.astype(np.float32)
    return torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)


def rollout(
    model: torch.nn.Module,
    mask_seq: Optional[np.ndarray],
    mask_static: np.ndarray,
    v0: torch.Tensor,
    p0: torch.Tensor,
    inflow_specs: List[InflowSpec],
    inflow_masks: List[torch.Tensor],
    inflow_phases: List[float],
    inflow_speed: float,
    dir_variation: float,
    dir_frequency: float,
    fan_strength: float,
    gt_v_seq: Optional[np.ndarray],
    gt_p_seq: Optional[np.ndarray],
    steps: int,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[float, float]]]:
    preds_v: List[np.ndarray] = []
    preds_p: List[np.ndarray] = []
    metrics: List[Tuple[float, float]] = []

    current_v = v0
    current_p = p0
    has_gt = gt_v_seq is not None and gt_p_seq is not None

    for step in range(steps):
        mask = mask_for_step(mask_seq, mask_static, step, device)
        fluid = 1.0 - mask
        current_v = current_v * fluid
        current_p = current_p * fluid
        current_v = apply_inflows(
            current_v,
            inflow_specs,
            inflow_masks,
            inflow_phases,
            inflow_speed,
            step,
            dir_variation,
            dir_frequency,
            fan_strength,
        )
        model_input = torch.cat([mask, current_v, current_p], dim=1)
        with torch.no_grad():
            pred = model(model_input)
        pred_v = pred[:, :3] * fluid
        pred_p = pred[:, 3:4] * fluid

        preds_v.append(pred_v.squeeze(0).permute(1, 2, 3, 0).cpu().numpy())
        preds_p.append(pred_p.squeeze(0).squeeze(0).cpu().numpy())

        if has_gt and gt_v_seq is not None and gt_p_seq is not None and step < gt_v_seq.shape[0]:
            tgt_v = torch.from_numpy(gt_v_seq[step]).permute(3, 0, 1, 2).unsqueeze(0).to(device) * fluid
            tgt_p_np = gt_p_seq[step]
            if tgt_p_np.ndim == 4:
                tgt_p_np = tgt_p_np[..., 0]
            tgt_p = torch.from_numpy(tgt_p_np).unsqueeze(0).unsqueeze(0).to(device) * fluid
            fluid_vol = fluid.sum().clamp_min(1.0)
            mse_v = ((pred_v - tgt_v) ** 2 * fluid).sum().div(fluid_vol).item()
            mse_p = ((pred_p - tgt_p) ** 2 * fluid).sum().div(fluid_vol).item()
            metrics.append((mse_v, mse_p))

        current_v = pred_v.detach()
        current_p = pred_p.detach()

    return preds_v, preds_p, metrics


def take_slice(volume: np.ndarray, axis: str, idx: int) -> np.ndarray:
    if axis == "x":
        return volume[idx, :, :]
    if axis == "y":
        return volume[:, idx, :]
    return volume[:, :, idx]


def build_phi_fields(velocity: np.ndarray, pressure: np.ndarray, bounds: object) -> Tuple[object, object]:
    Box, CenteredGrid, extrapolation, phi_math = require_phiflow()
    vel_tensor = phi_math.tensor(
        velocity,
        phi_math.spatial("x,y,z"),
        phi_math.channel("vector"),
    )
    p_tensor = phi_math.tensor(
        pressure,
        phi_math.spatial("x,y,z"),
    )
    vel_grid = CenteredGrid(
        vel_tensor,
        bounds=bounds,
        extrapolation=extrapolation.ZERO,
        x=velocity.shape[0],
        y=velocity.shape[1],
        z=velocity.shape[2],
    )
    p_grid = CenteredGrid(
        p_tensor,
        bounds=bounds,
        extrapolation=extrapolation.ZERO,
        x=pressure.shape[0],
        y=pressure.shape[1],
        z=pressure.shape[2],
    )
    return vel_grid, p_grid


def velocity_magnitude(velocity: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(velocity ** 2, axis=-1))


def build_frames(
    preds_v: List[np.ndarray],
    preds_p: List[np.ndarray],
    gt_v: Optional[np.ndarray],
    gt_p: Optional[np.ndarray],
    mask: np.ndarray,
    axis: str,
    slice_index: Optional[int],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    if slice_index is None:
        slice_index = mask.shape[axis_idx] // 2
    slice_idx = int(cast(int, slice_index))

    pred_vm_slices: List[np.ndarray] = []
    pred_p_slices: List[np.ndarray] = []
    gt_vm_slices: List[np.ndarray] = []
    gt_p_slices: List[np.ndarray] = []

    for step in range(len(preds_v)):
        pred_vm = velocity_magnitude(preds_v[step])
        pred_vm_slices.append(take_slice(pred_vm, axis, slice_idx))
        pred_p_slices.append(take_slice(preds_p[step], axis, slice_idx))

        if gt_v is not None and gt_p is not None and step < gt_v.shape[0]:
            gt_vm = velocity_magnitude(gt_v[step])
            gt_vm_slices.append(take_slice(gt_vm, axis, slice_idx))
            gt_p_step = gt_p[step]
            if gt_p_step.ndim == 4:
                gt_p_step = gt_p_step[..., 0]
            gt_p_slices.append(take_slice(gt_p_step, axis, slice_idx))

    return pred_vm_slices, pred_p_slices, gt_vm_slices, gt_p_slices


def animate_slices(
    pred_vm: List[np.ndarray],
    pred_p: List[np.ndarray],
    gt_vm: List[np.ndarray],
    gt_p: List[np.ndarray],
    axis: str,
    save_path: Optional[str],
    fps: int,
    show: bool,
) -> None:
    has_gt = len(gt_vm) > 0 and len(gt_p) > 0
    if has_gt:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax_pred_vm, ax_pred_p = axes[0]
        ax_gt_vm, ax_gt_p = axes[1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_pred_vm, ax_pred_p = axes
        ax_gt_vm, ax_gt_p = None, None

    def setup_ax(ax: Axes, title: str) -> None:
        ax.set_title(title)
        ax.set_xlabel(f"{axis} slice")
        ax.set_ylabel("index")

    setup_ax(ax_pred_vm, "Pred velocity |v|")
    setup_ax(ax_pred_p, "Pred pressure")
    if has_gt and ax_gt_vm is not None and ax_gt_p is not None:
        setup_ax(ax_gt_vm, "GT velocity |v|")
        setup_ax(ax_gt_p, "GT pressure")

    ims = []
    for step in range(len(pred_vm)):
        frame_artists = []
        im0 = ax_pred_vm.imshow(pred_vm[step].T, origin="lower", cmap="viridis")
        im1 = ax_pred_p.imshow(pred_p[step].T, origin="lower", cmap="coolwarm")
        frame_artists.extend([im0, im1])
        if has_gt and ax_gt_vm is not None and ax_gt_p is not None:
            im2 = ax_gt_vm.imshow(gt_vm[step].T, origin="lower", cmap="viridis")
            im3 = ax_gt_p.imshow(gt_p[step].T, origin="lower", cmap="coolwarm")
            frame_artists.extend([im2, im3])
        ims.append(frame_artists)

    fig.tight_layout()
    anim = animation.ArtistAnimation(fig, ims, interval=1000 / max(1, fps), blit=True)

    if save_path:
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        else:
            anim.save(save_path, writer="ffmpeg", fps=fps)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    sample_path = select_sample_path(data_dir, args.sample, args.sample_index)
    sample = load_sample(sample_path)
    meta = load_metadata(data_dir)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    mask, v0, p0 = prepare_tensors(sample, device)
    mask_np = sample["obstacle_mask"].astype(np.float32)

    inflow_specs, inflow_phases = parse_inflow_specs(sample)
    grid_size = mask_np.shape[0]
    inflow_masks_np = [build_inflow_mask_for_spec(grid_size, spec) for spec in inflow_specs]
    inflow_masks = [
        torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
        for mask in inflow_masks_np
    ]

    inflow_speed = float(meta.get("inflow_speed", 8.0))
    dir_variation = float(meta.get("inflow_direction_variation", 0.05))
    dir_frequency = float(meta.get("inflow_direction_frequency", 0.15))
    fan_strength = float(meta.get("inflow_fan_strength", 0.8))

    gt_v_seq = sample.get("velocity_tn")
    gt_p_seq = sample.get("pressure_tn")
    mask_seq = sample.get("obstacle_mask_tn")

    model = SteadyRolloutCFDUNet(SteadyRolloutCFDUNetCfg()).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if args.use_ema and "ema_state" in checkpoint:
        state = normalize_state_dict(checkpoint["ema_state"])
    else:
        state = normalize_state_dict(checkpoint.get("model_state", checkpoint))
    model.load_state_dict(state)
    model.eval()

    preds_v, preds_p, metrics = rollout(
        model,
        mask_seq,
        mask_np,
        v0,
        p0,
        inflow_specs,
        inflow_masks,
        inflow_phases,
        inflow_speed,
        dir_variation,
        dir_frequency,
        fan_strength,
        gt_v_seq,
        gt_p_seq,
        args.steps,
        device,
    )

    if metrics:
        mse_v = np.mean([m[0] for m in metrics])
        mse_p = np.mean([m[1] for m in metrics])
        print(f"Mean MSE velocity: {mse_v:.6e}, pressure: {mse_p:.6e}")

    # PhiFlow grids (for integration with PhiFlow tooling)
    Box, _, _, _ = require_phiflow()
    bounds = Box(x=(0, grid_size), y=(0, grid_size), z=(0, grid_size))  # type: ignore[call-arg]
    _ = [build_phi_fields(preds_v[i], preds_p[i], bounds) for i in range(len(preds_v))]

    pred_vm, pred_p, gt_vm, gt_p = build_frames(
        preds_v,
        preds_p,
        gt_v_seq,
        gt_p_seq,
        mask_np,
        args.slice_axis,
        args.slice_index,
    )

    animate_slices(
        pred_vm,
        pred_p,
        gt_vm,
        gt_p,
        args.slice_axis,
        args.save,
        args.fps,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
