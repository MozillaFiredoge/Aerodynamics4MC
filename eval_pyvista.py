#!/usr/bin/env python3
"""Evaluate a trained model and save 3D/2D animations using PyVista and Matplotlib."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib import animation

import pyvista as pv

from model import UNet3D, UNet3DConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model and render 3D/2D animations.")
    parser.add_argument("--checkpoint-dir", type=str, default="runs_128", help="Directory with checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path")
    parser.add_argument("--data-dir", type=str, default="val_data", help="Validation dataset directory")
    parser.add_argument("--sample-index", type=int, default=0, help="Index of sample_XXXXX.npy")
    parser.add_argument("--steps", type=int, default=20, help="Number of rollout steps")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument("--slice-axis", type=str, default="y", choices=["x", "y", "z"], help="2D slice axis")
    parser.add_argument("--format", type=str, default="gif", choices=["gif", "mp4", "both"], help="Animation format")
    parser.add_argument("--field", type=str, default="velocity", choices=["velocity", "pressure"], help="Field to render")
    return parser.parse_args()

def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict

def latest_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return checkpoints[-1]


def load_sample(data_dir: Path, index: int) -> Dict[str, np.ndarray]:
    samples = sorted(data_dir.glob("sample_*.npy"))
    if not samples:
        raise FileNotFoundError(f"No samples found in {data_dir}")
    if index >= len(samples):
        raise IndexError(f"sample-index {index} out of range (found {len(samples)})")
    return np.load(samples[index], allow_pickle=True).item()


def prepare_initial_state(sample: Dict[str, np.ndarray], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = torch.from_numpy(sample["obstacle_mask"].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    v0 = torch.from_numpy(sample["velocity_t"].astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    p0_np = sample["pressure_t"].astype(np.float32)
    if p0_np.ndim == 4:
        p0_np = p0_np[..., 0]
    p0 = torch.from_numpy(p0_np).unsqueeze(0).unsqueeze(0).to(device)
    return mask, v0, p0


def apply_inflow(velocity: torch.Tensor, speed: float = 5.0, width: int = 3) -> torch.Tensor:
    velocity = velocity.clone()
    velocity[:, 0, :width, :, :] = speed
    velocity[:, 1, :width, :, :] = 0.0
    velocity[:, 2, :width, :, :] = 0.0
    return velocity


def rollout(
    model: torch.nn.Module,
    mask_seq: np.ndarray,
    mask_static: np.ndarray,
    v0: torch.Tensor,
    p0: torch.Tensor,
    steps: int,
    device: torch.device,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    preds_v: List[np.ndarray] = []
    preds_p: List[np.ndarray] = []
    current_v = v0
    current_p = p0

    for step in range(steps):
        if mask_seq is not None and step < mask_seq.shape[0]:
            mask_np = mask_seq[step].astype(np.float32)
        else:
            mask_np = mask_static.astype(np.float32)
        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
        free = 1.0 - mask
        #current_v = apply_inflow(current_v)
        model_input = torch.cat([mask, current_v, current_p], dim=1)
        with torch.no_grad():
            pred = model(model_input)
        pred_v = pred[:, :3] * free
        pred_p = pred[:, 3:4] * free
        #pred_v = apply_inflow(pred_v)
        preds_v.append(pred_v.squeeze(0).permute(1, 2, 3, 0).cpu().numpy())
        preds_p.append(pred_p.squeeze(0).squeeze(0).cpu().numpy())
        current_v = pred_v.detach()
        current_p = pred_p.detach()

    return preds_v, preds_p


def velocity_magnitude(field: np.ndarray) -> np.ndarray:
    return np.linalg.norm(field, axis=-1)


def make_uniform_grid(data: np.ndarray) -> pv.ImageData:
    dims = np.array(data.shape) + 1
    grid = pv.ImageData(dimensions=dims, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    grid.cell_data["values"] = np.ascontiguousarray(data.ravel(order="F"))
    return grid


def obstacle_surface(mask: np.ndarray) -> pv.PolyData:
    grid = make_uniform_grid(mask.astype(np.float32))
    grid = grid.cell_data_to_point_data()
    return grid.contour([0.5])


def render_3d_animation(
    frames: List[np.ndarray],
    mask: np.ndarray,
    output_path: Path,
    fps: int,
    field_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True)
    surface = obstacle_surface(mask)
    plotter.add_mesh(surface, color="gray", opacity=0.35)

    vmin = float(np.min(frames))
    vmax = float(np.max(frames))
    grid = make_uniform_grid(frames[0])
    plotter.add_volume(grid, scalars="values", opacity="sigmoid", clim=(vmin, vmax))

    if output_path.suffix.lower() == ".gif":
        plotter.open_gif(str(output_path))
    else:
        plotter.open_movie(str(output_path), framerate=fps)

    for frame in frames:
        grid.cell_data["values"] = np.ascontiguousarray(frame.ravel(order="F"))
        plotter.write_frame()

    plotter.close()
    print(f"Saved 3D {field_label} animation to {output_path}")


def take_slice(volume: np.ndarray, axis: str, idx: int) -> np.ndarray:
    if axis == "x":
        return volume[idx, :, :]
    if axis == "y":
        return volume[:, idx, :]
    return volume[:, :, idx]


def render_2d_animation(
    frames: List[np.ndarray],
    mask: np.ndarray,
    axis: str,
    output_path: Path,
    fps: int,
    field_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    mid = mask.shape[axis_idx] // 2
    mask_slice = take_slice(mask.astype(np.float32), axis, mid)

    slices = [take_slice(frame, axis, mid) for frame in frames]
    vmin = float(np.min(slices))
    vmax = float(np.max(slices))

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(slices[0], origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.contour(mask_slice, levels=[0.5], colors="k", linewidths=0.5)
    ax.set_title(f"{field_label} slice ({axis}={mid})")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame_idx: int):
        im.set_data(slices[frame_idx])
        ax.set_xlabel(f"Frame {frame_idx + 1}/{len(slices)}")
        return [im]

    interval_ms = 1000 / max(fps, 1)
    ani = animation.FuncAnimation(fig, update, frames=len(slices), interval=interval_ms, blit=False)
    if output_path.suffix.lower() == ".gif":
        ani.save(output_path, writer="pillow", fps=fps)
    else:
        ani.save(output_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"Saved 2D {field_label} animation to {output_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data_dir = Path(args.data_dir)
    sample = load_sample(data_dir, args.sample_index)
    mask_static = sample["obstacle_mask"].astype(np.float32)
    mask_seq = sample.get("obstacle_mask_tn")

    steps = args.steps
    if "velocity_tn" in sample:
        steps = min(steps, sample["velocity_tn"].shape[0])

    mask, v0, p0 = prepare_initial_state(sample, device)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(Path(args.checkpoint_dir))
    model = UNet3D(UNet3DConfig(base_channels=32)).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model_state = normalize_state_dict(state.get("model_state", state))
    model.load_state_dict(model_state)
    model.eval()

    preds_v, preds_p = rollout(model, mask_seq, mask_static, v0, p0, steps, device)

    if args.field == "pressure":
        frames = preds_p
        field_label = "pressure"
    else:
        frames = [velocity_magnitude(v) for v in preds_v]
        field_label = "velocity"

    output_dir = Path(args.output_dir)
    formats = [args.format] if args.format != "both" else ["gif", "mp4"]
    for fmt in formats:
        out_3d = output_dir / f"rollout_3d_{field_label}.{fmt}"
        out_2d = output_dir / f"rollout_2d_{field_label}.{fmt}"
        render_3d_animation(frames, mask_static, out_3d, args.fps, field_label)
        render_2d_animation(frames, mask_static, args.slice_axis, out_2d, args.fps, field_label)


if __name__ == "__main__":
    main()
