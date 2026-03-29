#!/usr/bin/env python3
"""Evaluate a trained surrogate model and visualize rollout as an animation.

Features:
- Loads a checkpointed UNet3D model.
- Runs an auto-regressive rollout from an initial state in a dataset sample.
- Computes per-step MSE vs. ground-truth sequence (if present).
- Produces a 2x2 animated view: predicted vs. ground-truth velocity magnitude and pressure slices.
- Saves animation to GIF/MP4 (requires pillow/ffmpeg) or shows interactively.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation

from model import SteadyRolloutCFDUNet, SteadyRolloutCFDUNetCfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize surrogate rollout.")
    parser.add_argument("--checkpoint", type=str, default="runs/checkpoint_020.pt", help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/grid_32", help="Directory with sample_*.npy files")
    parser.add_argument("--sample", type=str, default=None, help="Optional explicit sample path")
    parser.add_argument("--sample-index", type=int, default=0, help="Index of sample_XXXXX.npy if --sample not given")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    parser.add_argument("--steps", type=int, default=10, help="Number of rollout steps to visualize")
    parser.add_argument("--slice-axis", type=str, default="y", choices=["x", "y", "z"], help="Axis for 2D slice")
    parser.add_argument("--save", type=str, default=None, help="Output path (.gif or .mp4). If omitted, shows interactively.")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second for animation")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    parser.add_argument("--inflow-speed", type=float, default=5.0, help="Inflow speed (matches data_gen default)")
    parser.add_argument(
        "--inflow-direction",
        type=float,
        nargs=3,
        default=(1.0, 0.0, 0.0),
        help="Inflow direction vector (x y z)",
    )
    parser.add_argument("--inflow-variation", type=float, default=0.05, help="Tangential inflow variation")
    parser.add_argument("--inflow-frequency", type=float, default=0.15, help="Tangential inflow frequency")
    parser.add_argument("--inflow-phase", type=float, default=0.0, help="Phase offset for tangential inflow")
    return parser.parse_args()

def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict

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


INFLOW_FACES = ("x+", "x-", "y+", "y-", "z+", "z-")


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def inflow_face_to_axis_sign(face: str) -> Tuple[str, int]:
    if face not in INFLOW_FACES:
        raise ValueError(f"Unknown inflow face: {face}")
    axis_name = face[0]
    sign = 1 if face[1] == "+" else -1
    return axis_name, sign


def inflow_face_from_direction(direction: np.ndarray) -> str:
    axis = int(np.argmax(np.abs(direction)))
    sign = "+" if direction[axis] >= 0 else "-"
    axis_name = ("x", "y", "z")[axis]
    return f"{axis_name}{sign}"


def build_inflow_mask(grid_size: int, face: str, thickness: int = 2) -> np.ndarray:
    inflow = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    axis_name, sign = inflow_face_to_axis_sign(face)
    if axis_name == "x":
        if sign > 0:
            inflow[-thickness:, :, :] = 1.0
        else:
            inflow[:thickness, :, :] = 1.0
    elif axis_name == "y":
        if sign > 0:
            inflow[:, -thickness:, :] = 1.0
        else:
            inflow[:, :thickness, :] = 1.0
    else:
        if sign > 0:
            inflow[:, :, -thickness:] = 1.0
        else:
            inflow[:, :, :thickness] = 1.0
    return inflow


def apply_inflow(
    velocity: torch.Tensor,
    inflow_mask: torch.Tensor,
    inflow_speed: float,
    t: int,
    inflow_face: str,
    dir_variation: float,
    dir_frequency: float,
    dir_phase: float,
) -> torch.Tensor:
    speed = inflow_speed * (0.5 + 0.5 * torch.sin(torch.tensor(0.2 * t, device=velocity.device)))
    axis_name, sign = inflow_face_to_axis_sign(inflow_face)
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map[axis_name]
    tangential_axes = [idx for idx in range(3) if idx != axis_idx]
    phase = dir_frequency * t + dir_phase
    tangential_amp = speed * dir_variation
    components = [torch.tensor(0.0, device=velocity.device) for _ in range(3)]
    components[axis_idx] = -speed * sign
    components[tangential_axes[0]] = tangential_amp * torch.sin(torch.tensor(phase, device=velocity.device))
    components[tangential_axes[1]] = tangential_amp * torch.cos(torch.tensor(phase, device=velocity.device))
    inflow = torch.zeros_like(velocity)
    inflow[:, 0, ...] = components[0]
    inflow[:, 1, ...] = components[1]
    inflow[:, 2, ...] = components[2]
    inflow_mask_staggered = inflow_mask.unsqueeze(1)
    return velocity * (1 - inflow_mask_staggered) + inflow * inflow_mask_staggered


def mask_for_step(
    mask_seq: Optional[np.ndarray],
    mask_static: np.ndarray,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    if mask_seq is not None and step < mask_seq.shape[0]:
        mask_np = mask_seq[step].astype(np.float32)
    else:
        mask_np = mask_static.astype(np.float32)
    return torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)


def rollout(
    model: torch.nn.Module,
    mask_seq: Optional[np.ndarray],
    mask_static: np.ndarray,
    inflow_mask: torch.Tensor,
    inflow_speed: float,
    inflow_face: str,
    dir_variation: float,
    dir_frequency: float,
    dir_phase: float,
    v0: torch.Tensor,
    p0: torch.Tensor,
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
        free = 1.0 - mask
        current_v = current_v * free
        current_p = current_p * free
        current_v = apply_inflow(
            current_v,
            inflow_mask,
            inflow_speed,
            step,
            inflow_face,
            dir_variation,
            dir_frequency,
            dir_phase,
        )
        model_input = torch.cat([mask, current_v, current_p], dim=1)
        with torch.no_grad():
            pred = model(model_input)
        pred_v = pred[:, :3] * free
        pred_p = pred[:, 3:4] * free
        pred_v = apply_inflow(
            pred_v,
            inflow_mask,
            inflow_speed,
            step,
            inflow_face,
            dir_variation,
            dir_frequency,
            dir_phase,
        )

        preds_v.append(pred_v.squeeze(0).permute(1, 2, 3, 0).cpu().numpy())
        preds_p.append(pred_p.squeeze(0).squeeze(0).cpu().numpy())

        if has_gt and gt_v_seq is not None and gt_p_seq is not None and step < gt_v_seq.shape[0]:
            tgt_v = torch.from_numpy(gt_v_seq[step]).permute(3, 0, 1, 2).unsqueeze(0).to(device) * free
            tgt_p_np = gt_p_seq[step]
            if tgt_p_np.ndim == 4:
                tgt_p_np = tgt_p_np[..., 0]
            tgt_p = torch.from_numpy(tgt_p_np).unsqueeze(0).unsqueeze(0).to(device) * free
            # tgt_v = apply_inflow(
            #     tgt_v,
            #     inflow_mask,
            #     inflow_speed,
            #     step,
            #     inflow_face,
            #     dir_variation,
            #     dir_frequency,
            #     dir_phase,
            # )
            mse_v = torch.mean((pred_v - tgt_v) ** 2).item()
            mse_p = torch.mean((pred_p - tgt_p) ** 2).item()
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


def build_frames(
    preds_v: List[np.ndarray],
    preds_p: List[np.ndarray],
    gt_v: Optional[np.ndarray],
    gt_p: Optional[np.ndarray],
    mask: np.ndarray,
    axis: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], float, float]:
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    mid = mask.shape[axis_idx] // 2

    pred_vm_slices: List[np.ndarray] = []
    pred_p_slices: List[np.ndarray] = []
    gt_vm_slices: List[np.ndarray] = []
    gt_p_slices: List[np.ndarray] = []

    for i in range(len(preds_v)):
        vm = np.linalg.norm(preds_v[i], axis=-1)
        pred_vm_slices.append(take_slice(vm, axis, mid))
        pred_p_slices.append(take_slice(preds_p[i], axis, mid))

        if gt_v is not None and i < gt_v.shape[0]:
            vm_gt = np.linalg.norm(gt_v[i], axis=-1)
            gt_vm_slices.append(take_slice(vm_gt, axis, mid))
        if gt_p is not None and i < gt_p.shape[0]:
            tgt_p = gt_p[i]
            if tgt_p.ndim == 4:
                tgt_p = tgt_p[..., 0]
            gt_p_slices.append(take_slice(tgt_p, axis, mid))

    vel_max = max(np.max(s) for s in pred_vm_slices + (gt_vm_slices if gt_vm_slices else [np.array([0])]))
    pres_abs_max = max(np.max(np.abs(s)) for s in pred_p_slices + (gt_p_slices if gt_p_slices else [np.array([0])]))

    return pred_vm_slices, pred_p_slices, gt_vm_slices, gt_p_slices, vel_max, pres_abs_max


def animate_frames(
    pred_vm: List[np.ndarray],
    pred_p: List[np.ndarray],
    gt_vm: List[np.ndarray],
    gt_p: List[np.ndarray],
    mask_slice: np.ndarray,
    vel_max: float,
    pres_abs_max: float,
    fps: int,
    save_path: Optional[Path],
    show: bool,
) -> None:
    steps = len(pred_vm)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    titles = ["Pred Vel |v|", "GT Vel |v|", "Pred Pressure", "GT Pressure"]

    # Ensure we always have 4 plots; fall back to predicted for missing GT
    gt_vm_safe = gt_vm if gt_vm else pred_vm
    gt_p_safe = gt_p if gt_p else pred_p

    ims = [
        axes[0, 0].imshow(pred_vm[0], origin="lower", vmin=0, vmax=vel_max + 1e-6, cmap="viridis"),
        axes[0, 1].imshow(gt_vm_safe[0], origin="lower", vmin=0, vmax=vel_max + 1e-6, cmap="viridis"),
        axes[1, 0].imshow(pred_p[0], origin="lower", vmin=-pres_abs_max, vmax=pres_abs_max, cmap="coolwarm"),
        axes[1, 1].imshow(gt_p_safe[0], origin="lower", vmin=-pres_abs_max, vmax=pres_abs_max, cmap="coolwarm"),
    ]

    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.contour(mask_slice, levels=[0.5], colors="k", linewidths=0.5)

    fig.tight_layout()

    def update(frame: int):
        ims[0].set_data(pred_vm[frame])
        ims[1].set_data(gt_vm_safe[min(frame, len(gt_vm_safe) - 1)])
        ims[2].set_data(pred_p[frame])
        ims[3].set_data(gt_p_safe[min(frame, len(gt_p_safe) - 1)])
        fig.suptitle(f"Frame {frame+1}/{steps}")
        return ims

    interval_ms = 1000 / max(fps, 1)
    ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval_ms, blit=False)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ext = save_path.suffix.lower()
        writer = "pillow" if ext == ".gif" else "ffmpeg"
        ani.save(save_path, writer=writer, fps=fps)
        print(f"Saved animation to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data_dir = Path(args.data_dir)
    sample_path = select_sample_path(data_dir, args.sample, args.sample_index)
    sample = load_sample(sample_path)

    # Ground-truth sequences (optional for metrics/plots)
    gt_v_seq = sample.get("velocity_tn")
    gt_p_seq = sample.get("pressure_tn")

    steps = min(args.steps, gt_v_seq.shape[0]) if gt_v_seq is not None else args.steps

    mask_static = sample["obstacle_mask"].astype(np.float32)
    mask_seq = sample.get("obstacle_mask_tn")
    mask_t, v0_t, p0_t = prepare_tensors(sample, device)
    inflow_face = sample.get("inflow_face")
    if inflow_face is None or not isinstance(inflow_face, str):
        base_dir = _normalize_vec(np.array(args.inflow_direction, dtype=np.float32))
        inflow_face = inflow_face_from_direction(base_dir)
    inflow_mask_np = build_inflow_mask(mask_static.shape[0], inflow_face)
    inflow_mask = torch.from_numpy(inflow_mask_np).unsqueeze(0).to(device)
    model = SteadyRolloutCFDUNet(SteadyRolloutCFDUNetCfg()).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model_state = normalize_state_dict(state.get("model_state", state))
    model.load_state_dict(model_state)
    model.eval()

    preds_v, preds_p, metrics = rollout(
        model,
        mask_seq,
        mask_static,
        inflow_mask,
        args.inflow_speed,
    inflow_face,
        args.inflow_variation,
        args.inflow_frequency,
        args.inflow_phase,
        v0_t,
        p0_t,
        gt_v_seq,
        gt_p_seq,
        steps,
        device,
    )

    if metrics:
        mse_v = np.mean([m[0] for m in metrics])
        mse_p = np.mean([m[1] for m in metrics])
        print(f"Per-step MSE (velocity): {mse_v:.6f} | (pressure): {mse_p:.6f}")
    else:
        print("No ground-truth sequence found; metrics skipped.")

    axis = args.slice_axis
    pred_vm_s, pred_p_s, gt_vm_s, gt_p_s, vel_max, pres_abs_max = build_frames(
        preds_v,
        preds_p,
        gt_v_seq,
        gt_p_seq,
        mask_static,
        axis,
    )
    mask_slice = take_slice(mask_static, axis, mask_static.shape[{"x":0,"y":1,"z":2}[axis]] // 2)

    save_path = Path(args.save) if args.save else None
    show_flag = not args.no_show if args.save is None else (not args.no_show)
    animate_frames(pred_vm_s, pred_p_s, gt_vm_s, gt_p_s, mask_slice, vel_max, pres_abs_max, args.fps, save_path, show_flag)


if __name__ == "__main__":
    main()
