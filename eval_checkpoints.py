#!/usr/bin/env python3
"""Evaluate all saved checkpoints on rollout metrics and pick the best epoch.

This script is intended for long-horizon model selection:
- Loads each checkpoint in a directory (e.g., every 5 epochs).
- Runs autoregressive rollout on validation samples.
- Reports per-checkpoint mean-step and final-step MSE for velocity/pressure.
- Ranks checkpoints by a configurable score and prints the best epoch.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from model import SteadyRolloutCFDUNet, SteadyRolloutCFDUNetCfg


_EPOCH_RE = re.compile(r"checkpoint_(\d+)\.pt$")


@dataclass
class EvalStats:
    epoch: int
    checkpoint: Path
    num_samples: int
    num_steps: int
    mean_v_mse: float
    mean_p_mse: float
    final_v_mse: float
    final_p_mse: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints and select the best epoch.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Directory containing checkpoint_XXX.pt")
    parser.add_argument("--data-dir", type=str, required=True, help="Validation sample directory")
    parser.add_argument("--checkpoint-pattern", type=str, default="checkpoint_*.pt", help="Glob pattern for checkpoints")
    parser.add_argument("--steps", type=int, default=16, help="Rollout steps to evaluate (<=0 means use full GT length)")
    parser.add_argument("--max-samples", type=int, default=0, help="Max number of samples (0 means all)")
    parser.add_argument("--sample-stride", type=int, default=1, help="Take every Nth sample")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    parser.add_argument("--use-ema", action="store_true", help="Use ema_state if present in checkpoint")
    parser.add_argument(
        "--select-by",
        type=str,
        choices=["mean", "final", "hybrid"],
        default="hybrid",
        help="Metric used to rank checkpoints",
    )
    parser.add_argument(
        "--pressure-weight",
        type=float,
        default=1.0,
        help="Relative weight of pressure MSE in score computation",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Print top-K checkpoints")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional CSV path for all checkpoint metrics")
    parser.add_argument(
        "--fail-on-incompatible",
        action="store_true",
        help="Fail immediately if a checkpoint cannot be loaded into current model code",
    )
    return parser.parse_args()


def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def infer_model_cfg_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> SteadyRolloutCFDUNetCfg:
    stem = state_dict.get("stem.conv.weight")
    head = state_dict.get("head.weight")
    if stem is None or head is None:
        return SteadyRolloutCFDUNetCfg()

    in_channels = int(stem.shape[1])
    base_channels = int(stem.shape[0])
    out_channels = int(head.shape[0])

    depth = 1
    down_indices: List[int] = []
    for key in state_dict.keys():
        if key.startswith("down."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                down_indices.append(int(parts[1]))
    if down_indices:
        depth = max(down_indices) + 2

    res_per_stage = 1
    res_indices: List[int] = []
    for key in state_dict.keys():
        if key.startswith("down.0.res."):
            parts = key.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                res_indices.append(int(parts[3]))
    if res_indices:
        res_per_stage = max(res_indices) + 1

    return SteadyRolloutCFDUNetCfg(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        depth=depth,
        res_per_stage=res_per_stage,
    )


def build_model_cfg(checkpoint_obj: Dict[str, object], model_state: Dict[str, torch.Tensor]) -> SteadyRolloutCFDUNetCfg:
    raw_cfg = checkpoint_obj.get("model_cfg")
    if isinstance(raw_cfg, dict):
        cfg_fields = {item.name for item in fields(SteadyRolloutCFDUNetCfg)}
        filtered = {k: v for k, v in raw_cfg.items() if k in cfg_fields}
        return SteadyRolloutCFDUNetCfg(**filtered)
    return infer_model_cfg_from_state_dict(model_state)


def extract_epoch(path: Path) -> int:
    match = _EPOCH_RE.search(path.name)
    if not match:
        raise ValueError(f"Unexpected checkpoint filename: {path.name}")
    return int(match.group(1))


def list_checkpoints(checkpoint_dir: Path, pattern: str) -> List[Path]:
    checkpoints = sorted(checkpoint_dir.glob(pattern), key=extract_epoch)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir} matching {pattern}")
    return checkpoints


def list_samples(data_dir: Path, max_samples: int, sample_stride: int) -> List[Path]:
    samples = sorted(data_dir.glob("sample_*.npy"))
    if not samples:
        samples = sorted(data_dir.rglob("sample_*.npy"))
    if not samples:
        raise FileNotFoundError(f"No sample_*.npy found under {data_dir}")
    stride = max(sample_stride, 1)
    samples = samples[::stride]
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples


def _to_pressure_channel(pressure: np.ndarray) -> np.ndarray:
    p = pressure.astype(np.float32)
    if p.ndim == 3:
        p = p[..., None]
    elif p.ndim == 4:
        # Supports both (D,H,W,1) and (T,D,H,W).
        # For sequence without channel, append channel at the end.
        if p.shape[-1] != 1:
            p = p[..., None]
    elif p.ndim == 5:
        if p.shape[-1] != 1:
            raise ValueError(f"Unsupported pressure shape {p.shape}: last dim must be 1 for 5D tensors")
    else:
        raise ValueError(f"Unsupported pressure tensor rank: {p.ndim}")
    return p


def _load_fan_fields(sample: Dict[str, np.ndarray], spatial_shape: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    fan_mask = sample.get("fan_mask")
    fan_velocity = sample.get("fan_velocity")
    if fan_mask is None:
        fan_mask_np = np.zeros(tuple(spatial_shape), dtype=np.float32)
    else:
        fan_mask_np = np.asarray(fan_mask, dtype=np.float32)
        if fan_mask_np.ndim == 4 and fan_mask_np.shape[-1] == 1:
            fan_mask_np = fan_mask_np[..., 0]
    if fan_velocity is None:
        fan_velocity_np = np.zeros((*spatial_shape, 3), dtype=np.float32)
    else:
        fan_velocity_np = np.asarray(fan_velocity, dtype=np.float32)
    if fan_mask_np.shape != tuple(spatial_shape):
        raise ValueError(f"fan_mask shape mismatch: expected {tuple(spatial_shape)}, got {fan_mask_np.shape}")
    if fan_velocity_np.shape != (*tuple(spatial_shape), 3):
        raise ValueError(
            f"fan_velocity shape mismatch: expected {(*tuple(spatial_shape), 3)}, got {fan_velocity_np.shape}"
        )
    return fan_mask_np, fan_velocity_np


def evaluate_sample(
    model: torch.nn.Module,
    sample: Dict[str, np.ndarray],
    steps: int,
    device: torch.device,
) -> Tuple[List[float], List[float]]:
    mask_static_np = sample["obstacle_mask"].astype(np.float32)
    mask_seq_np = sample.get("obstacle_mask_tn")
    gt_v = sample["velocity_tn"].astype(np.float32)
    gt_p = _to_pressure_channel(sample["pressure_tn"])

    spatial_shape = mask_static_np.shape
    fan_mask_np, fan_velocity_np = _load_fan_fields(sample, spatial_shape)

    v0 = torch.from_numpy(sample["velocity_t"].astype(np.float32)).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    p0 = torch.from_numpy(_to_pressure_channel(sample["pressure_t"])).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    fan_mask = torch.from_numpy(fan_mask_np).unsqueeze(0).unsqueeze(0).to(device)
    fan_velocity = torch.from_numpy(fan_velocity_np).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    total_steps = gt_v.shape[0]
    rollout_steps = total_steps if steps <= 0 else min(steps, total_steps)

    v_cur = v0
    p_cur = p0
    mse_v_steps: List[float] = []
    mse_p_steps: List[float] = []

    with torch.no_grad():
        for step in range(rollout_steps):
            if mask_seq_np is not None and step < mask_seq_np.shape[0]:
                mask_np = mask_seq_np[step].astype(np.float32)
            else:
                mask_np = mask_static_np
            mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device)
            fluid = 1.0 - mask

            v_cur = v_cur * fluid
            p_cur = p_cur * fluid
            source_mask = fan_mask * fluid
            source_velocity = fan_velocity * source_mask
            model_input = torch.cat([mask, source_mask, source_velocity, v_cur, p_cur], dim=1)

            pred = model(model_input)
            pred_v = pred[:, :3] * fluid
            pred_p = pred[:, 3:4] * fluid

            tgt_v = torch.from_numpy(gt_v[step]).permute(3, 0, 1, 2).unsqueeze(0).to(device) * fluid
            tgt_p = torch.from_numpy(gt_p[step]).permute(3, 0, 1, 2).unsqueeze(0).to(device) * fluid

            fluid_vol = fluid.sum().clamp_min(1.0)
            mse_v = ((pred_v - tgt_v) ** 2 * fluid).sum().div(fluid_vol).item()
            mse_p = ((pred_p - tgt_p) ** 2 * fluid).sum().div(fluid_vol).item()
            mse_v_steps.append(mse_v)
            mse_p_steps.append(mse_p)

            v_cur = pred_v.detach()
            p_cur = pred_p.detach()

    return mse_v_steps, mse_p_steps


def score_checkpoint(
    mean_v: float,
    mean_p: float,
    final_v: float,
    final_p: float,
    select_by: str,
    pressure_weight: float,
) -> float:
    pw = max(pressure_weight, 0.0)
    if select_by == "mean":
        return mean_v + pw * mean_p
    if select_by == "final":
        return final_v + pw * final_p
    # hybrid
    return 0.5 * (mean_v + final_v) + 0.5 * pw * (mean_p + final_p)


def evaluate_checkpoint(
    model: torch.nn.Module,
    sample_paths: Iterable[Path],
    steps: int,
    device: torch.device,
    epoch: int,
    checkpoint_path: Path,
    select_by: str,
    pressure_weight: float,
) -> EvalStats:
    mean_v_acc = 0.0
    mean_p_acc = 0.0
    final_v_acc = 0.0
    final_p_acc = 0.0
    sample_count = 0
    total_steps = 0

    for sample_path in sample_paths:
        sample = np.load(sample_path, allow_pickle=True).item()
        mse_v_steps, mse_p_steps = evaluate_sample(model, sample, steps, device)
        if not mse_v_steps:
            continue
        mean_v_acc += float(np.mean(mse_v_steps))
        mean_p_acc += float(np.mean(mse_p_steps))
        final_v_acc += float(mse_v_steps[-1])
        final_p_acc += float(mse_p_steps[-1])
        total_steps += len(mse_v_steps)
        sample_count += 1

    if sample_count == 0:
        raise RuntimeError(f"No valid samples evaluated for checkpoint {checkpoint_path}")

    mean_v = mean_v_acc / sample_count
    mean_p = mean_p_acc / sample_count
    final_v = final_v_acc / sample_count
    final_p = final_p_acc / sample_count
    score = score_checkpoint(mean_v, mean_p, final_v, final_p, select_by, pressure_weight)
    return EvalStats(
        epoch=epoch,
        checkpoint=checkpoint_path,
        num_samples=sample_count,
        num_steps=total_steps,
        mean_v_mse=mean_v,
        mean_p_mse=mean_p,
        final_v_mse=final_v,
        final_p_mse=final_p,
        score=score,
    )


def load_model_for_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    use_ema: bool,
) -> torch.nn.Module:
    state = torch.load(checkpoint_path, map_location=device)
    state_obj: Dict[str, object] = state if isinstance(state, dict) else {}

    if use_ema and isinstance(state, dict) and "ema_state" in state:
        model_state = normalize_state_dict(state["ema_state"])
    else:
        model_state = normalize_state_dict(state_obj.get("model_state", state))

    cfg = build_model_cfg(state_obj, model_state)
    model = SteadyRolloutCFDUNet(cfg).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return model


def write_csv(path: Path, rows: Sequence[EvalStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "checkpoint",
                "num_samples",
                "num_steps",
                "mean_v_mse",
                "mean_p_mse",
                "final_v_mse",
                "final_p_mse",
                "score",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.epoch,
                    str(row.checkpoint),
                    row.num_samples,
                    row.num_steps,
                    row.mean_v_mse,
                    row.mean_p_mse,
                    row.final_v_mse,
                    row.final_p_mse,
                    row.score,
                ]
            )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_dir = Path(args.checkpoint_dir)
    data_dir = Path(args.data_dir)

    checkpoints = list_checkpoints(checkpoint_dir, args.checkpoint_pattern)
    samples = list_samples(data_dir, args.max_samples, args.sample_stride)

    print(
        f"Evaluating {len(checkpoints)} checkpoints on {len(samples)} samples "
        f"(steps={args.steps}, device={device}, select_by={args.select_by})"
    )

    results: List[EvalStats] = []
    for ckpt in checkpoints:
        epoch = extract_epoch(ckpt)
        try:
            model = load_model_for_checkpoint(ckpt, device, use_ema=args.use_ema)
        except Exception as exc:
            summary = str(exc).splitlines()[0][:240]
            message = f"[epoch {epoch:>4}] skipped (incompatible checkpoint): {summary}"
            if args.fail_on_incompatible:
                raise RuntimeError(message) from exc
            print(message)
            continue
        stats = evaluate_checkpoint(
            model=model,
            sample_paths=samples,
            steps=args.steps,
            device=device,
            epoch=epoch,
            checkpoint_path=ckpt,
            select_by=args.select_by,
            pressure_weight=args.pressure_weight,
        )
        results.append(stats)
        print(
            f"[epoch {stats.epoch:>4}] score={stats.score:.6f} "
            f"mean(v={stats.mean_v_mse:.6f}, p={stats.mean_p_mse:.6f}) "
            f"final(v={stats.final_v_mse:.6f}, p={stats.final_p_mse:.6f})"
        )

    if not results:
        raise RuntimeError("No compatible checkpoints were evaluated.")

    ranked = sorted(results, key=lambda item: item.score)
    best = ranked[0]
    print("")
    print(
        f"BEST: epoch {best.epoch} | score={best.score:.6f} | "
        f"checkpoint={best.checkpoint}"
    )
    print(f"Top-{max(args.top_k, 1)}:")
    for row in ranked[: max(args.top_k, 1)]:
        print(
            f"  epoch {row.epoch:>4} | score={row.score:.6f} | "
            f"mean_v={row.mean_v_mse:.6f} mean_p={row.mean_p_mse:.6f} "
            f"final_v={row.final_v_mse:.6f} final_p={row.final_p_mse:.6f}"
        )

    if args.output_csv:
        csv_path = Path(args.output_csv)
        write_csv(csv_path, results)
        print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
