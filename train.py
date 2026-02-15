"""Training loop for 3D CFD surrogate with pushforward (unrolled) loss."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, cast, Optional

import numpy as np
import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from model import SteadyRolloutCFDUNet, SteadyRolloutCFDUNetCfg


@dataclass
class TrainConfig:
    data_dir: str = "data/grid_32"
    batch_size: int = 2
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-6
    unroll_steps: int = 20
    curriculum_start: int = 1
    curriculum_increase_every: int = 1
    curriculum_increment: int = 1
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 7
    log_interval: int = 10
    output_dir: str = "runs"
    div_weight: float = 1.0
    energy_weight: float = 0.01
    source_weight: float = 0.25
    pressure_mean_weight: float = 0.01
    late_step_weight: float = 1.0
    state_noise_std: float = 0.0
    p_tf_start: float = 1.0
    p_tf_end: float = 0.1
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    resume: bool = False
    checkpoint_path: str | None = None
    checkpoint_interval: int = 1
    grad_accum_steps: int = 1
    grad_checkpoint: bool = False
    multi_gpu: bool = False
    ddp: bool = False
    local_rank: int = 0
    amp: bool = False
    allow_missing_fan_fields: bool = False
    model_in_channels: int = 9
    model_base_channels: int = 48
    model_depth: int = 4
    model_res_per_stage: int = 2
    model_vel_delta_clip: float = 2.5
    model_pressure_delta_clip: float = 3.0
    model_source_inject_strength: float = 0.1


class FlowDataset(Dataset):
    def __init__(self, data_dir: Path, unroll_steps: int, allow_missing_fan_fields: bool = False) -> None:
        self.samples = sorted(data_dir.glob("sample_*.npy"))
        if not self.samples:
            raise FileNotFoundError(f"No samples found in {data_dir}")
        if unroll_steps < 1:
            raise ValueError("unroll_steps must be >= 1")
        self.unroll_steps = unroll_steps
        self.allow_missing_fan_fields = allow_missing_fan_fields
        if not self.allow_missing_fan_fields:
            probe = np.load(self.samples[0], allow_pickle=True).item()
            if "fan_mask" not in probe or "fan_velocity" not in probe:
                raise KeyError(
                    "Dataset is missing `fan_mask`/`fan_velocity`. "
                    "Regenerate with updated data_gen.py or pass --allow-missing-fan-fields."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, np.ndarray]]:
        path = self.samples[idx]
        try:
            sample = np.load(path, allow_pickle=True).item()

            obstacle_mask_tn = sample["obstacle_mask_tn"].astype(np.float32)
            fan_mask = sample.get("fan_mask")
            fan_velocity = sample.get("fan_velocity")
            if fan_mask is None or fan_velocity is None:
                if not self.allow_missing_fan_fields:
                    raise KeyError("Missing fan fields in sample.")
                spatial = obstacle_mask_tn.shape[1:]
                fan_mask = np.zeros(spatial, dtype=np.float32)
                fan_velocity = np.zeros((*spatial, 3), dtype=np.float32)
            fan_mask = np.asarray(fan_mask, dtype=np.float32)
            fan_velocity = np.asarray(fan_velocity, dtype=np.float32)
            if fan_mask.ndim == 4 and fan_mask.shape[-1] == 1:
                fan_mask = fan_mask[..., 0]
            if fan_mask.ndim != 3:
                raise ValueError(f"fan_mask must be (D,H,W), got {fan_mask.shape}")
            if fan_velocity.ndim != 4 or fan_velocity.shape[-1] != 3:
                raise ValueError(f"fan_velocity must be (D,H,W,3), got {fan_velocity.shape}")

            velocity_tn = sample["velocity_tn"].astype(np.float32)  # shape: (T, X, Y, Z, 3)
            pressure_tn = sample["pressure_tn"].astype(np.float32)  # shape: (T, X, Y, Z)

            velocity_t = sample["velocity_t"].astype(np.float32)  # shape: (X, Y, Z, 3)
            pressure_t = sample["pressure_t"].astype(np.float32)  # shape: (X, Y, Z)

            if pressure_tn.ndim == 4:
                pressure_tn = pressure_tn[..., None]
            if pressure_t.ndim == 3:
                pressure_t = pressure_t[..., None]

            T = velocity_tn.shape[0]  # total time steps
            if self.unroll_steps > T:
                raise ValueError(
                    f"unroll_steps ({self.unroll_steps}) greater than time steps in sample ({T})"
                )

            if T > self.unroll_steps:
                start_idx = np.random.randint(0, T - self.unroll_steps + 1)
            else:
                start_idx = 0

            velocity_seq = velocity_tn[start_idx:start_idx + self.unroll_steps]
            pressure_seq = pressure_tn[start_idx:start_idx + self.unroll_steps]
            mask_seq = obstacle_mask_tn[start_idx:start_idx + self.unroll_steps]
            if start_idx == 0:
                velocity_prev = velocity_t
                pressure_prev = pressure_t
            else:
                velocity_prev = velocity_tn[start_idx - 1]
                pressure_prev = pressure_tn[start_idx - 1]

            return {
                "mask_t": mask_seq,
                "fan_mask": fan_mask,
                "fan_velocity": fan_velocity,
                "velocity_t": velocity_seq,
                "pressure_t": pressure_seq,
                "velocity_prev": velocity_prev,
                "pressure_prev": pressure_prev,
            }
        except Exception as exc:
            print(f"Skipping sample {path}: {exc}")
            return None


def collate_skip_none(batch: List[Optional[Dict[str, np.ndarray]]]) -> Optional[Dict[str, torch.Tensor]]:
    batch_items: List[Dict[str, np.ndarray]] = [item for item in batch if item is not None]
    if not batch_items:
        return None
    keys = batch_items[0].keys()
    return {k: torch.from_numpy(np.stack([item[k] for item in batch_items], axis=0)) for k in keys}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_base_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        base_model = get_base_model(model)
        self.shadow = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        base_model = get_base_model(model)
        for key, value in base_model.state_dict().items():
            self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


def build_input(
    mask: torch.Tensor,
    fan_mask: torch.Tensor,
    fan_velocity: torch.Tensor,
    velocity: torch.Tensor,
    pressure: torch.Tensor,
) -> torch.Tensor:
    return torch.cat([mask, fan_mask, fan_velocity, velocity, pressure], dim=1)


def ddx_neumann(f: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    # f: (N,1,D,H,W)
    g = torch.empty_like(f)
    # interior: central
    g[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (2.0 * dx)
    # boundaries: one-sided (Neumann-friendly extension)
    g[..., 0] = (f[..., 1] - f[..., 0]) / dx
    g[..., -1] = (f[..., -1] - f[..., -2]) / dx
    return g

def ddy_neumann(f: torch.Tensor, dy: float = 1.0) -> torch.Tensor:
    g = torch.empty_like(f)
    g[..., 1:-1, :] = (f[..., 2:, :] - f[..., :-2, :]) / (2.0 * dy)
    g[..., 0, :] = (f[..., 1, :] - f[..., 0, :]) / dy
    g[..., -1, :] = (f[..., -1, :] - f[..., -2, :]) / dy
    return g

def ddz_neumann(f: torch.Tensor, dz: float = 1.0) -> torch.Tensor:
    g = torch.empty_like(f)
    g[..., 1:-1, :, :] = (f[..., 2:, :, :] - f[..., :-2, :, :]) / (2.0 * dz)
    g[..., 0, :, :] = (f[..., 1, :, :] - f[..., 0, :, :]) / dz
    g[..., -1, :, :] = (f[..., -1, :, :] - f[..., -2, :, :]) / dz
    return g
def div3d_neumann(u: torch.Tensor, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> torch.Tensor:
    """
    u: (N,3,D,H,W)
    returns divergence: (N,1,D,H,W)
    """
    ux = u[:, 0:1]
    uy = u[:, 1:2]
    uz = u[:, 2:3]
    return ddx_neumann(ux, dx=dx) + ddy_neumann(uy, dy=dy) + ddz_neumann(uz, dz=dz)

def unroll_loss(
    model: nn.Module,
    mask_seq: torch.Tensor,          # (N, steps, 1, D, H, W) or (N, steps, D,H,W) -> we'll handle
    fan_mask: torch.Tensor,          # (N, 1, D, H, W)
    fan_velocity: torch.Tensor,      # (N, 3, D, H, W)
    velocity: torch.Tensor,          # (N, 3, D, H, W)
    pressure: torch.Tensor,          # (N, 1, D, H, W)
    target_velocity: torch.Tensor,   # (N, steps, 3, D, H, W)
    target_pressure: torch.Tensor,   # (N, steps, 1, D, H, W)
    steps: int,
    div_weight: float,
    grad_checkpoint: bool,
    energy_weight: float = 0.01,
    source_weight: float = 0.25,
    pressure_mean_weight: float = 0.01,
    late_step_weight: float = 1.0,
    state_noise_std: float = 0.0,
    p_tf: float = 1.0,
) -> torch.Tensor:
    device = velocity.device
    total_loss = torch.zeros((), device=device)
    total_weight = torch.zeros((), device=device)

    cur_v = velocity
    cur_p = pressure
    if state_noise_std > 0 and model.training:
        start_mask = mask_seq[:, 0]
        if start_mask.dim() == 4:
            start_mask = start_mask.unsqueeze(1)
        start_fluid = 1.0 - start_mask.to(dtype=velocity.dtype).clamp(0, 1)
        cur_v = cur_v + state_noise_std * torch.randn_like(cur_v) * start_fluid
        cur_p = cur_p + state_noise_std * torch.randn_like(cur_p) * start_fluid

    fan_mask = fan_mask.to(dtype=velocity.dtype).clamp(0, 1)
    fan_velocity = fan_velocity.to(dtype=velocity.dtype)

    for step in range(steps):
        mask = mask_seq[:, step]
        if mask.dim() == 4:  # (N,D,H,W) -> (N,1,D,H,W)
            mask = mask.unsqueeze(1)
        mask = mask.to(dtype=velocity.dtype).clamp(0, 1)

        fluid = 1.0 - mask
        source_mask = fan_mask * fluid
        source_velocity = fan_velocity * source_mask
        model_input = build_input(mask, source_mask, source_velocity, cur_v, cur_p)

        if grad_checkpoint:
            def model_forward(x: torch.Tensor) -> torch.Tensor:
                return model(x)
            pred = cast(torch.Tensor, checkpoint.checkpoint(model_forward, model_input, use_reentrant=False))
        else:
            pred = model(model_input)

        pred_v = pred[:, :3]
        pred_p = pred[:, 3:4]

        # Hard mask outputs: obstacle region must be zero
        pred_v = pred_v * fluid
        pred_p = pred_p * fluid

        # Targets masked to fluid only
        tgt_v = target_velocity[:, step] * fluid
        tgt_p = target_pressure[:, step] * fluid

        # Data loss only on fluid region.
        # If your criterion is MSELoss(reduction="mean"), multiplying by fluid changes effective weighting with obstacle ratio.
        # Better: normalize by fluid volume so cases with small fluid don't get tiny gradients.
        fluid_vol = fluid.sum().clamp_min(1.0)

        mse_v = ((pred_v - tgt_v) ** 2 * fluid).sum() / fluid_vol
        mse_p = ((pred_p - tgt_p) ** 2 * fluid).sum() / fluid_vol
        data_loss = mse_v + mse_p

        # Divergence penalty (Neumann-consistent) only on fluid
        div = div3d_neumann(pred_v)  # (N,1,D,H,W)
        div_loss = (div**2).sum() / fluid_vol

        # Optional energy stabilizer
        if energy_weight > 0:
            e_pred = (pred_v**2).sum(dim=1, keepdim=True)
            e_tgt  = (tgt_v**2).sum(dim=1, keepdim=True)
            energy_loss = ((e_pred - e_tgt).abs() * fluid).sum() / fluid.sum().clamp_min(1.0)
        else:
            energy_loss = torch.zeros((), device=device)

        if source_weight > 0:
            source_vol = source_mask.sum().clamp_min(1.0)
            source_loss = ((pred_v - source_velocity) ** 2 * source_mask).sum() / source_vol
        else:
            source_loss = torch.zeros((), device=device)

        if pressure_mean_weight > 0:
            p_mean = (pred_p * fluid).sum() / fluid_vol
            pressure_mean_loss = p_mean.square()
        else:
            pressure_mean_loss = torch.zeros((), device=device)

        step_alpha = float(step) / float(max(steps - 1, 1))
        step_weight = 1.0 + late_step_weight * step_alpha
        step_loss = (
            data_loss
            + div_weight * div_loss
            + energy_weight * energy_loss
            + source_weight * source_loss
            + pressure_mean_weight * pressure_mean_loss
        )
        total_loss = total_loss + step_weight * step_loss
        total_weight = total_weight + step_weight
        # print(f"Step {step+1}/{steps} - data_loss: {data_loss.item():.6f}, div_loss: {div_loss.item():.6f}, energy_loss: {energy_loss.item():.6f}, fluid_vol/total_vol: {fluid_vol.item()/(mask.numel()):.4f}")
        # print(f"u_rms: {torch.sqrt((tgt_v**2).mean()).item():.4f}, p_rms: {torch.sqrt((tgt_p**2).mean()).item():.4f}")
        # print("div stats:", div.abs().max().item(), div.abs().mean().item())
        # print("pred_v stats:", pred_v.abs().max().item(), pred_v.abs().mean().item())
        # rollout state update
        if p_tf >= 1.0:
            cur_v = tgt_v
            cur_p = tgt_p
        elif p_tf <= 0.0:
            cur_v = pred_v
            cur_p = pred_p
        else:
            use_tf = (torch.rand((pred_v.shape[0], 1, 1, 1, 1), device=device) < p_tf).to(dtype=pred_v.dtype)
            cur_v = use_tf * tgt_v + (1.0 - use_tf) * pred_v
            cur_p = use_tf * tgt_p + (1.0 - use_tf) * pred_p

    return total_loss / total_weight.clamp_min(1.0)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    unroll_steps: int,
    log_interval: int,
    div_weight: float,
    energy_weight: float,
    source_weight: float,
    pressure_mean_weight: float,
    late_step_weight: float,
    state_noise_std: float,
    grad_accum_steps: int,
    grad_checkpoint: bool,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    p_tf: float,
    grad_clip_norm: float,
    ema: EMA | None,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        if batch is None:
            continue
        batch = {k: v.to(device) for k, v in batch.items()}
        mask_seq = batch["mask_t"][:, :, None]
        fan_mask = batch["fan_mask"][:, None]
        fan_velocity = batch["fan_velocity"].permute(0, 4, 1, 2, 3)
        velocity_prev = batch["velocity_prev"].permute(0, 4, 1, 2, 3)
        velocity_t1 = batch["velocity_t"].permute(0, 1, 5, 2, 3, 4)
        pressure_prev = batch["pressure_prev"].permute(0, 4, 1, 2, 3)
        pressure_t1 = batch["pressure_t"].permute(0, 1, 5, 2, 3, 4)
        with torch.cuda.amp.autocast(enabled=use_amp):
            raw_loss = unroll_loss(
                model,
                mask_seq,
                fan_mask,
                fan_velocity,
                velocity_prev,
                pressure_prev,
                velocity_t1,
                pressure_t1,
                unroll_steps,
                div_weight,
                grad_checkpoint,
                energy_weight,
                source_weight,
                pressure_mean_weight,
                late_step_weight,
                state_noise_std,
                p_tf,
            )
        loss = raw_loss / max(1, grad_accum_steps)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % max(1, grad_accum_steps) == 0:
            if scaler is not None:
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        total_loss += raw_loss.item()
        if (step + 1) % log_interval == 0:
            avg = total_loss / (step + 1)
            print(f"Step {step + 1}/{len(loader)} - loss: {avg:.6f}")

    if len(loader) % max(1, grad_accum_steps) != 0:
        if scaler is not None:
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        if ema is not None:
            ema.update(model)
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(len(loader), 1)


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model = get_base_model(model)
    model_state = base_model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "model_cfg": getattr(getattr(base_model, "cfg", None), "__dict__", None),
    }
    if ema is not None:
        checkpoint["ema_state"] = ema.state_dict()
    torch.save(checkpoint, output_dir / f"checkpoint_{epoch:03d}.pt")


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train 3D surrogate model with pushforward loss.")
    parser.add_argument("--data-dir", type=str, default="data/grid_32", help="Dataset directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--unroll-steps", type=int, default=5, help="Unrolled steps (k)")
    parser.add_argument(
        "--curriculum-start",
        type=int,
        default=1,
        help="Starting unroll steps for curriculum",
    )
    parser.add_argument(
        "--curriculum-increase-every",
        type=int,
        default=5,
        help="Increase unroll steps every N epochs",
    )
    parser.add_argument(
        "--curriculum-increment",
        type=int,
        default=1,
        help="Increase unroll steps by this amount",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    parser.add_argument("--output-dir", type=str, default="runs", help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    parser.add_argument("--div-weight", type=float, default=1.0, help="Divergence loss weight")
    parser.add_argument("--energy-weight", type=float, default=0.01, help="Energy loss weight for rollout stability")
    parser.add_argument("--source-weight", type=float, default=0.25, help="Fan source loss weight")
    parser.add_argument("--pressure-mean-weight", type=float, default=0.01, help="Pressure drift loss weight")
    parser.add_argument("--late-step-weight", type=float, default=1.0, help="Extra emphasis for later unroll steps")
    parser.add_argument("--state-noise-std", type=float, default=0.0, help="Gaussian noise std on initial rollout state")
    parser.add_argument("--p-tf-start", type=float, default=1.0, help="Teacher forcing probability start")
    parser.add_argument("--p-tf-end", type=float, default=0.1, help="Teacher forcing probability end")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay (0 to disable)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output dir")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path to resume")
    parser.add_argument("--checkpoint-interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--multi-gpu", action="store_true", help="Enable DataParallel multi-GPU training")
    parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel training")
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="DDP local rank")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--allow-missing-fan-fields", action="store_true", help="Allow old datasets without fan_mask/fan_velocity")
    parser.add_argument("--model-in-channels", type=int, default=9, choices=[5, 9], help="Model stem input channels (5 legacy / 9 fan-conditioned)")
    parser.add_argument("--model-base-channels", type=int, default=48, help="Base channels for U-Net")
    parser.add_argument("--model-depth", type=int, default=4, help="U-Net depth")
    parser.add_argument("--model-res-per-stage", type=int, default=2, help="Residual blocks per stage")
    parser.add_argument("--model-vel-delta-clip", type=float, default=2.5, help="Tanh clip for velocity potential delta")
    parser.add_argument("--model-pressure-delta-clip", type=float, default=3.0, help="Tanh clip for pressure delta")
    parser.add_argument("--model-source-inject-strength", type=float, default=0.1, help="Bias velocity toward fan source region")
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        unroll_steps=args.unroll_steps,
        curriculum_start=args.curriculum_start,
        curriculum_increase_every=args.curriculum_increase_every,
        curriculum_increment=args.curriculum_increment,
        num_workers=args.num_workers,
        device=device,
        seed=args.seed,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
        div_weight=args.div_weight,
        energy_weight=args.energy_weight,
        source_weight=args.source_weight,
        pressure_mean_weight=args.pressure_mean_weight,
        late_step_weight=args.late_step_weight,
        state_noise_std=args.state_noise_std,
        p_tf_start=args.p_tf_start,
        p_tf_end=args.p_tf_end,
        grad_clip_norm=args.grad_clip_norm,
        ema_decay=args.ema_decay,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        grad_accum_steps=args.grad_accum_steps,
        grad_checkpoint=args.grad_checkpoint,
        multi_gpu=args.multi_gpu,
        ddp=args.ddp,
        local_rank=args.local_rank,
        amp=args.amp,
        allow_missing_fan_fields=args.allow_missing_fan_fields,
        model_in_channels=args.model_in_channels,
        model_base_channels=args.model_base_channels,
        model_depth=args.model_depth,
        model_res_per_stage=args.model_res_per_stage,
        model_vel_delta_clip=args.model_vel_delta_clip,
        model_pressure_delta_clip=args.model_pressure_delta_clip,
        model_source_inject_strength=args.model_source_inject_strength,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = cfg.device

    use_ddp = cfg.ddp
    rank = 0
    if use_ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(cfg.local_rank)
        device = f"cuda:{cfg.local_rank}"

    data_dir = Path(cfg.data_dir)
    dataset = FlowDataset(data_dir, cfg.unroll_steps, allow_missing_fan_fields=cfg.allow_missing_fan_fields)
    sampler = None
    if use_ddp:
        sampler = DistributedSampler(dataset, shuffle=True)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=str(device).startswith("cuda"),
        drop_last=True,
        collate_fn=collate_skip_none,
    )

    model_cfg = SteadyRolloutCFDUNetCfg(
        in_channels=cfg.model_in_channels,
        base_channels=cfg.model_base_channels,
        depth=cfg.model_depth,
        res_per_stage=cfg.model_res_per_stage,
        vel_delta_clip=cfg.model_vel_delta_clip,
        pressure_delta_clip=cfg.model_pressure_delta_clip,
        source_inject_strength=cfg.model_source_inject_strength,
    )
    model = SteadyRolloutCFDUNet(model_cfg).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[cfg.local_rank])
    elif cfg.multi_gpu and device == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and str(device).startswith("cuda"))
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay and cfg.ema_decay > 0 else None

    start_epoch = 1
    if cfg.resume or cfg.checkpoint_path:
        ckpt_path = Path(cfg.checkpoint_path) if cfg.checkpoint_path else latest_checkpoint(Path(cfg.output_dir))
        if ckpt_path and ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device)
            model_state = normalize_state_dict(state.get("model_state", state))
            model_target = model.module if isinstance(model, (nn.DataParallel, DDP)) else model
            incompat = model_target.load_state_dict(model_state, strict=False)
            if rank == 0 and (incompat.missing_keys or incompat.unexpected_keys):
                print(
                    "Checkpoint loaded with non-strict matching. "
                    f"missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)}"
                )
            if "optimizer_state" in state:
                optimizer.load_state_dict(state["optimizer_state"])
            if ema is not None and "ema_state" in state:
                ema.load_state_dict(state["ema_state"])
            start_epoch = int(state.get("epoch", 0)) + 1
            if rank == 0:
                print(f"Resumed from {ckpt_path} at epoch {start_epoch}")
        else:
            if rank == 0:
                print("Resume requested but no checkpoint found. Starting fresh.")

    output_dir = Path(cfg.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "train_config.json", "w", encoding="utf-8") as handle:
            json.dump(cfg.__dict__, handle, indent=2)

    if start_epoch > cfg.epochs:
        if rank == 0:
            print(
                f"Start epoch {start_epoch} is greater than configured epochs {cfg.epochs}. "
                "Nothing to train. Increase --epochs to continue."
            )
        return

    for epoch in range(start_epoch, cfg.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        increments = max(0, (epoch - 1) // max(cfg.curriculum_increase_every, 1))
        current_steps = min(
            cfg.unroll_steps,
            cfg.curriculum_start + increments * cfg.curriculum_increment,
        )
        if cfg.epochs > 1:
            alpha = (epoch - 1) / max(cfg.epochs - 1, 1)
        else:
            alpha = 1.0
        p_tf = cfg.p_tf_start + alpha * (cfg.p_tf_end - cfg.p_tf_start)
        p_tf = float(max(0.0, min(1.0, p_tf)))
        loss = train_epoch(
            model,
            loader,
            optimizer,
            device,
            current_steps,
            cfg.log_interval,
            cfg.div_weight,
            cfg.energy_weight,
            cfg.source_weight,
            cfg.pressure_mean_weight,
            cfg.late_step_weight,
            cfg.state_noise_std,
            cfg.grad_accum_steps,
            cfg.grad_checkpoint,
            cfg.amp,
            scaler if cfg.amp else None,
            p_tf,
            cfg.grad_clip_norm,
            ema,
        )
        if rank == 0:
            print(
                f"Epoch {epoch}/{cfg.epochs} - loss: {loss:.6f} "
                f"(unroll_steps={current_steps}, p_tf={p_tf:.3f})"
            )
            if epoch % max(cfg.checkpoint_interval, 1) == 0 or epoch == cfg.epochs:
                save_checkpoint(output_dir, epoch, model, optimizer, ema)

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
