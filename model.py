"""3D U-Net with residual blocks for CFD surrogate modeling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Literal

import torch
from torch import nn
import torch.nn.functional as F


PaddingMode = Literal["replicate", "reflect", "circular", "constant"]


def pad3d(x: torch.Tensor, pad: int, mode: PaddingMode, value: float = 0.0) -> torch.Tensor:
    if pad <= 0:
        return x
    if mode == "constant":
        return F.pad(x, (pad, pad, pad, pad, pad, pad), mode=mode, value=value)
    return F.pad(x, (pad, pad, pad, pad, pad, pad), mode=mode)


def masked_mean_std(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: (N,C,D,H,W), mask: (N,1,D,H,W) with 0/1
    m = mask.to(dtype=x.dtype)
    m_sum = m.sum(dim=(0, 2, 3, 4), keepdim=True).clamp_min(eps)  # (1,1,1,1,1)
    mean = (x * m).sum(dim=(0, 2, 3, 4), keepdim=True) / m_sum     # (1,C,1,1,1)
    var = ((x - mean) ** 2 * m).sum(dim=(0, 2, 3, 4), keepdim=True) / m_sum
    std = torch.sqrt(var + eps)
    return mean, std


class MaskedRunningNorm3D(nn.Module):
    """EMA running stats computed only on fluid region (mask=1)."""
    def __init__(
        self,
        channels: int,
        momentum: float = 0.02,   # slightly higher for small grids
        eps: float = 1e-5,
        init_mean: Optional[Sequence[float]] = None,
        init_std: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        mean = torch.zeros(1, channels, 1, 1, 1, dtype=torch.float32)
        std = torch.ones(1, channels, 1, 1, 1, dtype=torch.float32)
        if init_mean is not None:
            mean[:, : len(init_mean)] = torch.tensor(init_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
        if init_std is not None:
            std[:, : len(init_std)] = torch.tensor(init_std, dtype=torch.float32).view(1, -1, 1, 1, 1)

        self.register_buffer("running_mean", mean)
        self.register_buffer("running_std", std)
        self.momentum = momentum
        self.eps = eps

    @torch.no_grad()
    def _update(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        mean, std = masked_mean_std(x, mask, eps=self.eps)
        self.running_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
        self.running_std.mul_(1 - self.momentum).add_(std * self.momentum)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update(x, mask)
        return (x - self.running_mean) / self.running_std.clamp_min(self.eps)

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.running_std + self.running_mean


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        pad: int = 1,
        groups: int = 8,
        act_slope: float = 0.1,
        padding_mode: PaddingMode = "replicate",
    ) -> None:
        super().__init__()
        self.pad = pad
        self.padding_mode = padding_mode
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=0, bias=False)  # no bias helps Neumann stability
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.LeakyReLU(act_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pad3d(x, self.pad, self.padding_mode)
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        return x


class ResBlock3D(nn.Module):
    def __init__(self, ch: int, groups: int = 8, padding_mode: PaddingMode = "replicate") -> None:
        super().__init__()
        self.c1 = ConvGNAct(ch, ch, 3, 1, groups, 0.1, padding_mode)
        self.pad = 1
        self.padding_mode = padding_mode
        self.c2 = nn.Conv3d(ch, ch, kernel_size=3, padding=0, bias=False)
        g = min(groups, ch)
        while ch % g != 0 and g > 1:
            g -= 1
        self.gn2 = nn.GroupNorm(g, ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.c1(x)
        x = pad3d(x, self.pad, self.padding_mode)
        x = self.c2(x)
        x = self.gn2(x)
        x = self.act(x + r)
        return x


class DownStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_res: int, groups: int, padding_mode: PaddingMode) -> None:
        super().__init__()
        self.in_proj = ConvGNAct(in_ch, out_ch, 3, 1, groups, 0.1, padding_mode)
        self.res = nn.Sequential(*[ResBlock3D(out_ch, groups, padding_mode) for _ in range(n_res)])
        # anti-alias downsample
        self.pool = nn.AvgPool3d(2, 2)
        self.post = ConvGNAct(out_ch, out_ch, 1, 0, groups, 0.1, padding_mode)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_proj(x)
        x = self.res(x)
        skip = x
        x = self.pool(x)
        x = self.post(x)
        return x, skip


class UpStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        n_res: int,
        groups: int,
        padding_mode: PaddingMode,
        up_mode: Literal["trilinear", "nearest", "transpose"] = "trilinear",
    ) -> None:
        super().__init__()
        self.up_mode = up_mode
        if up_mode == "transpose":
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=False)
        else:
            self.up = nn.Conv3d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)

        self.fuse = ConvGNAct(out_ch + skip_ch, out_ch, 3, 1, groups, 0.1, padding_mode)
        self.res = nn.Sequential(*[ResBlock3D(out_ch, groups, padding_mode) for _ in range(n_res)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.up_mode == "transpose":
            x = self.up(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.up_mode, align_corners=False if self.up_mode == "trilinear" else None)
            x = self.up(x)

        # crop safe
        if x.shape[-3:] != skip.shape[-3:]:
            d = min(x.shape[-3], skip.shape[-3])
            h = min(x.shape[-2], skip.shape[-2])
            w = min(x.shape[-1], skip.shape[-1])
            x = x[..., :d, :h, :w]
            skip = skip[..., :d, :h, :w]

        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.res(x)
        return x

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


def curl3d_neumann(A: torch.Tensor, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0) -> torch.Tensor:
    """
    A: (N,3,D,H,W) vector potential
    returns u = curl(A): (N,3,D,H,W)
    Neumann-friendly finite differences:
      - interior: central differences
      - boundary: one-sided differences (compatible with zero-gradient extension)
    """
    Ax = A[:, 0:1]
    Ay = A[:, 1:2]
    Az = A[:, 2:3]

    u = ddy_neumann(Az, dy=dy) - ddz_neumann(Ay, dz=dz)
    v = ddz_neumann(Ax, dz=dz) - ddx_neumann(Az, dx=dx)
    w = ddx_neumann(Ay, dx=dx) - ddy_neumann(Ax, dy=dy)
    return torch.cat([u, v, w], dim=1)


@dataclass
class SteadyRolloutCFDUNetCfg:
    # Prefer 9 channels:
    # [obstacle_mask, fan_mask, fan_vx, fan_vy, fan_vz, u, v, w, p]
    # Still supports 5-channel legacy input [obstacle_mask, u, v, w, p].
    in_channels: int = 9
    out_channels: int = 4  # (u,v,w,p)
    base_channels: int = 48
    depth: int = 4
    res_per_stage: int = 2
    groups: int = 8
    padding_mode: PaddingMode = "replicate"  # Neumann-friendly
    up_mode: Literal["trilinear", "nearest", "transpose"] = "trilinear"
    norm_momentum: float = 0.02
    vel_delta_clip: float = 2.5
    pressure_delta_clip: float = 3.0
    source_inject_strength: float = 0.1


class SteadyRolloutCFDUNet(nn.Module):
    """
    Designed for: incompressible, steady(ish) iterative rollout, Neumann BC, 32^3.
    - Input (preferred): [obstacle, fan_mask, fan_vx, fan_vy, fan_vz, u,v,w,p]
    - Input (legacy):    [obstacle, u,v,w,p]
    - Predicts residual update on top of current state for long-horizon stability.
    """
    def __init__(self, cfg: SteadyRolloutCFDUNetCfg) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.in_channels not in (5, 9):
            raise ValueError(f"in_channels must be 5 or 9, got {cfg.in_channels}")
        if cfg.out_channels != 4:
            raise ValueError(f"out_channels must be 4, got {cfg.out_channels}")

        chs = [cfg.base_channels * (2 ** i) for i in range(cfg.depth)]

        # normalize physical channels only (u,v,w,p)
        self.in_norm = MaskedRunningNorm3D(channels=4, momentum=cfg.norm_momentum)

        self.stem = ConvGNAct(cfg.in_channels, chs[0], 3, 1, cfg.groups, 0.1, cfg.padding_mode)

        self.down = nn.ModuleList()
        for i in range(cfg.depth - 1):
            self.down.append(DownStage(chs[i], chs[i + 1], cfg.res_per_stage, cfg.groups, cfg.padding_mode))

        self.bottleneck = nn.Sequential(*[ResBlock3D(chs[-1], cfg.groups, cfg.padding_mode) for _ in range(cfg.res_per_stage)])

        self.up = nn.ModuleList()
        for i in range(cfg.depth - 2, -1, -1):
            self.up.append(UpStage(chs[i + 1], chs[i + 1], chs[i], cfg.res_per_stage, cfg.groups, cfg.padding_mode, cfg.up_mode))

        # head predicts (Ax,Ay,Az,p)
        self.head = nn.Conv3d(chs[0], 4, kernel_size=1, padding=0, bias=False)

    @staticmethod
    def _split_channels(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns mask, fan_mask, fan_velocity(3), state(uvwp)
        if x.shape[1] >= 9:
            mask = x[:, 0:1]
            fan_mask = x[:, 1:2]
            fan_velocity = x[:, 2:5]
            state = x[:, 5:9]
            return mask, fan_mask, fan_velocity, state
        if x.shape[1] >= 5:
            mask = x[:, 0:1]
            state = x[:, 1:5]
            fan_mask = torch.zeros_like(mask)
            fan_velocity = torch.zeros_like(state[:, :3])
            return mask, fan_mask, fan_velocity, state
        raise ValueError(f"Expected input with at least 5 channels, got {x.shape[1]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask, fan_mask, fan_velocity, state = self._split_channels(x)
        mask = mask.clamp(0, 1)
        fluid_mask = 1.0 - mask
        fan_mask = fan_mask.clamp(0, 1) * fluid_mask
        fan_velocity = fan_velocity * fan_mask

        state_norm = self.in_norm(state, fluid_mask)
        if self.cfg.in_channels == 9:
            x = torch.cat([mask, fan_mask, fan_velocity, state_norm], dim=1)
        else:
            x = torch.cat([mask, state_norm], dim=1)

        x = self.stem(x)
        # soft gate to reduce solid leakage
        x = x * (0.05 + 0.95 * fluid_mask)

        skips: List[torch.Tensor] = []
        for d in self.down:
            x, s = d(x)
            s_fluid_mask = F.interpolate(fluid_mask, size=s.shape[-3:], mode="nearest")
            skips.append(s * (0.05 + 0.95 * s_fluid_mask))

        x = self.bottleneck(x)

        for u, s in zip(self.up, reversed(skips)):
            x = u(x, s)

        y = self.head(x)
        delta_A = torch.tanh(y[:, 0:3]) * self.cfg.vel_delta_clip
        delta_p = torch.tanh(y[:, 3:4]) * self.cfg.pressure_delta_clip
        delta_vel = curl3d_neumann(delta_A)

        cur_vel = state[:, 0:3]
        cur_p = state[:, 3:4]
        vel = cur_vel + delta_vel
        p = cur_p + delta_p
        if self.cfg.source_inject_strength > 0:
            vel = vel + self.cfg.source_inject_strength * fan_mask * fan_velocity

        out = torch.cat([vel, p], dim=1)
        # hard mask: solid region outputs = 0
        out = out * fluid_mask
        return out
