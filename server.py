#!/usr/bin/env python3
"""TCP PhiFlow backend server for Aerodynamics4MC.

Protocol:
- 4-byte big-endian uint32 length header
- Body: float32 array of shape (N, N, N, C)
  - preferred C=10:
    [obstacle_mask, fan_mask, fan_vx, fan_vy, fan_vz, v_x, v_y, v_z, pressure, thermal_source]
  - standard C=9:
    [obstacle_mask, fan_mask, fan_vx, fan_vy, fan_vz, v_x, v_y, v_z, pressure]
  - legacy C=5:
    [obstacle_mask, v_x, v_y, v_z, pressure]

Response:
- 4-byte big-endian uint32 length header
- Body: float32 array of shape (N, N, N, 4)
  - channels: [v_x, v_y, v_z, pressure]
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

try:
    from phi.flow import (
        Box,
        CenteredGrid,
        Obstacle,
        Solve,
        StaggeredGrid,
        advect,
        diffuse,
        extrapolation,
        fluid,
        math as phi_math,
    )
    from phi.geom import SDFGrid
    from phi.math import Diverged, NotConverged
except Exception as exc:
    raise RuntimeError(
        "PhiFlow is required. Install dependencies with `pip install phiflow torch`."
    ) from exc


LOGGER = logging.getLogger("aero-server")


class SDFGridWithVector(SDFGrid):
    """Compatibility adapter for PhiFlow obstacle projection API."""

    @property
    def vector(self) -> Any:
        return self.bounds.vector


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5001
    grid_size: int = 32
    dt: float = 1.0
    diffusivity: float = 1e-5
    sgs_model: str = "smagorinsky"
    sgs_cs: float = 0.12
    sgs_filter_width: float = 1.0
    sgs_nu_max: float = 0.05
    fan_strength: float = 0.8
    pressure_iterations: int = 24
    pressure_tolerance: float = 3e-3
    projection_interval: int = 4
    max_projection_interval: int = 12
    target_hz: float = 2.0
    log_interval: int = 30
    use_torch_backend: bool = False
    torch_device: str = "cpu"
    checkpoint: str = ""


@dataclass
class ClientRuntime:
    cfg: ServerConfig
    bounds: Any
    vel_extrapolation: Any
    step_index: int = 0
    projection_interval_runtime: int = 0
    obstacle_key: bytes | None = None
    obstacle: Any = None
    timings: deque[float] | None = None
    adapt_cooldown_steps: int = 30
    last_adapt_step: int = -10_000

    def __post_init__(self) -> None:
        self.projection_interval_runtime = max(0, self.cfg.projection_interval)
        self.timings = deque(maxlen=max(10, self.cfg.log_interval))

    def should_project(self) -> bool:
        interval = self.projection_interval_runtime
        return interval > 0 and (self.step_index % interval == 0)

    def update_obstacle(self, obstacle_mask: np.ndarray) -> Any:
        key = obstacle_mask.tobytes(order="C")
        if self.obstacle is not None and self.obstacle_key == key:
            return self.obstacle
        mask_tensor = phi_math.tensor(obstacle_mask, phi_math.spatial("x,y,z"))
        sdf = phi_math.where(mask_tensor > 0.5, -1.0, 1.0)
        self.obstacle = Obstacle(SDFGridWithVector(sdf, bounds=self.bounds))
        self.obstacle_key = key
        return self.obstacle

    def record_timing(self, dt_s: float) -> float:
        assert self.timings is not None
        self.timings.append(dt_s)
        if not self.timings:
            return 0.0
        mean_dt = float(sum(self.timings) / len(self.timings))
        return 1.0 / max(mean_dt, 1e-6)

    def maybe_adapt_projection(self, hz: float) -> None:
        if self.timings is None or len(self.timings) < 8:
            return
        if self.step_index - self.last_adapt_step < self.adapt_cooldown_steps:
            return
        if hz >= self.cfg.target_hz:
            return
        if self.projection_interval_runtime == 0:
            return
        if self.projection_interval_runtime < self.cfg.max_projection_interval:
            self.projection_interval_runtime += 1
            self.last_adapt_step = self.step_index
            LOGGER.warning(
                "Client step-rate %.2fHz < %.2fHz, relaxing projection interval to every %d steps",
                hz,
                self.cfg.target_hz,
                self.projection_interval_runtime,
            )
            return
        self.projection_interval_runtime = 0
        self.last_adapt_step = self.step_index
        LOGGER.warning(
            "Client step-rate %.2fHz < %.2fHz, disabling pressure projection for max speed",
            hz,
            self.cfg.target_hz,
        )


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < num_bytes:
        chunk = sock.recv(num_bytes - received)
        if not chunk:
            raise ConnectionError("Client disconnected")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> bytes:
    header = recv_exact(sock, 4)
    (length,) = struct.unpack(">I", header)
    return recv_exact(sock, length)


def send_message(sock: socket.socket, payload: bytes) -> None:
    header = struct.pack(">I", len(payload))
    sock.sendall(header + payload)


def _infer_grid_channels(float_count: int, expected_grid: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for channels in (10, 9, 5):
        if float_count % channels != 0:
            continue
        voxel_count = float_count // channels
        grid = int(round(voxel_count ** (1.0 / 3.0)))
        if grid * grid * grid == voxel_count:
            candidates.append((grid, channels))

    if not candidates:
        raise ValueError(
            f"Payload float count {float_count} cannot be interpreted as N^3*5/N^3*9/N^3*10"
        )

    for grid, channels in candidates:
        if grid == expected_grid:
            return grid, channels

    # Prefer richer packets when expected grid does not match.
    for grid, channels in candidates:
        if channels == 10:
            return grid, channels
    for grid, channels in candidates:
        if channels == 9:
            return grid, channels

    return candidates[0]


def parse_payload(payload: bytes, grid_size: int) -> tuple[np.ndarray, int]:
    data = np.frombuffer(payload, dtype="<f4")
    packet_grid, channels = _infer_grid_channels(int(data.size), grid_size)
    voxel_count = packet_grid * packet_grid * packet_grid

    packet = data.reshape(packet_grid, packet_grid, packet_grid, channels)
    if channels == 10:
        return packet[..., :9].astype(np.float32, copy=False), packet_grid
    if channels == 9:
        return packet.astype(np.float32, copy=False), packet_grid

    converted = np.zeros((packet_grid, packet_grid, packet_grid, 9), dtype=np.float32)
    converted[..., 0] = packet[..., 0]
    converted[..., 5] = packet[..., 1]
    converted[..., 6] = packet[..., 2]
    converted[..., 7] = packet[..., 3]
    converted[..., 8] = packet[..., 4]
    return converted, packet_grid


def setup_backend(cfg: ServerConfig) -> None:
    if not cfg.use_torch_backend:
        LOGGER.info("Using PhiFlow numpy backend")
        return

    if cfg.torch_device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    try:
        import phi.torch.flow  # noqa: F401
        from phi.math import backend as phi_backend
    except Exception as exc:
        raise RuntimeError(
            "Failed to enable PhiFlow torch backend. Start without `--torch-backend` or check torch/CUDA setup."
        ) from exc

    LOGGER.info(
        "Using PhiFlow torch backend (default=%s, backends=%s)",
        phi_backend.default_backend().name,
        [b.name for b in phi_backend.BACKENDS],
    )


def centered_vector_grid(values: np.ndarray, bounds: Any, vel_extrapolation: Any, grid_size: int) -> Any:
    return CenteredGrid(
        phi_math.tensor(values, phi_math.spatial("x,y,z"), phi_math.channel("vector")),
        bounds=bounds,
        extrapolation=vel_extrapolation,
        x=grid_size,
        y=grid_size,
        z=grid_size,
    )


def centered_scalar_grid(values: np.ndarray, bounds: Any, grid_size: int) -> Any:
    return CenteredGrid(
        phi_math.tensor(values, phi_math.spatial("x,y,z")),
        bounds=bounds,
        extrapolation=extrapolation.ZERO,
        x=grid_size,
        y=grid_size,
        z=grid_size,
    )


def smagorinsky_eddy_viscosity(
    velocity: np.ndarray,
    fluid_mask: np.ndarray,
    cs: float,
    filter_width: float,
    nu_max: float,
) -> np.ndarray:
    """Compute scalar SGS viscosity nu_t using a Smagorinsky closure."""
    ux = velocity[..., 0]
    uy = velocity[..., 1]
    uz = velocity[..., 2]

    dux_dx, dux_dy, dux_dz = np.gradient(ux, edge_order=1)
    duy_dx, duy_dy, duy_dz = np.gradient(uy, edge_order=1)
    duz_dx, duz_dy, duz_dz = np.gradient(uz, edge_order=1)

    s_xx = dux_dx
    s_yy = duy_dy
    s_zz = duz_dz
    s_xy = 0.5 * (dux_dy + duy_dx)
    s_xz = 0.5 * (dux_dz + duz_dx)
    s_yz = 0.5 * (duy_dz + duz_dy)

    # |S| = sqrt(2 * S_ij S_ij) with off-diagonal terms counted twice.
    s_mag = np.sqrt(
        2.0
        * (
            s_xx * s_xx
            + s_yy * s_yy
            + s_zz * s_zz
            + 2.0 * (s_xy * s_xy + s_xz * s_xz + s_yz * s_yz)
        )
    )
    delta = max(1e-6, float(filter_width))
    cs_eff = max(0.0, float(cs))
    nu_t = (cs_eff * delta) ** 2 * s_mag
    nu_t = nu_t.astype(np.float32, copy=False) * fluid_mask.astype(np.float32, copy=False)
    if nu_max > 0:
        nu_t = np.clip(nu_t, 0.0, float(nu_max))
    return nu_t


def laplacian_3d(field: np.ndarray) -> np.ndarray:
    padded = np.pad(field, ((1, 1), (1, 1), (1, 1)), mode="edge")
    center = padded[1:-1, 1:-1, 1:-1]
    return (
        padded[2:, 1:-1, 1:-1]
        + padded[:-2, 1:-1, 1:-1]
        + padded[1:-1, 2:, 1:-1]
        + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 1:-1, 2:]
        + padded[1:-1, 1:-1, :-2]
        - 6.0 * center
    ).astype(np.float32, copy=False)


def apply_smagorinsky_closure(
    velocity_centered: np.ndarray,
    fluid_mask: np.ndarray,
    dt: float,
    cs: float,
    filter_width: float,
    nu_max: float,
) -> np.ndarray:
    """Apply explicit SGS closure: u <- u + dt * nu_t * Laplacian(u)."""
    nu_t = smagorinsky_eddy_viscosity(
        velocity=velocity_centered,
        fluid_mask=fluid_mask,
        cs=cs,
        filter_width=filter_width,
        nu_max=nu_max,
    )
    lap_u = np.empty_like(velocity_centered, dtype=np.float32)
    lap_u[..., 0] = laplacian_3d(velocity_centered[..., 0])
    lap_u[..., 1] = laplacian_3d(velocity_centered[..., 1])
    lap_u[..., 2] = laplacian_3d(velocity_centered[..., 2])

    velocity_sgs = velocity_centered + float(dt) * nu_t[..., None] * lap_u
    velocity_sgs = velocity_sgs * fluid_mask[..., None]
    return velocity_sgs.astype(np.float32, copy=False)


def solve_step(packet: np.ndarray, runtime: ClientRuntime) -> np.ndarray:
    cfg = runtime.cfg
    n = cfg.grid_size

    obstacle_mask = np.clip(packet[..., 0], 0.0, 1.0).astype(np.float32, copy=False)
    fan_mask = np.clip(packet[..., 1], 0.0, 1.0).astype(np.float32, copy=False)
    fan_velocity = packet[..., 2:5].astype(np.float32, copy=False)
    velocity = packet[..., 5:8].astype(np.float32, copy=False)
    pressure = packet[..., 8].astype(np.float32, copy=False)

    fluid_mask = (obstacle_mask <= 0.5).astype(np.float32)
    velocity = velocity * fluid_mask[..., None]
    pressure = pressure * fluid_mask

    vel_centered = centered_vector_grid(velocity, runtime.bounds, runtime.vel_extrapolation, n)
    velocity_staggered = StaggeredGrid(
        vel_centered,
        bounds=runtime.bounds,
        extrapolation=runtime.vel_extrapolation,
        x=n,
        y=n,
        z=n,
    )
    pressure_centered = centered_scalar_grid(pressure, runtime.bounds, n)

    fan_mask_centered = centered_scalar_grid(fan_mask, runtime.bounds, n)
    fan_velocity_centered = centered_vector_grid(fan_velocity, runtime.bounds, runtime.vel_extrapolation, n)
    fan_velocity_staggered = StaggeredGrid(
        fan_velocity_centered,
        bounds=runtime.bounds,
        extrapolation=runtime.vel_extrapolation,
        x=n,
        y=n,
        z=n,
    )

    velocity_staggered = advect.semi_lagrangian(velocity_staggered, velocity_staggered, dt=cfg.dt)
    if cfg.sgs_model == "smagorinsky":
        velocity_centered_np = velocity_staggered.at_centers().values.numpy("x,y,z,vector").astype(np.float32, copy=False)
        velocity_centered_np = apply_smagorinsky_closure(
            velocity_centered=velocity_centered_np,
            fluid_mask=fluid_mask,
            dt=cfg.dt,
            cs=cfg.sgs_cs,
            filter_width=cfg.sgs_filter_width,
            nu_max=cfg.sgs_nu_max,
        )
        velocity_centered = centered_vector_grid(
            velocity_centered_np,
            runtime.bounds,
            runtime.vel_extrapolation,
            n,
        )
        velocity_staggered = StaggeredGrid(
            velocity_centered,
            bounds=runtime.bounds,
            extrapolation=runtime.vel_extrapolation,
            x=n,
            y=n,
            z=n,
        )
    if cfg.diffusivity > 0:
        velocity_staggered = diffuse.explicit(velocity_staggered, diffusivity=cfg.diffusivity, dt=cfg.dt)

    fan_mask_staggered = fan_mask_centered.at(velocity_staggered)
    fan_strength = float(max(0.0, min(1.0, cfg.fan_strength)))
    velocity_staggered = velocity_staggered + fan_mask_staggered * fan_strength * (fan_velocity_staggered - velocity_staggered)

    # Keep pressure cheap in fast path by semi-Lagrangian advection.
    pressure_centered = advect.semi_lagrangian(pressure_centered, velocity_staggered, dt=cfg.dt)

    if runtime.should_project():
        try:
            obstacles = runtime.update_obstacle(obstacle_mask)
            velocity_staggered, pressure_centered = fluid.make_incompressible(
                velocity_staggered,
                obstacles=obstacles,
                solve=Solve(
                    method="CG",
                    abs_tol=cfg.pressure_tolerance,
                    rel_tol=cfg.pressure_tolerance,
                    max_iterations=cfg.pressure_iterations,
                ),
            )
        except (NotConverged, Diverged) as exc:
            LOGGER.debug("Projection skipped due to solver issue: %s", exc)

    out_velocity = velocity_staggered.at_centers().values.numpy("x,y,z,vector").astype(np.float32, copy=False)
    out_pressure = pressure_centered.values.numpy("x,y,z").astype(np.float32, copy=False)
    out_velocity = out_velocity * fluid_mask[..., None]
    out_pressure = out_pressure * fluid_mask

    output = np.empty((n, n, n, 4), dtype=np.float32)
    output[..., :3] = out_velocity
    output[..., 3] = out_pressure
    return output


def handle_client(sock: socket.socket, cfg: ServerConfig) -> None:
    runtime: ClientRuntime | None = None
    try:
        while True:
            payload = recv_message(sock)
            packet, packet_grid = parse_payload(payload, cfg.grid_size)

            if runtime is None or runtime.cfg.grid_size != packet_grid:
                runtime_cfg = replace(cfg, grid_size=packet_grid)
                runtime = ClientRuntime(
                    cfg=runtime_cfg,
                    bounds=Box(x=packet_grid, y=packet_grid, z=packet_grid),
                    vel_extrapolation=extrapolation.combine_sides(
                        x=extrapolation.ZERO_GRADIENT,
                        y=extrapolation.ZERO_GRADIENT,
                        z=extrapolation.ZERO_GRADIENT,
                    ),
                )
                if packet_grid != cfg.grid_size:
                    LOGGER.warning(
                        "Client payload grid=%d differs from --grid-size=%d, using payload grid",
                        packet_grid,
                        cfg.grid_size,
                    )

            t0 = time.perf_counter()
            output = solve_step(packet, runtime)
            elapsed = time.perf_counter() - t0
            runtime.step_index += 1

            hz = runtime.record_timing(elapsed)
            runtime.maybe_adapt_projection(hz)

            if cfg.log_interval > 0 and runtime.step_index % cfg.log_interval == 0:
                LOGGER.info(
                    "Client step %d | dt=%.1fms | %.2fHz | projection_interval=%d",
                    runtime.step_index,
                    elapsed * 1000.0,
                    hz,
                    runtime.projection_interval_runtime,
                )

            send_message(sock, output.astype("<f4", copy=False).tobytes(order="C"))
    except (ConnectionError, ValueError) as exc:
        LOGGER.info("Client disconnected: %s", exc)
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        LOGGER.exception("Unhandled client error: %s", exc)
    finally:
        sock.close()


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="Aerodynamics4MC PhiFlow backend server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=5001, help="Bind port")
    parser.add_argument("--grid-size", type=int, default=32, help="Voxel grid size")
    parser.add_argument("--dt", type=float, default=1.0, help="Semi-Lagrangian timestep")
    parser.add_argument("--diffusivity", type=float, default=1e-5, help="Explicit diffusion coefficient")
    parser.add_argument(
        "--sgs-model",
        choices=["none", "smagorinsky"],
        default="smagorinsky",
        help="Subgrid closure model for LES diffusion",
    )
    parser.add_argument("--sgs-cs", type=float, default=0.12, help="Smagorinsky constant Cs")
    parser.add_argument("--sgs-filter-width", type=float, default=1.0, help="LES filter width in cell units")
    parser.add_argument("--sgs-nu-max", type=float, default=0.05, help="Upper bound for SGS eddy viscosity")
    parser.add_argument("--fan-strength", type=float, default=0.8, help="Relaxation strength toward fan velocity (0-1)")
    parser.add_argument("--pressure-iterations", type=int, default=24, help="Max CG iterations for projection")
    parser.add_argument("--pressure-tolerance", type=float, default=3e-3, help="CG abs/rel tolerance for projection")
    parser.add_argument(
        "--projection-interval",
        type=int,
        default=4,
        help="Run pressure projection every N steps (0 disables projection for max speed)",
    )
    parser.add_argument(
        "--max-projection-interval",
        type=int,
        default=12,
        help="Auto-tuning upper bound before projection is disabled",
    )
    parser.add_argument("--target-hz", type=float, default=2.0, help="Minimum target refresh rate")
    parser.add_argument("--log-interval", type=int, default=30, help="Log performance every N steps")
    parser.add_argument("--torch-backend", action="store_true", help="Use PhiFlow torch backend instead of numpy")
    parser.add_argument("--torch-device", choices=["cpu", "cuda"], default="cpu", help="Torch backend device hint")
    parser.add_argument("--checkpoint", type=str, default="", help="Ignored (compatibility only)")
    args = parser.parse_args()
    return ServerConfig(
        host=args.host,
        port=args.port,
        grid_size=args.grid_size,
        dt=args.dt,
        diffusivity=args.diffusivity,
        sgs_model=args.sgs_model,
        sgs_cs=max(0.0, args.sgs_cs),
        sgs_filter_width=max(1e-6, args.sgs_filter_width),
        sgs_nu_max=max(0.0, args.sgs_nu_max),
        fan_strength=args.fan_strength,
        pressure_iterations=max(0, args.pressure_iterations),
        pressure_tolerance=max(1e-8, args.pressure_tolerance),
        projection_interval=max(0, args.projection_interval),
        max_projection_interval=max(0, args.max_projection_interval),
        target_hz=max(0.1, args.target_hz),
        log_interval=max(0, args.log_interval),
        use_torch_backend=args.torch_backend,
        torch_device=args.torch_device,
        checkpoint=args.checkpoint,
    )


def serve(cfg: ServerConfig) -> None:
    setup_backend(cfg)
    if cfg.checkpoint:
        LOGGER.info("Ignoring --checkpoint=%s (PhiFlow backend does not use model checkpoints)", cfg.checkpoint)
    LOGGER.info(
        "PhiFlow server config: grid=%d, projection_interval=%d, pressure_iter=%d, target_hz=%.2f, sgs=%s",
        cfg.grid_size,
        cfg.projection_interval,
        cfg.pressure_iterations,
        cfg.target_hz,
        cfg.sgs_model,
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((cfg.host, cfg.port))
        server.listen()
        LOGGER.info("Listening on %s:%s", cfg.host, cfg.port)
        while True:
            client, addr = server.accept()
            LOGGER.info("Client connected: %s:%s", addr[0], addr[1])
            thread = threading.Thread(target=handle_client, args=(client, cfg), daemon=True)
            thread.start()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    cfg = parse_args()
    serve(cfg)


if __name__ == "__main__":
    main()
