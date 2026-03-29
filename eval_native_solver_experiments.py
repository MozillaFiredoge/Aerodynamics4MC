#!/usr/bin/env python3
"""Benchmark the native C++ solver and generate an Experiments-style report."""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-aerodynamics4mc")

import matplotlib.pyplot as plt
import numpy as np


INPUT_CHANNELS = 10
OUTPUT_CHANNELS = 4
NATIVE_VELOCITY_SCALE = 30.0

BENCHMARK_ABI_VERSION = 1

CYLINDER_BENCHMARK_LENGTH = 2.2
CYLINDER_BENCHMARK_HEIGHT = 0.41
CYLINDER_BENCHMARK_DIAMETER = 0.1
CYLINDER_BENCHMARK_RADIUS = 0.5 * CYLINDER_BENCHMARK_DIAMETER
CYLINDER_BENCHMARK_CENTER_X = 0.2
CYLINDER_BENCHMARK_CENTER_Y = 0.2
CYLINDER_BENCHMARK_REYNOLDS = 100.0
CYLINDER_BENCHMARK_UMEAN = 0.08
CYLINDER_BENCHMARK_UMAX = 1.5 * CYLINDER_BENCHMARK_UMEAN
CYLINDER_BENCHMARK_ST_RANGE = (0.1, 0.5)

AERO_LBM_BENCHMARK_PRESET_NONE = 0
AERO_LBM_BENCHMARK_PRESET_TAYLOR_GREEN_3D = 1
AERO_LBM_BENCHMARK_PRESET_LID_DRIVEN_CAVITY_2D = 2
AERO_LBM_BENCHMARK_PRESET_LID_DRIVEN_CAVITY_3D = 3
AERO_LBM_BENCHMARK_PRESET_CYLINDER_CROSSFLOW_2D = 4
AERO_LBM_BENCHMARK_PRESET_DIFFERENTIALLY_HEATED_CAVITY_2D = 5
AERO_LBM_BENCHMARK_PRESET_DIFFERENTIALLY_HEATED_CAVITY_3D = 6

CH_OBS = 0
CH_FAN_MASK = 1
CH_FAN_VX = 2
CH_FAN_VY = 3
CH_FAN_VZ = 4
CH_VX = 5
CH_VY = 6
CH_VZ = 7
CH_P = 8
CH_THERM = 9

MODE_AUTO = "auto"
MODE_CPU = "cpu"
MODES = (MODE_AUTO, MODE_CPU)

WORKLOAD_QUIESCENT = "quiescent"
WORKLOAD_MIXED = "mixed_flow"
PERF_WORKLOADS = (WORKLOAD_QUIESCENT, WORKLOAD_MIXED)

CASE_QUIESCENT = "quiescent_stability"
CASE_SYMMETRY = "dual_jet_symmetry"
CASE_OBSTACLE = "obstacle_impermeability"
CASE_BUOYANCY = "buoyancy_plume"
CASE_TAYLOR_GREEN = "taylor_green_decay"
CASE_LID_CAVITY_2D = "lid_driven_cavity_2d"
CASE_CYLINDER_CROSSFLOW = "cylinder_crossflow_2d"
DIAGNOSTIC_CASES = (
    CASE_QUIESCENT,
    CASE_SYMMETRY,
    CASE_OBSTACLE,
    CASE_BUOYANCY,
    CASE_TAYLOR_GREEN,
    CASE_LID_CAVITY_2D,
    CASE_CYLINDER_CROSSFLOW,
)


class AeroLbmTimingSnapshot(ctypes.Structure):
    _fields_ = [
        ("ticks", ctypes.c_uint64),
        ("last_payload_copy_ms", ctypes.c_double),
        ("last_solver_ms", ctypes.c_double),
        ("last_readback_ms", ctypes.c_double),
        ("last_total_ms", ctypes.c_double),
        ("avg_payload_copy_ms", ctypes.c_double),
        ("avg_solver_ms", ctypes.c_double),
        ("avg_readback_ms", ctypes.c_double),
        ("avg_total_ms", ctypes.c_double),
    ]


class AeroLbmBoundaryFaceConfig(ctypes.Structure):
    _fields_ = [
        ("hydrodynamic_kind", ctypes.c_int),
        ("thermal_kind", ctypes.c_int),
        ("velocity", ctypes.c_float * 3),
        ("pressure", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("heat_flux", ctypes.c_float),
    ]


class AeroLbmBenchmarkConfig(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("enabled", ctypes.c_int),
        ("preset", ctypes.c_int),
        ("reserved0", ctypes.c_int),
        ("reynolds_number", ctypes.c_float),
        ("rayleigh_number", ctypes.c_float),
        ("prandtl_number", ctypes.c_float),
        ("mach_number", ctypes.c_float),
        ("reference_density", ctypes.c_float),
        ("reference_temperature", ctypes.c_float),
        ("reference_length", ctypes.c_float),
        ("body_force", ctypes.c_float * 3),
        ("gravity", ctypes.c_float * 3),
        ("initial_velocity", ctypes.c_float * 3),
        ("initial_pressure", ctypes.c_float),
        ("initial_temperature", ctypes.c_float),
        ("x_min", AeroLbmBoundaryFaceConfig),
        ("x_max", AeroLbmBoundaryFaceConfig),
        ("y_min", AeroLbmBoundaryFaceConfig),
        ("y_max", AeroLbmBoundaryFaceConfig),
        ("z_min", AeroLbmBoundaryFaceConfig),
        ("z_max", AeroLbmBoundaryFaceConfig),
        ("reserved", ctypes.c_uint32 * 8),
    ]


@dataclass
class RuntimeMeasurement:
    mode: str
    runtime_info: str
    grid_size: int
    grid_shape: GridShape
    workload: str
    warmup_steps: int
    measure_steps: int
    external_step_ms_mean: float
    external_step_ms_std: float
    external_step_ms_p95: float
    native_solver_ms_mean: float
    native_total_ms_mean: float
    mcups: float
    mlups: float
    last_max_speed: float
    final_kinetic_energy: float
    mean_divergence_rms: float


@dataclass
class DiagnosticMeasurement:
    mode: str
    runtime_info: str
    grid_size: int
    grid_shape: GridShape
    case: str
    steps: int
    all_finite: bool
    final_divergence_rms: float
    max_speed: float
    history: list[dict[str, Any]]
    extra_summary: dict[str, float]


@dataclass(frozen=True)
class CylinderChannelGeometry:
    y_min: float
    y_max: float
    height: float
    center: tuple[float, float]
    radius: float
    diameter: float


GridShape = tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the native Aero LBM solver and generate a report.")
    parser.add_argument(
        "--library",
        type=str,
        default="",
        help="Explicit path to native library. If omitted, auto-detect under fabric-mod/native.",
    )
    parser.add_argument(
        "--perf-grid-sizes",
        type=str,
        default="16,24,32,48",
        help="Comma-separated grid sizes for performance sweep.",
    )
    parser.add_argument(
        "--diag-grid-size",
        type=int,
        default=32,
        help="Grid size for numerical diagnostics.",
    )
    parser.add_argument("--perf-warmup", type=int, default=12, help="Warmup steps per performance run.")
    parser.add_argument("--perf-steps", type=int, default=48, help="Measured steps per performance run.")
    parser.add_argument("--diag-steps", type=int, default=48, help="Steps per diagnostic case.")
    parser.add_argument(
        "--modes",
        type=str,
        default="auto,cpu",
        help="Comma-separated runtime modes: auto,cpu",
    )
    parser.add_argument(
        "--diag-cases",
        type=str,
        default="all",
        help=f"Comma-separated diagnostic cases or 'all': {','.join(DIAGNOSTIC_CASES)}",
    )
    parser.add_argument(
        "--skip-perf",
        action="store_true",
        help="Skip the performance sweep and run diagnostics only.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/native_solver_experiments",
        help="Directory for JSON, figures, and report.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Run experiments and save JSON only.",
    )
    return parser.parse_args()


def parse_modes(spec: str) -> list[str]:
    modes = [part.strip() for part in spec.split(",") if part.strip()]
    if not modes:
        raise ValueError("No runtime modes selected")
    bad = [mode for mode in modes if mode not in MODES]
    if bad:
        raise ValueError(f"Unsupported mode(s): {', '.join(bad)}")
    return modes


def parse_grid_sizes(spec: str) -> list[int]:
    grids = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        grids.append(int(part))
    if not grids:
        raise ValueError("No grid sizes selected")
    return grids


def parse_diag_cases(spec: str) -> list[str]:
    raw = [part.strip() for part in spec.split(",") if part.strip()]
    if not raw or raw == ["all"]:
        return list(DIAGNOSTIC_CASES)
    bad = [case for case in raw if case not in DIAGNOSTIC_CASES]
    if bad:
        raise ValueError(f"Unsupported diagnostic case(s): {', '.join(bad)}")
    return raw


def grid_shape_to_str(shape: GridShape) -> str:
    return f"{shape[0]}x{shape[1]}x{shape[2]}"


def detect_library_path(explicit: str) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    candidates = [
        Path("fabric-mod/native/build/libaero_lbm.so"),
        Path("fabric-mod/native/build/libaero_lbm.dylib"),
        Path("fabric-mod/native/build/aero_lbm.dll"),
        Path("fabric-mod/native/build-opencl-test/libaero_lbm.so"),
        Path("fabric-mod/native/build-cpu-test/libaero_lbm.so"),
        Path("fabric-mod/native/build-check/libaero_lbm.so"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not auto-detect native library under fabric-mod/native")


class NativeSolverLib:
    def __init__(self, library_path: Path) -> None:
        self.library_path = library_path.resolve()
        self.lib = ctypes.CDLL(str(self.library_path))

        self.lib.aero_lbm_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.aero_lbm_init.restype = ctypes.c_int
        self.lib.aero_lbm_init_rect.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.aero_lbm_init_rect.restype = ctypes.c_int

        self.lib.aero_lbm_step.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.aero_lbm_step.restype = ctypes.c_int
        self.lib.aero_lbm_step_rect.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.aero_lbm_step_rect.restype = ctypes.c_int

        self.lib.aero_lbm_get_last_force.argtypes = [
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.aero_lbm_get_last_force.restype = ctypes.c_int

        self.lib.aero_lbm_release_context.argtypes = [ctypes.c_longlong]
        self.lib.aero_lbm_release_context.restype = None

        self.lib.aero_lbm_shutdown.argtypes = []
        self.lib.aero_lbm_shutdown.restype = None

        self.lib.aero_lbm_runtime_info.argtypes = []
        self.lib.aero_lbm_runtime_info.restype = ctypes.c_char_p

        self.lib.aero_lbm_timing_info.argtypes = []
        self.lib.aero_lbm_timing_info.restype = ctypes.c_char_p

        self.lib.aero_lbm_reset_timing.argtypes = []
        self.lib.aero_lbm_reset_timing.restype = None

        self.lib.aero_lbm_get_timing_snapshot.argtypes = [ctypes.POINTER(AeroLbmTimingSnapshot)]
        self.lib.aero_lbm_get_timing_snapshot.restype = ctypes.c_int

        self.lib.aero_lbm_benchmark_default_config.argtypes = [ctypes.POINTER(AeroLbmBenchmarkConfig)]
        self.lib.aero_lbm_benchmark_default_config.restype = None

        self.lib.aero_lbm_benchmark_default_preset_config.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(AeroLbmBenchmarkConfig),
        ]
        self.lib.aero_lbm_benchmark_default_preset_config.restype = ctypes.c_int

        self.lib.aero_lbm_benchmark_set_config.argtypes = [ctypes.POINTER(AeroLbmBenchmarkConfig)]
        self.lib.aero_lbm_benchmark_set_config.restype = ctypes.c_int

        self.lib.aero_lbm_benchmark_get_config.argtypes = [ctypes.POINTER(AeroLbmBenchmarkConfig)]
        self.lib.aero_lbm_benchmark_get_config.restype = ctypes.c_int

        self.lib.aero_lbm_benchmark_reset_config.argtypes = []
        self.lib.aero_lbm_benchmark_reset_config.restype = None

        self.lib.aero_lbm_benchmark_info.argtypes = []
        self.lib.aero_lbm_benchmark_info.restype = ctypes.c_char_p

    def initialize(self, grid_shape: GridShape, mode: str) -> str:
        if mode == MODE_CPU:
            os.environ["AERO_LBM_CPU_ONLY"] = "1"
        else:
            os.environ.pop("AERO_LBM_CPU_ONLY", None)
        nx, ny, nz = grid_shape
        if nx == ny == nz:
            ok = self.lib.aero_lbm_init(nx, INPUT_CHANNELS, OUTPUT_CHANNELS)
        else:
            ok = self.lib.aero_lbm_init_rect(nx, ny, nz, INPUT_CHANNELS, OUTPUT_CHANNELS)
        if ok != 1:
            raise RuntimeError(f"aero_lbm_init failed for grid={grid_shape} mode={mode}")
        return self.runtime_info()

    def runtime_info(self) -> str:
        raw = self.lib.aero_lbm_runtime_info()
        return raw.decode("utf-8") if raw else "unknown"

    def timing_info(self) -> str:
        raw = self.lib.aero_lbm_timing_info()
        return raw.decode("utf-8") if raw else "ticks=0"

    def reset_timing(self) -> None:
        self.lib.aero_lbm_reset_timing()

    def timing_snapshot(self) -> AeroLbmTimingSnapshot:
        snapshot = AeroLbmTimingSnapshot()
        ok = self.lib.aero_lbm_get_timing_snapshot(ctypes.byref(snapshot))
        if ok != 1:
            raise RuntimeError("aero_lbm_get_timing_snapshot failed")
        return snapshot

    def benchmark_default_config(self) -> AeroLbmBenchmarkConfig:
        config = AeroLbmBenchmarkConfig()
        self.lib.aero_lbm_benchmark_default_config(ctypes.byref(config))
        return config

    def benchmark_default_preset_config(self, preset: int) -> AeroLbmBenchmarkConfig:
        config = AeroLbmBenchmarkConfig()
        ok = self.lib.aero_lbm_benchmark_default_preset_config(preset, ctypes.byref(config))
        if ok != 1:
            raise RuntimeError(f"aero_lbm_benchmark_default_preset_config failed for preset={preset}")
        return config

    def benchmark_set_config(self, config: AeroLbmBenchmarkConfig) -> None:
        config.abi_version = BENCHMARK_ABI_VERSION
        config.struct_size = ctypes.sizeof(AeroLbmBenchmarkConfig)
        ok = self.lib.aero_lbm_benchmark_set_config(ctypes.byref(config))
        if ok != 1:
            raise RuntimeError("aero_lbm_benchmark_set_config failed")

    def benchmark_get_config(self) -> AeroLbmBenchmarkConfig:
        config = AeroLbmBenchmarkConfig()
        ok = self.lib.aero_lbm_benchmark_get_config(ctypes.byref(config))
        if ok != 1:
            raise RuntimeError("aero_lbm_benchmark_get_config failed")
        return config

    def benchmark_reset_config(self) -> None:
        self.lib.aero_lbm_benchmark_reset_config()

    def benchmark_info(self) -> str:
        raw = self.lib.aero_lbm_benchmark_info()
        return raw.decode("utf-8") if raw else "benchmark=unavailable"

    def step(self, packet: np.ndarray, context_key: int, output: np.ndarray) -> bool:
        packet_ptr = packet.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        nx, ny, nz = packet.shape[:3]
        if nx == ny == nz:
            return self.lib.aero_lbm_step(packet_ptr, nx, context_key, output_ptr) == 1
        return self.lib.aero_lbm_step_rect(packet_ptr, nx, ny, nz, context_key, output_ptr) == 1

    def release_context(self, context_key: int) -> None:
        self.lib.aero_lbm_release_context(context_key)

    def get_last_force(self, context_key: int) -> tuple[float, float, float]:
        fx = ctypes.c_float()
        fy = ctypes.c_float()
        fz = ctypes.c_float()
        ok = self.lib.aero_lbm_get_last_force(
            ctypes.c_longlong(context_key),
            ctypes.byref(fx),
            ctypes.byref(fy),
            ctypes.byref(fz),
        )
        if ok != 1:
            raise RuntimeError("aero_lbm_get_last_force failed")
        return float(fx.value), float(fy.value), float(fz.value)

    def shutdown(self) -> None:
        self.lib.aero_lbm_shutdown()


def idx(x: int, y: int, z: int, n: int) -> int:
    return (x * n + y) * n + z


def add_box_obstacle(packet: np.ndarray, lo: tuple[int, int, int], hi: tuple[int, int, int]) -> None:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    packet[x0:x1, y0:y1, z0:z1, CH_OBS] = 1.0


def add_fan_disk(packet: np.ndarray, center: tuple[int, int, int], axis: str, sign: int, radius: int, speed: float, thickness: int = 1) -> None:
    n = packet.shape[0]
    cx, cy, cz = center
    grid_x, grid_y, grid_z = np.indices((n, n, n))
    radius_sq = radius * radius
    speed = speed if sign >= 0 else -speed
    thickness = max(1, thickness)

    if axis == "x":
        mask = (np.abs(grid_x - cx) < thickness) & (((grid_y - cy) ** 2 + (grid_z - cz) ** 2) <= radius_sq)
        packet[mask, CH_FAN_MASK] = 1.0
        packet[mask, CH_FAN_VX] = speed
    elif axis == "y":
        mask = (np.abs(grid_y - cy) < thickness) & (((grid_x - cx) ** 2 + (grid_z - cz) ** 2) <= radius_sq)
        packet[mask, CH_FAN_MASK] = 1.0
        packet[mask, CH_FAN_VY] = speed
    elif axis == "z":
        mask = (np.abs(grid_z - cz) < thickness) & (((grid_x - cx) ** 2 + (grid_y - cy) ** 2) <= radius_sq)
        packet[mask, CH_FAN_MASK] = 1.0
        packet[mask, CH_FAN_VZ] = speed
    else:
        raise ValueError(f"Unsupported axis: {axis}")


def add_thermal_blob(packet: np.ndarray, center: tuple[int, int, int], radius: int, source: float) -> None:
    n = packet.shape[0]
    cx, cy, cz = center
    grid_x, grid_y, grid_z = np.indices((n, n, n))
    dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2 + (grid_z - cz) ** 2
    mask = dist_sq <= radius * radius
    packet[mask, CH_THERM] = np.maximum(packet[mask, CH_THERM], source)


def add_cylinder_obstacle(packet: np.ndarray, center_xy: tuple[float, float], radius: float) -> None:
    nx, ny = packet.shape[0], packet.shape[1]
    cx, cy = center_xy
    grid_x, grid_y = np.indices((nx, ny))
    dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    mask_2d = dist_sq <= radius * radius
    packet[mask_2d, :, CH_OBS] = 1.0


def benchmark_grid_shape(case_name: str, grid_size: int) -> GridShape:
    if case_name == CASE_CYLINDER_CROSSFLOW:
        nx = int(grid_size)
        ny = max(16, int(round(CYLINDER_BENCHMARK_HEIGHT / CYLINDER_BENCHMARK_LENGTH * (nx - 1))) + 1)
        nz = 8
        return (nx, ny, nz)
    if case_name == CASE_LID_CAVITY_2D:
        return (int(grid_size), int(grid_size), 8)
    return (int(grid_size), int(grid_size), int(grid_size))


def cylinder_channel_geometry(shape: GridShape) -> CylinderChannelGeometry:
    nx, ny, _ = shape
    length_cells = float(max(1, nx - 1))
    height_cells = float(max(1, ny - 1))
    diameter = max(
        3.0,
        CYLINDER_BENCHMARK_DIAMETER / CYLINDER_BENCHMARK_LENGTH * length_cells,
    )
    radius = 0.5 * diameter
    center_x = CYLINDER_BENCHMARK_CENTER_X / CYLINDER_BENCHMARK_LENGTH * length_cells
    center_y = CYLINDER_BENCHMARK_CENTER_Y / CYLINDER_BENCHMARK_HEIGHT * height_cells
    return CylinderChannelGeometry(
        y_min=0.0,
        y_max=height_cells,
        height=height_cells,
        center=(center_x, center_y),
        radius=radius,
        diameter=2.0 * radius,
    )


def cylinder_geometry(shape: GridShape) -> tuple[tuple[float, float], float]:
    geom = cylinder_channel_geometry(shape)
    return geom.center, geom.radius


def cylinder_inlet_profile(shape: GridShape, u_max: float = CYLINDER_BENCHMARK_UMAX) -> np.ndarray:
    geom = cylinder_channel_geometry(shape)
    _, ny, _ = shape
    coords = np.arange(ny, dtype=np.float64)
    s = coords / max(geom.height, 1.0)
    profile = 4.0 * u_max * s * (1.0 - s)
    return profile.astype(np.float32)


def apply_cylinder_seed_perturbation(packet: np.ndarray, center_xy: tuple[float, float], radius: float, step: int) -> None:
    if step >= 192:
        return
    cx, cy = center_xy
    nx, ny = packet.shape[0], packet.shape[1]
    grid_x, grid_y = np.indices((nx, ny), dtype=np.float64)
    sigma_x = max(2.0, 1.5 * radius)
    sigma_y = max(2.0, 2.0 * radius)
    amplitude = 1.5e-2 * math.exp(-step / 48.0)
    perturb = np.exp(-((grid_x - (cx + 2.0 * radius)) ** 2) / (2.0 * sigma_x * sigma_x))
    perturb *= np.exp(-((grid_y - cy) ** 2) / (2.0 * sigma_y * sigma_y))
    perturb *= (grid_y - cy) / max(sigma_y, 1.0)
    perturb *= amplitude
    packet[:, :, :, CH_VY] += perturb[:, :, None].astype(np.float32)
    solid = packet[..., CH_OBS] > 0.5
    packet[solid, CH_VY] = 0.0


def build_taylor_green_packet(shape: GridShape, amplitude: float = 0.03) -> np.ndarray:
    nx, ny, nz = shape
    if nx != ny or ny != nz:
        raise ValueError("Taylor-Green benchmark requires a cubic grid")
    packet = np.zeros((nx, ny, nz, INPUT_CHANNELS), dtype=np.float32)
    coords = np.arange(nx, dtype=np.float64)
    x = (2.0 * math.pi * coords / nx)[:, None, None]
    y = (2.0 * math.pi * coords / ny)[None, :, None]
    z = (2.0 * math.pi * coords / nz)[None, None, :]
    packet[..., CH_VX] = (amplitude * np.sin(x) * np.cos(y) * np.cos(z)).astype(np.float32)
    packet[..., CH_VY] = (-amplitude * np.cos(x) * np.sin(y) * np.cos(z)).astype(np.float32)
    return packet


def build_case_packet(case_name: str, shape: GridShape, step: int) -> np.ndarray:
    nx, ny, nz = shape
    packet = np.zeros((nx, ny, nz, INPUT_CHANNELS), dtype=np.float32)
    n = nx
    radius = max(2, min(nx, ny, nz) // 8)
    thickness = max(1, min(nx, ny, nz) // 32)

    if case_name == WORKLOAD_QUIESCENT or case_name == CASE_QUIESCENT:
        return packet

    if case_name == WORKLOAD_MIXED:
        add_fan_disk(packet, (1, n // 2, n // 2), "x", 1, radius, 8.0, thickness)
        add_box_obstacle(packet, (n // 2 - max(2, n // 10), n // 2 - max(3, n // 6), n // 2 - max(3, n // 6)),
                         (n // 2 + max(2, n // 10), n // 2 + max(3, n // 6), n // 2 + max(3, n // 6)))
        add_thermal_blob(packet, (max(2, n // 6), max(2, n // 5), n // 2), max(2, n // 10), 0.004)
        return packet

    if case_name == CASE_SYMMETRY:
        z_a = n // 3
        z_b = n - 1 - z_a
        add_fan_disk(packet, (1, n // 2, z_a), "x", 1, radius, 8.0, thickness)
        add_fan_disk(packet, (1, n // 2, z_b), "x", 1, radius, 8.0, thickness)
        return packet

    if case_name == CASE_OBSTACLE:
        add_fan_disk(packet, (1, n // 2, n // 2), "x", 1, radius, 8.0, thickness)
        add_box_obstacle(packet, (n // 2 - max(2, n // 12), n // 2 - max(3, n // 8), n // 2 - max(3, n // 8)),
                         (n // 2 + max(2, n // 12), n // 2 + max(3, n // 8), n // 2 + max(3, n // 8)))
        return packet

    if case_name == CASE_BUOYANCY:
        add_thermal_blob(packet, (n // 4, max(2, n // 6), n // 2), max(2, n // 8), 0.005)
        add_fan_disk(packet, (1, max(2, n // 6), n // 2), "x", 1, max(2, n // 10), 8.0, thickness)
        add_fan_disk(packet, (2, max(2, n // 6), n // 2), "x", -1, max(2, n // 10), 8.0, thickness)
        return packet

    if case_name == CASE_TAYLOR_GREEN:
        return build_taylor_green_packet(shape)

    if case_name == CASE_LID_CAVITY_2D:
        return packet

    if case_name == CASE_CYLINDER_CROSSFLOW:
        geom = cylinder_channel_geometry(shape)
        add_cylinder_obstacle(packet, geom.center, geom.radius)
        inlet = cylinder_inlet_profile(shape)
        packet[..., CH_VX] = inlet[None, :, None]
        packet[packet[..., CH_OBS] > 0.5, CH_VX:CH_VZ + 1] = 0.0
        return packet

    raise ValueError(f"Unsupported case: {case_name}")


def world_velocity(output_native: np.ndarray) -> np.ndarray:
    velocity = output_native[..., :3].astype(np.float64, copy=True)
    velocity *= NATIVE_VELOCITY_SCALE
    return velocity


def lattice_velocity(output_native: np.ndarray) -> np.ndarray:
    return output_native[..., :3].astype(np.float64, copy=True)


def pressure_field(output_native: np.ndarray) -> np.ndarray:
    return output_native[..., 3].astype(np.float64, copy=False)


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


def kinetic_energy(velocity: np.ndarray, obstacle_mask: np.ndarray) -> float:
    fluid = obstacle_mask <= 0.5
    if not np.any(fluid):
        return 0.0
    speed_sq = np.sum(velocity * velocity, axis=-1)
    return 0.5 * float(np.mean(speed_sq[fluid]))


def configure_benchmark_for_case(lib: NativeSolverLib, case_name: str, shape: GridShape) -> None:
    lib.benchmark_reset_config()
    if case_name == CASE_TAYLOR_GREEN:
        config = lib.benchmark_default_preset_config(AERO_LBM_BENCHMARK_PRESET_TAYLOR_GREEN_3D)
        lib.benchmark_set_config(config)
    elif case_name == CASE_LID_CAVITY_2D:
        config = lib.benchmark_default_preset_config(AERO_LBM_BENCHMARK_PRESET_LID_DRIVEN_CAVITY_2D)
        config.reference_length = ctypes.c_float(float(shape[0] - 1)).value
        lib.benchmark_set_config(config)
    elif case_name == CASE_CYLINDER_CROSSFLOW:
        config = lib.benchmark_default_preset_config(AERO_LBM_BENCHMARK_PRESET_CYLINDER_CROSSFLOW_2D)
        geom = cylinder_channel_geometry(shape)
        config.reference_length = ctypes.c_float(float(geom.diameter)).value
        config.reynolds_number = ctypes.c_float(CYLINDER_BENCHMARK_REYNOLDS).value
        config.initial_velocity[0] = ctypes.c_float(CYLINDER_BENCHMARK_UMEAN).value
        config.x_min.velocity[0] = ctypes.c_float(CYLINDER_BENCHMARK_UMAX).value
        lib.benchmark_set_config(config)


def taylor_green_expected_energy(step: int, grid_size: int, initial_energy: float) -> float:
    tau_shear = 0.502
    viscosity = (tau_shear - 0.5) / 3.0
    wave_number = 2.0 * math.pi / grid_size
    decay = math.exp(-4.0 * viscosity * (wave_number ** 2) * step)
    return initial_energy * decay


def cavity_centerline_profiles_lattice(velocity_lattice: np.ndarray) -> tuple[list[float], list[float]]:
    nx, ny, nz = velocity_lattice.shape[:3]
    mid_x = nx // 2
    mid_y = ny // 2
    mid_z = nz // 2
    u_profile = velocity_lattice[mid_x, :, mid_z, 0].astype(float).tolist()
    v_profile = velocity_lattice[:, mid_y, mid_z, 1].astype(float).tolist()
    return u_profile, v_profile


def benchmark_reference_speed_from_config(config: AeroLbmBenchmarkConfig) -> float:
    if config.preset == AERO_LBM_BENCHMARK_PRESET_CYLINDER_CROSSFLOW_2D:
        initial_speed = math.sqrt(sum(float(config.initial_velocity[i]) ** 2 for i in range(3)))
        if initial_speed > 1e-8:
            return initial_speed
        face_speed = math.sqrt(sum(float(config.x_min.velocity[i]) ** 2 for i in range(3)))
        return (2.0 / 3.0) * face_speed
    speed = math.sqrt(sum(float(config.initial_velocity[i]) ** 2 for i in range(3)))
    for face_name in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        face = getattr(config, face_name)
        face_speed = math.sqrt(sum(float(face.velocity[i]) ** 2 for i in range(3)))
        speed = max(speed, face_speed)
    return speed


def benchmark_viscosity_from_config(config: AeroLbmBenchmarkConfig) -> float:
    u_ref = benchmark_reference_speed_from_config(config)
    if config.reynolds_number <= 1e-8 or u_ref <= 1e-8:
        return (0.502 - 0.5) / 3.0
    return float(u_ref * float(config.reference_length) / float(config.reynolds_number))


def bilinear_sample(field: np.ndarray, x: float, y: float) -> float:
    nx, ny = field.shape
    x = min(max(x, 0.0), nx - 1.0)
    y = min(max(y, 0.0), ny - 1.0)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(nx - 1, x0 + 1)
    y1 = min(ny - 1, y0 + 1)
    fx = x - x0
    fy = y - y0
    c00 = field[x0, y0]
    c10 = field[x1, y0]
    c01 = field[x0, y1]
    c11 = field[x1, y1]
    c0 = c00 + (c10 - c00) * fx
    c1 = c01 + (c11 - c01) * fx
    return float(c0 + (c1 - c0) * fy)


def cylinder_pressure_probe_points(shape: GridShape) -> tuple[tuple[float, float], tuple[float, float]]:
    geom = cylinder_channel_geometry(shape)
    length_cells = float(max(1, shape[0] - 1))
    y = geom.center[1]
    a1 = (0.15 / CYLINDER_BENCHMARK_LENGTH * length_cells, y)
    a2 = (0.25 / CYLINDER_BENCHMARK_LENGTH * length_cells, y)
    return a1, a2


def cylinder_pressure_difference(pressure_lattice: np.ndarray, shape: GridShape) -> float:
    mid = pressure_lattice.shape[2] // 2
    p = pressure_lattice[:, :, mid]
    a1, a2 = cylinder_pressure_probe_points(shape)
    return bilinear_sample(p, *a1) - bilinear_sample(p, *a2)


def cylinder_force_coefficients_contour(
    velocity_lattice: np.ndarray,
    pressure_lattice: np.ndarray,
    center: tuple[float, float],
    radius: float,
    inflow_speed: float,
    viscosity: float,
    samples: int = 256,
) -> tuple[float, float]:
    mid = velocity_lattice.shape[2] // 2
    u = velocity_lattice[:, :, mid, 0]
    v = velocity_lattice[:, :, mid, 1]
    p = pressure_lattice[:, :, mid]

    du_dx, du_dy = np.gradient(u, axis=(0, 1), edge_order=1)
    dv_dx, dv_dy = np.gradient(v, axis=(0, 1), edge_order=1)

    cx, cy = center
    fx_total = 0.0
    fy_total = 0.0
    dtheta = 2.0 * math.pi / samples
    sample_radius = radius + 0.25

    for i in range(samples):
        theta = (i + 0.5) * dtheta
        nx = math.cos(theta)
        ny = math.sin(theta)
        x = cx + sample_radius * nx
        y = cy + sample_radius * ny
        p_s = bilinear_sample(p, x, y)
        du_dx_s = bilinear_sample(du_dx, x, y)
        du_dy_s = bilinear_sample(du_dy, x, y)
        dv_dx_s = bilinear_sample(dv_dx, x, y)
        dv_dy_s = bilinear_sample(dv_dy, x, y)

        sigma_xx = -p_s + 2.0 * viscosity * du_dx_s
        sigma_xy = viscosity * (du_dy_s + dv_dx_s)
        sigma_yy = -p_s + 2.0 * viscosity * dv_dy_s

        tx = sigma_xx * nx + sigma_xy * ny
        ty = sigma_xy * nx + sigma_yy * ny
        ds = radius * dtheta
        fx_total += tx * ds
        fy_total += ty * ds

    diameter = 2.0 * radius
    denom = max(0.5 * inflow_speed * inflow_speed * diameter, 1e-12)
    return fx_total / denom, fy_total / denom


def cylinder_force_coefficients_from_native_force(
    force_xyz: tuple[float, float, float],
    inflow_speed: float,
    diameter: float,
) -> tuple[float, float]:
    fx, fy, _ = force_xyz
    denom = max(0.5 * inflow_speed * inflow_speed * diameter, 1e-12)
    return fx / denom, fy / denom


def estimate_strouhal(
    lift_coeffs: list[float],
    inflow_speed: float,
    diameter: float,
    st_range: tuple[float, float] | None = None,
) -> float:
    if len(lift_coeffs) < 16 or inflow_speed <= 1e-8 or diameter <= 1e-8:
        return float("nan")
    signal = np.asarray(lift_coeffs, dtype=np.float64)
    signal = signal[len(signal) // 2 :]
    signal = signal - np.mean(signal)
    if np.allclose(signal, 0.0):
        return float("nan")
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0)
    if spectrum.size <= 1:
        return float("nan")
    amplitudes = np.abs(spectrum)
    if st_range is not None:
        st_min, st_max = st_range
        f_min = max(0.0, st_min * inflow_speed / diameter)
        f_max = max(f_min, st_max * inflow_speed / diameter)
        mask = (freqs >= f_min) & (freqs <= f_max)
        mask[0] = False
        if np.any(mask):
            masked = np.where(mask, amplitudes, -1.0)
            peak_idx = int(np.argmax(masked))
        else:
            peak_idx = int(np.argmax(amplitudes[1:]) + 1)
    else:
        peak_idx = int(np.argmax(amplitudes[1:]) + 1)
    frequency = float(freqs[peak_idx])
    return frequency * diameter / inflow_speed


def detect_cylinder_cycle_window(
    lift_coeffs: list[float],
    inflow_speed: float,
    diameter: float,
    st_range: tuple[float, float] = CYLINDER_BENCHMARK_ST_RANGE,
) -> tuple[int, int] | None:
    if len(lift_coeffs) < 32 or inflow_speed <= 1e-8 or diameter <= 1e-8:
        return None
    signal = np.asarray(lift_coeffs, dtype=np.float64)
    start = len(signal) // 2
    signal = signal[start:]
    if signal.size < 16:
        return None
    st_min, st_max = st_range
    min_period = max(4, int(math.floor(diameter / (inflow_speed * max(st_max, 1e-8)))))
    max_period = max(min_period + 1, int(math.ceil(diameter / (inflow_speed * max(st_min, 1e-8)))))
    minima: list[int] = []
    for i in range(1, signal.size - 1):
        if signal[i] <= signal[i - 1] and signal[i] < signal[i + 1]:
            minima.append(i)
    for i in range(len(minima) - 2, -1, -1):
        period = minima[i + 1] - minima[i]
        if min_period <= period <= max_period:
            return start + minima[i], start + minima[i + 1]
    return None


def obstacle_flux_stats(velocity: np.ndarray, obstacle_mask: np.ndarray) -> tuple[float, float, float]:
    speed = np.linalg.norm(velocity, axis=-1)
    solid = obstacle_mask > 0.5
    inside_speed = float(np.max(speed[solid])) if np.any(solid) else 0.0

    interface_values: list[np.ndarray] = []
    for axis in range(3):
        lo = [slice(None), slice(None), slice(None)]
        hi = [slice(None), slice(None), slice(None)]
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)

        solid_lo = solid[tuple(lo)]
        solid_hi = solid[tuple(hi)]
        fluid_lo = ~solid_lo
        fluid_hi = ~solid_hi
        vel_lo = np.abs(velocity[tuple(lo)][..., axis])
        vel_hi = np.abs(velocity[tuple(hi)][..., axis])

        mask_lo = fluid_lo & solid_hi
        mask_hi = solid_lo & fluid_hi
        if np.any(mask_lo):
            interface_values.append(vel_lo[mask_lo])
        if np.any(mask_hi):
            interface_values.append(vel_hi[mask_hi])

    if not interface_values:
        return inside_speed, 0.0, 0.0
    values = np.concatenate(interface_values, axis=0)
    return inside_speed, float(np.mean(values)), float(np.max(values))


def mirror_symmetry_error(velocity: np.ndarray, mirror_axis: int, flip_component: int) -> float:
    mirrored = np.flip(velocity, axis=mirror_axis).copy()
    mirrored[..., flip_component] *= -1.0
    denom = float(np.sqrt(np.mean(np.square(velocity))) + 1e-8)
    diff = velocity - mirrored
    return float(np.sqrt(np.mean(np.square(diff))) / denom)


def average_upper_vy(velocity: np.ndarray, y_min: int, y_max: int, z_center: int, radius: int) -> float:
    n = velocity.shape[0]
    y0 = max(0, min(n - 1, y_min))
    y1 = max(0, min(n - 1, y_max))
    if y1 < y0:
        return 0.0
    z0 = max(0, z_center - radius)
    z1 = min(n - 1, z_center + radius)
    x1 = max(2, n // 4)
    samples = []
    for x in range(0, min(n, x1 + 1)):
        for y in range(y0, y1 + 1):
            for z in range(z0, z1 + 1):
                dz = z - z_center
                if dz * dz > radius * radius:
                    continue
                samples.append(velocity[x, y, z, 1])
    return float(mean(samples)) if samples else 0.0


def mcups(shape: GridShape, mean_ms: float) -> float:
    cells = shape[0] * shape[1] * shape[2]
    return (cells / max(mean_ms, 1e-9)) / 1e3


def native_output_view(output_flat: np.ndarray, shape: GridShape) -> np.ndarray:
    nx, ny, nz = shape
    return output_flat.reshape(nx, ny, nz, OUTPUT_CHANNELS)


def initialize_packet_state(packet: np.ndarray, output_flat: np.ndarray) -> None:
    out = output_flat.reshape(packet.shape[0], packet.shape[1], packet.shape[2], OUTPUT_CHANNELS)
    packet[..., CH_VX:CH_VZ + 1] = out[..., :3]
    packet[..., CH_P] = out[..., 3]
    solid = packet[..., CH_OBS] > 0.5
    packet[solid, CH_VX:CH_VZ + 1] = 0.0
    packet[solid, CH_P] = 0.0


def apply_static_fields(packet: np.ndarray, fresh: np.ndarray) -> None:
    packet[..., CH_OBS] = fresh[..., CH_OBS]
    packet[..., CH_FAN_MASK:CH_FAN_VZ + 1] = fresh[..., CH_FAN_MASK:CH_FAN_VZ + 1]
    packet[..., CH_THERM] = fresh[..., CH_THERM]


def run_performance_case(lib: NativeSolverLib, mode: str, grid_size: int, workload: str, warmup_steps: int, measure_steps: int, context_seed: int) -> RuntimeMeasurement:
    lib.benchmark_reset_config()
    shape = benchmark_grid_shape(workload, grid_size)
    runtime_info = lib.initialize(shape, mode)
    context_key = context_seed
    packet = build_case_packet(workload, shape, 0)
    output = np.empty((shape[0] * shape[1] * shape[2]) * OUTPUT_CHANNELS, dtype=np.float32)

    for step in range(warmup_steps):
        apply_static_fields(packet, build_case_packet(workload, shape, step))
        if not lib.step(packet, context_key, output):
            lib.release_context(context_key)
            lib.shutdown()
            raise RuntimeError(
                f"Native step failed during warmup: mode={mode} grid={grid_shape_to_str(shape)} workload={workload}"
            )
        initialize_packet_state(packet, output)

    lib.reset_timing()
    external_times = []
    divergences = []
    kinetic_energies = []
    last_max_speed = 0.0
    for step in range(measure_steps):
        apply_static_fields(packet, build_case_packet(workload, shape, warmup_steps + step))
        t0 = time.perf_counter()
        ok = lib.step(packet, context_key, output)
        external_times.append((time.perf_counter() - t0) * 1000.0)
        if not ok:
            lib.release_context(context_key)
            lib.shutdown()
            raise RuntimeError(
                f"Native step failed during measurement: mode={mode} grid={grid_shape_to_str(shape)} workload={workload}"
            )
        view = native_output_view(output, shape)
        velocity = world_velocity(view)
        obstacle = packet[..., CH_OBS]
        divergences.append(divergence_rms(velocity, obstacle))
        kinetic_energies.append(kinetic_energy(velocity, obstacle))
        last_max_speed = float(np.max(np.linalg.norm(velocity, axis=-1)))
        initialize_packet_state(packet, output)

    snapshot = lib.timing_snapshot()
    lib.release_context(context_key)
    lib.shutdown()
    p95 = float(np.percentile(external_times, 95))
    return RuntimeMeasurement(
        mode=mode,
        runtime_info=runtime_info,
        grid_size=grid_size,
        grid_shape=shape,
        workload=workload,
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        external_step_ms_mean=float(mean(external_times)),
        external_step_ms_std=float(pstdev(external_times)) if len(external_times) > 1 else 0.0,
        external_step_ms_p95=p95,
        native_solver_ms_mean=float(snapshot.avg_solver_ms),
        native_total_ms_mean=float(snapshot.avg_total_ms),
        mcups=mcups(shape, float(snapshot.avg_solver_ms if snapshot.avg_solver_ms > 0 else mean(external_times))),
        mlups=mcups(shape, float(snapshot.avg_solver_ms if snapshot.avg_solver_ms > 0 else mean(external_times))),
        last_max_speed=last_max_speed,
        final_kinetic_energy=kinetic_energies[-1] if kinetic_energies else 0.0,
        mean_divergence_rms=float(mean(divergences)) if divergences else 0.0,
    )


def run_diagnostic_case(lib: NativeSolverLib, mode: str, grid_size: int, case_name: str, steps: int, context_seed: int) -> DiagnosticMeasurement:
    shape = benchmark_grid_shape(case_name, grid_size)
    configure_benchmark_for_case(lib, case_name, shape)
    benchmark_config = lib.benchmark_get_config()
    runtime_info = lib.initialize(shape, mode)
    context_key = context_seed
    packet = build_case_packet(case_name, shape, 0)
    output = np.empty((shape[0] * shape[1] * shape[2]) * OUTPUT_CHANNELS, dtype=np.float32)
    history: list[dict[str, Any]] = []
    all_finite = True
    initial_energy = kinetic_energy(packet[..., CH_VX:CH_VZ + 1].astype(np.float64), packet[..., CH_OBS]) if case_name == CASE_TAYLOR_GREEN else 0.0
    cylinder_center, cylinder_radius = cylinder_geometry(shape) if case_name == CASE_CYLINDER_CROSSFLOW else ((0.0, 0.0), 0.0)
    inflow_speed = benchmark_reference_speed_from_config(benchmark_config)
    benchmark_viscosity = benchmark_viscosity_from_config(benchmark_config)

    for step in range(steps):
        fresh = build_case_packet(case_name, shape, step)
        apply_static_fields(packet, fresh)
        if case_name == CASE_CYLINDER_CROSSFLOW:
            apply_cylinder_seed_perturbation(packet, cylinder_center, cylinder_radius, step)
        t0 = time.perf_counter()
        ok = lib.step(packet, context_key, output)
        step_ms = (time.perf_counter() - t0) * 1000.0
        if not ok:
            all_finite = False
            break
        view = native_output_view(output, shape)
        finite = bool(np.isfinite(view).all())
        all_finite &= finite

        velocity = lattice_velocity(view)
        pressure = pressure_field(view)
        obstacle = packet[..., CH_OBS]
        speed = np.linalg.norm(velocity, axis=-1)
        entry: dict[str, Any] = {
            "step": step,
            "step_ms": float(step_ms),
            "all_finite": finite,
            "divergence_rms": divergence_rms(velocity, obstacle),
            "kinetic_energy": kinetic_energy(velocity, obstacle),
            "max_speed": float(np.max(speed)),
            "max_abs_pressure": float(np.max(np.abs(pressure))),
        }

        if case_name == CASE_SYMMETRY:
            entry["symmetry_error"] = mirror_symmetry_error(velocity, mirror_axis=2, flip_component=2)
        elif case_name == CASE_OBSTACLE:
            inside_speed, mean_flux, max_flux = obstacle_flux_stats(velocity, obstacle)
            entry["inside_obstacle_speed"] = inside_speed
            entry["mean_interface_normal_flux"] = mean_flux
            entry["max_interface_normal_flux"] = max_flux
        elif case_name == CASE_BUOYANCY:
            entry["upper_vy"] = average_upper_vy(
                velocity,
                y_min=shape[1] // 2,
                y_max=shape[1] - 2,
                z_center=shape[2] // 2,
                radius=max(2, min(shape[0], shape[1], shape[2]) // 8),
            )
        elif case_name == CASE_TAYLOR_GREEN:
            lattice_vel = lattice_velocity(view)
            energy = kinetic_energy(lattice_vel, obstacle)
            analytic = taylor_green_expected_energy(step + 1, grid_size, initial_energy)
            entry["kinetic_energy_lattice"] = energy
            entry["analytic_kinetic_energy_lattice"] = analytic
            entry["relative_energy_error"] = float(abs(energy - analytic) / max(abs(analytic), 1e-12))
        elif case_name == CASE_LID_CAVITY_2D:
            lattice_vel = lattice_velocity(view)
            mid_x = shape[0] // 2
            mid_y = shape[1] // 2
            mid_z = shape[2] // 2
            entry["center_u"] = float(lattice_vel[mid_x, mid_y, mid_z, 0])
            entry["center_v"] = float(lattice_vel[mid_x, mid_y, mid_z, 1])
        elif case_name == CASE_CYLINDER_CROSSFLOW:
            cd, cl = cylinder_force_coefficients_from_native_force(
                lib.get_last_force(context_key),
                inflow_speed,
                2.0 * cylinder_radius,
            )
            entry["drag_coefficient"] = float(cd)
            entry["lift_coefficient"] = float(cl)
            entry["pressure_difference"] = float(cylinder_pressure_difference(pressure, shape))

        history.append(entry)
        initialize_packet_state(packet, output)

    extra_summary: dict[str, float] = {}
    if case_name == CASE_SYMMETRY and history:
        values = [float(h["symmetry_error"]) for h in history]
        extra_summary["mean_symmetry_error"] = float(mean(values))
        extra_summary["final_symmetry_error"] = values[-1]
    elif case_name == CASE_OBSTACLE and history:
        inside = [float(h["inside_obstacle_speed"]) for h in history]
        leak = [float(h["mean_interface_normal_flux"]) for h in history]
        extra_summary["max_inside_obstacle_speed"] = float(max(inside))
        extra_summary["mean_interface_normal_flux"] = float(mean(leak))
    elif case_name == CASE_BUOYANCY and history:
        upper = [float(h["upper_vy"]) for h in history]
        extra_summary["peak_upper_vy"] = float(max(upper))
        extra_summary["final_upper_vy"] = upper[-1]
    elif case_name == CASE_TAYLOR_GREEN and history:
        rel = [float(h["relative_energy_error"]) for h in history]
        extra_summary["final_relative_energy_error"] = rel[-1]
        extra_summary["mean_relative_energy_error"] = float(mean(rel))
    elif case_name == CASE_LID_CAVITY_2D and history:
        final_view = native_output_view(output, shape)
        final_lattice = lattice_velocity(final_view)
        u_profile, v_profile = cavity_centerline_profiles_lattice(final_lattice)
        extra_summary["final_center_u"] = float(history[-1]["center_u"])
        extra_summary["final_center_v"] = float(history[-1]["center_v"])
        extra_summary["u_centerline_profile"] = u_profile
        extra_summary["v_centerline_profile"] = v_profile
    elif case_name == CASE_CYLINDER_CROSSFLOW and history:
        drag = [float(h["drag_coefficient"]) for h in history]
        lift = [float(h["lift_coefficient"]) for h in history]
        pressure_diff = [float(h["pressure_difference"]) for h in history]
        cycle_window = detect_cylinder_cycle_window(lift, inflow_speed, 2.0 * cylinder_radius)
        if cycle_window is not None:
            cycle_start, cycle_end = cycle_window
            drag_window = np.asarray(drag[cycle_start:cycle_end + 1], dtype=np.float64)
            lift_window = np.asarray(lift[cycle_start:cycle_end + 1], dtype=np.float64)
            pressure_window = np.asarray(pressure_diff[cycle_start:cycle_end + 1], dtype=np.float64)
            period_steps = max(1, cycle_end - cycle_start)
            st = (2.0 * cylinder_radius) / max(inflow_speed * period_steps, 1e-12)
            extra_summary["cycle_start_step"] = float(cycle_start)
            extra_summary["cycle_end_step"] = float(cycle_end)
            extra_summary["cycle_period_steps"] = float(period_steps)
        else:
            tail_start = (3 * len(drag)) // 4
            drag_window = np.asarray(drag[tail_start:], dtype=np.float64)
            lift_window = np.asarray(lift[tail_start:], dtype=np.float64)
            pressure_window = np.asarray(pressure_diff[tail_start:], dtype=np.float64)
            st = estimate_strouhal(
                lift,
                inflow_speed,
                2.0 * cylinder_radius,
                st_range=CYLINDER_BENCHMARK_ST_RANGE,
            )
            extra_summary["cycle_start_step"] = float(tail_start)
            extra_summary["cycle_end_step"] = float(len(drag) - 1)
            extra_summary["cycle_period_steps"] = float(len(drag_window))

        extra_summary["mean_drag_coefficient"] = float(np.mean(drag_window))
        extra_summary["min_drag_coefficient"] = float(np.min(drag_window))
        extra_summary["max_drag_coefficient"] = float(np.max(drag_window))
        extra_summary["amp_drag_coefficient"] = float(np.max(drag_window) - np.min(drag_window))
        extra_summary["mean_lift_coefficient"] = float(np.mean(lift_window))
        extra_summary["min_lift_coefficient"] = float(np.min(lift_window))
        extra_summary["max_lift_coefficient"] = float(np.max(lift_window))
        extra_summary["amp_lift_coefficient"] = float(np.max(lift_window) - np.min(lift_window))
        extra_summary["lift_rms"] = float(np.sqrt(np.mean(np.square(lift_window - np.mean(lift_window)))))
        extra_summary["mean_pressure_difference"] = float(np.mean(pressure_window))
        extra_summary["min_pressure_difference"] = float(np.min(pressure_window))
        extra_summary["max_pressure_difference"] = float(np.max(pressure_window))
        extra_summary["amp_pressure_difference"] = float(np.max(pressure_window) - np.min(pressure_window))
        extra_summary["strouhal_number"] = float(st)

    lib.release_context(context_key)
    lib.shutdown()
    return DiagnosticMeasurement(
        mode=mode,
        runtime_info=runtime_info,
        grid_size=grid_size,
        grid_shape=shape,
        case=case_name,
        steps=len(history),
        all_finite=all_finite and bool(history),
        final_divergence_rms=float(history[-1]["divergence_rms"]) if history else float("nan"),
        max_speed=max(float(h["max_speed"]) for h in history) if history else float("nan"),
        history=history,
        extra_summary=extra_summary,
    )


def to_dict_runtime(measurement: RuntimeMeasurement) -> dict[str, Any]:
    data = measurement.__dict__.copy()
    data["grid_shape"] = list(measurement.grid_shape)
    return data


def to_dict_diag(measurement: DiagnosticMeasurement) -> dict[str, Any]:
    return {
        "mode": measurement.mode,
        "runtime_info": measurement.runtime_info,
        "grid_size": measurement.grid_size,
        "grid_shape": list(measurement.grid_shape),
        "case": measurement.case,
        "steps": measurement.steps,
        "all_finite": measurement.all_finite,
        "final_divergence_rms": measurement.final_divergence_rms,
        "max_speed": measurement.max_speed,
        "extra_summary": measurement.extra_summary,
        "history": measurement.history,
    }


def system_info() -> dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        uname = subprocess.check_output(["uname", "-a"], text=True).strip()
        info["uname"] = uname
    except Exception:
        pass
    return info


def plot_performance(perf_results: list[RuntimeMeasurement], fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    for workload in PERF_WORKLOADS:
        subset = [r for r in perf_results if r.workload == workload]
        if not subset:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for mode in MODES:
            mode_rows = sorted((r for r in subset if r.mode == mode), key=lambda r: r.grid_size)
            if not mode_rows:
                continue
            grids = [r.grid_size for r in mode_rows]
            solver_ms = [r.native_solver_ms_mean if r.native_solver_ms_mean > 0 else r.external_step_ms_mean for r in mode_rows]
            throughput = [r.mcups for r in mode_rows]
            axes[0].plot(grids, solver_ms, marker="o", label=mode)
            axes[1].plot(grids, throughput, marker="o", label=mode)

        axes[0].set_title(f"{workload}: step time")
        axes[0].set_xlabel("Grid size")
        axes[0].set_ylabel("Mean solver step time (ms)")
        axes[0].grid(alpha=0.3)

        axes[1].set_title(f"{workload}: throughput")
        axes[1].set_xlabel("Grid size")
        axes[1].set_ylabel("MCUPS")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        out_path = fig_dir / f"perf_{workload}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        saved.append(out_path.name)

    return saved


def plot_diagnostics(diag_results: list[DiagnosticMeasurement], fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    metric_map = {
        CASE_QUIESCENT: ("max_speed", "Residual max speed"),
        CASE_SYMMETRY: ("symmetry_error", "Symmetry error"),
        CASE_OBSTACLE: ("mean_interface_normal_flux", "Mean interface normal flux"),
        CASE_BUOYANCY: ("upper_vy", "Upper-column $v_y$"),
        CASE_TAYLOR_GREEN: ("relative_energy_error", "Relative kinetic-energy error"),
        CASE_LID_CAVITY_2D: ("center_u", "Centerline velocity"),
        CASE_CYLINDER_CROSSFLOW: ("drag_coefficient", "Force coefficient"),
    }

    for case_name, (metric_key, ylabel) in metric_map.items():
        subset = [r for r in diag_results if r.case == case_name]
        if not subset:
            continue
        if case_name == CASE_TAYLOR_GREEN:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for mode in MODES:
                mode_rows = [r for r in subset if r.mode == mode]
                if not mode_rows:
                    continue
                row = mode_rows[0]
                steps = [h["step"] + 1 for h in row.history]
                numeric = [h["kinetic_energy_lattice"] for h in row.history]
                analytic = [h["analytic_kinetic_energy_lattice"] for h in row.history]
                rel = [h["relative_energy_error"] for h in row.history]
                axes[0].plot(steps, numeric, label=f"{mode} numeric")
                axes[0].plot(steps, analytic, linestyle="--", label=f"{mode} analytic")
                axes[1].plot(steps, rel, label=f"{mode}")
            axes[0].set_title(case_name)
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Kinetic energy (lattice units)")
            axes[0].grid(alpha=0.3)
            axes[0].legend()
            axes[1].set_title(f"{case_name}: relative error")
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Relative error")
            axes[1].set_yscale("log")
            axes[1].grid(alpha=0.3)
            axes[1].legend()
            fig.tight_layout()
            out_path = fig_dir / f"diag_{case_name}.png"
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            saved.append(out_path.name)
            continue
        if case_name == CASE_LID_CAVITY_2D:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for mode in MODES:
                mode_rows = [r for r in subset if r.mode == mode]
                if not mode_rows:
                    continue
                row = mode_rows[0]
                steps = [h["step"] + 1 for h in row.history]
                center_u = [h["center_u"] for h in row.history]
                center_v = [h["center_v"] for h in row.history]
                axes[0].plot(steps, center_u, label=f"{mode} center u")
                axes[0].plot(steps, center_v, linestyle="--", label=f"{mode} center v")

                u_profile = row.extra_summary.get("u_centerline_profile")
                v_profile = row.extra_summary.get("v_centerline_profile")
                if u_profile and v_profile:
                    coords = np.linspace(0.0, 1.0, len(u_profile))
                    axes[1].plot(coords, u_profile, label=f"{mode} u(y)")
                    axes[1].plot(coords, v_profile, linestyle="--", label=f"{mode} v(x)")
            axes[0].set_title(case_name)
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Velocity (lattice units)")
            axes[0].grid(alpha=0.3)
            axes[0].legend()
            axes[1].set_title(f"{case_name}: centerlines")
            axes[1].set_xlabel("Normalized centerline coordinate")
            axes[1].set_ylabel("Velocity (lattice units)")
            axes[1].grid(alpha=0.3)
            axes[1].legend()
            fig.tight_layout()
            out_path = fig_dir / f"diag_{case_name}.png"
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            saved.append(out_path.name)
            continue
        if case_name == CASE_CYLINDER_CROSSFLOW:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for mode in MODES:
                mode_rows = [r for r in subset if r.mode == mode]
                if not mode_rows:
                    continue
                row = mode_rows[0]
                steps = [h["step"] + 1 for h in row.history]
                cd = [h["drag_coefficient"] for h in row.history]
                cl = [h["lift_coefficient"] for h in row.history]
                axes[0].plot(steps, cd, label=f"{mode} Cd")
                axes[0].plot(steps, cl, linestyle="--", label=f"{mode} Cl")
                axes[1].plot(steps, [h["divergence_rms"] for h in row.history], label=f"{mode} div")
            axes[0].set_title(case_name)
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Coefficient")
            axes[0].grid(alpha=0.3)
            axes[0].legend()
            axes[1].set_title(f"{case_name}: divergence")
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Divergence RMS")
            axes[1].set_yscale("log")
            axes[1].grid(alpha=0.3)
            axes[1].legend()
            fig.tight_layout()
            out_path = fig_dir / f"diag_{case_name}.png"
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            saved.append(out_path.name)
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for mode in MODES:
            mode_rows = [r for r in subset if r.mode == mode]
            if not mode_rows:
                continue
            row = mode_rows[0]
            steps = [h["step"] for h in row.history]
            series = [h[metric_key] for h in row.history]
            divergence = [h["divergence_rms"] for h in row.history]
            axes[0].plot(steps, series, label=f"{mode} ({row.runtime_info.split(':')[0]})")
            axes[1].plot(steps, divergence, label=f"{mode} ({row.runtime_info.split(':')[0]})")

        axes[0].set_title(case_name)
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel(ylabel)
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].set_title(f"{case_name}: divergence")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Divergence RMS")
        axes[1].set_yscale("log")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        out_path = fig_dir / f"diag_{case_name}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        saved.append(out_path.name)

    return saved


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def generate_report(
    report_path: Path,
    library_path: Path,
    perf_results: list[RuntimeMeasurement],
    diag_results: list[DiagnosticMeasurement],
    perf_figs: list[str],
    diag_figs: list[str],
    args: argparse.Namespace,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fig_dir = report_path.parent / "figures"

    runtime_lines = []
    for mode in MODES:
        rows = [r for r in perf_results if r.mode == mode]
        if rows:
            runtime_lines.append(f"- `{mode}`: `{rows[0].runtime_info}`")

    perf_rows = []
    for result in sorted(perf_results, key=lambda r: (r.workload, r.grid_size, r.mode)):
        perf_rows.append([
            result.workload,
            result.mode,
            grid_shape_to_str(result.grid_shape),
            f"{result.native_solver_ms_mean:.3f}",
            f"{result.external_step_ms_mean:.3f}",
            f"{result.mcups:.2f}",
            f"{result.mean_divergence_rms:.3e}",
        ])

    diag_rows = []
    for result in sorted(diag_results, key=lambda r: (r.case, r.mode)):
        summary_value = ""
        if "mean_symmetry_error" in result.extra_summary:
            summary_value = f"sym={result.extra_summary['mean_symmetry_error']:.3e}"
        elif "max_inside_obstacle_speed" in result.extra_summary:
            summary_value = f"leak={result.extra_summary['max_inside_obstacle_speed']:.3e}"
        elif "peak_upper_vy" in result.extra_summary:
            summary_value = f"peakVy={result.extra_summary['peak_upper_vy']:.3e}"
        diag_rows.append([
            result.case,
            result.mode,
            grid_shape_to_str(result.grid_shape),
            "yes" if result.all_finite else "no",
            f"{result.final_divergence_rms:.3e}",
            f"{result.max_speed:.3e}",
            summary_value,
        ])

    lines = [
        "# Native Solver Experiments",
        "",
        "## Experimental Setup",
        "",
        f"- Native library: `{library_path}`",
        f"- Performance grid sizes: `{args.perf_grid_sizes}`",
        f"- Diagnostic grid size: `{args.diag_grid_size}`",
        f"- Performance protocol: `{args.perf_warmup}` warmup steps + `{args.perf_steps}` measured steps",
        f"- Diagnostic protocol: `{args.diag_steps}` steps per case",
        "",
        "### Runtime Selection",
        "",
        *runtime_lines,
        "",
        "### Host System",
        "",
        *[f"- `{key}`: `{value}`" for key, value in system_info().items()],
        "",
        "## Performance Results",
        "",
        markdown_table(
            ["Workload", "Mode", "Grid", "Solver ms", "External ms", "MCUPS", "Mean div RMS"],
            perf_rows,
        ),
        "",
        "The table reports both internal solver timing and end-to-end Python-to-native call timing. "
        "MCUPS is computed from the internal solver time when available.",
        "",
    ]

    for fig_name in perf_figs:
        lines.extend([
            f"![{fig_name}](figures/{fig_name})",
            "",
        ])

    lines.extend([
        "## Numerical Diagnostics",
        "",
        markdown_table(
            ["Case", "Mode", "Grid", "Finite", "Final div RMS", "Max speed", "Key summary"],
            diag_rows,
        ),
        "",
        "These diagnostics emphasize numerical behavior instead of raw throughput: residual drift in quiescent flow, "
        "symmetry preservation, obstacle leakage, and buoyancy-driven vertical transport.",
        "",
    ])

    for fig_name in diag_figs:
        lines.extend([
            f"![{fig_name}](figures/{fig_name})",
            "",
        ])

    auto_rows = [r for r in perf_results if r.mode == MODE_AUTO and r.workload == WORKLOAD_MIXED]
    cpu_rows = [r for r in perf_results if r.mode == MODE_CPU and r.workload == WORKLOAD_MIXED]
    speedup_lines = []
    if auto_rows and cpu_rows:
        cpu_by_grid = {r.grid_size: r for r in cpu_rows}
        for auto_row in auto_rows:
            cpu_row = cpu_by_grid.get(auto_row.grid_size)
            if not cpu_row:
                continue
            denom = max(auto_row.native_solver_ms_mean, 1e-9)
            speedup = cpu_row.native_solver_ms_mean / denom
            speedup_lines.append(
                f"- grid `{auto_row.grid_size}`: `{speedup:.2f}x` speedup of `auto` relative to forced CPU "
                f"on the mixed-flow workload."
            )

    lines.extend([
        "## Discussion",
        "",
        "The native benchmark separates throughput from numerical behavior. "
        "Performance is reported over a representative mixed-flow workload, while diagnostics are evaluated on targeted canonical cases.",
        "",
    ])
    if speedup_lines:
        lines.extend(speedup_lines + [""])
    else:
        lines.extend([
            "- No mode-to-mode speedup summary was produced, either because only one mode was run or because a matching grid sweep was unavailable.",
            "",
        ])

    lines.extend([
        "## Limitations",
        "",
        "- These experiments benchmark the native C++ solver in isolation, not the full Minecraft integration path.",
        "- The benchmark uses synthetic canonical scenes rather than logged in-game windows.",
        "- If `auto` falls back to CPU, the report still remains valid, but it becomes a CPU-path characterization instead of an OpenCL acceleration study.",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    perf_grid_sizes = parse_grid_sizes(args.perf_grid_sizes)
    modes = parse_modes(args.modes)
    diag_cases = parse_diag_cases(args.diag_cases)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    library_path = detect_library_path(args.library)
    lib = NativeSolverLib(library_path)

    perf_results: list[RuntimeMeasurement] = []
    diag_results: list[DiagnosticMeasurement] = []

    context_seed = 10_000
    for mode in modes:
        if not args.skip_perf:
            for grid_size in perf_grid_sizes:
                for workload in PERF_WORKLOADS:
                    perf_results.append(
                        run_performance_case(
                            lib=lib,
                            mode=mode,
                            grid_size=grid_size,
                            workload=workload,
                            warmup_steps=args.perf_warmup,
                            measure_steps=args.perf_steps,
                            context_seed=context_seed,
                        )
                    )
                    context_seed += 1

        for case_name in diag_cases:
            diag_results.append(
                run_diagnostic_case(
                    lib=lib,
                    mode=mode,
                    grid_size=args.diag_grid_size,
                    case_name=case_name,
                    steps=args.diag_steps,
                    context_seed=context_seed,
                )
            )
            context_seed += 1

    results_json = {
        "library_path": str(library_path),
        "perf_results": [to_dict_runtime(r) for r in perf_results],
        "diag_results": [to_dict_diag(r) for r in diag_results],
        "args": vars(args),
        "system": system_info(),
    }
    json_path = output_dir / "native_solver_results.json"
    json_path.write_text(json.dumps(results_json, indent=2), encoding="utf-8")

    fig_dir = output_dir / "figures"
    perf_figs = plot_performance(perf_results, fig_dir)
    diag_figs = plot_diagnostics(diag_results, fig_dir)

    if args.skip_report:
        print(f"Saved results to {json_path}")
        print(f"Saved figures to {fig_dir}")
        return

    report_path = output_dir / "native_solver_report.md"
    generate_report(report_path, library_path, perf_results, diag_results, perf_figs, diag_figs, args)

    print(f"Saved JSON report to {json_path}")
    print(f"Saved Markdown report to {report_path}")
    print(f"Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
