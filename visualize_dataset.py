#!/usr/bin/env python3
"""Visualize generated PhiFlow 3D flow dataset.

Visualizes:
1. Obstacle mask (3D volume rendering)
2. Velocity field (vector arrows on slices)
3. Pressure field (contours on slices)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_sample(sample_path: Path) -> Dict[str, np.ndarray]:
    """Load a single sample from .npy file."""
    data = np.load(sample_path, allow_pickle=True).item()
    return data


def load_metadata(data_dir: Path) -> Dict:
    """Load dataset metadata."""
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {}


def visualize_obstacle_mask_3d(obstacle_mask: np.ndarray, ax: Axes3D, alpha: float = 0.3):
    """Visualize obstacle mask as 3D voxels."""
    # Create a 3D grid
    x, y, z = np.indices(obstacle_mask.shape)
    
    # Get voxel coordinates where mask is 1
    mask_voxels = obstacle_mask > 0.5
    
    # Set colors
    colors = np.empty(mask_voxels.shape + (4,))
    colors[..., 0] = 0.8  # R
    colors[..., 1] = 0.2  # G
    colors[..., 2] = 0.2  # B
    colors[..., 3] = alpha  # Alpha
    
    # Plot the voxels
    ax.voxels(mask_voxels, facecolors=colors, edgecolors='k', linewidth=0.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Obstacle Mask (3D)')


def visualize_obstacle_mask_slices(obstacle_mask: np.ndarray, figsize: tuple = (15, 5)):
    """Visualize obstacle mask as orthogonal slices."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Mid slices
    mid_x = obstacle_mask.shape[0] // 2
    mid_y = obstacle_mask.shape[1] // 2
    mid_z = obstacle_mask.shape[2] // 2
    
    # XY slice (constant Z)
    im0 = axes[0].imshow(obstacle_mask[:, :, mid_z].T, 
                         cmap='binary_r', origin='lower')
    axes[0].set_title(f'XY Slice (Z={mid_z})')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im0, ax=axes[0], label='Obstacle')
    
    # XZ slice (constant Y)
    im1 = axes[1].imshow(obstacle_mask[:, mid_y, :].T,
                         cmap='binary_r', origin='lower')
    axes[1].set_title(f'XZ Slice (Y={mid_y})')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    plt.colorbar(im1, ax=axes[1], label='Obstacle')
    
    # YZ slice (constant X)
    im2 = axes[2].imshow(obstacle_mask[mid_x, :, :].T,
                         cmap='binary_r', origin='lower')
    axes[2].set_title(f'YZ Slice (X={mid_x})')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    plt.colorbar(im2, ax=axes[2], label='Obstacle')
    
    plt.tight_layout()
    return fig


def visualize_velocity_field(velocity: np.ndarray, figsize: tuple = (15, 10)):
    """Visualize 3D velocity field on orthogonal slices with vector arrows."""
    # velocity shape: (x, y, z, 3)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Mid slices
    mid_x = velocity.shape[0] // 2
    mid_y = velocity.shape[1] // 2
    mid_z = velocity.shape[2] // 2
    
    # Downsample for vectors
    step = max(1, velocity.shape[0] // 8)
    
    # XY slice (constant Z)
    X, Y = np.meshgrid(np.arange(velocity.shape[0]), 
                       np.arange(velocity.shape[1]), indexing='ij')
    
    # Magnitude for color
    vel_mag = np.sqrt(np.sum(velocity**2, axis=3))
    
    # Plot velocity magnitude as background
    im0 = axes[0, 0].imshow(vel_mag[:, :, mid_z].T, 
                           cmap='viridis', origin='lower')
    axes[0, 0].set_title(f'XY Slice (Z={mid_z}) - Velocity Magnitude')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im0, ax=axes[0, 0], label='|V|')
    
    # Plot vector arrows (downsampled)
    axes[0, 1].quiver(X[::step, ::step], Y[::step, ::step],
                     velocity[::step, ::step, mid_z, 0].T,
                     velocity[::step, ::step, mid_z, 1].T,
                     color='white', scale=20)
    axes[0, 1].set_title(f'XY Slice - Velocity Vectors (U,V)')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_aspect('equal')
    
    # Plot W component
    im0_w = axes[0, 2].imshow(velocity[:, :, mid_z, 2].T,
                             cmap='coolwarm', origin='lower')
    axes[0, 2].set_title(f'XY Slice - W Component')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    plt.colorbar(im0_w, ax=axes[0, 2], label='W')
    
    # XZ slice (constant Y)
    X, Z = np.meshgrid(np.arange(velocity.shape[0]),
                       np.arange(velocity.shape[2]), indexing='ij')
    
    im1 = axes[1, 0].imshow(vel_mag[:, mid_y, :].T,
                           cmap='viridis', origin='lower')
    axes[1, 0].set_title(f'XZ Slice (Y={mid_y}) - Velocity Magnitude')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Z')
    plt.colorbar(im1, ax=axes[1, 0], label='|V|')
    
    axes[1, 1].quiver(X[::step, ::step], Z[::step, ::step],
                     velocity[::step, mid_y, ::step, 0].T,
                     velocity[::step, mid_y, ::step, 2].T,
                     color='white', scale=20)
    axes[1, 1].set_title(f'XZ Slice - Velocity Vectors (U,W)')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Z')
    axes[1, 1].set_aspect('equal')
    
    im1_v = axes[1, 2].imshow(velocity[:, mid_y, :, 1].T,
                             cmap='coolwarm', origin='lower')
    axes[1, 2].set_title(f'XZ Slice - V Component')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Z')
    plt.colorbar(im1_v, ax=axes[1, 2], label='V')
    
    plt.tight_layout()
    return fig


def visualize_pressure_field(pressure: np.ndarray, figsize: tuple = (15, 5)):
    """Visualize pressure field on orthogonal slices."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Mid slices
    mid_x = pressure.shape[0] // 2
    mid_y = pressure.shape[1] // 2
    mid_z = pressure.shape[2] // 2
    
    # XY slice (constant Z)
    im0 = axes[0].imshow(pressure[:, :, mid_z].T,
                        cmap='RdBu_r', origin='lower')
    axes[0].set_title(f'XY Slice (Z={mid_z}) - Pressure')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im0, ax=axes[0], label='Pressure')
    
    # XZ slice (constant Y)
    im1 = axes[1].imshow(pressure[:, mid_y, :].T,
                        cmap='RdBu_r', origin='lower')
    axes[1].set_title(f'XZ Slice (Y={mid_y}) - Pressure')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    plt.colorbar(im1, ax=axes[1], label='Pressure')
    
    # YZ slice (constant X)
    im2 = axes[2].imshow(pressure[mid_x, :, :].T,
                        cmap='RdBu_r', origin='lower')
    axes[2].set_title(f'YZ Slice (X={mid_x}) - Pressure')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    plt.colorbar(im2, ax=axes[2], label='Pressure')
    
    plt.tight_layout()
    return fig


def visualize_sample_comparison(sample_t: Dict[str, np.ndarray], 
                               sample_t1: Optional[Dict[str, np.ndarray]] = None,
                               show_3d: bool = False):
    """Visualize comparison between time steps."""
    
    if show_3d:
        # Create 3D visualization
        fig_3d = plt.figure(figsize=(15, 10))
        
        # Obstacle mask in 3D
        ax1 = fig_3d.add_subplot(231, projection='3d')
        visualize_obstacle_mask_3d(sample_t['obstacle_mask'], ax1)
        
        # Velocity magnitude isosurface (simplified as slice in 3D)
        ax2 = fig_3d.add_subplot(232, projection='3d')
        vel_mag = np.sqrt(np.sum(sample_t['velocity_t']**2, axis=3))
        mid_z = vel_mag.shape[2] // 2
        
        # Create surface plot
        X, Y = np.meshgrid(range(vel_mag.shape[0]), range(vel_mag.shape[1]))
        Z = np.ones_like(X) * mid_z
        
        surf = ax2.plot_surface(X, Y, vel_mag[:, :, mid_z].T, 
                               cmap='viridis', alpha=0.8)
        ax2.set_title(f'Velocity Magnitude at Z={mid_z}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('|V|')
        
        # Vector field in 3D (downsampled)
        ax3 = fig_3d.add_subplot(233, projection='3d')
        step = max(1, sample_t['velocity_t'].shape[0] // 6)
        X, Y, Z = np.meshgrid(np.arange(0, sample_t['velocity_t'].shape[0], step),
                             np.arange(0, sample_t['velocity_t'].shape[1], step),
                             np.arange(0, sample_t['velocity_t'].shape[2], step), indexing='ij')
        
        U = sample_t['velocity_t'][::step, ::step, ::step, 0].flatten()
        V = sample_t['velocity_t'][::step, ::step, ::step, 1].flatten()
        W = sample_t['velocity_t'][::step, ::step, ::step, 2].flatten()
        
        # Color by magnitude
        colors = np.sqrt(U**2 + V**2 + W**2)
        
        quiver = ax3.quiver(X.flatten(), Y.flatten(), Z.flatten(),
                           U, V, W, length=1, 
                           normalize=True, cmap='viridis')
        ax3.set_title('3D Velocity Vectors')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        plt.colorbar(quiver, ax=ax3, label='Velocity Magnitude')
        
        plt.tight_layout()
        plt.show()
    
    # Create 2D visualizations
    print("Visualizing sample data...")
    
    # Obstacle mask slices
    fig_obs = visualize_obstacle_mask_slices(sample_t['obstacle_mask'])
    fig_obs.suptitle('Obstacle Mask', fontsize=16)
    
    # Velocity at time t
    fig_vel_t = visualize_velocity_field(sample_t['velocity_t'])
    fig_vel_t.suptitle('Velocity Field at Time T', fontsize=16)
    
    # Velocity at time t+1 (if available)
    if sample_t1 is not None:
        fig_vel_t1 = visualize_velocity_field(sample_t1['velocity_t1'])
        fig_vel_t1.suptitle('Velocity Field at Time T+1', fontsize=16)
    
    # Pressure at time t+1
    if 'pressure_t1' in sample_t:
        fig_pressure = visualize_pressure_field(sample_t['pressure_t1'])
        fig_pressure.suptitle('Pressure Field at Time T+1', fontsize=16)
    
    plt.show()


def visualize_statistics(data_dir: Path, num_samples: int = 10):
    """Visualize statistics across multiple samples."""
    sample_files = sorted(data_dir.glob("sample_*.npy"))[:num_samples]
    
    if not sample_files:
        print(f"No sample files found in {data_dir}")
        return
    
    print(f"Analyzing {len(sample_files)} samples...")
    
    # Collect statistics
    velocities_t = []
    velocities_t1 = []
    pressures = []
    obstacle_densities = []
    
    for sample_file in sample_files:
        sample = load_sample(sample_file)
        
        # Velocity statistics
        vel_t = sample['velocity_t']
        vel_t1 = sample['velocity_t1']
        velocities_t.append(vel_t)
        velocities_t1.append(vel_t1)
        
        # Pressure statistics
        if 'pressure_t1' in sample:
            pressures.append(sample['pressure_t1'])
        
        # Obstacle density
        obstacle_density = sample['obstacle_mask'].mean()
        obstacle_densities.append(obstacle_density)
    
    # Convert to arrays for analysis
    velocities_t = np.array(velocities_t)
    velocities_t1 = np.array(velocities_t1)
    
    # Create statistics plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Obstacle density distribution
    axes[0, 0].hist(obstacle_densities, bins=20, edgecolor='black')
    axes[0, 0].set_xlabel('Obstacle Density')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Obstacle Density Distribution')
    axes[0, 0].axvline(np.mean(obstacle_densities), color='red', 
                      linestyle='--', label=f'Mean: {np.mean(obstacle_densities):.3f}')
    axes[0, 0].legend()
    
    # 2. Velocity magnitude at T
    vel_mag_t = np.sqrt(np.sum(velocities_t**2, axis=-1))
    axes[0, 1].hist(vel_mag_t.flatten(), bins=50, alpha=0.7, 
                   label='T', edgecolor='black')
    axes[0, 1].set_xlabel('Velocity Magnitude')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Velocity Magnitude Distribution at T')
    axes[0, 1].legend()
    
    # 3. Velocity magnitude at T+1
    vel_mag_t1 = np.sqrt(np.sum(velocities_t1**2, axis=-1))
    axes[0, 2].hist(vel_mag_t1.flatten(), bins=50, alpha=0.7,
                   label='T+1', edgecolor='black')
    axes[0, 2].set_xlabel('Velocity Magnitude')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Velocity Magnitude Distribution at T+1')
    axes[0, 2].legend()
    
    # 4. Velocity component distributions
    components = ['U', 'V', 'W']
    colors = ['red', 'green', 'blue']
    
    for i, (comp, color) in enumerate(zip(components, colors)):
        axes[1, 0].hist(velocities_t[..., i].flatten(), bins=50, 
                       alpha=0.5, color=color, label=f'{comp} at T',
                       edgecolor='black', density=True)
        axes[1, 0].hist(velocities_t1[..., i].flatten(), bins=50,
                       alpha=0.3, color=color, label=f'{comp} at T+1',
                       edgecolor='black', linestyle='dashed', 
                       histtype='step', density=True)
    axes[1, 0].set_xlabel('Velocity Component Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Velocity Component Distributions')
    axes[1, 0].legend()
    
    # 5. Pressure distribution (if available)
    if pressures:
        pressures = np.array(pressures)
        axes[1, 1].hist(pressures.flatten(), bins=50, edgecolor='black')
        axes[1, 1].set_xlabel('Pressure')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Pressure Distribution')
        axes[1, 1].axvline(np.mean(pressures), color='red', 
                          linestyle='--', label=f'Mean: {np.mean(pressures):.3f}')
        axes[1, 1].legend()
    
    # 6. Scatter: obstacle density vs velocity magnitude
    mean_vel_mag_t = np.mean(vel_mag_t, axis=(1, 2, 3))
    mean_vel_mag_t1 = np.mean(vel_mag_t1, axis=(1, 2, 3))
    
    axes[1, 2].scatter(obstacle_densities, mean_vel_mag_t, 
                      alpha=0.7, label='Mean |V| at T')
    axes[1, 2].scatter(obstacle_densities, mean_vel_mag_t1,
                      alpha=0.7, marker='x', label='Mean |V| at T+1')
    axes[1, 2].set_xlabel('Obstacle Density')
    axes[1, 2].set_ylabel('Mean Velocity Magnitude')
    axes[1, 2].set_title('Obstacle Density vs Velocity')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(f'Dataset Statistics ({len(sample_files)} samples)', fontsize=16)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize PhiFlow 3D flow dataset")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Path to dataset directory (contains sample_*.npy files)")
    parser.add_argument("--sample-idx", type=int, default=0,
                       help="Sample index to visualize (default: 0)")
    parser.add_argument("--show-3d", action="store_true",
                       help="Show 3D visualizations (requires matplotlib 3D)")
    parser.add_argument("--stats", action="store_true",
                       help="Show dataset statistics instead of individual sample")
    parser.add_argument("--num-stats-samples", type=int, default=10,
                       help="Number of samples to use for statistics")
    parser.add_argument("--compare-time-steps", action="store_true",
                       help="Compare velocity at T and T+1 for the same sample")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # Load metadata
    metadata = load_metadata(data_dir)
    if metadata:
        print("Dataset metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    if args.stats:
        # Show dataset statistics
        visualize_statistics(data_dir, args.num_stats_samples)
    else:
        # Load specific sample
        sample_files = sorted(data_dir.glob("sample_*.npy"))
        
        if not sample_files:
            print(f"No sample files found in {data_dir}")
            return
        
        if args.sample_idx >= len(sample_files):
            print(f"Error: Sample index {args.sample_idx} out of range (max: {len(sample_files)-1})")
            return
        
        sample_path = sample_files[args.sample_idx]
        print(f"Loading sample {args.sample_idx}: {sample_path.name}")
        
        sample_t = load_sample(sample_path)
        
        if args.compare_time_steps:
            # For comparison, we need to load the same sample but show both time steps
            # In our dataset, each sample contains both T and T+1
            visualize_sample_comparison(sample_t, sample_t, args.show_3d)
        else:
            visualize_sample_comparison(sample_t, show_3d=args.show_3d)


if __name__ == "__main__":
    main()