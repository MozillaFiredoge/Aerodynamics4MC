import argparse
import json
import numpy as np
from pathlib import Path
import sys

try:
    import openvdb as vdb
except ImportError:
    print("错误: 找不到 'pyopenvdb'。请安装: conda install -c conda-forge pyopenvdb")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert PhiFlow NPY to VDB (Fixed Layout)")
    parser.add_argument("--data-dir", type=str, required=True, help="Input directory")
    parser.add_argument("--out-dir", type=str, default="vdb_output", help="Output directory")
    parser.add_argument("--scale", type=float, default=1.0, help="Grid scale")
    return parser.parse_args()

def process_grid(data, name, scale):
    """
    通用网格处理函数：处理转置、内存布局和 VDB 创建
    """
    # ---------------------------------------------------------
    # 核心修复逻辑
    # ---------------------------------------------------------
    # PhiFlow 导出的是 (x, y, z)
    # OpenVDB/NumPy 交互通常偏好 (z, y, x) 的层级顺序来保持空间连续性
    # 如果不转置，数据就会变成“雪花/噪点”
    
    # 1. 矢量场处理 (X, Y, Z, 3) -> (Z, Y, X, 3)
    if data.ndim == 4 and data.shape[-1] == 3:
        # Transpose spatial dims: 0,1,2 -> 2,1,0. Keep vector dim (3) at position 3
        data_fixed = data.transpose(2, 1, 0, 3)
        data_fixed = np.ascontiguousarray(data_fixed, dtype=np.float32)
        
        grid = vdb.Vec3SGrid()
        grid.copyFromArray(data_fixed)
    
    # 2. 标量场处理 (X, Y, Z) -> (Z, Y, X)
    elif data.ndim == 3:
        # Transpose spatial dims: 0,1,2 -> 2,1,0
        data_fixed = data.transpose(2, 1, 0)
        data_fixed = np.ascontiguousarray(data_fixed, dtype=np.float32)
        
        grid = vdb.FloatGrid()
        grid.copyFromArray(data_fixed)
        
    else:
        print(f"警告: 跳过无法识别的数据形状 {data.shape}")
        return None

    grid.name = name
    # 设置变换矩阵，保证缩放正确
    grid.transform = vdb.createLinearTransform(voxelSize=scale)
    return grid

def numpy_to_vdb(npy_data, scale=1.0):
    grids = []
    
    # 1. Obstacle (标量)
    if "obstacle_mask" in npy_data:
        g = process_grid(npy_data["obstacle_mask"].astype(np.float32), "obstacle", scale)
        if g: grids.append(g)

    # 2. Velocity (矢量)
    if "velocity_t" in npy_data:
        vel_data = npy_data["velocity_t"].astype(np.float32)
        g_vel = process_grid(vel_data, "velocity", scale)
        if g_vel: grids.append(g_vel)
        
        # 3. Density/Speed (用于预览的标量)
        # 计算速度模长，方便在 Blender 默认材质中直接看到形状
        speed_data = np.linalg.norm(vel_data, axis=-1)
        g_speed = process_grid(speed_data, "density", scale)
        if g_speed: grids.append(g_speed)

    # 4. Pressure (标量)
    if "pressure_t" in npy_data:
        g_press = process_grid(npy_data["pressure_t"].astype(np.float32), "pressure", scale)
        if g_press: grids.append(g_press)

    return grids

def main():
    args = parse_args()
    in_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(list(in_dir.glob("sample_*.npy")))
    print(f"Found {len(sample_files)} samples in {in_dir}")

    for i, npy_file in enumerate(sample_files):
        try:
            payload = np.load(npy_file, allow_pickle=True).item()
            grids = numpy_to_vdb(payload, scale=args.scale)
            
            out_name = f"fluid_{i:04d}.vdb"
            vdb.write(str(out_dir / out_name), grids=grids)
            
            if i == 0:
                print(f"Sample 0 converted. Grids: {[g.name for g in grids]}")
                
        except Exception as e:
            print(f"Error converting {npy_file}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()