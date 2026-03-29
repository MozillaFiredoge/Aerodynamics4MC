import numpy as np
import pyvista as pv
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import sys
import os
from tqdm import tqdm
import imageio
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

# 设置全局主题
pv.global_theme.allow_empty_mesh = True
pv.global_theme.interactive = True

def load_sample(sample_path):
    """加载样本数据"""
    data = np.load(sample_path, allow_pickle=True).item()
    return data

def create_pyvista_grid(grid_size=32, bounds=None):
    """创建PyVista网格对象"""
    if bounds is None:
        bounds = [0, grid_size, 0, grid_size, 0, grid_size]
    
    grid = pv.ImageData()
    grid.dimensions = (grid_size, grid_size, grid_size)
    grid.origin = (bounds[0], bounds[2], bounds[4])
    grid.spacing = ((bounds[1]-bounds[0])/(grid_size-1),
                    (bounds[3]-bounds[2])/(grid_size-1),
                    (bounds[5]-bounds[4])/(grid_size-1))
    return grid

def add_scalar_field(grid, scalar_data, name):
    """添加标量场到网格"""
    if scalar_data.ndim == 3:
        flattened = scalar_data.flatten(order='F')
        grid[name] = flattened
    else:
        print(f"警告: {name} 的形状 {scalar_data.shape} 不正确")
    return grid

def add_vector_field(grid, vector_data, name):
    """添加矢量场到网格"""
    if vector_data.ndim == 4 and vector_data.shape[3] == 3:
        flattened = vector_data.reshape(-1, 3, order='F')
        grid[name] = flattened
    else:
        print(f"警告: {name} 的形状 {vector_data.shape} 不正确")
    return grid

class CFDAnimation:
    """CFD数据动画播放器"""
    
    def __init__(self, data, grid_size=32):
        self.data = data
        self.grid_size = grid_size
        
        # 检查数据形状
        self.check_data_shapes()
        
        # 获取时间步数
        self.num_timesteps = self.get_num_timesteps()
        print(f"数据包含 {self.num_timesteps} 个时间步")
        
        # 创建网格
        self.grid = create_pyvista_grid(grid_size)
        
        # 添加初始数据
        self.update_time_step(0)
        
        # 创建绘图器
        self.plotter = None
        self.current_time = 0
        
    def check_data_shapes(self):
        """检查数据形状"""
        print("数据形状检查:")
        for key, value in self.data.items():
            print(f"  {key}: {value.shape}")
            
    def get_num_timesteps(self):
        """获取时间步数"""
        if 'velocity_tn' in self.data:
            return len(self.data['velocity_tn'])
        elif 'obstacle_mask_tn' in self.data:
            return len(self.data['obstacle_mask_tn'])
        else:
            return 1  # 只有单个时间步
    
    def update_time_step(self, time_idx):
        """更新到指定时间步的数据"""
        if time_idx < 0 or time_idx >= self.num_timesteps:
            raise ValueError(f"时间索引 {time_idx} 超出范围")
            
        self.current_time = time_idx
        
        # 清除旧数据
        for key in list(self.grid.array_names):
            if key != 'vtkOriginalPointIds':
                self.grid.RemoveAllObservers()
        
        # 添加障碍物掩码
        if 'obstacle_mask_tn' in self.data:
            obstacle_data = self.data['obstacle_mask_tn'][time_idx]
        elif 'obstacle_mask' in self.data:
            obstacle_data = self.data['obstacle_mask']
        else:
            obstacle_data = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        
        self.grid = add_scalar_field(self.grid, obstacle_data.astype(float), 'obstacle_mask')
        
        # 添加速度场
        if 'velocity_tn' in self.data:
            velocity_data = self.data['velocity_tn'][time_idx]
            self.grid = add_vector_field(self.grid, velocity_data, 'velocity')
            
            # 计算速度大小
            speed = np.sqrt(np.sum(velocity_data**2, axis=3))
            self.grid = add_scalar_field(self.grid, speed, 'speed')
        
        # 添加压力场
        if 'pressure_tn' in self.data:
            pressure_data = self.data['pressure_tn'][time_idx]
            self.grid = add_scalar_field(self.grid, pressure_data, 'pressure')
    
    def create_interactive_player(self):
        """创建交互式动画播放器"""
        if self.plotter is not None:
            self.plotter.close()
            
        self.plotter = pv.Plotter(shape=(2, 2), window_size=(1400, 800))
        self.plotter.add_key_event('Right', self.next_time_step)
        self.plotter.add_key_event('Left', self.prev_time_step)
        self.plotter.add_key_event('Space', self.toggle_animation)
        self.plotter.add_key_event('r', self.reset_view)
        
        # 创建初始可视化
        self.update_visualization()
        
        # 添加时间步显示
        self.time_text = self.plotter.add_text(
            f"时间步: {self.current_time}/{self.num_timesteps-1}\n"
            f"控制: ←/→ 切换时间步, 空格 播放/暂停, R 重置视图",
            position='upper_left',
            font_size=10
        )
        
        # 动画控制
        self.animating = False
        self.animation_timer = None
        
        return self.plotter
    
    def update_visualization(self):
        """更新可视化"""
        if self.plotter is None:
            return
            
        # 清除所有actors（除了时间文本）
        for actor in self.plotter.renderer.actors.values():
            #if actor != self.time_text:
            self.plotter.remove_actor(actor)
        
        # 1. 障碍物3D视图
        self.plotter.subplot(0, 0)
        self.plotter.add_text("障碍物", font_size=12, position='upper_edge')
        
        # 提取障碍物表面
        obstacle_threshold = self.grid.threshold([0.5, 1.5], scalars='obstacle_mask')
        if obstacle_threshold.n_points > 0:
            self.plotter.add_mesh(obstacle_threshold, color='tan', 
                                opacity=0.8, show_edges=True, line_width=1)
        
        self.plotter.add_axes()
        
        # 2. 速度大小等值面
        self.plotter.subplot(0, 1)
        self.plotter.add_text("速度大小等值面", font_size=12, position='upper_edge')
        
        if 'speed' in self.grid.array_names:
            # 创建等值面
            contours = self.grid.contour(isosurfaces=5, scalars='speed')
            if contours.n_points > 0:
                self.plotter.add_mesh(contours, scalars='speed', 
                                    cmap='viridis', opacity=0.7,
                                    show_scalar_bar=True, 
                                    scalar_bar_args={'title': '速度大小'})
        
        self.plotter.add_axes()
        
        # 3. 压力场切片
        self.plotter.subplot(1, 0)
        self.plotter.add_text("压力场切片", font_size=12, position='upper_edge')
        
        if 'pressure' in self.grid.array_names:
            # 在三个方向上添加切片
            slices = self.grid.slice_orthogonal(x=self.grid_size//2, 
                                              y=self.grid_size//2, 
                                              z=self.grid_size//2)
            self.plotter.add_mesh(slices, scalars='pressure', 
                                cmap='RdBu_r', opacity=0.8,
                                show_scalar_bar=True, 
                                scalar_bar_args={'title': '压力'})
        
        self.plotter.add_axes()
        
        # 4. 速度流线
        self.plotter.subplot(1, 1)
        self.plotter.add_text("速度流线", font_size=12, position='upper_edge')
        
        if 'velocity' in self.grid.array_names:
            # 创建流线
            try:
                n_seeds = 50
                seed_points = np.zeros((n_seeds, 3))
                seed_points[:, 0] = 2  # x坐标固定为2（流入区域）
                seed_points[:, 1] = np.random.uniform(0, self.grid_size, n_seeds)
                seed_points[:, 2] = np.random.uniform(0, self.grid_size, n_seeds)
                
                streams = self.grid.streamlines(
                    vectors='velocity',
                    source_center=(2, self.grid_size/2, self.grid_size/2),
                    source_radius=self.grid_size,
                    n_points=n_seeds,
                    max_time=200.0,
                    integration_direction='forward'
                )
                
                if streams.n_points > 0:
                    self.plotter.add_mesh(streams, line_width=2, color='cyan')
            except Exception as e:
                print(f"流线生成失败: {e}")
        
        self.plotter.add_axes()
        
        # 更新时间文本
        #self.time_text.SetText(
        #    f"时间步: {self.current_time}/{self.num_timesteps-1}\n"
        #    f"控制: ←/→ 切换时间步, 空格 播放/暂停, R 重置视图"
        #)
    
    def next_time_step(self):
        """下一时间步"""
        if self.current_time < self.num_timesteps - 1:
            self.current_time += 1
            self.update_time_step(self.current_time)
            self.update_visualization()
            self.plotter.update()
    
    def prev_time_step(self):
        """上一时间步"""
        if self.current_time > 0:
            self.current_time -= 1
            self.update_time_step(self.current_time)
            self.update_visualization()
            self.plotter.update()
    
    def toggle_animation(self):
        """切换动画播放/暂停"""
        self.animating = not self.animating
        if self.animating:
            self.start_animation()
        else:
            self.stop_animation()
    
    def start_animation(self):
        """开始动画"""
        def animate():
            if self.animating and self.current_time < self.num_timesteps - 1:
                self.next_time_step()
                self.plotter.app.process_events()
                self.animation_timer = self.plotter.app.call_later(0.1, animate)
            else:
                self.animating = False
        
        self.animation_timer = self.plotter.app.call_later(0.1, animate)
    
    def stop_animation(self):
        """停止动画"""
        self.animating = False
        if self.animation_timer:
            self.animation_timer.stop()
    
    def reset_view(self):
        """重置视图"""
        for i in range(4):
            self.plotter.subplot(i // 2, i % 2)
            self.plotter.reset_camera()
        self.plotter.update()
    
    def show(self):
        """显示交互式播放器"""
        if self.plotter is None:
            self.create_interactive_player()
        self.plotter.show()
    
    def create_video(self, output_path="cfd_animation.mp4", fps=10):
        """创建视频文件"""
        print(f"开始创建视频，帧率: {fps} fps")
        
        # 创建临时目录保存帧
        temp_dir = Path("temp_frames")
        temp_dir.mkdir(exist_ok=True)
        
        # 创建简单的可视化
        plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
        
        frames = []
        for t in tqdm.tqdm(range(self.num_timesteps), desc="生成视频帧"):
            # 更新到当前时间步
            self.update_time_step(t)
            
            # 清除旧actors
            plotter.clear()
            
            # 添加障碍物
            obstacle_threshold = self.grid.threshold([0.5, 1.5], scalars='obstacle_mask')
            if obstacle_threshold.n_points > 0:
                plotter.add_mesh(obstacle_threshold, color='tan', 
                               opacity=0.8, show_edges=False)
            
            # 添加速度等值面
            if 'speed' in self.grid.array_names:
                contours = self.grid.contour(isosurfaces=3, scalars='speed')
                if contours.n_points > 0:
                    plotter.add_mesh(contours, scalars='speed', 
                                   cmap='viridis', opacity=0.7)
            
            # 添加时间步文本
            plotter.add_text(f"时间步: {t}/{self.num_timesteps-1}", 
                           position='upper_left', font_size=12)
            
            # 添加标题
            plotter.add_text("CFD 时间演化", position='upper_edge', font_size=16)
            
            # 设置相机
            plotter.camera_position = 'iso'
            plotter.camera.zoom(1.2)
            
            # 渲染并保存帧
            frame_path = temp_dir / f"frame_{t:04d}.png"
            plotter.screenshot(str(frame_path))
            frames.append(str(frame_path))
        
        plotter.close()
        
        # 使用imageio创建视频
        print("正在编码视频...")
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame_path in tqdm.tqdm(frames, desc="编码视频"):
                image = imageio.imread(frame_path)
                writer.append_data(image)
        
        # 清理临时文件
        for frame_path in frames:
            os.remove(frame_path)
        temp_dir.rmdir()
        
        print(f"视频已保存至: {output_path}")
        return output_path
    
    def create_matplotlib_animation(self, output_path="cfd_animation.html"):
        """创建基于matplotlib的交互式动画（可在Jupyter中播放）"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # 预计算所有时间步的切片数据
        print("正在预处理数据...")
        obstacle_slices = []
        speed_slices = []
        pressure_slices = []
        vector_fields = []
        
        for t in tqdm.tqdm(range(self.num_timesteps), desc="预处理"):
            self.update_time_step(t)
            
            # 获取中间切片
            mid_slice = self.grid_size // 2
            
            # 障碍物切片
            obstacle_slice = self.data.get('obstacle_mask_tn', [self.data.get('obstacle_mask', np.zeros((self.grid_size, self.grid_size, self.grid_size)))] * self.num_timesteps)[t][:, :, mid_slice]
            obstacle_slices.append(obstacle_slice)
            
            # 速度大小切片
            if 'velocity_tn' in self.data:
                speed = np.sqrt(np.sum(self.data['velocity_tn'][t]**2, axis=3))
                speed_slices.append(speed[:, :, mid_slice])
            else:
                speed_slices.append(np.zeros((self.grid_size, self.grid_size)))
            
            # 压力切片
            if 'pressure_tn' in self.data:
                pressure_slices.append(self.data['pressure_tn'][t][:, :, mid_slice])
            else:
                pressure_slices.append(np.zeros((self.grid_size, self.grid_size)))
            
            # 矢量场（稀疏采样）
            if 'velocity_tn' in self.data:
                step = self.grid_size // 8
                x, y = np.meshgrid(np.arange(0, self.grid_size, step),
                                  np.arange(0, self.grid_size, step))
                u = self.data['velocity_tn'][t][x, y, mid_slice, 0]
                v = self.data['velocity_tn'][t][x, y, mid_slice, 1]
                vector_fields.append((x, y, u, v))
        
        # 创建更新函数
        def update_frame(t):
            # 清除所有axes
            for ax in axes:
                ax.clear()
            
            # 1. 障碍物
            axes[0].imshow(obstacle_slices[t], cmap='binary', interpolation='nearest')
            axes[0].set_title(f"障碍物 (t={t})")
            axes[0].set_xlabel("X")
            axes[0].set_ylabel("Y")
            axes[0].grid(False)
            
            # 2. 速度大小
            im_speed = axes[1].imshow(speed_slices[t], cmap='viridis', interpolation='bilinear')
            axes[1].set_title("速度大小")
            axes[1].set_xlabel("X")
            axes[1].set_ylabel("Y")
            axes[1].grid(False)
            plt.colorbar(im_speed, ax=axes[1], fraction=0.046, pad=0.04)
            
            # 3. 压力场
            pressure_data = pressure_slices[t]
            vmax = max(abs(pressure_data.min()), abs(pressure_data.max())) or 1.0
            im_pressure = axes[2].imshow(pressure_data, cmap='RdBu_r', 
                                        vmin=-vmax, vmax=vmax, interpolation='bilinear')
            axes[2].set_title("压力场")
            axes[2].set_xlabel("X")
            axes[2].set_ylabel("Y")
            axes[2].grid(False)
            plt.colorbar(im_pressure, ax=axes[2], fraction=0.046, pad=0.04)
            
            # 4. 速度矢量场
            if vector_fields:
                x, y, u, v = vector_fields[t]
                axes[3].quiver(x, y, u, v, color='red', scale=50, width=0.002)
                axes[3].set_xlim(0, self.grid_size)
                axes[3].set_ylim(0, self.grid_size)
                axes[3].set_aspect('equal')
                axes[3].set_title("速度矢量场")
                axes[3].set_xlabel("X")
                axes[3].set_ylabel("Y")
                axes[3].grid(True, alpha=0.3)
            
            plt.suptitle(f"CFD 时间演化 - 时间步: {t}/{self.num_timesteps-1}", fontsize=16)
            plt.tight_layout()
        
        # 创建动画
        print("正在创建动画...")
        anim = FuncAnimation(fig, update_frame, frames=self.num_timesteps,
                           interval=200, repeat=True)
        
        # 保存为HTML（可在Jupyter中播放）
        if output_path.endswith('.html'):
            print("正在保存为HTML...")
            html = anim.to_jshtml()
            with open(output_path, 'w') as f:
                f.write(html)
            print(f"HTML动画已保存至: {output_path}")
            
            # 在Jupyter中显示
            plt.close(fig)
            return HTML(html)
        else:
            # 保存为视频
            print("正在保存为视频...")
            anim.save(output_path, writer='ffmpeg', fps=5, dpi=100)
            print(f"视频已保存至: {output_path}")
            plt.close(fig)
            return output_path

def visualize_single_sample_with_animation(data, sample_idx=0, mode='interactive'):
    """可视化单个样本（支持动画）"""
    print(f"可视化样本 {sample_idx}")
    
    # 创建动画播放器
    grid_size = data.get('obstacle_mask', np.zeros((32, 32, 32))).shape[0]
    player = CFDAnimation(data, grid_size)
    
    if mode == 'interactive':
        # 交互式播放器（可暂停、跳转）
        player.show()
    elif mode == 'video':
        # 生成视频文件
        player.create_video(f"cfd_sample_{sample_idx}_animation.mp4")
    elif mode == 'web':
        # 生成网页动画（适用于Jupyter）
        return player.create_matplotlib_animation(f"cfd_sample_{sample_idx}_animation.html")
    elif mode == 'simple':
        # 简单的时间步浏览
        return visualize_time_step_browser(data, sample_idx)
    else:
        print(f"未知模式: {mode}")

def visualize_time_step_browser(data, sample_idx):
    """时间步浏览器（简单的matplotlib实现）"""
    import ipywidgets as widgets
    from IPython.display import display
    
    # 获取时间步数
    if 'velocity_tn' in data:
        num_timesteps = len(data['velocity_tn'])
    elif 'obstacle_mask_tn' in data:
        num_timesteps = len(data['obstacle_mask_tn'])
    else:
        num_timesteps = 1
    
    grid_size = data.get('obstacle_mask', np.zeros((32, 32, 32))).shape[0]
    
    # 创建控件
    time_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=num_timesteps-1,
        step=1,
        description='时间步:',
        continuous_update=False
    )
    
    play_button = widgets.Play(
        interval=200,
        value=0,
        min=0,
        max=num_timesteps-1,
        step=1,
        description="播放"
    )
    
    widgets.jslink((play_button, 'value'), (time_slider, 'value'))
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    def update_plot(t):
        # 清除所有axes
        for ax in axes:
            ax.clear()
        
        # 获取中间切片
        mid_slice = grid_size // 2
        
        # 1. 障碍物
        if 'obstacle_mask_tn' in data:
            obstacle_data = data['obstacle_mask_tn'][t]
        else:
            obstacle_data = data.get('obstacle_mask', np.zeros((grid_size, grid_size, grid_size)))
        
        axes[0].imshow(obstacle_data[:, :, mid_slice], cmap='binary')
        axes[0].set_title(f"障碍物 (t={t})")
        axes[0].axis('off')
        
        # 2. 速度大小
        if 'velocity_tn' in data:
            speed = np.sqrt(np.sum(data['velocity_tn'][t]**2, axis=3))
            im = axes[1].imshow(speed[:, :, mid_slice], cmap='viridis')
            axes[1].set_title("速度大小")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # 3. 压力场
        if 'pressure_tn' in data:
            pressure_data = data['pressure_tn'][t][:, :, mid_slice]
            vmax = max(abs(pressure_data.min()), abs(pressure_data.max())) or 1.0
            im = axes[2].imshow(pressure_data, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[2].set_title("压力场")
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.suptitle(f"样本 {sample_idx} - 时间步: {t}/{num_timesteps-1}", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # 初始显示
    update_plot(0)
    
    # 连接控件
    widgets.interactive(update_plot, t=time_slider)
    
    # 显示控件
    controls = widgets.HBox([play_button, time_slider])
    display(controls)
    
    return controls

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CFD数据可视化工具（支持动画）")
    parser.add_argument("--data-dir", type=str, default="data_fixed/grid_32",
                       help="数据目录路径")
    parser.add_argument("--sample-idx", type=int, default=0,
                       help="样本索引")
    parser.add_argument("--mode", type=str, default="interactive",
                       choices=["interactive", "video", "web", "simple", "browser"],
                       help="可视化模式: interactive=交互式3D, video=生成视频, web=网页动画, simple=简单视图, browser=时间步浏览器")
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        return
    
    # 加载元数据
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        print(f"数据集信息: {metadata}")
    
    # 加载样本
    sample_files = sorted(data_dir.glob("sample_*.npy"))
    if args.sample_idx >= len(sample_files):
        print(f"错误: 样本索引 {args.sample_idx} 超出范围 (总共 {len(sample_files)} 个样本)")
        return
    
    data = load_sample(sample_files[args.sample_idx])
    
    # 根据模式选择可视化方式
    if args.mode == "browser":
        # 在Jupyter环境中使用
        visualize_time_step_browser(data, args.sample_idx)
    else:
        # 其他可视化模式
        result = visualize_single_sample_with_animation(
            data, 
            args.sample_idx, 
            mode=args.mode
        )
        
        if args.output and hasattr(result, 'save'):
            result.save(args.output)
            print(f"结果已保存至: {args.output}")

if __name__ == "__main__":
    # 检查必要的库
    try:
        import pyvista
        print(f"PyVista版本: {pv.__version__}")
    except ImportError:
        print("错误: 请先安装PyVista: pip install pyvista")
        sys.exit(1)
    
    try:
        import tqdm
    except ImportError:
        print("警告: 未安装tqdm，进度显示将不可用。安装: pip install tqdm")
    
    main()