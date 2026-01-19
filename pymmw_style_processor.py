# -*- coding: utf-8 -*-
"""
pymmw 风格的雷达信号处理器
生成 Azimuth-Range 和 Doppler-Range 热图，支持视频输出
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.animation as animation
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (numADCSamples, numLoopsPerFrame, numTxAntennas, numRxAntennas,
                    startFreq_GHz, freqSlope_MHz_us, idleTime_us, rampEndTime_us, 
                    SPEED_OF_LIGHT, BIN_FILE_PATH, adcSampleRate_ksps)
from bin_reader import BinFileReader


class PymmwStyleProcessor:
    def __init__(self, display_range_max=10.0, display_velocity_max=1.0):
        self.numADCSamples = numADCSamples
        self.numLoopsPerFrame = numLoopsPerFrame
        self.numTxAntennas = numTxAntennas
        self.numRxAntennas = numRxAntennas
        
        self.display_range_max = display_range_max
        self.display_velocity_max = display_velocity_max
        
        self.numRangeBins = self._pow2_ceil(numADCSamples)
        self.numDopplerBins = numLoopsPerFrame
        self.numAngleBins = 64
        self.numVirtualAntennas = 2 * numRxAntennas
        
        self.chirp_time = idleTime_us + rampEndTime_us
        # 正确计算最大距离: range_max = (Fs * c) / (2 * S)
        # Fs = adcSampleRate_ksps * 1000, S = freqSlope_MHz_us * 1e12
        self.range_max = (adcSampleRate_ksps * 1e3 * SPEED_OF_LIGHT) / (2 * freqSlope_MHz_us * 1e12)
        print(f"计算的最大距离: {self.range_max:.2f} m")
        
        frame_chirp_time = self.chirp_time * numTxAntennas
        self.doppler_max = SPEED_OF_LIGHT / (4 * startFreq_GHz * 1e9 * frame_chirp_time * 1e-6)
        
        print(f"\n=== 处理器参数 ===")
        print(f"Range bins: {self.numRangeBins}, Angle bins: {self.numAngleBins}")
        print(f"显示范围: 0-{display_range_max}m, ±{display_velocity_max}m/s")
        
    def _pow2_ceil(self, x):
        if x < 0: return 0
        x -= 1
        x |= x >> 1
        x |= x >> 2
        x |= x >> 4
        x |= x >> 8
        x |= x >> 16
        return x + 1
    
    def process_frame(self, data_cube):
        """处理一帧数据 - 使用完整的 3D FFT (Range-Doppler-Azimuth)"""
        # TDM MIMO chirp 顺序 (根据 18xx.mmwave.json):
        # chirp 0: TX0 (0x1) - 方位
        # chirp 1: TX2 (0x4) - 方位  
        # chirp 2: TX1 (0x2) - 仰角
        # 用于方位成像的是 TX0 和 TX2 (chirp 0 和 chirp 1)
        tx0_indices = np.arange(0, data_cube.shape[1], self.numTxAntennas)  # chirp 0, 3, 6, ...
        tx2_indices = np.arange(1, data_cube.shape[1], self.numTxAntennas)  # chirp 1, 4, 7, ... (TX2)
        tx0_data = data_cube[:, tx0_indices, :]  # (4, 16, 256)
        tx2_data = data_cube[:, tx2_indices, :]  # (4, 16, 256)
        
        # ========== Range-Doppler (保持不变) ==========
        range_doppler_sum = None
        for rx_idx in range(self.numRxAntennas):
            rx_data = tx0_data[rx_idx, :, :]
            range_fft = np.fft.fft(rx_data, n=self.numRangeBins, axis=1)
            range_doppler = np.fft.fftshift(np.fft.fft(range_fft, axis=0), axes=0)
            if range_doppler_sum is None:
                range_doppler_sum = np.abs(range_doppler) ** 2
            else:
                range_doppler_sum += np.abs(range_doppler) ** 2
        
        # ========== 3D FFT: Range-Doppler-Azimuth ==========
        # 步骤1: Range FFT
        tx0_range = np.fft.fft(tx0_data, n=self.numRangeBins, axis=2)  # (4, 16, 256)
        tx2_range = np.fft.fft(tx2_data, n=self.numRangeBins, axis=2)  # (4, 16, 256)
        
        # 步骤2: Doppler FFT
        tx0_doppler = np.fft.fft(tx0_range, axis=1)  # (4, 16, 256)
        tx2_doppler = np.fft.fft(tx2_range, axis=1)  # (4, 16, 256)
        tx0_doppler = np.fft.fftshift(tx0_doppler, axes=1)
        tx2_doppler = np.fft.fftshift(tx2_doppler, axes=1)
        
        # 步骤3: 构建虚拟阵列并做 Angle FFT
        # 拼接 TX0 和 TX2 的数据: (8, 16, 256)
        virtual_3d = np.concatenate([tx0_doppler, tx2_doppler], axis=0)  # (8, 16, 256)
        
        # 对每个 (Doppler, Range) bin 做 Angle FFT
        # 转置为 (16, 256, 8) 方便处理
        virtual_3d = np.transpose(virtual_3d, (1, 2, 0))  # (16, 256, 8)
        
        # Angle FFT: 对最后一维 (8 个虚拟天线) 做 FFT
        range_doppler_azimuth = np.fft.fft(virtual_3d, n=self.numAngleBins, axis=2)  # (16, 256, 64)
        range_doppler_azimuth = np.abs(range_doppler_azimuth)
        range_doppler_azimuth = np.fft.fftshift(range_doppler_azimuth, axes=2)  # 角度居中
        
        # 步骤4: 取所有 Doppler bin 的最大值投影 (Maximum Intensity Projection)
        # 这样可以同时显示静态和移动目标
        range_azimuth = np.max(range_doppler_azimuth, axis=0)  # (256, 64)
        range_azimuth = range_azimuth.T  # (64, 256)
        
        return range_doppler_sum, range_azimuth


def create_video(reader, processor, output_path, fps=10):
    """生成视频"""
    print(f"\n正在生成视频，共 {reader.n_frames} 帧...")
    
    # 计算显示范围
    range_bins = int(processor.display_range_max / processor.range_max * processor.numRangeBins)
    range_bins = min(range_bins, processor.numRangeBins // 2)
    doppler_half = processor.numDopplerBins // 2
    vel_ratio = min(processor.display_velocity_max / processor.doppler_max, 1.0)
    doppler_bins = max(int(doppler_half * vel_ratio), 1)
    lateral_max = processor.display_range_max  # X轴范围 ±10m
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 初始化左图 - 形状 (range_bins, angle_bins) 对应 (Y=距离, X=角度)
    ax1 = axes[0]
    im1 = ax1.imshow(np.zeros((range_bins, processor.numAngleBins)), cmap='jet', aspect='auto',
                      extent=[-lateral_max, lateral_max, 0, processor.display_range_max],
                      origin='lower', interpolation='bilinear')
    ax1.plot([0, 0], [0, processor.display_range_max], 'w:', linewidth=0.5, alpha=0.7)
    ax1.plot([0, -lateral_max], [0, lateral_max], 'w:', linewidth=0.5, alpha=0.7)
    ax1.plot([0, lateral_max], [0, lateral_max], 'w:', linewidth=0.5, alpha=0.7)
    ax1.set_xlim([-lateral_max, lateral_max])
    ax1.set_ylim([0, processor.display_range_max])
    ax1.set_xlabel('Lateral distance along [m]')
    ax1.set_ylabel('Longitudinal distance along [m]')
    title1 = ax1.set_title('Azimuth-Range FFT Heatmap - Frame 0')
    
    # 初始化右图
    ax2 = axes[1]
    im2 = ax2.imshow(np.zeros((doppler_bins*2, range_bins)), cmap='jet', aspect='auto',
                      extent=[0, processor.display_range_max, 
                             -processor.display_velocity_max, processor.display_velocity_max],
                      origin='lower', interpolation='bilinear')
    ax2.grid(color='white', linestyle=':', linewidth=0.5, alpha=0.7)
    ax2.axhline(y=0, color='white', linestyle=':', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('Longitudinal distance [m]')
    ax2.set_ylabel('Radial velocity [m/s]')
    title2 = ax2.set_title('Doppler-Range FFT Heatmap - Frame 0')
    
    plt.tight_layout()
    
    def update(frame_idx):
        """更新函数"""
        frame_data = reader.get_frame(frame_idx)
        range_doppler, range_azimuth = processor.process_frame(frame_data)
        
        # 更新数据
        # 转置使得：行=距离(Y轴)，列=角度(X轴)
        # origin='lower'会让数组第一行在底部(近距离)，无需额外翻转
        azimuth_display = range_azimuth[:, :range_bins].T
        # 不翻转Y轴，因为origin='lower'已经处理了
        doppler_display = range_doppler[doppler_half-doppler_bins:doppler_half+doppler_bins, :range_bins]
        doppler_display = doppler_display.T  # 只转置，不翻转
        
        im1.set_array(azimuth_display)
        im1.autoscale()
        im2.set_array(doppler_display)
        im2.autoscale()
        
        title1.set_text(f'Azimuth-Range FFT Heatmap - Frame {frame_idx}')
        title2.set_text(f'Doppler-Range FFT Heatmap - Frame {frame_idx}')
        
        if frame_idx % 10 == 0:
            print(f"  处理帧: {frame_idx}/{reader.n_frames}")
        
        return [im1, im2, title1, title2]
    
    # 创建动画
    anim = animation.FuncAnimation(fig, update, frames=reader.n_frames, 
                                    interval=1000//fps, blit=True)
    
    # 保存视频
    print(f"正在保存视频到: {output_path}")
    
    # 尝试使用 ffmpeg，如果没有则使用 pillow 保存为 gif
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(output_path, writer=writer)
        print(f"视频已保存: {output_path}")
    except Exception as e:
        print(f"FFmpeg 不可用 ({e})，尝试保存为 GIF...")
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            writer = animation.PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer)
            print(f"GIF 已保存: {gif_path}")
        except Exception as e2:
            print(f"保存失败: {e2}")
            print("请安装 ffmpeg 或 pillow")
    
    plt.close(fig)
    return anim


def main():
    print("=" * 50)
    print("pymmw 风格雷达热图 + 视频生成")
    print("=" * 50)
    
    reader = BinFileReader()
    processor = PymmwStyleProcessor(display_range_max=15.0, display_velocity_max=1.0)
    
    # 生成单帧图像
    frame_data = reader.get_frame(0)
    range_doppler, range_azimuth = processor.process_frame(frame_data)
    print(f"Range-Azimuth 形状: {range_azimuth.shape}")
    
    # 生成视频
    output_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(output_dir, 'radar_heatmap.mp4')
    
    create_video(reader, processor, video_path, fps=10)
    
    print("\n完成!")


if __name__ == "__main__":
    main()
