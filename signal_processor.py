# -*- coding: utf-8 -*-
"""
雷达信号处理模块
用于生成 Range-Doppler 热图
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (numADCSamples, numLoopsPerFrame, numTxAntennas, numRxAntennas,
                    rangeResolution_m, startFreq_GHz, freqSlope_MHz_us, 
                    idleTime_us, rampEndTime_us, SPEED_OF_LIGHT)
from bin_reader import BinFileReader


class RangeDopplerProcessor:
    """
    Range-Doppler 信号处理器
    """
    
    def __init__(self):
        self.numADCSamples = numADCSamples
        self.numLoopsPerFrame = numLoopsPerFrame
        self.numTxAntennas = numTxAntennas
        self.numRxAntennas = numRxAntennas
        
        # 计算 Doppler 相关参数
        self.chirp_time = idleTime_us + rampEndTime_us  # chirp 周期 (us)
        self.lambda_wave = SPEED_OF_LIGHT / (startFreq_GHz * 1e9)  # 波长 (m)
        
        # 对于 TDM MIMO，有效 chirp 周期是单个 TX 的周期
        self.effective_chirp_time = self.chirp_time * numTxAntennas  # us
        
        # 最大无模糊速度 (m/s)
        self.max_velocity = self.lambda_wave / (4 * self.effective_chirp_time * 1e-6)
        
        # 速度分辨率 (m/s)
        self.velocity_resolution = self.lambda_wave / (2 * numLoopsPerFrame * self.effective_chirp_time * 1e-6)
        
        print(f"\n=== 信号处理参数 ===")
        print(f"波长: {self.lambda_wave*1000:.2f} mm")
        print(f"最大速度: {self.max_velocity:.2f} m/s ({self.max_velocity * 3.6:.2f} km/h)")
        print(f"速度分辨率: {self.velocity_resolution:.3f} m/s")
        
    def apply_window(self, data, axis, window_type='hamming'):
        """
        应用窗函数
        
        Args:
            data: 输入数据
            axis: 应用窗函数的轴
            window_type: 窗函数类型 ('hamming', 'hanning', 'blackman')
        """
        n = data.shape[axis]
        
        if window_type == 'hamming':
            window = np.hamming(n)
        elif window_type == 'hanning':
            window = np.hanning(n)
        elif window_type == 'blackman':
            window = np.blackman(n)
        else:
            window = np.ones(n)
            
        # 扩展窗函数维度以匹配数据
        shape = [1] * data.ndim
        shape[axis] = n
        window = window.reshape(shape)
        
        return data * window
    
    def process_range_doppler(self, data_cube, rx_idx=0, tx_idx=0):
        """
        处理 Range-Doppler 图
        
        Args:
            data_cube: 形状为 (numRx, numChirps, numADCSamples) 的复数数据
            rx_idx: 使用的 RX 天线索引
            tx_idx: 使用的 TX 天线索引 (用于 TDM MIMO)
            
        Returns:
            range_doppler: Range-Doppler 图 (dB)
            range_axis: 距离轴 (m)
            velocity_axis: 速度轴 (m/s)
        """
        # 对于 TDM MIMO，提取特定 TX 的 chirps
        # chirps 索引: tx_idx, tx_idx + numTx, tx_idx + 2*numTx, ...
        chirp_indices = np.arange(tx_idx, data_cube.shape[1], self.numTxAntennas)
        rx_data = data_cube[rx_idx, chirp_indices, :]  # (numLoops, numADCSamples)
        
        # 应用 Range FFT 窗函数
        rx_data_windowed = self.apply_window(rx_data, axis=1, window_type='hamming')
        
        # Range FFT (沿 ADC 采样方向)
        range_fft = np.fft.fft(rx_data_windowed, axis=1)
        # 只取正频率部分
        range_fft = range_fft[:, :self.numADCSamples // 2]
        
        # 应用 Doppler FFT 窗函数
        range_fft_windowed = self.apply_window(range_fft, axis=0, window_type='hamming')
        
        # Doppler FFT (沿 chirp 方向)
        range_doppler = np.fft.fftshift(np.fft.fft(range_fft_windowed, axis=0), axes=0)
        
        # 转换为 dB
        range_doppler_db = 20 * np.log10(np.abs(range_doppler) + 1e-10)
        
        # 生成轴标注
        num_range_bins = self.numADCSamples // 2
        range_axis = np.arange(num_range_bins) * rangeResolution_m
        
        num_doppler_bins = len(chirp_indices)
        velocity_axis = np.linspace(-self.max_velocity, self.max_velocity, num_doppler_bins)
        
        return range_doppler_db, range_axis, velocity_axis
    
    def process_range_angle(self, data_cube, tx_idx=0, doppler_idx=None):
        """
        处理 Range-Angle 图 (波束形成)
        
        Args:
            data_cube: 形状为 (numRx, numChirps, numADCSamples) 的复数数据
            tx_idx: 使用的 TX 天线索引
            doppler_idx: 使用的 Doppler bin 索引，None 表示对所有 Doppler 求和
            
        Returns:
            range_angle: Range-Angle 图 (dB)
            range_axis: 距离轴 (m)
            angle_axis: 角度轴 (度)
        """
        # 对于 TDM MIMO，提取特定 TX 的 chirps
        chirp_indices = np.arange(tx_idx, data_cube.shape[1], self.numTxAntennas)
        
        # 提取所有 RX 天线的数据
        # 形状: (numRx, numLoops, numADCSamples)
        rx_data = data_cube[:, chirp_indices, :]
        
        # Range FFT
        rx_data_windowed = self.apply_window(rx_data, axis=2, window_type='hamming')
        range_fft = np.fft.fft(rx_data_windowed, axis=2)
        range_fft = range_fft[:, :, :self.numADCSamples // 2]
        
        # Doppler FFT
        range_fft_windowed = self.apply_window(range_fft, axis=1, window_type='hamming')
        range_doppler = np.fft.fftshift(np.fft.fft(range_fft_windowed, axis=1), axes=1)
        
        # 选择 Doppler bin 或对所有 Doppler 求和
        if doppler_idx is not None:
            range_angle_data = range_doppler[:, doppler_idx, :]
        else:
            # 对 Doppler 维度求和 (非相干积累)
            range_angle_data = np.sum(np.abs(range_doppler), axis=1)
        
        # Angle FFT (波束形成)
        # 对 RX 天线维度进行 FFT，增加零填充以提高角度分辨率
        num_angle_bins = 64
        range_angle = np.fft.fftshift(np.fft.fft(range_angle_data, n=num_angle_bins, axis=0), axes=0)
        
        # 转换为 dB
        range_angle_db = 20 * np.log10(np.abs(range_angle) + 1e-10)
        
        # 生成轴标注
        num_range_bins = self.numADCSamples // 2
        range_axis = np.arange(num_range_bins) * rangeResolution_m
        
        # 角度轴 (假设半波长间距)
        angle_axis = np.arcsin(np.linspace(-1, 1, num_angle_bins)) * 180 / np.pi
        
        return range_angle_db, range_axis, angle_axis


def plot_range_doppler(range_doppler_db, range_axis, velocity_axis, title="Range-Doppler Map"):
    """绘制 Range-Doppler 热图"""
    plt.figure(figsize=(12, 8))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 转置使得 x 轴为距离，y 轴为速度
    plt.imshow(range_doppler_db, aspect='auto', origin='lower',
               extent=[range_axis[0], range_axis[-1], 
                      velocity_axis[0], velocity_axis[-1]],
               cmap='jet')
    
    plt.colorbar(label='幅度 (dB)')
    plt.xlabel('距离 (m)')
    plt.ylabel('速度 (m/s)')
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()


def plot_range_angle(range_angle_db, range_axis, angle_axis, title="Range-Angle Map"):
    """绘制 Range-Angle 热图"""
    plt.figure(figsize=(12, 8))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 转置使得 x 轴为距离，y 轴为角度
    plt.imshow(range_angle_db, aspect='auto', origin='lower',
               extent=[range_axis[0], range_axis[-1], 
                      angle_axis[0], angle_axis[-1]],
               cmap='jet')
    
    plt.colorbar(label='幅度 (dB)')
    plt.xlabel('距离 (m)')
    plt.ylabel('角度 (°)')
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()


def main():
    """主函数：处理 bin 文件并生成热图"""
    print("=" * 50)
    print("雷达信号处理 - Range-Doppler 热图生成")
    print("=" * 50)
    
    # 创建读取器
    reader = BinFileReader()
    
    # 创建处理器
    processor = RangeDopplerProcessor()
    
    # 读取第一帧数据
    frame_data = reader.get_frame(0)
    print(f"\n帧数据形状: {frame_data.shape}")
    
    # 处理 Range-Doppler 图
    range_doppler_db, range_axis, velocity_axis = processor.process_range_doppler(
        frame_data, rx_idx=0, tx_idx=0)
    
    print(f"Range-Doppler 图形状: {range_doppler_db.shape}")
    print(f"距离范围: {range_axis[0]:.2f} - {range_axis[-1]:.2f} m")
    print(f"速度范围: {velocity_axis[0]:.2f} - {velocity_axis[-1]:.2f} m/s")
    
    # 绘制 Range-Doppler 图
    fig1 = plot_range_doppler(range_doppler_db, range_axis, velocity_axis, 
                               title="Range-Doppler 热图 (帧 0)")
    
    # 处理 Range-Angle 图
    range_angle_db, range_axis2, angle_axis = processor.process_range_angle(
        frame_data, tx_idx=0, doppler_idx=None)
    
    print(f"\nRange-Angle 图形状: {range_angle_db.shape}")
    print(f"角度范围: {angle_axis[0]:.1f} - {angle_axis[-1]:.1f} °")
    
    # 绘制 Range-Angle 图
    fig2 = plot_range_angle(range_angle_db, range_axis2, angle_axis,
                            title="Range-Angle 热图 (帧 0)")
    
    # 保存图像
    output_dir = os.path.dirname(os.path.abspath(__file__))
    fig1.savefig(os.path.join(output_dir, 'range_doppler.png'), dpi=150)
    fig2.savefig(os.path.join(output_dir, 'range_angle.png'), dpi=150)
    print(f"\n图像已保存到: {output_dir}")
    
    plt.show()


if __name__ == "__main__":
    main()
