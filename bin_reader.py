# -*- coding: utf-8 -*-
"""
TI mmWave 雷达 bin 文件读取器
用于读取 mmWaveStudio + DCA1000 采集的 adc_data.bin 文件
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (numADCSamples, numTxAntennas, numRxAntennas, 
                    numLoopsPerFrame, numRangeBins, BIN_FILE_PATH)


class BinFileReader:
    """
    读取 TI mmWave 雷达的 bin 文件
    
    数据格式说明:
    - IWR1843 使用 16-bit 有符号整数存储 ADC 数据
    - 复数数据: I (实部) 和 Q (虚部) 交替存储
    - 通道非交织模式 (chInterleave = 1)
    """
    
    def __init__(self, filename=None):
        """
        初始化 bin 文件读取器
        
        Args:
            filename: bin 文件路径，默认使用配置文件中的路径
        """
        self.filename = filename if filename else BIN_FILE_PATH
        self.numADCSamples = numADCSamples
        self.numTxAntennas = numTxAntennas
        self.numRxAntennas = numRxAntennas
        self.numLoopsPerFrame = numLoopsPerFrame
        self.numRangeBins = numRangeBins
        
        # 每帧的采样点数 (复数数据，所以乘以2)
        # 对于 TDM MIMO: samples_per_frame = numRx * numLoops * numTx * numADCSamples * 2 (I/Q)
        self.samples_per_frame = (self.numRxAntennas * self.numLoopsPerFrame * 
                                   self.numTxAntennas * self.numADCSamples * 2)
        
        # 读取原始数据
        self._load_data()
        
        self.current_frame = 0
        
    def _load_data(self):
        """加载 bin 文件数据"""
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"找不到文件: {self.filename}")
        
        # 读取为 16-bit 有符号整数
        self.raw_data = np.fromfile(self.filename, dtype=np.int16)
        
        # 计算帧数
        self.n_frames = int(len(self.raw_data) / self.samples_per_frame)
        
        print(f"=== bin 文件信息 ===")
        print(f"文件路径: {self.filename}")
        print(f"文件大小: {len(self.raw_data) * 2 / 1024 / 1024:.2f} MB")
        print(f"总采样点数: {len(self.raw_data)}")
        print(f"每帧采样点数: {self.samples_per_frame}")
        print(f"总帧数: {self.n_frames}")
        
    def get_frame(self, frame_idx):
        """
        获取指定帧的数据
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            data_cube: 形状为 (numRx, numChirps, numADCSamples) 的复数数组
        """
        if frame_idx >= self.n_frames:
            raise IndexError(f"帧索引 {frame_idx} 超出范围 (总帧数: {self.n_frames})")
        
        # 提取该帧的原始数据
        start_idx = frame_idx * self.samples_per_frame
        end_idx = start_idx + self.samples_per_frame
        frame_data = self.raw_data[start_idx:end_idx]
        
        # 重组数据为复数格式
        # 数据排列: I0, Q0, I1, Q1, ... 
        num_complex_samples = len(frame_data) // 2
        complex_data = frame_data[0::2] + 1j * frame_data[1::2]
        
        # 重塑为数据立方体
        # 对于 TDM MIMO，数据排列为: [chirp0_rx0, chirp0_rx1, ..., chirp1_rx0, ...]
        # 重塑为 (numTx * numLoops, numRx, numADCSamples)
        try:
            data_cube = complex_data.reshape(
                (self.numLoopsPerFrame * self.numTxAntennas, 
                 self.numRxAntennas, 
                 self.numADCSamples)
            )
            # 转置为 (numRx, numChirps, numADCSamples)
            data_cube = np.transpose(data_cube, (1, 0, 2))
        except ValueError as e:
            print(f"数据重塑失败: {e}")
            print(f"复数采样点数: {num_complex_samples}")
            print(f"期望大小: {self.numLoopsPerFrame * self.numTxAntennas * self.numRxAntennas * self.numADCSamples}")
            raise
            
        return data_cube
    
    def get_next_frame(self):
        """获取下一帧数据"""
        if self.current_frame >= self.n_frames:
            print("已到达文件末尾，重置到开始")
            self.current_frame = 0
            
        data_cube = self.get_frame(self.current_frame)
        self.current_frame += 1
        return data_cube
    
    def reset(self):
        """重置帧计数器"""
        self.current_frame = 0
        

def test_reader():
    """测试读取器功能"""
    try:
        reader = BinFileReader()
        
        # 读取第一帧
        frame_data = reader.get_frame(0)
        print(f"\n=== 第一帧数据信息 ===")
        print(f"数据形状: {frame_data.shape}")
        print(f"数据类型: {frame_data.dtype}")
        print(f"数据范围: [{frame_data.real.min():.0f}, {frame_data.real.max():.0f}] (实部)")
        print(f"数据范围: [{frame_data.imag.min():.0f}, {frame_data.imag.max():.0f}] (虚部)")
        
        return reader
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_reader()
