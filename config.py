# -*- coding: utf-8 -*-
"""
雷达参数配置文件
根据 mmWaveStudio 的 18xx.mmwave.json 配置提取
与 pymmw 的 x8_mmw-xWR18xx.cfg 配置一致
"""

# ============ 雷达配置参数 ============

# ADC 采样点数
numADCSamples = 256

# 发射天线数 (txChannelEn = 0x7)
numTxAntennas = 3

# 接收天线数 (rxChannelEn = 0xF)
numRxAntennas = 4

# 每帧循环数 (numLoops) - 与 pymmw 一致
numLoopsPerFrame = 16

# 距离 bin 数
numRangeBins = int(numADCSamples / 2)

# 总帧数
numFrames = 50

# ============ 文件路径配置 ============
BIN_FILE_PATH = r"E:\TI\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\adc_data.bin"

# ============ 雷达物理参数 ============
# 起始频率 (GHz)
startFreq_GHz = 77.0

# 频率斜率 (MHz/us) - 与 mmWave Studio 一致
freqSlope_MHz_us = 29.982

# ADC 采样率 (ksps) - digOutSampleRate
adcSampleRate_ksps = 10000

# 空闲时间 (us)
idleTime_us = 100.0

# 斜坡结束时间 (us)
rampEndTime_us = 60.0

# 帧周期 (ms)
framePeriod_ms = 80.0

# ============ 计算派生参数 ============
SPEED_OF_LIGHT = 3e8

# 有效带宽 (MHz)
bandwidth_MHz = freqSlope_MHz_us * rampEndTime_us

# 距离分辨率 (m) - 约 4.4 cm
rangeResolution_m = SPEED_OF_LIGHT / (2 * bandwidth_MHz * 1e6)

# 最大距离 (m) - 约 9 m
maxRange_m = (adcSampleRate_ksps * 1e3 * SPEED_OF_LIGHT) / (2 * freqSlope_MHz_us * 1e12)

# Chirp 周期
chirpTime_us = idleTime_us + rampEndTime_us

# 最大速度 (m/s) - 约 1 m/s
lambda_m = SPEED_OF_LIGHT / (startFreq_GHz * 1e9)
maxVelocity_mps = lambda_m / (4 * chirpTime_us * 1e-6 * numTxAntennas)

print(f"=== 雷达配置信息 (pymmw 兼容) ===")
print(f"ADC 采样点数: {numADCSamples}")
print(f"TX/RX 天线: {numTxAntennas}/{numRxAntennas}")
print(f"每帧循环数: {numLoopsPerFrame}")
print(f"距离分辨率: {rangeResolution_m*100:.2f} cm")
print(f"最大距离: {maxRange_m:.2f} m")
print(f"最大速度: {maxVelocity_mps:.2f} m/s")
