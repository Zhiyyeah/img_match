#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取Landsat 8/9 NetCDF文件中的Rrs数据
"""

import netCDF4 as nc
import numpy as np
import os

# 文件路径
file_path = r"D:\Py_Code\SR_Imagery\LC09_L1TP_116035_20250504_20250504_02_T1\output\L9_OLI_2025_05_04_02_11_09_116035_L2W.nc"
# Landsat 8/9 OLI波长
wavelengths = np.array([443, 483, 561, 592, 613, 655, 865, 1609, 2201])
rrs_variables = [f"Rrs_{wavelength}" for wavelength in wavelengths]

# 初始化3D数组 [rows, cols, bands]
# 先读取第一个变量来获取维度
with nc.Dataset(file_path, 'r') as dataset:
    first_var = dataset.variables[rrs_variables[0]]
    rows, cols = first_var.shape
    num_bands = len(wavelengths)
    rrs_data = np.zeros((rows, cols, num_bands), dtype=np.float32)

# 读取数据
print('读取Rrs数据...')
with nc.Dataset(file_path, 'r') as dataset:
    for i, var_name in enumerate(rrs_variables):
        print(f'读取 {var_name}...')
        rrs_data[:, :, i] = dataset.variables[var_name][:]

print(f'读取完成！数据形状: {rrs_data.shape}')
        

