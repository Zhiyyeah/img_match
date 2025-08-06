#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版：读取GOCI-II NetCDF文件中的所有Rrs数据
"""

import numpy as np
import netCDF4 as nc

# 文件路径
filename = "H:\GK2B_GOCI2_L2_20250323_021530_LA_S009_AC.nc"

# 定义Rrs变量名
rrs_variables = [
    'Rrs_380', 'Rrs_412', 'Rrs_443', 'Rrs_490',
    'Rrs_510', 'Rrs_555', 'Rrs_620', 'Rrs_660',
    'Rrs_680', 'Rrs_709', 'Rrs_745', 'Rrs_865'
]

# 对应波长
wavelengths = np.array([380, 412, 443, 490, 510, 555, 620, 660, 680, 709, 745, 865])

# 初始化3D数组 [rows, cols, bands]
rrs_data = np.zeros((2780, 2780, 12), dtype=np.float32)

# 读取数据
print('读取Rrs数据...')
with nc.Dataset(filename, 'r') as dataset:
    rrs_group = dataset.groups['geophysical_data'].groups['Rrs']
    
    for i, var_name in enumerate(rrs_variables):
        print(f'读取 {var_name}...')
        rrs_data[:, :, i] = rrs_group.variables[var_name][:]

print(f'读取完成！数据形状: {rrs_data.shape}')

# RGB可视化
import matplotlib.pyplot as plt

def create_rgb_image(rrs_data, wavelengths):
    """创建RGB合成图像"""
    
    # 选择RGB波段 (接近真彩色)
    # 红色: 660nm (索引7), 绿色: 555nm (索引5), 蓝色: 443nm (索引2)
    red_idx = 7    # 660nm
    green_idx = 5  # 555nm  
    blue_idx = 2   # 443nm
    
    print(f'RGB波段选择: R={wavelengths[red_idx]}nm, G={wavelengths[green_idx]}nm, B={wavelengths[blue_idx]}nm')
    
    # 提取RGB波段
    red = rrs_data[:, :, red_idx].copy()
    green = rrs_data[:, :, green_idx].copy()
    blue = rrs_data[:, :, blue_idx].copy()
    
    # 处理无效值，将-999替换为0
    red[red == -999] = 0
    green[green == -999] = 0
    blue[blue == -999] = 0
    
    # 将负值设为0
    red[red < 0] = 0
    green[green < 0] = 0
    blue[blue < 0] = 0
    
    # 数据归一化和对比度拉伸
    def normalize_band(band, percentile=98):
        """归一化单个波段"""
        # 使用98%分位数进行拉伸，避免极值影响
        valid_data = band[band > 0]
        if len(valid_data) > 0:
            max_val = np.percentile(valid_data, percentile)
            band_norm = np.clip(band / max_val, 0, 1)
        else:
            band_norm = band
        return band_norm
    
    # 归一化各波段
    red_norm = normalize_band(red)
    green_norm = normalize_band(green)
    blue_norm = normalize_band(blue)
    
    # 组合RGB图像
    rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=2)
    
    return rgb_image

# 创建RGB图像
rgb_image = create_rgb_image(rrs_data, wavelengths)

# 显示RGB图像
plt.figure(figsize=(12, 10))
plt.imshow(rgb_image)
plt.title('GOCI-II Rrs RGB composite image\n(R:660nm, G:555nm, B:443nm)')
plt.xlabel('pixel column')
plt.ylabel('pixel row')
plt.axis('equal')
plt.tight_layout()
plt.show()

# 显示各个波段的单独图像
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 显示RGB分量
im1 = axes[0,0].imshow(rgb_image[:,:,0], cmap='Reds')
axes[0,0].set_title('red band (660nm)')
plt.colorbar(im1, ax=axes[0,0])

im2 = axes[0,1].imshow(rgb_image[:,:,1], cmap='Greens')
axes[0,1].set_title('green band (555nm)')
plt.colorbar(im2, ax=axes[0,1])

im3 = axes[1,0].imshow(rgb_image[:,:,2], cmap='Blues')
axes[1,0].set_title('blue band (443nm)')
plt.colorbar(im3, ax=axes[1,0])

# RGB合成
axes[1,1].imshow(rgb_image)
axes[1,1].set_title('RGB composite image')

plt.tight_layout()
plt.show()

print("RGB可视化完成！")
print("数据使用说明：")
print("- rrs_data: 原始三维数据 [2780, 2780, 12]")
print("- rgb_image: RGB合成图像 [2780, 2780, 3]")
print("- wavelengths: 波长信息")
print("- rrs_variables: 变量名列表")