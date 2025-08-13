#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landsat 9 C2 L1 辐射定标（DN -> 辐亮度）
- 读取 MTL.txt 的 RADIANCE_MULT/ADD 系数
- 对各个 B1..B11 (若存在) 进行 L = ML * DN + AL
- 输出每个波段的辐射定标TIF文件
- 绘制B2波段的空间图
- 跳过 QA/质量控制相关影像
"""

import os
import re
import numpy as np
import rasterio
import platform
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle

# === 用户路径（按需修改） ===

system_type = platform.system()

if system_type == "Windows":
    root = r"D:\Py_Code\SR_Imagery\LC09_L1TP_116035_20250504_20250504_02_T1"

elif system_type == "Darwin":
    root = "/Users/zy/Python_code/My_Git/SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1"

else:
    root = "/public/home/zyye/SR/Image_match_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1"
print(f"当前系统: {system_type}")

def find_mtl_path(folder):
    for fn in os.listdir(folder):
        if fn.endswith("_MTL.txt") or fn.endswith("_MTL.TXT"):
            return os.path.join(folder, fn)
    raise FileNotFoundError("未找到 MTL 元数据文件（*_MTL.txt）。")

def parse_mtl_radiance_coeffs(mtl_path):
    """
    解析 MTL，提取每个波段的 RADIANCE_MULT_BAND_x 与 RADIANCE_ADD_BAND_x
    返回: {band_num: {"ML": float, "AL": float}}
    """
    coeffs = {}
    pattern_mult = re.compile(r'RADIANCE_MULT_BAND_(\d+)\s=\s([Ee0-9\.\-\+]+)')
    pattern_add  = re.compile(r'RADIANCE_ADD_BAND_(\d+)\s=\s([Ee0-9\.\-\+]+)')
    with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    for m in pattern_mult.finditer(text):
        b = int(m.group(1))
        coeffs.setdefault(b, {})["ML"] = float(m.group(2))
    for m in pattern_add.finditer(text):
        b = int(m.group(1))
        coeffs.setdefault(b, {})["AL"] = float(m.group(2))

    coeffs = {b: v for b, v in coeffs.items() if "ML" in v and "AL" in v}
    if not coeffs:
        raise ValueError("在 MTL 中未解析到 RADIANCE_MULT/ADD 系数。")
    return coeffs

def is_band_tif(filename):
    """
    简单判断是否为光谱波段影像：
    - 名称形如 *_B1.TIF, *_B2.TIF, ... 或 *_B10.TIF, *_B11.TIF
    - 排除 QA/角度影像
    """
    if not filename.upper().endswith(".TIF"):
        return False
    name = filename.upper()
    if "QA" in name or "ANG" in name:
        return False
    return re.search(r"_B(\d{1,2})\.TIF$", name) is not None

def extract_band_num(filename):
    m = re.search(r"_B(\d{1,2})\.TIF$", filename.upper())
    return int(m.group(1)) if m else None

def calibrate_to_radiance(in_tif, ml, al, nodata_out=-9999.0):
    with rasterio.open(in_tif) as src:
        profile = src.profile.copy()
        dn = src.read(1).astype(np.float32)

        # DN==0 通常为填充值；也可能存在其他无效码，用户可按需扩展
        mask = (dn == 0)

        # Radiance: L = ML * DN + AL
        L = ml * dn + al

        # 写 nodata
        L = np.where(mask, nodata_out, L).astype(np.float32)

        return L, src.transform, src.crs, profile

def save_radiance_tif(radiance_data, transform, crs, profile, output_path, band_num):
    """
    保存辐射定标后的TIF文件，保持原始坐标系统
    """
    # 更新profile以保存辐射定标后的数据
    profile.update(
        dtype='float32',
        count=1,
        nodata=-9999.0,
        compress='lzw'  # 使用LZW压缩
    )
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(radiance_data, 1)
        # 确保坐标系统信息被正确保存
        if crs:
            dst.crs = crs
        dst.transform = transform
        
    print(f"波段 {band_num} 辐射定标TIF已保存: {output_path}")

def plot_b2_spatial(radiance_data, transform, crs, ml, al):
    """
    绘制B2波段的空间图
    """
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建掩码，排除无效值
    valid_mask = radiance_data != -9999.0
    valid_data = radiance_data[valid_mask]
    
    if len(valid_data) == 0:
        print("警告：B2波段没有有效数据")
        return
    
    # 计算统计信息
    vmin = np.percentile(valid_data, 2)  # 2%分位数作为最小值
    vmax = np.percentile(valid_data, 98)  # 98%分位数作为最大值
    
    # 计算空间范围并转换为地理坐标
    height, width = radiance_data.shape
    bounds = rasterio.transform.array_bounds(height, width, transform)
    
    # 检查坐标系统并转换
    if crs and not crs.is_geographic:
        # 如果是投影坐标系，转换为地理坐标系
        from rasterio.warp import transform_bounds
        geographic_bounds = transform_bounds(crs, 'EPSG:4326', *bounds)
        print(f"原始投影坐标范围: {bounds}")
        print(f"转换后地理坐标范围: {geographic_bounds}")
        print(f"坐标系统: {crs}")
        
        # 使用转换后的地理坐标
        extent = [geographic_bounds[0], geographic_bounds[2], 
                 geographic_bounds[1], geographic_bounds[3]]
        xlabel = 'Longitude'
        ylabel = 'Latitude'
    else:
        # 如果已经是地理坐标系
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        xlabel = 'X Coordinate'
        ylabel = 'Y Coordinate'
        print(f"当前坐标范围: {bounds}")
        print(f"坐标系统: {crs}")
    
    # 绘制空间图 - 修复像元比例问题
    im = ax.imshow(radiance_data, 
                   extent=extent,
                   cmap='viridis',
                   vmin=vmin, vmax=vmax,
                   interpolation='nearest',
                   aspect='equal')  # 确保像元为正方形
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Radiance (W/(m²·sr·μm))', fontsize=12)
    
    # 设置标题和标签 - 移除定标系数
    ax.set_title('Landsat 9 Band 2 Spatial Distribution', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息文本框
    stats_text = f'Statistics:\nMin: {valid_data.min():.4f}\nMax: {valid_data.max():.4f}\nMean: {valid_data.mean():.4f}\nStd: {valid_data.std():.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main(folder):
    mtl_path = find_mtl_path(folder)
    coeffs = parse_mtl_radiance_coeffs(mtl_path)

    inputs = [fn for fn in os.listdir(folder) if is_band_tif(fn)]
    if not inputs:
        raise FileNotFoundError("未找到任何波段 TIF（形如 *_B1.TIF）。")

    # 创建输出目录
    output_dir = os.path.join(folder, "radiance_calibrated")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 处理所有波段
    for fn in inputs:
        band_num = extract_band_num(fn)
        if band_num not in coeffs:
            print(f"跳过波段 {band_num}: 未找到定标系数")
            continue
            
        in_path = os.path.join(folder, fn)
        ml = coeffs[band_num]["ML"]
        al = coeffs[band_num]["AL"]
        
        print(f"正在处理波段 {band_num}: {fn}")
        print(f"辐射定标系数: ML={ml:.6e}, AL={al:.6e}")
        
        # 计算辐射值
        L, transform, crs, profile = calibrate_to_radiance(in_path, ml, al)
        
        # 生成输出文件名
        base_name = os.path.splitext(fn)[0]
        output_filename = f"{base_name}_radiance.tif"
        output_path = os.path.join(output_dir, output_filename)
        
        # 保存TIF文件
        save_radiance_tif(L, transform, crs, profile, output_path, band_num)
        
        # 如果是B2波段，绘制空间图
        if band_num == 2:
            print("正在绘制B2波段空间图...")
            plot_b2_spatial(L, transform, crs, ml, al)
    
    print(f"\n所有波段处理完成！输出文件保存在: {output_dir}")

if __name__ == "__main__":
    main(root)
