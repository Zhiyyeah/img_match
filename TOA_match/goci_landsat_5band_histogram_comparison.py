#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOCI2和Landsat 5波段辐亮度直方图对比脚本
- 输入：GOCI2 L1B原始文件和Landsat TOA辐亮度文件（cal_L_TOA_rad_ref.py输出）
- 处理：5个对应波段 [443, 490, 555, 660, 865] nm
- 输出：每个波段的直方图对比图
- 功能：通过Landsat范围对GOCI进行裁剪，生成重叠区域的直方图对比
"""

import os
import platform
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
import matplotlib.font_manager as fm

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 波段映射配置
# ----------------------------

# GOCI2波段映射 (波段索引 -> 波长)
GOCI_BAND_MAPPING = {
    3: "443", 4: "490", 6: "555", 8: "660", 12: "865"
}

# Landsat波段映射 (波段号 -> 波长)
LANDSAT_BAND_MAPPING = {
    1: "443", 2: "490", 3: "555", 4: "660", 5: "865"
}

# 对应关系：GOCI波长 -> Landsat波段号
WAVELENGTH_TO_LANDSAT_BAND = {
    "443": 1, "490": 2, "555": 3, "660": 4, "865": 5
}

# ----------------------------
# 数据读取函数
# ----------------------------

def read_goci_band_from_l1b(goci_file, wavelength):
    """
    从GOCI2 L1B文件中读取指定波长的TOA辐亮度数据
    Args:
        goci_file: GOCI2 L1B文件路径
        wavelength: 波长字符串，如"443"
    Returns:
        data, lat, lon: 数据数组和经纬度
    """
    print(f"读取GOCI2文件: {goci_file}")
    print(f"读取波段: L_TOA_{wavelength}")
    
    with nc.Dataset(goci_file, 'r') as dataset:
        # 直接读取geophysical_data组中的TOA辐亮度数据
        band_var = f'L_TOA_{wavelength}'
        g = dataset.groups['geophysical_data']
        data = g.variables[band_var][:]
        print(f"在geophysical_data组中找到{band_var}")
        
        # 直接读取经纬度数据
        n = dataset.groups['navigation_data']
        lat = n.variables['latitude'][:]
        lon = n.variables['longitude'][:]
        
        print(f"GOCI2 {wavelength}nm波段数据形状: {data.shape}")
        print(f"经纬度数据形状: lat={lat.shape}, lon={lon.shape}")
        
        return data, lat, lon

def read_landsat_band_from_multiband(landsat_file, band_number):
    """
    从多波段Landsat文件中读取指定波段数据
    Args:
        landsat_file: Landsat多波段文件路径
        band_number: 波段号 (1-5)
    Returns:
        data, meta: 数据数组和元数据
    """
    print(f"读取Landsat文件: {landsat_file}")
    print(f"读取波段: {band_number}")
    
    with rasterio.open(landsat_file) as src:
        data = src.read(band_number)
        meta = {
            "crs": src.crs,
            "bounds": src.bounds,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "count": src.count
        }
        
        print(f"Landsat波段{band_number}数据形状: {data.shape}")
        return data, meta

# ----------------------------
# 裁剪工具函数
# ----------------------------

def get_landsat_lonlat_bounds(landsat_meta):
    """
    将Landsat边界转换为WGS84经纬度
    """
    b = landsat_meta["bounds"]
    crs = landsat_meta["crs"]
    lonmin, latmin, lonmax, latmax = transform_bounds(
        crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21
    )
    lonmin, lonmax = (min(lonmin, lonmax), max(lonmin, lonmax))
    latmin, latmax = (min(latmin, latmax), max(latmin, latmax))
    print(f"Landsat WGS84范围: lon[{lonmin:.6f}, {lonmax:.6f}], lat[{latmin:.6f}, {latmax:.6f}]")
    return lonmin, latmin, lonmax, latmax

def crop_goci_to_landsat_extent(goci_data, goci_lat, goci_lon, landsat_meta):
    """
    使用Landsat范围裁剪GOCI数据
    Returns: cropped_data, cropped_lat, cropped_lon, crop_bounds
    """
    lonmin, latmin, lonmax, latmax = get_landsat_lonlat_bounds(landsat_meta)

    # 确保2D经纬度网格
    if goci_lat.ndim == 1 and goci_lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(goci_lon, goci_lat)
    else:
        lat2d, lon2d = goci_lat, goci_lon

    # 创建掩膜
    inside = (lon2d >= lonmin) & (lon2d <= lonmax) & (lat2d >= latmin) & (lat2d <= latmax)

    # 找到有效区域的行列范围
    rows = np.any(inside, axis=1)
    cols = np.any(inside, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # 添加边界填充
    pad = 1
    rmin = max(0, rmin - pad)
    rmax = min(goci_data.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(goci_data.shape[1] - 1, cmax + pad)

    # 裁剪数据
    goci_cut = goci_data[rmin:rmax+1, cmin:cmax+1].astype(np.float64, copy=False)
    lat_cut = lat2d[rmin:rmax+1, cmin:cmax+1]
    lon_cut = lon2d[rmin:rmax+1, cmin:cmax+1]

    # 将严格超出范围的部分设为NaN
    inside_cut = (lon_cut >= lonmin) & (lon_cut <= lonmax) & (lat_cut >= latmin) & (lat_cut <= latmax)
    goci_cut = np.where(inside_cut, goci_cut, np.nan)

    print(f"裁剪后GOCI形状: {goci_cut.shape}")
    crop_bounds = (lonmin, latmin, lonmax, latmax)
    return goci_cut, lat_cut, lon_cut, crop_bounds

def read_landsat_window_in_wgs84(landsat_path, bounds_wgs84, band_number):
    """
    读取Landsat指定波段在给定WGS84边界内的数据
    """
    with rasterio.open(landsat_path) as src:
        # 将WGS84边界投影到Landsat坐标系
        l, b, r, t = transform_bounds("EPSG:4326", src.crs, *bounds_wgs84, densify_pts=21)
        win = from_bounds(l, b, r, t, transform=src.transform)
        
        # 与完整图像窗口相交
        full = rasterio.windows.Window(0, 0, src.width, src.height)
        win = win.intersection(full)

        arr = src.read(band_number, window=win)
        nodata = src.nodata

        # 计算实际窗口边界（WGS84）
        win_bounds_proj = rasterio.windows.bounds(win, src.transform)
        win_bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326",
                                            win_bounds_proj[0], win_bounds_proj[1],
                                            win_bounds_proj[2], win_bounds_proj[3],
                                            densify_pts=21)
        return arr, win_bounds_wgs84, nodata

# ----------------------------
# 直方图对比函数
# ----------------------------

def plot_histogram_comparison(goci_cropped, landsat_path, bounds_wgs84, wavelength, 
                             landsat_band, bins=100, clip_percent=(1, 99), 
                             outfile="hist_comparison.png"):
    """
    绘制GOCI和Landsat的直方图对比
    """
    print(f"创建{wavelength}nm波段直方图对比...")

    # 读取Landsat重叠窗口数据
    landsat_arr, _, ls_nodata = read_landsat_window_in_wgs84(landsat_path, bounds_wgs84, landsat_band)

    # 准备有效像素数据
    goci_vals = goci_cropped[np.isfinite(goci_cropped)].ravel()
    if ls_nodata is None:
        landsat_mask = np.isfinite(landsat_arr)
    else:
        landsat_mask = np.isfinite(landsat_arr) & (landsat_arr != ls_nodata)
    landsat_vals = landsat_arr[landsat_mask].ravel()

    # 统一分位数范围
    lo_g, hi_g = np.nanpercentile(goci_vals, clip_percent)
    lo_l, hi_l = np.nanpercentile(landsat_vals, clip_percent)
    lo = min(lo_g, lo_l)
    hi = max(hi_g, hi_l)

    # 计算统计信息
    def stats(arr):
        return dict(
            n=arr.size, 
            mean=float(np.nanmean(arr)), 
            std=float(np.nanstd(arr)),
            p1=float(np.nanpercentile(arr, 1)), 
            p99=float(np.nanpercentile(arr, 99))
        )
    
    s_g = stats(goci_vals)
    s_l = stats(landsat_vals)
    print(f"GOCI统计: {s_g}")
    print(f"Landsat统计: {s_l}")
    print(f"统一直方图范围: [{lo:.6f}, {hi:.6f}]")

    # 绘制直方图
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 设置颜色
    goci_color = '#1f77b4'  # 蓝色
    landsat_color = '#ff7f0e'  # 橙色
    
    ax.hist(goci_vals, bins=bins, range=(lo, hi), alpha=0.7, density=True, 
            label=f"GOCI-2 L_TOA_{wavelength}nm (Cropped)", color=goci_color, edgecolor='black', linewidth=0.5)
    ax.hist(landsat_vals, bins=bins, range=(lo, hi), alpha=0.7, density=True, 
            label=f"Landsat B{landsat_band} (Overlap Region)", color=landsat_color, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel("Radiance (W m⁻² sr⁻¹ μm⁻¹)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Radiance Histogram Comparison: GOCI-2 {wavelength}nm vs Landsat B{landsat_band}", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 添加统计信息文本框
    stats_text = (f"GOCI-2 {wavelength}nm:\n"
                  f"  Pixels: {s_g['n']:,}\n"
                  f"  Mean: {s_g['mean']:.4f}\n"
                  f"  Std: {s_g['std']:.4f}\n"
                  f"  1%: {s_g['p1']:.4f}\n"
                  f"  99%: {s_g['p99']:.4f}\n\n"
                  f"Landsat B{landsat_band}:\n"
                  f"  Pixels: {s_l['n']:,}\n"
                  f"  Mean: {s_l['mean']:.4f}\n"
                  f"  Std: {s_l['std']:.4f}\n"
                  f"  1%: {s_l['p1']:.4f}\n"
                  f"  99%: {s_l['p99']:.4f}")
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='gray'),
            fontsize=10, family='monospace')

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"直方图已保存: {outfile}")
    return fig

# ----------------------------
# 主函数
# ----------------------------

def main():
    """主函数"""
    # 系统路径设置
    system_type = platform.system()
    
    if system_type == "Windows":
        goci_file = r"D:\Py_Code\SR_Imagery\GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
        landsat_file = r"D:\Py_Code\SR_Imagery\LC09_L1TP_116035_20250504_20250504_02_T1\LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif"
        output_dir = r"D:\Py_Code\SR_Imagery\histogram_comparison_output"
    elif system_type == "Darwin":  # macOS
        goci_file = "/Users/zy/python_code/My_Git/SR_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
        landsat_file = "/Users/zy/python_code/My_Git/SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif"
        output_dir = "/Users/zy/python_code/My_Git/SR_Imagery/histogram_comparison_output"
    else:  # Linux服务器
        goci_file = "/public/home/zyye/SR/Image_match_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
        landsat_file = "/public/home/zyye/SR/Image_match_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif"
        output_dir = "/public/home/zyye/SR/Image_match_Imagery/histogram_comparison_output"

    print(f"当前系统: {system_type}")
    print(f"GOCI2文件: {goci_file}")
    print(f"Landsat文件: {landsat_file}")
    print(f"输出目录: {output_dir}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理波长列表
    wavelengths = ["443", "490", "555", "660", "865"]
    
    try:
        # 首先读取Landsat元数据（用于裁剪）
        print("\n=== 读取Landsat元数据 ===")
        landsat_data_sample, landsat_meta = read_landsat_band_from_multiband(landsat_file, 1)

        # 处理每个波段
        for wavelength in wavelengths:
            print(f"\n{'='*50}")
            print(f"处理波段: {wavelength}nm")
            print(f"{'='*50}")
            
            # 获取对应的Landsat波段号
            landsat_band = WAVELENGTH_TO_LANDSAT_BAND[wavelength]
            
            # 1. 读取GOCI2数据
            goci_data, goci_lat, goci_lon = read_goci_band_from_l1b(goci_file, wavelength)

            # 2. 裁剪GOCI2到Landsat范围
            cropped_goci_data, cropped_lat, cropped_lon, crop_bounds = crop_goci_to_landsat_extent(
                goci_data, goci_lat, goci_lon, landsat_meta
            )

            # 3. 生成直方图对比
            output_file = os.path.join(output_dir, f"hist_{wavelength}nm_comparison.png")
            fig = plot_histogram_comparison(
                goci_cropped=cropped_goci_data,
                landsat_path=landsat_file,
                bounds_wgs84=crop_bounds,
                wavelength=wavelength,
                landsat_band=landsat_band,
                bins=100,
                clip_percent=(1, 99),
                outfile=output_file
            )
            
            plt.close(fig)
            print(f"✅ {wavelength}nm波段处理完成")

        print(f"\n{'='*50}")
        print("所有波段处理完成！")
        print(f"输出目录: {output_dir}")
        print(f"{'='*50}")

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 