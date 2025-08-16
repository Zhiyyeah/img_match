#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的443波段对比图（新增：按Landsat范围裁剪GOCI）
对比GOCI的443波段和Landsat的B2波段辐亮度
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import rasterio
from rasterio.warp import transform_bounds

def read_goci_443_band(goci_file):
    """读取GOCI的443波段数据与经纬度"""
    print(f"读取GOCI文件: {goci_file}")

    with nc.Dataset(goci_file, 'r') as dataset:
        # 1) 查找L_TOA_443
        data = None
        if 'L_TOA_443' in dataset.variables:
            data = dataset.variables['L_TOA_443'][:]
        else:
            for group in dataset.groups.values():
                if 'L_TOA_443' in group.variables:
                    data = group.variables['L_TOA_443'][:]
                    break
        if data is None:
            print("未找到L_TOA_443变量")
            return None, None, None

        # 2) 查找经纬度
        lat = None; lon = None
        for var_name in ['latitude', 'lat', 'Latitude', 'Lat']:
            if var_name in dataset.variables:
                lat = dataset.variables[var_name][:]; break
        for var_name in ['longitude', 'lon', 'Longitude', 'Lon']:
            if var_name in dataset.variables:
                lon = dataset.variables[var_name][:]; break

        if lat is None or lon is None:
            for group in dataset.groups.values():
                if lat is None:
                    for var_name in ['latitude', 'lat', 'Latitude', 'Lat']:
                        if var_name in group.variables:
                            lat = group.variables[var_name][:]; break
                if lon is None:
                    for var_name in ['longitude', 'lon', 'Longitude', 'Lon']:
                        if var_name in group.variables:
                            lon = group.variables[var_name][:]; break
                if lat is not None and lon is not None:
                    break

        print(f"GOCI 443波段数据形状: {data.shape}")
        return data, lat, lon

def read_landsat_b2_radiance(landsat_file):
    """读取Landsat的B2波段辐亮度数据 + 元数据（用于获取范围与CRS）"""
    print(f"读取Landsat文件: {landsat_file}")
    src = rasterio.open(landsat_file)
    data = src.read(1)
    print(f"Landsat B2波段数据形状: {data.shape}")
    return data, src

def get_landsat_lonlat_bounds(src):
    """
    将Landsat的外接矩形范围转换到经纬度（EPSG:4326）坐标。
    自动处理UTM等投影到经纬度的转换。
    """
    bounds_proj = src.bounds
    crs = src.crs
    # 转换为经纬度
    lonmin, latmin, lonmax, latmax = transform_bounds(crs, "EPSG:4326",
                                                      bounds_proj.left, bounds_proj.bottom,
                                                      bounds_proj.right, bounds_proj.top,
                                                      densify_pts=21)
    # 规范化：确保最小/最大顺序正确
    lonmin, lonmax = (min(lonmin, lonmax), max(lonmin, lonmax))
    latmin, latmax = (min(latmin, latmax), max(latmin, latmax))
    return lonmin, latmin, lonmax, latmax

def crop_goci_to_landsat_extent(goci_data, lat, lon, landsat_src):
    """
    按 Landsat 覆盖范围，对 GOCI 进行裁剪（经纬度空间裁剪）。
    返回裁剪后的 GOCI 数据以及对应的经纬度子集。
    - 若经纬度为2D网格（常见），按经纬度框做布尔掩膜；并进一步缩减到最小包络行列窗口。
    - 若经纬度为1D（少见），也能处理。
    """
    if goci_data is None or lat is None or lon is None:
        return None, None, None

    lonmin, latmin, lonmax, latmax = get_landsat_lonlat_bounds(landsat_src)
    print(f"Landsat经纬度范围: lon[{lonmin:.6f}, {lonmax:.6f}], lat[{latmin:.6f}, {latmax:.6f}]")

    # 统一为2D网格
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)  # [rows, cols]
    else:
        lat2d, lon2d = lat, lon

    # 构建掩膜并找到最小行列包络
    inside = (lon2d >= lonmin) & (lon2d <= lonmax) & (lat2d >= latmin) & (lat2d <= latmax)

    if not np.any(inside):
        print("警告：Landsat范围与GOCI经纬度无重叠，返回原始数据（全部设为NaN掩膜）")
        goci_cropped = np.full_like(goci_data, np.nan, dtype=np.float64)
        return goci_cropped, lat2d, lon2d

    rows = np.any(inside, axis=1)
    cols = np.any(inside, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # 多留一圈边界，避免边界像素被截掉（可选）
    pad = 1
    rmin = max(0, rmin - pad); rmax = min(goci_data.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad); cmax = min(goci_data.shape[1] - 1, cmax + pad)

    goci_cut = goci_data[rmin:rmax+1, cmin:cmax+1].astype(np.float64, copy=False)
    lat_cut  = lat2d[rmin:rmax+1, cmin:cmax+1]
    lon_cut  = lon2d[rmin:rmax+1, cmin:cmax+1]

    # 将 Landsat 范围外的像素置为 NaN，确保严格裁剪
    inside_cut = (lon_cut >= lonmin) & (lon_cut <= lonmax) & (lat_cut >= latmin) & (lat_cut <= latmax)
    goci_cut = np.where(inside_cut, goci_cut, np.nan)

    print(f"GOCI裁剪后形状: {goci_cut.shape}")
    return goci_cut, lat_cut, lon_cut

def create_simple_comparison(goci_data, landsat_data, lat=None, lon=None, title_suffix=""):
    """创建简单的对比图"""
    print("创建对比图...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'443波段对比: GOCI（裁剪） vs Landsat{title_suffix}', fontsize=16, fontweight='bold')

    # 左图：GOCI 443波段（已裁剪）
    ax1 = axes[0]
    ax1.set_title('GOCI L_TOA_443（按Landsat范围裁剪）')

    goci_plot = np.where(np.isfinite(goci_data), goci_data, np.nan)
    valid_goci = goci_plot[np.isfinite(goci_plot)]
    if valid_goci.size > 0:
        vmin_goci = np.nanpercentile(valid_goci, 2)
        vmax_goci = np.nanpercentile(valid_goci, 98)
    else:
        vmin_goci, vmax_goci = 0, 1

    if lat is not None and lon is not None:
        extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
        im1 = ax1.imshow(goci_plot, cmap='viridis', vmin=vmin_goci, vmax=vmax_goci,
                         extent=extent, aspect='auto', origin='upper')
        ax1.set_xlabel('Longitude'); ax1.set_ylabel('Latitude')
    else:
        im1 = ax1.imshow(goci_plot, cmap='viridis', vmin=vmin_goci, vmax=vmax_goci, origin='upper')
        ax1.set_xlabel('Column Index'); ax1.set_ylabel('Row Index')
    plt.colorbar(im1, ax=ax1, label='Raw Value')

    if valid_goci.size > 0:
        stats_text = f'Min: {np.nanmin(valid_goci):.4f}\nMax: {np.nanmax(valid_goci):.4f}\nMean: {np.nanmean(valid_goci):.4f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                 va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 右图：Landsat B2辐亮度
    ax2 = axes[1]
    ax2.set_title('Landsat B2 Radiance (W/(m²·sr·μm))')

    landsat_plot = np.where(landsat_data != -9999.0, landsat_data, np.nan)
    valid_landsat = landsat_plot[np.isfinite(landsat_plot)]
    if valid_landsat.size > 0:
        vmin_landsat = np.nanpercentile(valid_landsat, 2)
        vmax_landsat = np.nanpercentile(valid_landsat, 98)
    else:
        vmin_landsat, vmax_landsat = 0, 1

    im2 = ax2.imshow(landsat_plot, cmap='viridis', vmin=vmin_landsat, vmax=vmax_landsat, origin='upper')
    plt.colorbar(im2, ax=ax2, label='Radiance (W/(m²·sr·μm))')

    if valid_landsat.size > 0:
        stats_text = f'Min: {np.nanmin(valid_landsat):.4f}\nMax: {np.nanmax(valid_landsat):.4f}\nMean: {np.nanmean(valid_landsat):.4f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                 va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig

def main():
    """主函数"""
    # 文件路径（按需修改）
    goci_file = r"D:\Py_Code\SR_Imagery\GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
    landsat_file = r"D:\Py_Code\SR_Imagery\LC09_L1TP_116035_20250504_20250504_02_T1\radiance_calibrated\LC09_L1TP_116035_20250504_20250504_02_T1_B2_radiance.tif"

    if not os.path.exists(goci_file):
        print(f"错误：GOCI文件不存在: {goci_file}"); return
    if not os.path.exists(landsat_file):
        print(f"错误：Landsat文件不存在: {landsat_file}"); return

    try:
        # 读取数据
        goci_data, lat, lon = read_goci_443_band(goci_file)
        if goci_data is None:
            print("无法读取GOCI数据"); return

        landsat_data, landsat_src = read_landsat_b2_radiance(landsat_file)
        if landsat_data is None:
            print("无法读取Landsat数据"); return

        # 用 Landsat 空间范围裁剪 GOCI
        goci_cut, lat_cut, lon_cut = crop_goci_to_landsat_extent(goci_data, lat, lon, landsat_src)
        if goci_cut is None:
            print("GOCI裁剪失败"); return

        # 创建对比图（左：裁剪后的GOCI，右：原Landsat）
        fig = create_simple_comparison(goci_cut, landsat_data, lat_cut, lon_cut, title_suffix="（按范围裁剪）")

        # 保存与展示
        output_file = "band_443_comparison_cropped.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"对比图已保存: {output_file}")
        plt.show()

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
