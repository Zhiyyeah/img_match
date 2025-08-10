#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的443波段对比图
对比GOCI的443波段和Landsat的B2波段辐亮度
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import rasterio

def read_goci_443_band(goci_file):
    """读取GOCI的443波段数据"""
    print(f"读取GOCI文件: {goci_file}")
    
    with nc.Dataset(goci_file, 'r') as dataset:
        # 查找L_TOA_443变量
        if 'L_TOA_443' in dataset.variables:
            data = dataset.variables['L_TOA_443'][:]
        else:
            # 在组中查找
            for group in dataset.groups.values():
                if 'L_TOA_443' in group.variables:
                    data = group.variables['L_TOA_443'][:]
                    break
            else:
                print("未找到L_TOA_443变量")
                return None, None, None
        
        # 读取经纬度信息
        lat = None
        lon = None
        
        # 查找纬度变量
        for var_name in ['latitude', 'lat', 'Latitude', 'Lat']:
            if var_name in dataset.variables:
                lat = dataset.variables[var_name][:]
                break
        
        # 查找经度变量
        for var_name in ['longitude', 'lon', 'Longitude', 'Lon']:
            if var_name in dataset.variables:
                lon = dataset.variables[var_name][:]
                break
        
        # 如果没找到，尝试在组中查找
        if lat is None or lon is None:
            for group in dataset.groups.values():
                for var_name in ['latitude', 'lat', 'Latitude', 'Lat']:
                    if var_name in group.variables:
                        lat = group.variables[var_name][:]
                        break
                for var_name in ['longitude', 'lon', 'Longitude', 'Lon']:
                    if var_name in group.variables:
                        lon = group.variables[var_name][:]
                        break
                if lat is not None and lon is not None:
                    break
        
        print(f"GOCI 443波段数据形状: {data.shape}")
        return data, lat, lon

def read_landsat_b2_radiance(landsat_file):
    """读取Landsat的B2波段辐亮度数据"""
    print(f"读取Landsat文件: {landsat_file}")
    
    with rasterio.open(landsat_file) as src:
        data = src.read(1)  # 读取第一个波段
        print(f"Landsat B2波段数据形状: {data.shape}")
        return data

def create_simple_comparison(goci_data, landsat_data, lat=None, lon=None):
    """创建简单的对比图"""
    print("创建对比图...")
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('443波段对比: GOCI vs Landsat', fontsize=16, fontweight='bold')
    
    # 左图：GOCI 443波段
    ax1 = axes[0]
    ax1.set_title('GOCI L_TOA_443 (原始值)')
    
    # 处理无效值
    goci_plot = np.where(np.isfinite(goci_data), goci_data, np.nan)
    
    # 计算显示范围
    valid_goci = goci_data[np.isfinite(goci_data)]
    if len(valid_goci) > 0:
        vmin_goci = np.nanpercentile(valid_goci, 2)
        vmax_goci = np.nanpercentile(valid_goci, 98)
    else:
        vmin_goci, vmax_goci = 0, 1
    
    # 如果有经纬度信息，使用地理坐标显示
    if lat is not None and lon is not None:
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
        im1 = ax1.imshow(goci_plot, cmap='viridis', vmin=vmin_goci, vmax=vmax_goci, 
                         extent=extent, aspect='auto', origin='upper')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
    else:
        # 如果没有经纬度，使用像素索引
        im1 = ax1.imshow(goci_plot, cmap='viridis', vmin=vmin_goci, vmax=vmax_goci, origin='upper')
        ax1.set_xlabel('Column Index')
        ax1.set_ylabel('Row Index')
    
    plt.colorbar(im1, ax=ax1, label='Raw Value')
    
    # 添加统计信息
    if len(valid_goci) > 0:
        stats_text = f'Min: {valid_goci.min():.4f}\nMax: {valid_goci.max():.4f}\nMean: {valid_goci.mean():.4f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 右图：Landsat B2波段
    ax2 = axes[1]
    ax2.set_title('Landsat B2 Radiance (W/(m²·sr·μm))')
    
    # 处理无效值
    landsat_plot = np.where(landsat_data != -9999.0, landsat_data, np.nan)
    
    # 计算显示范围
    valid_landsat = landsat_data[landsat_data != -9999.0]
    if len(valid_landsat) > 0:
        vmin_landsat = np.nanpercentile(valid_landsat, 2)
        vmax_landsat = np.nanpercentile(valid_landsat, 98)
    else:
        vmin_landsat, vmax_landsat = 0, 1
    
    im2 = ax2.imshow(landsat_plot, cmap='viridis', vmin=vmin_landsat, vmax=vmax_landsat, origin='upper')
    plt.colorbar(im2, ax=ax2, label='Radiance (W/(m²·sr·μm))')
    
    # 添加统计信息
    if len(valid_landsat) > 0:
        stats_text = f'Min: {valid_landsat.min():.4f}\nMax: {valid_landsat.max():.4f}\nMean: {valid_landsat.mean():.4f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    # 文件路径
    goci_file = r"D:\Py_Code\SR_Imagery\GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
    landsat_file = r"D:\Py_Code\SR_Imagery\LC09_L1TP_116035_20250504_20250504_02_T1\radiance_calibrated\LC09_L1TP_116035_20250504_20250504_02_T1_B2_radiance.tif"
    
    # 检查文件是否存在
    if not os.path.exists(goci_file):
        print(f"错误：GOCI文件不存在: {goci_file}")
        return
    
    if not os.path.exists(landsat_file):
        print(f"错误：Landsat文件不存在: {landsat_file}")
        return
    
    try:
        # 读取数据
        goci_data, lat, lon = read_goci_443_band(goci_file)
        if goci_data is None:
            print("无法读取GOCI数据")
            return
            
        landsat_data = read_landsat_b2_radiance(landsat_file)
        if landsat_data is None:
            print("无法读取Landsat数据")
            return
        
        # 创建对比图
        fig = create_simple_comparison(goci_data, landsat_data, lat, lon)
        
        # 保存图像
        output_file = "band_443_comparison.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"对比图已保存: {output_file}")
        
        # 显示图像
        plt.show()
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 