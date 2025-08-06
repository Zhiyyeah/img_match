#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取和分析goci2_upsampled_multiband.nc文件
功能：
1. 读取多波段nc文件
2. 分波段可视化数据
3. 根据波段灵活设置colorbar
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')


def read_upsampled_nc(file_path):
    """
    读取上采样后的多波段nc文件
    """
    print("="*60)
    print("读取GOCI2上采样多波段nc文件")
    print("="*60)
    
    with nc.Dataset(file_path, 'r') as dataset:
        # 显示文件基本信息
        print(f"\n文件信息:")
        print(f"  文件名: {file_path}")
        
        # 显示维度信息
        print(f"\n维度信息:")
        for dim_name, dim in dataset.dimensions.items():
            print(f"  {dim_name}: {len(dim)} {'(unlimited)' if dim.isunlimited() else ''}")
        
        # 读取变量信息
        print(f"\n变量信息:")
        variables_info = {}
        
        for var_name, var in dataset.variables.items():
            print(f"  {var_name}: {var.shape}")
            
            # 读取数据
            data = var[:]
            variables_info[var_name] = {
                'data': data,
                'shape': var.shape,
                'dtype': var.dtype
            }
        
        return variables_info


def visualize_bands_separately(variables_info, output_prefix='upsampled_analysis', output_dir=None, date_str=None):
    """
    分波段可视化数据，每个波段单独设置colorbar
    """
    print(f"\n生成分波段可视化图像...")
    
    # 获取Rrs波段变量（排除经纬度）
    rrs_vars = {name: info for name, info in variables_info.items() 
               if name.startswith('Rrs_') and name not in ['lon', 'lat']}
    
    if not rrs_vars:
        print("  未找到Rrs波段数据")
        return
    
    # 获取经纬度
    lon = variables_info['lon']['data']
    lat = variables_info['lat']['data']
    
    # 为每个波段创建单独的图像
    for var_name, var_info in rrs_vars.items():
        print(f"  处理波段: {var_name}")
        
        # 创建单个波段的图像
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        data = var_info['data']
        
        # 清理数据 - 处理异常值
        # 统计异常值
        original_valid = np.sum(np.isfinite(data) & (data > 0))
        large_values = np.sum((data > 10) & np.isfinite(data))
        
        cleaned_data = np.where(np.isfinite(data) & (data > 0) & (data <= 10), data, np.nan)
        
        # 计算该波段的颜色范围
        valid_data = cleaned_data[np.isfinite(cleaned_data)]
        if len(valid_data) > 0:
            vmin, vmax = np.percentile(valid_data, [2, 98])
            # 确保vmax不超过10
            vmax = min(vmax, 10.0)
            print(f"    原始有效像素: {original_valid}, 异常值(>10): {large_values}, 清理后有效像素: {len(valid_data)}")
            print(f"    颜色范围: [{vmin:.4f}, {vmax:.4f}]")
        else:
            vmin, vmax = 0, 0.05
            print(f"    原始有效像素: {original_valid}, 异常值(>10): {large_values}, 清理后无有效数据")
        
        # 绘制图像
        if lon.ndim == 2 and lat.ndim == 2:
            im = ax.pcolormesh(lon, lat, cleaned_data, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        else:
            extent = [lon.min(), lon.max(), lat.min(), lat.max()]
            im = ax.imshow(cleaned_data, extent=extent, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto', origin='upper')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Rrs (sr⁻¹)', fontsize=12)
        
        # 设置标题
        band_wavelength = var_name.replace('Rrs_', '')
        ax.set_title(f'GOCI2 Upsampled - {var_name} ({band_wavelength}nm)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude (°)', fontsize=12)
        ax.set_ylabel('Latitude (°)', fontsize=12)
        
        # 添加简单的覆盖率信息
        valid_mask = np.isfinite(cleaned_data)
        if np.sum(valid_mask) > 0:
            coverage = np.sum(valid_mask) / cleaned_data.size * 100
            ax.text(0.02, 0.98, f'Coverage: {coverage:.1f}%', 
                   transform=ax.transAxes, 
                   va='top', 
                   fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图像
        if output_dir and date_str:
            output_file = os.path.join(output_dir, f'{output_prefix}_{var_name}_{date_str}.png')
        else:
            output_file = f'{output_prefix}_{var_name}.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"    保存图像: {output_file}")
        
        plt.close(fig)  # 关闭图像以释放内存
    
    print(f"  所有波段图像生成完成！")


def analyze_data_quality(variables_info):
    """
    分析数据质量
    """
    print(f"\n数据质量分析:")
    print("="*40)
    
    rrs_vars = {name: info for name, info in variables_info.items() 
               if name.startswith('Rrs_') and name not in ['lon', 'lat']}
    
    if not rrs_vars:
        print("  未找到Rrs波段数据")
        return
    
    # 创建质量分析表格
    print(f"{'波段':<12} {'有效像素':<10} {'覆盖率':<10} {'数据范围':<20}")
    print("-" * 55)
    
    for var_name, var_info in rrs_vars.items():
        data = var_info['data']
        valid_data = data[np.isfinite(data) & (data > 0) & (data <= 10)]
        
        if len(valid_data) > 0:
            valid_count = len(valid_data)
            coverage = valid_count / data.size * 100
            data_range = f"{np.min(valid_data):.4f} - {np.max(valid_data):.4f}"
            
            print(f"{var_name:<12} {valid_count:<10} {coverage:<10.2f} {data_range:<20}")
        else:
            print(f"{var_name:<12} {0:<10} {0:<10.2f} {'N/A':<20}")


def main():
    """
    主函数
    """
    # 文件路径
    nc_file = 'goci2_upsampled_multiband.nc'
    
    # 从文件名中提取日期
    date_match = None
    if os.path.exists(nc_file):
        # 尝试从nc文件名中提取日期
        for part in nc_file.split('_'):
            if len(part) == 8 and part.isdigit():
                date_match = part
                break
    
    if date_match is None:
        # 如果没有找到日期，使用当前日期
        date_match = datetime.now().strftime('%Y%m%d')
        print(f"警告：无法从文件名提取日期，使用当前日期: {date_match}")
    
    # 创建输出目录
    output_dir = f"{date_match}_match"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    else:
        print(f"输出目录已存在: {output_dir}")
    
    try:
        # 1. 读取nc文件
        variables_info = read_upsampled_nc(nc_file)
        
        # 2. 分析数据质量
        analyze_data_quality(variables_info)
        
        # 3. 生成分波段可视化
        visualize_bands_separately(variables_info, output_dir=output_dir, date_str=date_match)
        
        print(f"\n{'='*60}")
        print("文件读取和分析完成！")
        print(f"输出目录: {output_dir}")
        print("="*60)
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {nc_file}")
        print("请确保文件存在于当前目录中")
    except Exception as e:
        print(f"错误：{str(e)}")


if __name__ == "__main__":
    main() 