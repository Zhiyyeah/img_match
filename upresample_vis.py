#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的GOCI2上采样可视化工具
核心功能：
1. 对GOCI2进行上采样到Landsat分辨率
2. 可视化采样前后的对比结果
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from pyresample import geometry, kd_tree
import warnings
warnings.filterwarnings('ignore')


def find_matching_bands(goci_bands, landsat_bands, tolerance=20):
    """
    找出GOCI2和Landsat中相近的波段对
    tolerance: 波段匹配的容差范围（nm）
    """
    matching_pairs = []
    
    for g_band in goci_bands:
        for l_band in landsat_bands:
            if abs(g_band - l_band) <= tolerance:
                matching_pairs.append((g_band, l_band))
                print(f"  匹配波段对: GOCI2 {g_band}nm ↔ Landsat {l_band}nm (差异: {abs(g_band - l_band)}nm)")
    
    return matching_pairs


def load_satellite_data(goci_file, landsat_file):
    """
    加载GOCI2和Landsat的数据
    自动匹配所有相近的波段（差异在20nm以内）
    """
    print("加载卫星数据...")
    
    # GOCI2的所有可用波段
    goci_bands = [380, 412, 443, 490, 510, 555, 620, 660, 680, 709, 745, 865]
    # Landsat的所有可用波段
    landsat_bands = [443, 483, 561, 592, 613, 655, 865, 1609, 2201]
    
    # 找出相近的波段对
    print("\n查找相近波段对（容差±20nm）...")
    band_pairs = find_matching_bands(goci_bands, landsat_bands, tolerance=20)
    
    # 读取GOCI2数据
    goci_data = {'band_pairs': band_pairs}
    with nc.Dataset(goci_file, 'r') as dataset:
        # 读取经纬度
        nav_group = dataset.groups['navigation_data']
        goci_data['lon'] = nav_group.variables['longitude'][:]
        goci_data['lat'] = nav_group.variables['latitude'][:]
        
        # 读取Rrs数据
        print("\n读取GOCI2数据...")
        rrs_group = dataset.groups['geophysical_data'].groups['Rrs']
        for g_band, l_band in band_pairs:
            var_name = f"Rrs_{g_band}"
            if var_name in rrs_group.variables:
                goci_data[var_name] = rrs_group.variables[var_name][:]
                print(f"  ✓ 读取GOCI2 {var_name}")
    
    # 读取Landsat数据
    landsat_data = {'band_pairs': band_pairs}
    with nc.Dataset(landsat_file, 'r') as dataset:
        # 读取经纬度
        landsat_data['lon'] = dataset.variables['lon'][:]
        landsat_data['lat'] = dataset.variables['lat'][:]
        
        # 读取Rrs数据
        print("\n读取Landsat数据...")
        for g_band, l_band in band_pairs:
            var_name = f"Rrs_{l_band}"
            if var_name in dataset.variables:
                landsat_data[var_name] = dataset.variables[var_name][:]
                print(f"  ✓ 读取Landsat {var_name}")
    
    return goci_data, landsat_data


def upsample_goci2_to_landsat(goci_data, landsat_data):
    """
    将GOCI2数据上采样到Landsat的空间分辨率
    使用高斯加权重采样方法
    """
    print("\n执行GOCI2上采样...")
    
    # 创建几何定义
    source_geo = geometry.SwathDefinition(lons=goci_data['lon'], lats=goci_data['lat'])
    target_geo = geometry.SwathDefinition(lons=landsat_data['lon'], lats=landsat_data['lat'])
    
    # 准备上采样后的数据
    upsampled_data = {
        'lon': landsat_data['lon'],
        'lat': landsat_data['lat'],
        'band_pairs': goci_data['band_pairs']
    }
    
    # 对每个匹配的波段对进行上采样
    for g_band, l_band in goci_data['band_pairs']:
        goci_var = f"Rrs_{g_band}"
        landsat_var = f"Rrs_{l_band}"
        
        if goci_var in goci_data:
            print(f"  上采样 GOCI2 {g_band}nm → Landsat {l_band}nm 分辨率...")
            
            # landsat范围
            lon_min, lon_max = np.nanmin(landsat_data['lon']), np.nanmax(landsat_data['lon'])
            lat_min, lat_max = np.nanmin(landsat_data['lat']), np.nanmax(landsat_data['lat'])

            # GOCI2原始shape
            g_lon = goci_data['lon']
            g_lat = goci_data['lat']
            g_rrs = goci_data[goci_var]

            # mask shape和g_lon/g_lat一致
            mask = (g_lon >= lon_min) & (g_lon <= lon_max) & (g_lat >= lat_min) & (g_lat <= lat_max)

            # 用mask索引，结果都是一维，shape完全一致
            goci_lon_crop = g_lon[mask]
            goci_lat_crop = g_lat[mask]
            goci_rrs_crop = g_rrs[mask]

            # 检查shape
            assert goci_lon_crop.shape == goci_lat_crop.shape == goci_rrs_crop.shape

            # SwathDefinition
            source_geo = geometry.SwathDefinition(lons=goci_lon_crop, lats=goci_lat_crop)

            # 插值
            upsampled = kd_tree.resample_gauss(
                source_geo_def=source_geo,
                data=goci_rrs_crop,
                target_geo_def=target_geo,
                radius_of_influence=3000,
                sigmas=1000,
                fill_value=np.nan
            )
            
            # 存储上采样结果，使用对应的Landsat波段名称
            upsampled_data[landsat_var] = upsampled
            # 同时保存原始GOCI2波段名称的映射
            upsampled_data[f"from_{goci_var}"] = landsat_var
            
            print(f"    完成！形状：{goci_rrs_crop.shape} → {upsampled.shape}")
    
    return upsampled_data


def visualize_upsampling_comparison(goci_data, landsat_data, upsampled_data, band_pair):
    """
    可视化上采样前后的对比结果
    显示：GOCI2原始图像 | Landsat参考图像 | GOCI2上采样图像
    """
    g_band, l_band = band_pair
    goci_var = f"Rrs_{g_band}"
    landsat_var = f"Rrs_{l_band}"
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'GOCI2 - GOCI2 {g_band}nm → Landsat {l_band}nm', 
                 fontsize=16, fontweight='bold')
    
    # 数据集配置
    datasets = [
        (f'GOCI2 imagery\n{g_band}nm (250m)', goci_data, goci_var, 0),
        (f'Landsat imagery\n{l_band}nm (30m)', landsat_data, landsat_var, 1),
        (f'GOCI2 upsample imagery\n{g_band}→{l_band}nm (30m)', upsampled_data, landsat_var, 2)
    ]
    
    # 计算统一的颜色范围
    all_valid_data = []
    for _, data, var_name, _ in datasets:
        if var_name in data:
            valid_data = data[var_name][np.isfinite(data[var_name]) & (data[var_name] > 0)]
            if len(valid_data) > 0:
                all_valid_data.extend(valid_data.flatten())
    
    vmin, vmax = np.percentile(all_valid_data, [2, 98]) if all_valid_data else (0, 0.05)
    
    # 绘制每个数据集
    for title, data, var_name, idx in datasets:
        ax = axes[idx]
        
        if var_name in data:
            # 准备数据
            rrs_data = data[var_name]
            lon, lat = data['lon'], data['lat']
            
            # 清理数据
            cleaned_data = np.where(np.isfinite(rrs_data) & (rrs_data > 0), rrs_data, np.nan)
            
            # 判断经纬度是否为2维，如果是则直接用
            if lon.ndim == 2 and lat.ndim == 2:
                im = ax.pcolormesh(lon, lat, cleaned_data, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
            else:
                # 如果是一维的，退回到imshow+extent
                extent = [lon.min(), lon.max(), lat.min(), lat.max()]
                im = ax.imshow(cleaned_data, extent=extent, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto', origin='upper')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Rrs (sr⁻¹)', fontsize=10)
            
            # 设置标题和标签
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Lon (°)', fontsize=10)
            ax.set_ylabel('Lat (°)', fontsize=10)
            ax.tick_params(axis='x', labelrotation=30)
            
            # 添加统计信息
            valid_mask = np.isfinite(cleaned_data)
            if np.sum(valid_mask) > 0:
                mean_val = np.mean(cleaned_data[valid_mask])
                coverage = np.sum(valid_mask) / cleaned_data.size * 100
                
                info_text = (f'mean: {mean_val:.4f}\n'
                           f'pixel number: {cleaned_data.shape[0]}×{cleaned_data.shape[1]}\n'
                           f'cover: {coverage:.1f}%')
                
                ax.text(0.02, 0.98, info_text, 
                       transform=ax.transAxes, 
                       va='top', 
                       fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def main(goci_file, landsat_file):
    """
    主函数：执行GOCI2上采样并可视化结果
    """
    print("="*60)
    print("GOCI2上采样可视化工具")
    print("="*60)
    
    # 1. 加载数据
    goci_data, landsat_data = load_satellite_data(goci_file, landsat_file)
    
    # 检查是否找到匹配的波段
    if not goci_data['band_pairs']:
        print("\n错误：没有找到相近的波段对！")
        return None
    
    print(f"\n找到 {len(goci_data['band_pairs'])} 个匹配的波段对")
    
    # 2. 执行上采样
    upsampled_data = upsample_goci2_to_landsat(goci_data, landsat_data)
    
    # 3. 可视化对比结果
    print("\n生成可视化对比图...")
    
    # 对每个匹配的波段对生成对比图
    for band_pair in goci_data['band_pairs']:
        g_band, l_band = band_pair
        print(f"\n处理波段对: GOCI2 {g_band}nm ↔ Landsat {l_band}nm...")
        
        fig = visualize_upsampling_comparison(
            goci_data, landsat_data, upsampled_data, 
            band_pair=band_pair
        )
        
        # 保存图像
        output_file = f'goci2_upsampling_comparison_{g_band}to{l_band}nm_lonlat_lim.png'
        fig.savefig(output_file, dpi=500, bbox_inches='tight')
        print(f"  保存图像: {output_file}")
        
       # plt.show()
    
    print("\n处理完成！")
    print(f"共处理了 {len(goci_data['band_pairs'])} 个波段对的上采样")
    return upsampled_data


# 使用示例
if __name__ == "__main__":
    # 输入文件路径
    goci_file = r"H:\GK2B_GOCI2_L2_20250323_021530_LA_S009_AC.nc"
    landsat_file = r"H:\LC08_L1TP_118041_20250323_20250331_02_T1\output\L8_OLI_2025_03_23_02_25_53_118041_L2W.nc"
    
    # 执行上采样和可视化
    upsampled_data = main(goci_file, landsat_file)