#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复内存问题的GOCI2上采样可视化工具
专门解决pcolormesh内存不足的问题
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from pyresample import geometry, kd_tree
import gc
import warnings
warnings.filterwarnings('ignore')


def find_matching_bands(goci_bands, landsat_bands, tolerance=20):
    """
    找出GOCI2和Landsat中相近的波段对
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


def crop_data_to_region(lon_data, lat_data, rrs_data, target_bounds):
    """
    Crop data to specified region while maintaining 2D structure
    """
    print(f"  裁剪前shape: {rrs_data.shape}")
    
    lon_min, lon_max, lat_min, lat_max = target_bounds
    
    # Find pixels within target range
    in_range = (lon_data >= lon_min) & (lon_data <= lon_max) & (lat_data >= lat_min) & (lat_data <= lat_max)
    
    if np.sum(in_range) > 0:
        # Find minimum rectangle containing all valid pixels
        valid_rows = np.where(np.any(in_range, axis=1))[0]
        valid_cols = np.where(np.any(in_range, axis=0))[0]
        
        if len(valid_rows) > 0 and len(valid_cols) > 0:
            # Crop data while maintaining 2D structure
            cropped_lon = lon_data[valid_rows[0]:valid_rows[-1]+1, valid_cols[0]:valid_cols[-1]+1]
            cropped_lat = lat_data[valid_rows[0]:valid_rows[-1]+1, valid_cols[0]:valid_cols[-1]+1]
            cropped_rrs = rrs_data[valid_rows[0]:valid_rows[-1]+1, valid_cols[0]:valid_cols[-1]+1]
            
            print(f"  裁剪后shape: {cropped_rrs.shape}")
            return cropped_lon, cropped_lat, cropped_rrs, True
    
    return None, None, None, False


def upsample_goci2_to_landsat(goci_data, landsat_data):
    """
    将GOCI2数据上采样到Landsat的空间分辨率
    """
    print("\n执行GOCI2上采样...")
    
    # 获取Landsat的范围
    lon_min, lon_max = np.nanmin(landsat_data['lon']), np.nanmax(landsat_data['lon'])
    lat_min, lat_max = np.nanmin(landsat_data['lat']), np.nanmax(landsat_data['lat'])
    target_bounds = (lon_min, lon_max, lat_min, lat_max)
    print(f"目标范围: lon[{lon_min:.4f}, {lon_max:.4f}], lat[{lat_min:.4f}, {lat_max:.4f}]")
    
    # 创建目标几何定义（Landsat网格）
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
            print(f"\n处理波段对: GOCI2 {g_band}nm → Landsat {l_band}nm...")
            
            # 裁剪GOCI2数据到Landsat范围
            cropped_lon, cropped_lat, cropped_rrs, success = crop_data_to_region(
                goci_data['lon'], 
                goci_data['lat'], 
                goci_data[goci_var], 
                target_bounds
            )
            
            if not success:
                print(f"  警告：无法裁剪GOCI2 {g_band}nm数据到目标范围")
                upsampled_data[landsat_var] = np.full_like(landsat_data['lon'], np.nan)
                continue
            
            # 创建裁剪后的源几何定义
            source_geo = geometry.SwathDefinition(lons=cropped_lon, lats=cropped_lat)
            
            # 执行上采样
            print(f"  执行高斯插值上采样...")
            try:
                upsampled = kd_tree.resample_gauss(
                    source_geo_def=source_geo,
                    data=cropped_rrs,
                    target_geo_def=target_geo,
                    radius_of_influence=3000,
                    sigmas=1000,
                    fill_value=np.nan
                )
                
                # 存储上采样结果
                upsampled_data[landsat_var] = upsampled
                upsampled_data[f"from_{goci_var}"] = landsat_var
                
                print(f"  上采样完成！形状：{cropped_rrs.shape} → {upsampled.shape}")
                
            except Exception as e:
                print(f"  上采样失败: {str(e)}")
                upsampled_data[landsat_var] = np.full_like(landsat_data['lon'], np.nan)
    
    return upsampled_data


def visualize_single_band_safely(goci_data, landsat_data, upsampled_data, band_pair):
    """
    安全地可视化单个波段，避免内存问题
    """
    g_band, l_band = band_pair
    goci_var = f"Rrs_{g_band}"
    landsat_var = f"Rrs_{l_band}"
    
    print(f"  生成可视化对比图...")
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'GOCI2 {g_band}nm → Landsat {l_band}nm', 
                 fontsize=16, fontweight='bold')
    
    # 数据集配置
    datasets = [
        (f'GOCI2 {g_band}nm\n(250m)', goci_data, goci_var, 0),
        (f'Landsat {l_band}nm\n(30m)', landsat_data, landsat_var, 1),
        (f'GOCI2上采样\n{g_band}→{l_band}nm (30m)', upsampled_data, landsat_var, 2)
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
            
            # 检查数据大小，如果太大则降采样
            total_pixels = cleaned_data.size
            if total_pixels > 500000:  # 降低阈值，更保守
                # 计算降采样因子
                downsample_factor = int(np.sqrt(total_pixels / 500000))
                print(f"    数据过大 ({total_pixels:,} 像素)，降采样因子: {downsample_factor}")
                
                # 降采样数据
                if lon.ndim == 2 and lat.ndim == 2:
                    lon_down = lon[::downsample_factor, ::downsample_factor]
                    lat_down = lat[::downsample_factor, ::downsample_factor]
                    data_down = cleaned_data[::downsample_factor, ::downsample_factor]
                else:
                    lon_down = lon[::downsample_factor]
                    lat_down = lat[::downsample_factor]
                    data_down = cleaned_data[::downsample_factor, ::downsample_factor]
                
                print(f"    降采样后: {data_down.shape[0]}×{data_down.shape[1]} = {data_down.size:,} 像素")
                lon, lat, cleaned_data = lon_down, lat_down, data_down
            
            # 使用imshow而不是pcolormesh来避免内存问题
            try:
                if lon.ndim == 2 and lat.ndim == 2:
                    # 对于2D网格，使用imshow更安全
                    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
                    im = ax.imshow(cleaned_data, extent=extent, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto', origin='upper')
                else:
                    # 对于1D网格，也使用imshow
                    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
                    im = ax.imshow(cleaned_data, extent=extent, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto', origin='upper')
                
                print(f"    使用imshow绘制 {title}")
                
            except Exception as e:
                print(f"    绘制失败: {str(e)}")
                # 如果还是失败，尝试更激进的降采样
                if cleaned_data.size > 100000:
                    factor = int(np.sqrt(cleaned_data.size / 100000))
                    cleaned_data = cleaned_data[::factor, ::factor]
                    if lon.ndim == 2 and lat.ndim == 2:
                        lon = lon[::factor, ::factor]
                        lat = lat[::factor, ::factor]
                    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
                    im = ax.imshow(cleaned_data, extent=extent, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto', origin='upper')
                    print(f"    使用激进降采样后绘制成功")
            
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
                           f'pixels: {cleaned_data.shape[0]}×{cleaned_data.shape[1]}\n'
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
    print("内存优化的GOCI2上采样可视化工具")
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
    
    # 3. 逐个处理波段对的可视化
    print("\n生成可视化对比图...")
    
    for i, band_pair in enumerate(goci_data['band_pairs'], 1):
        g_band, l_band = band_pair
        print(f"\n进度: {i}/{len(goci_data['band_pairs'])}")
        print(f"处理波段对: GOCI2 {g_band}nm ↔ Landsat {l_band}nm...")
        
        try:
            # 生成可视化
            fig = visualize_single_band_safely(
                goci_data, landsat_data, upsampled_data, 
                band_pair=band_pair
            )
            
            # 保存图像
            output_file = f'goci2_upsampling_{g_band}to{l_band}nm_memory_fixed.png'
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  保存图像: {output_file}")
            
            # 立即关闭图形释放内存
            plt.close(fig)
            
            # 强制垃圾回收
            gc.collect()
            
        except Exception as e:
            print(f"  处理波段对 {g_band}nm ↔ {l_band}nm 时出错: {str(e)}")
            # 继续处理下一个波段对
            continue
    
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