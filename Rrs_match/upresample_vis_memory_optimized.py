#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化的GOCI2上采样可视化工具
核心功能：
1. 分波段处理，避免同时加载所有数据
2. 及时释放内存
3. 对GOCI2进行上采样到Landsat分辨率
4. 可视化采样前后的对比结果
5. 输出多波段nc文件
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
    tolerance: 波段匹配的容差范围（nm）
    """
    matching_pairs = []
    
    for g_band in goci_bands:
        for l_band in landsat_bands:
            if abs(g_band - l_band) <= tolerance:
                matching_pairs.append((g_band, l_band))
                print(f"  匹配波段对: GOCI2 {g_band}nm ↔ Landsat {l_band}nm (差异: {abs(g_band - l_band)}nm)")
    
    return matching_pairs


def load_landsat_data(landsat_file, band_pairs):
    """
    只加载Landsat数据（作为目标网格）
    """
    print("加载Landsat数据（目标网格）...")
    
    landsat_data = {'band_pairs': band_pairs}
    with nc.Dataset(landsat_file, 'r') as dataset:
        # 读取经纬度
        landsat_data['lon'] = dataset.variables['lon'][:]
        landsat_data['lat'] = dataset.variables['lat'][:]
        print(f"  Landsat网格大小: {landsat_data['lon'].shape}")
    
    return landsat_data


def load_goci_band_data(goci_file, g_band):
    """
    只加载单个GOCI2波段的数据
    """
    print(f"  加载GOCI2 {g_band}nm波段数据...")
    
    with nc.Dataset(goci_file, 'r') as dataset:
        # 读取经纬度
        nav_group = dataset.groups['navigation_data']
        lon = nav_group.variables['longitude'][:]
        lat = nav_group.variables['latitude'][:]
        
        # 读取Rrs数据
        rrs_group = dataset.groups['geophysical_data'].groups['Rrs']
        var_name = f"Rrs_{g_band}"
        if var_name in rrs_group.variables:
            rrs_data = rrs_group.variables[var_name][:]
            print(f"    GOCI2 {var_name} 形状: {rrs_data.shape}")
            return lon, lat, rrs_data
        else:
            print(f"    警告：GOCI2中未找到 {var_name}")
            return None, None, None


def load_landsat_band_data(landsat_file, l_band):
    """
    只加载单个Landsat波段的数据
    """
    print(f"  加载Landsat {l_band}nm波段数据...")
    
    with nc.Dataset(landsat_file, 'r') as dataset:
        var_name = f"Rrs_{l_band}"
        if var_name in dataset.variables:
            rrs_data = dataset.variables[var_name][:]
            print(f"    Landsat {var_name} 形状: {rrs_data.shape}")
            return rrs_data
        else:
            print(f"    警告：Landsat中未找到 {var_name}")
            return None


def upsample_single_band(goci_lon, goci_lat, goci_rrs, landsat_lon, landsat_lat, g_band, l_band):
    """
    对单个波段进行上采样
    """
    print(f"  执行GOCI2 {g_band}nm → Landsat {l_band}nm上采样...")
    
    # 获取Landsat的范围
    lon_min, lon_max = np.nanmin(landsat_lon), np.nanmax(landsat_lon)
    lat_min, lat_max = np.nanmin(landsat_lat), np.nanmax(landsat_lat)
    
    # 裁剪GOCI2数据到Landsat范围
    mask = (goci_lon >= lon_min) & (goci_lon <= lon_max) & (goci_lat >= lat_min) & (goci_lat <= lat_max)
    
    if np.sum(mask) == 0:
        print(f"    警告：GOCI2数据与Landsat范围无重叠")
        return np.full_like(landsat_lon, np.nan)
    
    # 提取重叠区域的数据
    goci_lon_crop = goci_lon[mask]
    goci_lat_crop = goci_lat[mask]
    goci_rrs_crop = goci_rrs[mask]
    
    print(f"    裁剪后有效像素数: {len(goci_rrs_crop)}")
    
    # 创建几何定义
    source_geo = geometry.SwathDefinition(lons=goci_lon_crop, lats=goci_lat_crop)
    target_geo = geometry.SwathDefinition(lons=landsat_lon, lats=landsat_lat)
    
    # 执行上采样
    try:
        upsampled = kd_tree.resample_gauss(
            source_geo_def=source_geo,
            data=goci_rrs_crop,
            target_geo_def=target_geo,
            radius_of_influence=3000,
            sigmas=1000,
            fill_value=np.nan
        )
        
        print(f"    上采样完成！目标形状: {upsampled.shape}")
        return upsampled
        
    except Exception as e:
        print(f"    上采样失败: {str(e)}")
        return np.full_like(landsat_lon, np.nan)


def visualize_single_band_comparison(goci_lon, goci_lat, goci_rrs, 
                                   landsat_lon, landsat_lat, landsat_rrs,
                                   upsampled_rrs, g_band, l_band):
    """
    可视化单个波段的对比结果
    """
    print(f"  生成可视化对比图...")
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'GOCI2 {g_band}nm → Landsat {l_band}nm comparison', 
                 fontsize=16, fontweight='bold')
    
    # 数据集配置
    datasets = [
        (f'GOCI2 {g_band}nm\n(250m)', goci_lon, goci_lat, goci_rrs, 0),
        (f'Landsat {l_band}nm\n(30m)', landsat_lon, landsat_lat, landsat_rrs, 1),
        (f'GOCI2 upsampled\n{g_band}→{l_band}nm (30m)', landsat_lon, landsat_lat, upsampled_rrs, 2)
    ]
    
    # 计算统一的colorbar范围 - 不改变数据，只调整colorbar显示
    all_valid_data = []
    for _, lon, lat, data, _ in datasets:
        if data is not None:
            # 只过滤有限值，不改变数据范围
            valid_data = data[np.isfinite(data)]
            if len(valid_data) > 0:
                all_valid_data.extend(valid_data.flatten())
    
    if all_valid_data:
        vmin, vmax = np.nanpercentile(all_valid_data, [0.1, 99.9])
        print(f"    动态colorbar范围: [{vmin:.6f}, {vmax:.6f}]")
        print(f"    数据统计: 最小值={np.min(all_valid_data):.6f}, 最大值={np.max(all_valid_data):.6f}")
    else:
        vmin, vmax = 0, 0.05
        print(f"    使用默认colorbar范围: [{vmin}, {vmax}]")
    
    # 获取Landsat范围用于画框
    landsat_lon_min, landsat_lon_max = np.nanmin(landsat_lon), np.nanmax(landsat_lon)
    landsat_lat_min, landsat_lat_max = np.nanmin(landsat_lat), np.nanmax(landsat_lat)
    
    # 绘制每个数据集
    for title, lon, lat, data, idx in datasets:
        ax = axes[idx]
        
        if data is not None:
            # 清理数据 - 只处理NaN值，不改变有效数据
            cleaned_data = np.where(np.isfinite(data), data, np.nan)
            
            # 判断经纬度是否为2维
            if lon.ndim == 2 and lat.ndim == 2:
                im = ax.pcolormesh(lon, lat, cleaned_data, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
            else:
                # 如果是一维的，使用imshow
                extent = [lon.min(), lon.max(), lat.min(), lat.max()]
                im = ax.imshow(cleaned_data, extent=extent, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto', origin='upper')
            
            # 在GOCI2原始图中画红框表示Landsat范围
            if idx == 0:  # GOCI2原始图
                # 画红色矩形框
                rect = plt.Rectangle((landsat_lon_min, landsat_lat_min), 
                                   landsat_lon_max - landsat_lon_min, 
                                   landsat_lat_max - landsat_lat_min,
                                   linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                # 添加标签
                ax.text(landsat_lon_min, landsat_lat_max, 'Landsat8\ncoverage', 
                       color='red', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Rrs (sr⁻¹)', fontsize=10)
            
            # 验证colorbar范围设置正确
            print(f"    子图{idx} colorbar范围: [{im.norm.vmin:.6f}, {im.norm.vmax:.6f}]")
            
            # 设置标题和标签
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Longitude (°)', fontsize=10)
            ax.set_ylabel('Latitude (°)', fontsize=10)
            ax.tick_params(axis='x', labelrotation=30)
            
            # 添加统计信息（删除覆盖率计算）
            valid_mask = np.isfinite(cleaned_data)
            if np.sum(valid_mask) > 0:
                mean_val = np.mean(cleaned_data[valid_mask])
                
                info_text = (f'mean: {mean_val:.4f}\n'
                           f'shape: {cleaned_data.shape[0]}×{cleaned_data.shape[1]}')
                
                ax.text(0.02, 0.98, info_text, 
                       transform=ax.transAxes, 
                       va='top', 
                       fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def process_single_band_pair(goci_file, landsat_file, landsat_data, band_pair, output_nc_file):
    """
    处理单个波段对：加载、上采样、可视化、保存、释放内存
    """
    g_band, l_band = band_pair
    print(f"\n{'='*50}")
    print(f"处理波段对: GOCI2 {g_band}nm ↔ Landsat {l_band}nm")
    print(f"{'='*50}")
    
    try:
        # 1. 加载GOCI2波段数据
        goci_lon, goci_lat, goci_rrs = load_goci_band_data(goci_file, g_band)
        if goci_rrs is None:
            print(f"  跳过波段对 {g_band}nm ↔ {l_band}nm（数据不可用）")
            return None
        
        # 2. 加载Landsat波段数据
        landsat_rrs = load_landsat_band_data(landsat_file, l_band)
        if landsat_rrs is None:
            print(f"  跳过波段对 {g_band}nm ↔ {l_band}nm（数据不可用）")
            return None
        
        # 3. 执行上采样
        upsampled_rrs = upsample_single_band(
            goci_lon, goci_lat, goci_rrs,
            landsat_data['lon'], landsat_data['lat'],
            g_band, l_band
        )
        
        # 4. 生成可视化
        fig = visualize_single_band_comparison(
            goci_lon, goci_lat, goci_rrs,
            landsat_data['lon'], landsat_data['lat'], landsat_rrs,
            upsampled_rrs, g_band, l_band
        )
        
        # 5. 保存图像
        output_file = f'goci2_upsampling_{g_band}to{l_band}nm_memory_optimized.png'
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  保存图像: {output_file}")
        
        # 6. 关闭图形释放内存
        plt.close(fig)
        
        # 7. 返回上采样结果用于保存到多波段文件
        return {
            'band_name': f'Rrs_{l_band}',
            'data': upsampled_rrs,
            'long_name': f'Remote sensing reflectance from GOCI2 {g_band}nm upsampled to Landsat {l_band}nm',
            'units': 'sr^-1'
        }
        
    except Exception as e:
        print(f"  处理波段对 {g_band}nm ↔ {l_band}nm 时出错: {str(e)}")
        return None
    
    finally:
        # 8. 强制垃圾回收释放内存
        gc.collect()
        print(f"  内存已清理")


def save_multiband_nc(output_nc_file, landsat_data, band_results):
    """
    保存多波段nc文件
    """
    print(f"\n保存多波段nc文件: {output_nc_file}")
    
    with nc.Dataset(output_nc_file, 'w') as dataset:
        # 创建维度
        dataset.createDimension('y', landsat_data['lon'].shape[0])
        dataset.createDimension('x', landsat_data['lon'].shape[1])
        
        # 创建经纬度变量
        lon_var = dataset.createVariable('lon', 'f4', ('y', 'x'))
        lat_var = dataset.createVariable('lat', 'f4', ('y', 'x'))
        
        lon_var[:] = landsat_data['lon']
        lat_var[:] = landsat_data['lat']
        
        # 添加经纬度属性
        lon_var.long_name = 'Longitude'
        lon_var.units = 'degrees_east'
        lat_var.long_name = 'Latitude'
        lat_var.units = 'degrees_north'
        
        # 创建每个波段变量
        for band_result in band_results:
            if band_result is not None:
                var_name = band_result['band_name']
                rrs_var = dataset.createVariable(var_name, 'f4', ('y', 'x'))
                rrs_var[:] = band_result['data']
                rrs_var.long_name = band_result['long_name']
                rrs_var.units = band_result['units']
                rrs_var.fill_value = np.nan
                
                print(f"  ✓ 保存波段: {var_name}")
        
        # 添加全局属性
        dataset.title = 'GOCI2 upsampled to Landsat8 resolution'
        dataset.description = 'Remote sensing reflectance data from GOCI2 upsampled to Landsat8 spatial resolution'
        dataset.source = 'GOCI2 satellite data upsampled using pyresample'
        dataset.history = 'Created by upresample_vis_memory_optimized.py'
    
    print(f"  多波段nc文件保存完成！")


def main(goci_file, landsat_file):
    """
    主函数：分波段处理GOCI2上采样并可视化结果
    """
    print("="*60)
    print("内存优化的GOCI2上采样可视化工具")
    print("="*60)
    
    # GOCI2和Landsat的波段定义
    goci_bands = [380, 412, 443, 490, 510, 555, 620, 660, 680, 709, 745, 865]
    # landsat_bands = [443, 483, 561, 592, 613, 655, 865, 1609, 2201]
    landsat_bands = sorted([int(var.split('_')[1]) for var in nc.Dataset(landsat_file).variables if var.startswith('Rrs_')])
    print(landsat_bands)
    
    # 找出相近的波段对
    print("\n查找相近波段对（容差±20nm）...")
    band_pairs = find_matching_bands(goci_bands, landsat_bands, tolerance=20)
    
    if not band_pairs:
        print("\n错误：没有找到相近的波段对！")
        return
    
    print(f"\n找到 {len(band_pairs)} 个匹配的波段对")
    
    # 加载Landsat网格数据（只加载一次）
    landsat_data = load_landsat_data(landsat_file, band_pairs)
    
    # 准备输出文件名
    output_nc_file = 'goci2_upsampled_multiband.nc'
    
    # 存储所有波段的结果
    band_results = []
    
    # 逐个处理波段对
    for i, band_pair in enumerate(band_pairs, 1):
        print(f"\n进度: {i}/{len(band_pairs)}")
        result = process_single_band_pair(goci_file, landsat_file, landsat_data, band_pair, output_nc_file)
        band_results.append(result)
    
    # 保存多波段nc文件
    save_multiband_nc(output_nc_file, landsat_data, band_results)
    
    print(f"\n{'='*60}")
    print("所有波段对处理完成！")
    print(f"共处理了 {len(band_pairs)} 个波段对")
    print(f"多波段nc文件: {output_nc_file}")
    print("="*60)


# 使用示例
if __name__ == "__main__":
    # 输入文件路径
    goci_file = r"D:\Py_Code\SR_Imagery\GK2B_GOCI2_L2_20250309_021530_LA_S007_AC.nc"
    landsat_file = r"D:\Py_Code\SR_Imagery\LC08_L1TP_116036_20250309_20250324_02_T1\output\L8_OLI_2025_03_09_02_11_42_116036_L2W.nc"
    
    # 执行分波段上采样和可视化
    main(goci_file, landsat_file) 