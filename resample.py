#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOCI2和Landsat数据重采样处理
将高分辨率Landsat数据重采样到GOCI2网格
"""

import numpy as np
import netCDF4 as nc
import xarray as xr
from pyresample import geometry, kd_tree
from scipy.interpolate import interp1d

def read_goci2_data(filename):
    """
    读取GOCI2数据（经纬度和Rrs数据）
    """
    print("读取GOCI2数据...")
    
    # GOCI2波长
    wavelengths = np.array([380, 412, 443, 490, 510, 555, 620, 660, 680, 709, 745, 865])
    rrs_variables = [f"Rrs_{wavelength}" for wavelength in wavelengths]
    
    with nc.Dataset(filename, 'r') as dataset:
        # 读取经纬度数据
        try:
            # GOCI2经纬度通常在navigation_data组中
            nav_group = dataset.groups['navigation_data']
            lon = nav_group.variables['longitude'][:]
            lat = nav_group.variables['latitude'][:]
        except:
            # 如果不在navigation_data中，尝试其他位置
            try:
                lon = dataset.variables['longitude'][:]
                lat = dataset.variables['latitude'][:]
            except:
                # 如果没有经纬度数据，需要根据实际文件结构调整
                print("警告：未找到经纬度数据，请检查文件结构")
                lon = lat = None
        
        # 读取Rrs数据
        rrs_group = dataset.groups['geophysical_data'].groups['Rrs']
        rrs_data = np.zeros((2780, 2780, 12), dtype=np.float32)
        
        for i, var_name in enumerate(rrs_variables):
            print(f'读取 GOCI2 {var_name}...')
            rrs_data[:, :, i] = rrs_group.variables[var_name][:]
    
    return {
        'lon': lon,
        'lat': lat,
        'wavelengths': wavelengths,
        'rrs_data': rrs_data,
        'rrs_variables': rrs_variables
    }

def read_landsat_data(filename):
    """
    读取Landsat数据（经纬度和Rrs数据）
    """
    print("读取Landsat数据...")
    
    # Landsat 8/9 OLI波长
    wavelengths = np.array([443, 483, 561, 592, 613, 655, 865, 1609, 2201])
    rrs_variables = [f"Rrs_{wavelength}" for wavelength in wavelengths]
    
    with nc.Dataset(filename, 'r') as dataset:
        # 读取经纬度数据
        try:
            lon = dataset.variables['longitude'][:]
            lat = dataset.variables['latitude'][:]
        except:
            try:
                lon = dataset.variables['lon'][:]
                lat = dataset.variables['lat'][:]
            except:
                print("警告：未找到Landsat经纬度数据，请检查文件结构")
                lon = lat = None
        
        # 获取数据维度
        first_var = dataset.variables[rrs_variables[0]]
        rows, cols = first_var.shape
        num_bands = len(wavelengths)
        
        # 读取Rrs数据
        rrs_data = np.zeros((rows, cols, num_bands), dtype=np.float32)
        for i, var_name in enumerate(rrs_variables):
            print(f'读取 Landsat {var_name}...')
            rrs_data[:, :, i] = dataset.variables[var_name][:]
    
    return {
        'lon': lon,
        'lat': lat,
        'wavelengths': wavelengths,
        'rrs_data': rrs_data,
        'rrs_variables': rrs_variables
    }

def resample_high2low(lon_high, lat_high, array_high, lon_low, lat_low, radius=1000):
    """
    将高分辨率数据重采样到低分辨率网格
    
    参数:
    - lon_high, lat_high: 高分辨率数据的经纬度
    - array_high: 高分辨率数据数组
    - lon_low, lat_low: 低分辨率网格的经纬度
    - radius: 影响半径(米)
    """
    # 创建几何定义
    source_geo = geometry.SwathDefinition(lons=lon_high, lats=lat_high)
    target_geo = geometry.SwathDefinition(lons=lon_low, lats=lat_low)
    
    # 重采样
    resampled_data = kd_tree.resample_nearest(
        source_geo, array_high, target_geo,
        radius_of_influence=radius,
        fill_value=np.nan
    )
    
    return resampled_data

def spectral_matching(landsat_wavelengths, landsat_rrs, target_wavelengths):
    """
    光谱匹配：将Landsat波段插值到目标波长
    
    参数:
    - landsat_wavelengths: Landsat波长数组
    - landsat_rrs: Landsat Rrs数据 [rows, cols, bands]
    - target_wavelengths: 目标波长数组
    
    返回:
    - 插值后的Rrs数据
    """
    rows, cols, _ = landsat_rrs.shape
    target_rrs = np.full((rows, cols, len(target_wavelengths)), np.nan, dtype=np.float32)
    
    # 只使用可见光-近红外波段进行插值（排除短波红外）
    vis_nir_mask = landsat_wavelengths <= 900  # 只用到865nm波段
    landsat_wl_vis = landsat_wavelengths[vis_nir_mask]
    
    print(f"使用Landsat波段: {landsat_wl_vis}")
    print(f"目标GOCI2波段: {target_wavelengths}")
    
    for i in range(rows):
        if i % 100 == 0:
            print(f"处理光谱匹配: {i}/{rows}")
        
        for j in range(cols):
            # 获取当前像素的Rrs值
            pixel_rrs = landsat_rrs[i, j, vis_nir_mask]
            
            # 跳过无效数据
            if np.all(np.isnan(pixel_rrs)) or np.all(pixel_rrs <= 0):
                continue
            
            # 创建插值函数
            valid_mask = ~np.isnan(pixel_rrs) & (pixel_rrs > 0)
            if np.sum(valid_mask) >= 2:  # 至少需要2个有效点
                interp_func = interp1d(
                    landsat_wl_vis[valid_mask], 
                    pixel_rrs[valid_mask],
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                
                # 插值到目标波长
                for k, target_wl in enumerate(target_wavelengths):
                    if landsat_wl_vis.min() <= target_wl <= landsat_wl_vis.max():
                        target_rrs[i, j, k] = interp_func(target_wl)
    
    return target_rrs

def process_goci_landsat_data(goci_file, landsat_file, output_file):
    """
    主处理函数：将Landsat数据重采样并匹配到GOCI2网格
    """
    print("=== GOCI2-Landsat数据重采样处理 ===\n")
    
    # 1. 读取数据
    goci_data = read_goci2_data(goci_file)
    landsat_data = read_landsat_data(landsat_file)
    
    if goci_data['lon'] is None or landsat_data['lon'] is None:
        print("错误：缺少经纬度信息，无法进行重采样")
        return
    
    print(f"GOCI2数据形状: {goci_data['rrs_data'].shape}")
    print(f"Landsat数据形状: {landsat_data['rrs_data'].shape}")
    
    # 2. 光谱匹配：选择GOCI2和Landsat共同的波段
    # 找到相近的波长匹配
    goci_target_wl = []
    goci_target_idx = []
    
    # 匹配规则（根据传感器特性调整）
    wl_matches = {
        443: 443,   # 蓝
        490: 483,   # 蓝绿
        555: 561,   # 绿
        620: 613,   # 红
        660: 655,   # 红
        865: 865    # 近红外
    }
    
    for goci_wl, landsat_wl in wl_matches.items():
        if goci_wl in goci_data['wavelengths'] and landsat_wl in landsat_data['wavelengths']:
            goci_target_wl.append(goci_wl)
            goci_idx = np.where(goci_data['wavelengths'] == goci_wl)[0][0]
            goci_target_idx.append(goci_idx)
    
    goci_target_wl = np.array(goci_target_wl)
    print(f"匹配的波长: {goci_target_wl}")
    
    # 3. 对每个匹配的波段进行重采样
    resampled_data = {}
    
    for i, target_wl in enumerate(goci_target_wl):
        print(f"\n处理波长 {target_wl}nm...")
        
        # 找到Landsat对应波段
        landsat_wl = wl_matches[target_wl]
        landsat_idx = np.where(landsat_data['wavelengths'] == landsat_wl)[0][0]
        
        # 提取Landsat该波段数据
        landsat_band = landsat_data['rrs_data'][:, :, landsat_idx]
        
        # 重采样到GOCI2网格
        resampled_band = resample_high2low(
            landsat_data['lon'], landsat_data['lat'], landsat_band,
            goci_data['lon'], goci_data['lat'],
            radius=2000  # 2km影响半径
        )
        
        resampled_data[f'Rrs_{target_wl}_resample'] = resampled_band
    
    # 4. 创建输出Dataset
    print("\n创建输出文件...")
    
    # 准备数据字典
    data_vars = {
        'longitude': (['y', 'x'], goci_data['lon']),
        'latitude': (['y', 'x'], goci_data['lat'])
    }
    
    # 添加原始GOCI2数据
    for i, wl in enumerate(goci_data['wavelengths']):
        data_vars[f'GOCI2_Rrs_{wl}'] = (['y', 'x'], goci_data['rrs_data'][:, :, i])
    
    # 添加重采样的Landsat数据
    for var_name, data in resampled_data.items():
        data_vars[var_name] = (['y', 'x'], data)
    
    # 创建Dataset
    ds_output = xr.Dataset(data_vars)
    
    # 添加属性
    ds_output.attrs['title'] = 'GOCI2-Landsat重采样数据'
    ds_output.attrs['description'] = 'Landsat数据重采样到GOCI2网格'
    ds_output.attrs['goci_wavelengths'] = str(goci_data['wavelengths'].tolist())
    ds_output.attrs['landsat_wavelengths'] = str(landsat_data['wavelengths'].tolist())
    ds_output.attrs['matched_wavelengths'] = str(goci_target_wl.tolist())
    
    # 保存文件
    ds_output.to_netcdf(output_file, format='NETCDF4')
    print(f"\n处理完成！输出文件: {output_file}")
    
    return ds_output

# 使用示例
if __name__ == "__main__":
    # 文件路径
    goci_file = r"H:\GK2B_GOCI2_L2_20250323_021530_LA_S009_AC.nc"
    landsat_file = r"H:\LC08_L1TP_118041_20250323_20250331_02_T1\output\L8_OLI_2025_03_23_02_25_53_118041_L2W.nc"
    output_file = r"H:\GOCI2_Landsat_resampled.nc"
    
    # 执行处理
    result = process_goci_landsat_data(goci_file, landsat_file, output_file)
    
    if result is not None:
        print("\n=== 处理结果摘要 ===")
        print(f"输出数据变量: {list(result.data_vars.keys())}")
        print(f"数据形状: {result.dims}")