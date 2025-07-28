#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOCI2超分辨率预处理
将低分辨率GOCI2数据上采样到高分辨率Landsat网格
用于超分辨率算法的训练和评估
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
        'rrs_variables': rrs_variables,
        'shape': (rows, cols)
    }

def upsample_low2high(lon_low, lat_low, array_low, lon_high, lat_high, radius=2000, method='nearest'):
    """
    将低分辨率数据上采样到高分辨率网格
    
    参数:
    - lon_low, lat_low: 低分辨率数据的经纬度
    - array_low: 低分辨率数据数组
    - lon_high, lat_high: 高分辨率网格的经纬度  
    - radius: 影响半径(米)
    - method: 插值方法 ('nearest', 'bilinear', 'gaussian')
    
    返回:
    - 上采样后的高分辨率数据
    """
    print(f"上采样：{array_low.shape} → {lon_high.shape}")
    
    # 创建几何定义
    source_geo = geometry.SwathDefinition(lons=lon_low, lats=lat_low)
    target_geo = geometry.SwathDefinition(lons=lon_high, lats=lat_high)
    
    # 根据方法选择重采样算法
    if method == 'nearest':
        upsampled_data = kd_tree.resample_nearest(
            source_geo, array_low, target_geo,
            radius_of_influence=radius,
            fill_value=np.nan
        )
    elif method == 'bilinear':
        # 双线性插值（更平滑）
        upsampled_data = kd_tree.resample_gauss(
            source_geo, array_low, target_geo,
            radius_of_influence=radius,
            sigmas=radius/3,  # 高斯核标准差
            fill_value=np.nan
        )
    elif method == 'gaussian':
        # 高斯加权插值
        upsampled_data = kd_tree.resample_gauss(
            source_geo, array_low, target_geo,
            radius_of_influence=radius,
            sigmas=radius/2,
            fill_value=np.nan
        )
    else:
        raise ValueError(f"不支持的插值方法: {method}")
    
    return upsampled_data

def match_spectral_bands(goci_wavelengths, landsat_wavelengths):
    """
    匹配GOCI2和Landsat的光谱波段
    返回匹配的波长对应关系
    """
    # 定义波长匹配规则（允许一定误差范围）
    matches = []
    tolerance = 20  # 20nm容差
    
    for goci_wl in goci_wavelengths:
        # 找到最接近的Landsat波长
        differences = np.abs(landsat_wavelengths - goci_wl)
        min_diff_idx = np.argmin(differences)
        min_diff = differences[min_diff_idx]
        
        if min_diff <= tolerance:
            landsat_wl = landsat_wavelengths[min_diff_idx]
            matches.append({
                'goci_wl': goci_wl,
                'landsat_wl': landsat_wl,
                'goci_idx': np.where(goci_wavelengths == goci_wl)[0][0],
                'landsat_idx': np.where(landsat_wavelengths == landsat_wl)[0][0],
                'diff': min_diff
            })
            print(f"匹配波长: GOCI2 {goci_wl}nm ↔ Landsat {landsat_wl}nm (差异: {min_diff}nm)")
    
    return matches

def create_superresolution_dataset(goci_file, landsat_file, output_file, interp_method='bilinear'):
    """
    创建超分辨率训练数据集
    将GOCI2数据上采样到Landsat网格，同时保存原始Landsat作为真值
    
    参数:
    - goci_file: GOCI2文件路径
    - landsat_file: Landsat文件路径  
    - output_file: 输出文件路径
    - interp_method: 插值方法 ('nearest', 'bilinear', 'gaussian')
    """
    print("=== GOCI2超分辨率数据集创建 ===\n")
    
    # 1. 读取数据
    goci_data = read_goci2_data(goci_file)
    landsat_data = read_landsat_data(landsat_file)
    
    if goci_data['lon'] is None or landsat_data['lon'] is None:
        print("错误：缺少经纬度信息，无法进行重采样")
        return
    
    print(f"GOCI2数据形状: {goci_data['rrs_data'].shape}")
    print(f"Landsat数据形状: {landsat_data['rrs_data'].shape}")
    print(f"分辨率提升倍数: {landsat_data['shape'][0] / 2780:.1f}x")
    
    # 2. 光谱波段匹配
    band_matches = match_spectral_bands(goci_data['wavelengths'], landsat_data['wavelengths'])
    
    if not band_matches:
        print("错误：没有找到匹配的光谱波段")
        return
    
    # 3. 对每个匹配波段进行上采样
    print(f"\n开始上采样处理，使用 {interp_method} 插值...")
    upsampled_data = {}
    
    for match in band_matches:
        goci_wl = match['goci_wl']
        landsat_wl = match['landsat_wl']
        goci_idx = match['goci_idx']
        landsat_idx = match['landsat_idx']
        
        print(f"\n处理波长对: GOCI2 {goci_wl}nm → Landsat {landsat_wl}nm 网格")
        
        # 提取GOCI2该波段数据
        goci_band = goci_data['rrs_data'][:, :, goci_idx]
        
        # 上采样到Landsat网格
        upsampled_band = upsample_low2high(
            goci_data['lon'], goci_data['lat'], goci_band,
            landsat_data['lon'], landsat_data['lat'],
            radius=3000,  # 3km影响半径，适合上采样
            method=interp_method
        )
        
        # 提取对应的Landsat真值数据
        landsat_band = landsat_data['rrs_data'][:, :, landsat_idx]
        
        # 保存数据
        upsampled_data[f'GOCI2_Rrs_{goci_wl}_upsampled'] = upsampled_band
        upsampled_data[f'Landsat_Rrs_{landsat_wl}_truth'] = landsat_band
        upsampled_data[f'wavelength_pair_{goci_wl}_{landsat_wl}'] = {
            'goci_wl': goci_wl,
            'landsat_wl': landsat_wl,
            'spectral_diff': match['diff']
        }
    
    # 4. 计算上采样质量指标
    print("\n计算上采样质量...")
    quality_metrics = {}
    
    for match in band_matches:
        goci_wl = match['goci_wl']
        landsat_wl = match['landsat_wl']
        
        upsampled = upsampled_data[f'GOCI2_Rrs_{goci_wl}_upsampled']
        truth = upsampled_data[f'Landsat_Rrs_{landsat_wl}_truth']
        
        # 计算有效像素区域的统计
        valid_mask = ~np.isnan(upsampled) & ~np.isnan(truth) & (truth > 0) & (upsampled > 0)
        
        if np.sum(valid_mask) > 1000:  # 至少1000个有效像素
            upsampled_valid = upsampled[valid_mask]
            truth_valid = truth[valid_mask]
            
            # 基本统计
            bias = np.mean(upsampled_valid - truth_valid)
            rmse = np.sqrt(np.mean((upsampled_valid - truth_valid)**2))
            mae = np.mean(np.abs(upsampled_valid - truth_valid))
            correlation = np.corrcoef(upsampled_valid, truth_valid)[0, 1]
            
            quality_metrics[f'band_{goci_wl}_{landsat_wl}'] = {
                'bias': bias,
                'rmse': rmse,
                'mae': mae,
                'correlation': correlation,
                'valid_pixels': np.sum(valid_mask)
            }
            
            print(f"波长 {goci_wl}-{landsat_wl}nm: RMSE={rmse:.6f}, R={correlation:.3f}, 有效像素={np.sum(valid_mask)}")
    
    # 5. 创建输出Dataset
    print("\n创建输出数据集...")
    
    # 准备数据变量
    data_vars = {
        'longitude': (['y', 'x'], landsat_data['lon']),
        'latitude': (['y', 'x'], landsat_data['lat'])
    }
    
    # 添加上采样和真值数据
    for var_name, data in upsampled_data.items():
        if isinstance(data, np.ndarray):
            data_vars[var_name] = (['y', 'x'], data)
    
    # 创建Dataset
    ds_output = xr.Dataset(data_vars)
    
    # 添加元数据属性
    ds_output.attrs.update({
        'title': 'GOCI2超分辨率训练数据集',
        'description': f'GOCI2数据上采样到Landsat网格，插值方法: {interp_method}',
        'goci_resolution': '~500m (2780x2780)',
        'landsat_resolution': '30m',
        'upsampling_ratio': f'{landsat_data["shape"][0] / 2780:.1f}x',
        'interpolation_method': interp_method,
        'goci_wavelengths': str(goci_data['wavelengths'].tolist()),
        'landsat_wavelengths': str(landsat_data['wavelengths'].tolist()),
        'matched_bands': str([(m['goci_wl'], m['landsat_wl']) for m in band_matches]),
        'quality_metrics': str(quality_metrics)
    })
    
    # 保存文件
    print(f"保存到: {output_file}")
    ds_output.to_netcdf(output_file, format='NETCDF4', engine='netcdf4')
    
    # 输出处理摘要
    print("\n=== 处理完成摘要 ===")
    print(f"输出文件: {output_file}")
    print(f"数据维度: {ds_output.dims}")
    print(f"变量数量: {len(ds_output.data_vars)}")
    print(f"匹配波段: {len(band_matches)}个")
    print(f"插值方法: {interp_method}")
    
    return ds_output, quality_metrics

def compare_interpolation_methods(goci_file, landsat_file, output_dir):
    """
    比较不同插值方法的效果
    """
    methods = ['nearest', 'bilinear', 'gaussian']
    results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"测试插值方法: {method}")
        print(f"{'='*50}")
        
        output_file = f"{output_dir}/goci2_upsampled_{method}.nc"
        ds, metrics = create_superresolution_dataset(
            goci_file, landsat_file, output_file, method
        )
        results[method] = {
            'dataset': ds,
            'metrics': metrics,
            'output_file': output_file
        }
    
    # 输出比较结果
    print(f"\n{'='*50}")
    print("插值方法比较结果")
    print(f"{'='*50}")
    
    # 汇总各方法的平均RMSE
    for method, result in results.items():
        rmse_values = [m['rmse'] for m in result['metrics'].values()]
        if rmse_values:
            avg_rmse = np.mean(rmse_values)
            print(f"{method:10s}: 平均RMSE = {avg_rmse:.6f}")
    
    return results

# 使用示例
if __name__ == "__main__":
    # 文件路径
    goci_file = r"H:\GK2B_GOCI2_L2_20250323_021530_LA_S009_AC.nc"
    landsat_file = r"H:\LC08_L1TP_118041_20250323_20250331_02_T1\output\L8_OLI_2025_03_23_02_25_53_118041_L2W.nc"
    
    # 单一插值方法处理
    output_file = r"H:\GOCI2_Landsat_superres_dataset.nc"
    
    print("创建超分辨率训练数据集...")
    dataset, metrics = create_superresolution_dataset(
        goci_file, landsat_file, output_file, 
        interp_method='bilinear'  # 推荐使用bilinear获得更平滑结果
    )
    
    # 可选：比较多种插值方法
    # output_dir = r"H:\superres_comparison"
    # results = compare_interpolation_methods(goci_file, landsat_file, output_dir)