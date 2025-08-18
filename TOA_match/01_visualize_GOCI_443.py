import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt

# 文件路径
file_path = r"D:\Py_Code\SR_Imagery\GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"

# 读取文件
with nc.Dataset(file_path, 'r') as dataset:
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
            exit()
    
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

# 可视化
plt.figure(figsize=(8, 6))

# 如果有经纬度信息，使用地理坐标显示
if lat is not None and lon is not None:
    # 使用经纬度范围显示
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    im = plt.imshow(data, cmap='viridis', vmin=0, vmax=99, extent=extent, aspect='auto')
    plt.title('L_TOA_443 Band')
    plt.colorbar(im, label='TOA Radiance')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
else:
    # 如果没有经纬度，使用像素索引
    im = plt.imshow(data, cmap='viridis', vmin=0, vmax=99)
    plt.title('L_TOA_443 Band')
    plt.colorbar(im, label='TOA Radiance')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
plt.show()

print(f"数据形状: {data.shape}")
print(f"数据范围: {np.nanmin(data):.6f} - {np.nanmax(data):.6f}")
print(f"数据均值: {np.nanmean(data):.6f}") 