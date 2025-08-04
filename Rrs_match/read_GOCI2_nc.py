
import numpy as np
import netCDF4 as nc

# 文件路径
filename = "H:\GK2B_GOCI2_L2_20250323_021530_LA_S009_AC.nc"

# 对应波长
wavelengths = np.array([380, 412, 443, 490, 510, 555, 620, 660, 680, 709, 745, 865])
rrs_variables = [f"Rrs_{wavelength}" for wavelength in wavelengths]

# 初始化3D数组 [rows, cols, bands]
rrs_data = np.zeros((2780, 2780, 12), dtype=np.float32)

# 读取数据
print('读取Rrs数据...')
with nc.Dataset(filename, 'r') as dataset:
    rrs_group = dataset.groups['geophysical_data'].groups['Rrs']
    
    for i, var_name in enumerate(rrs_variables):
        print(f'读取 {var_name}...')
        rrs_data[:, :, i] = rrs_group.variables[var_name][:]

print(f'读取完成！数据形状: {rrs_data.shape}')