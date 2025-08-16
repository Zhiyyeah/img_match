#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract selected GOCI-2 L1B bands into a new NetCDF (simplified, fixed groups)
- 输入结构固定:
    geophysical_data/L_TOA_380 ... L_TOA_865
    navigation_data/latitude, navigation_data/longitude
- 提取波段: [3,4,6,8,12] -> [443, 490, 555, 660, 865]
- 输出: float32, zlib 压缩, 含 2D latitude/longitude 与简单 WGS84 grid mapping
"""

import os
import platform
import numpy as np
from netCDF4 import Dataset

# -------- 路径设置（按系统自动切换，可自行改为命令行参数）--------
system = platform.system()
if system == "Windows":
    in_nc  = r"D:\Py_Code\SR_Imagery\GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
    out_nc = r"D:\Py_Code\SR_Imagery\GK2_GOCI2_L1B_subset_b3-4-6-8-12.nc"
elif system == "Darwin":
    in_nc  = "/Users/zy/python_code/My_Git/SR_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
    out_nc = "/Users/zy/python_code/My_Git/SR_Imagery/GK2_GOCI2_L1B_subset_b3-4-6-8-12.nc"
else:
    in_nc  = "/public/home/zyye/SR/Image_match_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
    out_nc = "/public/home/zyye/SR/Image_match_Imagery/GK2_GOCI2_L1B_subset_b3-4-6-8-12.nc"

# -------- 固定映射与设置 --------
idx_to_wvl = {
    1:"380", 2:"412", 3:"443", 4:"490", 5:"510", 6:"555",
    7:"620", 8:"660", 9:"680", 10:"709", 11:"745", 12:"865"
}
select_idx = [3, 4, 6, 8, 12]
select_wvl = [idx_to_wvl[i] for i in select_idx]
FILL = -999.0

def main():
    if not os.path.exists(in_nc):
        raise FileNotFoundError(f"输入文件不存在: {in_nc}")

    with Dataset(in_nc, "r") as src:
        g = src.groups["geophysical_data"]
        n = src.groups["navigation_data"]

        # 读取 2D 坐标（若为 1D 则网格化）
        lat = np.array(n.variables["latitude"][:])
        lon = np.array(n.variables["longitude"][:])
        if lat.ndim == 1 and lon.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
        elif lat.ndim == 2 and lon.ndim == 2:
            lat2d, lon2d = lat, lon
        else:
            # 其他组合尽量容错
            if lat.ndim == 2 and lon.ndim == 1:
                lon2d = np.tile(lon[None, :], (lat.shape[0], 1))
                lat2d = lat
            elif lat.ndim == 1 and lon.ndim == 2:
                lat2d = np.tile(lat[:, None], (1, lon.shape[1]))
                lon2d = lon
            else:
                raise RuntimeError("无法统一 latitude/longitude 为 2D。")

        ny, nx = lat2d.shape

        # 读取选定波段数据
        data_dict = {}
        for w in select_wvl:
            varname = f"L_TOA_{w}"
            if varname not in g.variables:
                raise KeyError(f"缺少变量 {varname} 于 geophysical_data 组")
            arr = np.array(g.variables[varname][:], dtype=np.float32)
            if arr.shape != (ny, nx):
                raise RuntimeError(f"{varname} 尺寸不匹配: {arr.shape} vs {(ny, nx)}")
            data_dict[w] = arr

        # 写出新 NC
        if os.path.exists(out_nc):
            os.remove(out_nc)
        with Dataset(out_nc, "w") as dst:
            # 维度
            dst.createDimension("y", ny)
            dst.createDimension("x", nx)

            # 坐标变量（2D）
            vlat = dst.createVariable("latitude", "f4", ("y","x"), zlib=True, complevel=4, fill_value=FILL)
            vlon = dst.createVariable("longitude","f4", ("y","x"), zlib=True, complevel=4, fill_value=FILL)
            vlat[:] = lat2d.astype(np.float32)
            vlon[:] = lon2d.astype(np.float32)
            vlat.long_name = "latitude";  vlat.units = "degrees_north"
            vlon.long_name = "longitude"; vlon.units = "degrees_east"

            # 简单 CRS / grid mapping
            crs = dst.createVariable("crs", "i4")
            crs.grid_mapping_name = "latitude_longitude"
            crs.longitude_of_prime_meridian = 0.0
            crs.semi_major_axis = 6378137.0
            crs.inverse_flattening = 298.257223563
            crs.datum = "WGS84"

            # 波段变量
            for w in select_wvl:
                varname = f"L_TOA_{w}"
                src_var = g.variables[varname]
                v = dst.createVariable(varname, "f4", ("y","x"), zlib=True, complevel=4, fill_value=FILL)
                v[:] = data_dict[w]
                v.coordinates = "latitude longitude"
                v.grid_mapping = "crs"
                # 复制常见属性（若存在）
                for key in ("long_name","standard_name","units","valid_min","valid_max","scale_factor","add_offset"):
                    if hasattr(src_var, key):
                        setattr(v, key, getattr(src_var, key))
                if not hasattr(v, "units"):
                    v.units = "W m-2 sr-1 um-1"

            # 复制少量全局属性（可按需扩充）
            for key in ("title","summary","product_name","product_version","history","naming_authority","id"):
                if key in src.ncattrs():
                    dst.setncattr(key, src.getncattr(key))
            dst.setncattr("source_file", os.path.basename(in_nc))
            dst.setncattr("note", "Subset bands [443,490,555,660,865] with 2D lat/lon")

    print(f"✅ 已保存: {out_nc}")

if __name__ == "__main__":
    main()
