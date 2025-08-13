#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOCI-2 L1B 排查脚本：
- 列出疑似 SZA/Sun Elevation 变量（含路径、shape、单位、scale_factor、_FillValue 等）
- 列出 latitude/longitude（shape）
- 列出 L_TOA_443/490/555/660/865（shape）
- 检查 SZA 是否与经纬/影像形状匹配；若不匹配，说明不能逐像元计算反射率
- 检查 Esun（每波段）是否可获取
- 计算一个 cos(SZA) 的统计（若 SZA 有二维）
"""

import os
import platform
import numpy as np
import netCDF4 as nc

# ===== 路径自动选择 =====
system_type = platform.system()
if system_type == "Windows":
    goci_file = r"D:\Py_Code\SR_Imagery\GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
elif system_type == "Darwin":
    goci_file = "/Users/zy/python_code/My_Git/SR_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
else:
    goci_file = "/public/home/zyye/SR/Image_match_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"

print(f"当前系统: {system_type}")
print(f"输入 L1B 文件: {goci_file}")
assert os.path.exists(goci_file), "❌ 文件不存在"

# 目标波段
keep_wls = [443, 490, 555, 660, 865]

# 候选变量名
SZA_CANDIDATES = ["solar_zenith", "sun_zenith", "SZA", "sza",
                  "solar_zenith_angle", "sun_zenith_angle"]
SELEV_CANDIDATES = ["solar_elevation", "sun_elevation"]
LAT_CANDIDATES = ["latitude", "lat"]
LON_CANDIDATES = ["longitude", "lon"]
ESUN_CANDIDATES_TABLE = ["ESUN", "F0", "SOLAR_IRRADIANCE", "solar_irradiance", "esun"]

GROUP_GUESS = ["", "geophysical_data", "navigation_data",
               "geolocation_data", "solar_irradiance", "sensor", "geometry"]

def list_hits(ds, names):
    hits = []
    # 根级
    for n in names:
        if n in ds.variables:
            hits.append(("/", n, ds.variables[n]))
    # 常见分组
    for grp_name in GROUP_GUESS:
        if not grp_name:
            continue
        if grp_name in ds.groups:
            g = ds.groups[grp_name]
            for n in names:
                if n in g.variables:
                    hits.append(("/" + grp_name, n, g.variables[n]))
    return hits

def show_var_info(prefix, path, name, var):
    shp = tuple(var.shape)
    dtype = str(var.dtype)
    attrs = {k: getattr(var, k) for k in var.ncattrs()}
    sf = attrs.get("scale_factor", None)
    ao = attrs.get("add_offset", None)
    fv = attrs.get("_FillValue", attrs.get("missing_value", None))
    units = attrs.get("units", None)
    print(f"{prefix} {path}/{name}  shape={shp}  dtype={dtype}  units={units}  "
          f"scale_factor={sf}  add_offset={ao}  _FillValue/missing={fv}")

def try_read_2d(var):
    arr = var[:]
    if arr.ndim == 2:
        return arr
    return None

def find_esun_for_wl(ds, wl):
    # 先找显式命名 ESUN_443 这类
    for grp_name in [""] + GROUP_GUESS:
        g = ds if grp_name=="" else ds.groups.get(grp_name, None)
        if g is None:
            continue
        for key in [f"ESUN_{wl}", f"F0_{wl}", f"SOLAR_IRRADIANCE_{wl}",
                    f"Esun_{wl}", f"solar_irradiance_{wl}"]:
            if key in (g.variables if hasattr(g, "variables") else {}):
                v = g.variables[key][:]
                try:
                    return float(np.array(v).astype(np.float32).mean())
                except Exception:
                    pass
    # 再找表
    for tn in ESUN_CANDIDATES_TABLE:
        for grp_name in [""] + GROUP_GUESS:
            g = ds if grp_name=="" else ds.groups.get(grp_name, None)
            if g is None:
                continue
            if tn in g.variables:
                vv = g.variables[tn][:]
                # 尝试配套的 wavelength
                wl_var = None
                for wname in ["wavelength", "bands_wavelength", "band_wavelength"]:
                    if wname in g.variables:
                        wl_var = g.variables[wname][:]
                        break
                if wl_var is not None and vv.ndim == 1 and wl_var.shape[0] == vv.shape[0]:
                    wls = np.array(wl_var).round().astype(int)
                    idx = np.where(wls == wl)[0]
                    if idx.size > 0:
                        return float(np.array(vv[idx[0]]).astype(np.float32))
                # 标量表（极少见）
                if vv.ndim == 0:
                    return float(vv[:])
    return None

with nc.Dataset(goci_file) as ds:
    print("\n=== 1) 经纬度变量 ===")
    lat_hits = list_hits(ds, LAT_CANDIDATES)
    lon_hits = list_hits(ds, LON_CANDIDATES)
    if not lat_hits or not lon_hits:
        print("❌ 未找到 latitude/longitude 变量")
    else:
        for p, n, v in lat_hits:
            show_var_info("[LAT]", p, n, v)
        for p, n, v in lon_hits:
            show_var_info("[LON]", p, n, v)

    # 取第一个 lat/lon 变量，读出形状
    lat_var = lat_hits[0][2] if lat_hits else None
    lon_var = lon_hits[0][2] if lon_hits else None
    lat2d = try_read_2d(lat_var) if lat_var else None
    lon2d = try_read_2d(lon_var) if lon_var else None

    if lat2d is not None and lon2d is not None:
        print(f"→ lat/lon 2D 形状: {lat2d.shape}/{lon2d.shape}  相等? {lat2d.shape==lon2d.shape}")
    else:
        print("⚠️ lat/lon 不是二维数组（可能是 1D 或更复杂维度），主脚本默认要求 2D。")

    print("\n=== 2) L_TOA 目标波段（443/490/555/660/865） ===")
    L_shapes = {}
    for wl in keep_wls:
        # 在常见组里搜 L_TOA_wl
        hits = list_hits(ds, [f"L_TOA_{wl}"])
        if not hits:
            print(f"[L_TOA] 未找到 L_TOA_{wl}")
            continue
        for p, n, v in hits:
            show_var_info("[L_TOA]", p, n, v)
            if v.ndim >= 2:
                L_shapes[wl] = tuple(v.shape[-2:])
            else:
                L_shapes[wl] = tuple(v.shape)

    print("\n=== 3) 太阳角度变量（SZA/太阳高度） ===")
    sza_hits = list_hits(ds, SZA_CANDIDATES)
    selev_hits = list_hits(ds, SELEV_CANDIDATES)
    if not sza_hits and not selev_hits:
        print("❌ 未找到 SZA 或太阳高度角变量（名称可能不同）")
    else:
        for p, n, v in sza_hits:
            show_var_info("[SZA]", p, n, v)
        for p, n, v in selev_hits:
            show_var_info("[SELEV]", p, n, v)

    # 取一个最可能可用的角度场，并检查形状
    sza_2d = None
    label = None
    if sza_hits:
        # 优先 SZA
        for p, n, v in sza_hits:
            if v.ndim == 2:
                sza_2d = v[:].astype(np.float32)
                label = f"{p}/{n}"
                break
    if sza_2d is None and selev_hits:
        # 次选：太阳高度，转 SZA
        for p, n, v in selev_hits:
            if v.ndim == 2:
                elev = v[:].astype(np.float32)
                sza_2d = 90.0 - elev
                label = f"{p}/{n} (converted)"
                break

    if sza_2d is not None:
        print(f"→ 选用角度字段: {label}  shape={sza_2d.shape}")
        if lat2d is not None:
            print(f"→ 与 lat/lon 匹配? {sza_2d.shape == lat2d.shape}")
        # 简要统计
        finite = np.isfinite(sza_2d)
        if finite.any():
            mn, mx = float(np.nanmin(sza_2d)), float(np.nanmax(sza_2d))
            print(f"→ SZA 范围: min={mn:.3f}, max={mx:.3f}")
            cos_sza = np.cos(np.deg2rad(sza_2d))
            valid_cos = np.isfinite(cos_sza) & (cos_sza > 0)
            print(f"→ cos(SZA) > 0 的像元数: {int(valid_cos.sum())} / {int(finite.sum())}")
        else:
            print("⚠️ SZA 全为非有限值（NaN/Inf），请检查 scale_factor/_FillValue。")
    else:
        print("⚠️ 未找到二维 SZA/SELEV，可用情况下主脚本会退化为辐亮度。")

    print("\n=== 4) Esun 查找（每波段） ===")
    for wl in keep_wls:
        val = None
        # 先找显式命名/表
        val = None or find_esun_for_wl(ds, wl)
        if val is not None:
            print(f"[Esun] {wl} nm: {val} (W m^-2 um^-1)")
        else:
            print(f"[Esun] {wl} nm: 未在文件中找到（主脚本会用兜底表或仅输出辐亮度）")

    print("\n=== 5) 形状匹配结论 ===")
    if lat2d is None or lon2d is None:
        print("结论：经纬度不是二维，或缺失 → 主脚本无法确定逐像元地理/角度匹配。")
    else:
        ok = True
        # 要求 L_TOA 的最后两个维度与 lat2d 相同
        for wl, shp in L_shapes.items():
            if shp != lat2d.shape:
                print(f"❌ L_TOA_{wl} 的像元尺寸 {shp} 与 lat/lon {lat2d.shape} 不一致")
                ok = False
        if sza_2d is None:
            print("⚠️ 无二维 SZA → 只能输出辐亮度")
        else:
            if sza_2d.shape != lat2d.shape:
                print(f"❌ SZA 形状 {sza_2d.shape} 与 lat/lon {lat2d.shape} 不一致")
                ok = False
        if ok:
            print("✅ 形状层面没有发现阻碍逐像元计算反射率的问题。")
        else:
            print("➡️ 请根据上面的不一致项调整读取或做插值/重采样。")
