#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOCI-2 L1B -> (TOA 反射率优先 / TOA 辐亮度回退) 并重投影到 WGS84 的多波段 GeoTIFF
- 读取 geophysical_data/L_TOA_XXX（单位通常为 W·m^-2·sr^-1·um^-1）
- 若能获取 SZA(太阳天顶角) + Esun(每波段太阳辐照度)，计算 TOA 反射率:
    rho = pi * L * d^2 / (Esun * cos(SZA))
- 仅保留波段索引 3,4,6,8,12（对应 443, 490, 555, 660, 865 nm）
- 从文件读取 latitude/longitude（若是规则经纬网，直接写为 WGS84；若不规则，则做重采样到规则经纬网）
- 无效值一律写入 NaN（GeoTIFF 不设置 nodata），避免 SNAP 拉伸异常
- 目标分辨率：自动根据经纬度栅格的中位步长估计（可按需固定）
"""

import os
import math
import platform
import datetime as dt
import numpy as np

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling, calculate_default_transform
import netCDF4 as nc

# ---- 可选：若安装了 SciPy，可用于更鲁棒的最近邻重采样（处理不规则 lat/lon）----
try:
    from scipy.spatial import cKDTree
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ===================== 路径与配置 =====================
system_type = platform.system()
if system_type == "Windows":
    goci_file = r"D:\Py_Code\SR_Imagery\GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
elif system_type == "Darwin":  # macOS
    goci_file = "/Users/zy/python_code/My_Git/SR_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
else:  # Linux 服务器
    goci_file = "/public/home/zyye/SR/Image_match_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"

print(f"当前系统: {system_type}")
print(f"输入 L1B 文件: {goci_file}")

# 只保留的“波段索引”（1-基）
keep_band_indices = [3, 4, 6, 8, 12]

# 索引到波长（nm）的映射（1-基）
INDEX2NM = {
    1: 380, 2: 412, 3: 443, 4: 490, 5: 510, 6: 555,
    7: 620, 8: 660, 9: 680, 10: 709, 11: 745, 12: 865
}
keep_wavelengths = [INDEX2NM[i] for i in keep_band_indices]

# 反射率可视化裁剪（可设为 None 关闭）
CLIP_MIN, CLIP_MAX = 0.0, 1.5

# Esun 兜底表（若文件无提供，以官方表为准，以下为近似占位）
FALLBACK_ESUN = {
    380: 1670.0, 412: 1870.0, 443: 1874.0, 490: 1959.0, 510: 1920.0,
    555: 1825.0, 620: 1630.0, 660: 1536.0, 680: 1480.0, 709: 1410.0,
    745: 1350.0, 865: 1068.0,
}

# ===================== 工具函数 =====================

def parse_time_from_name(fname: str) -> dt.datetime | None:
    base = os.path.basename(fname)
    parts = base.split("_")
    for i, p in enumerate(parts):
        if len(p) == 8 and p.isdigit():
            if i + 1 < len(parts) and len(parts[i+1]) == 6 and parts[i+1].isdigit():
                try:
                    return dt.datetime.strptime(parts[i] + parts[i+1], "%Y%m%d%H%M%S")
                except Exception:
                    return None
    return None

def earth_sun_distance_au(date: dt.datetime) -> float:
    if date is None:
        return 1.0
    doy = int(date.strftime("%j"))
    return 1.0 - 0.01672 * math.cos(math.radians(0.9856 * (doy - 4)))

def try_get_variable(ds, names):
    # 根级
    for name in names:
        if name in ds.variables:
            return ds.variables[name]
    # 常见组
    for grp in ["geophysical_data", "navigation_data", "geolocation_data", "solar_irradiance", "geometry"]:
        if grp in ds.groups:
            g = ds.groups[grp]
            for name in names:
                if name in g.variables:
                    return g.variables[name]
    return None

def try_get_attr(container, keys, default=None):
    for k in keys:
        if k in getattr(container, "ncattrs", lambda: [])():
            return getattr(container, k)
        if hasattr(container, k):
            return getattr(container, k)
    return default

def read_scaled(var) -> np.ndarray:
    arr_raw = var[:]
    arr = arr_raw.astype(np.float32)
    sf = try_get_attr(var, ["scale_factor"], 1.0) or 1.0
    ao = try_get_attr(var, ["add_offset"], 0.0) or 0.0
    fill = try_get_attr(var, ["_FillValue", "missing_value"], None)
    arr = arr * np.float32(sf) + np.float32(ao)
    if fill is not None:
        arr = np.where(arr_raw == fill, np.nan, arr)
    # optional: valid_min/max
    vmin = try_get_attr(var, ["valid_min"], None)
    vmax = try_get_attr(var, ["valid_max"], None)
    if vmin is not None:
        arr = np.where(arr < float(vmin), np.nan, arr)
    if vmax is not None:
        arr = np.where(arr > float(vmax), np.nan, arr)
    return arr

def get_sza_deg(ds) -> np.ndarray | None:
    var = try_get_variable(ds, ["solar_zenith", "sun_zenith", "SZA", "sza"])
    if var is not None:
        return read_scaled(var)
    var = try_get_variable(ds, ["solar_elevation", "sun_elevation"])
    if var is not None:
        elev = read_scaled(var)
        return 90.0 - elev
    # 标量也尝试
    sza_attr = try_get_attr(ds, ["solar_zenith", "sun_zenith", "SZA"], None)
    if sza_attr is not None:
        # 估尺寸
        H = try_get_attr(ds, ["number_of_lines"], None)
        W = try_get_attr(ds, ["number_of_columns"], None)
        if H is None or W is None:
            # 从任一 L_TOA 获取
            for wl in keep_wavelengths:
                v = try_get_variable(ds, [f"L_TOA_{wl}"])
                if v is not None:
                    H, W = v.shape[-2], v.shape[-1]
                    break
        if H and W:
            return np.full((int(H), int(W)), float(sza_attr), np.float32)
    elev_attr = try_get_attr(ds, ["solar_elevation", "sun_elevation"], None)
    if elev_attr is not None:
        H = try_get_attr(ds, ["number_of_lines"], None)
        W = try_get_attr(ds, ["number_of_columns"], None)
        if H and W:
            return np.full((int(H), int(W)), 90.0 - float(elev_attr), np.float32)
    return None

def get_esun_for_band(ds, wl_nm: int) -> float | None:
    candidates = [
        f"ESUN_{wl_nm}", f"F0_{wl_nm}", f"SOLAR_IRRADIANCE_{wl_nm}",
        f"Esun_{wl_nm}", f"solar_irradiance_{wl_nm}"
    ]
    var = try_get_variable(ds, candidates)
    if var is not None:
        return float(np.nanmean(np.array(var[:]).astype(np.float32)))
    table_names = ["ESUN", "F0", "SOLAR_IRRADIANCE", "solar_irradiance", "esun"]
    for tn in table_names:
        v = try_get_variable(ds, [tn])
        if v is None:
            continue
        wl_var = try_get_variable(ds, ["wavelength", "bands_wavelength", "band_wavelength"])
        if wl_var is not None:
            wls = np.array(wl_var[:]).astype(np.float32).round().astype(int)
            vals = np.array(v[:]).astype(np.float32)
            if vals.ndim == 1 and vals.shape[0] == wls.shape[0]:
                idx = np.where(wls == wl_nm)[0]
                if idx.size > 0:
                    return float(vals[idx[0]])
        if v.ndim == 0:
            return float(v[:])
    val = try_get_attr(ds, [f"ESUN_{wl_nm}", f"F0_{wl_nm}", f"SOLAR_IRRADIANCE_{wl_nm}"], None)
    if val is not None:
        try:
            return float(val)
        except Exception:
            pass
    return None

def estimate_lonlat_affine(lon2d: np.ndarray, lat2d: np.ndarray) -> Affine | None:
    """
    若经纬度为规则网格（行向经度近似线性、列向纬度近似线性），估计仿射变换。
    返回 affine（像素->地理坐标），否则 None。
    """
    H, W = lat2d.shape
    # 取第一行与第一列估步长
    lon_row = lon2d[0, :]
    lat_col = lat2d[:, 0]
    # 要求单调且近似等步长
    def approx_step(x):
        dif = np.diff(x)
        return np.nanmedian(dif), np.nanstd(dif)
    dlon, slon = approx_step(lon_row)
    dlat, slat = approx_step(lat_col)
    if np.isnan(dlon) or np.isnan(dlat):
        return None
    # 容忍一定非规则度
    if slon > abs(dlon) * 0.1 or slat > abs(dlat) * 0.1:
        return None
    # 左上像元中心的经纬度
    lon0 = lon2d[0, 0]
    lat0 = lat2d[0, 0]
    # 构建 affine（注意 y 方向步长为负：上->下纬度通常减小）
    a = dlon
    e = dlat
    # 由像素中心对齐到 GDAL 的左上角需减去半个像素
    transform = Affine.translation(lon0 - a / 2.0, lat0 - e / 2.0) * Affine.scale(a, e)
    return transform

def build_regular_lonlat_grid(lon2d: np.ndarray, lat2d: np.ndarray, res_deg=None):
    """
    根据原始经纬度范围构建规则经纬网（WGS84），默认分辨率取中位步长。
    返回 (dst_lon, dst_lat, transform, width, height)
    """
    lon_min = float(np.nanmin(lon2d))
    lon_max = float(np.nanmax(lon2d))
    lat_min = float(np.nanmin(lat2d))
    lat_max = float(np.nanmax(lat2d))

    if res_deg is None:
        # 用中位步长估计分辨率（取经纬中的较小者）
        med_dlon = float(np.nanmedian(np.abs(np.diff(lon2d, axis=1))))
        med_dlat = float(np.nanmedian(np.abs(np.diff(lat2d, axis=0))))
        res_deg = max(min(med_dlon, med_dlat), 1e-4)  # 至少 1e-4 度 ~ 11 m，别太夸张

    width = max(1, int(round((lon_max - lon_min) / res_deg)))
    height = max(1, int(round((lat_max - lat_min) / res_deg)))

    # 目标 transform（左上角像元左上）
    transform = Affine.translation(lon_min, lat_max) * Affine.scale(res_deg, -res_deg)
    # 目标网格中心经纬
    xs = lon_min + (np.arange(width) + 0.5) * res_deg
    ys = lat_max - (np.arange(height) + 0.5) * res_deg
    dst_lon, dst_lat = np.meshgrid(xs, ys)
    return dst_lon, dst_lat, transform, width, height, res_deg

def resample_by_kdtree(src_lon, src_lat, src_img, dst_lon, dst_lat):
    """
    使用 KDTree 做最近邻配准（需要 SciPy）。src/dst 都是 2D。
    """
    Hs, Ws = src_lon.shape
    valid = np.isfinite(src_lon) & np.isfinite(src_lat) & np.isfinite(src_img)
    pts_src = np.column_stack([src_lon[valid].ravel(), src_lat[valid].ravel()])
    tree = cKDTree(pts_src)
    pts_dst = np.column_stack([dst_lon.ravel(), dst_lat.ravel()])
    dist, idx = tree.query(pts_dst, k=1, distance_upper_bound=1e-2)  # 约 ~1km 容忍（按度粗估）
    out = np.full(dst_lon.size, np.nan, np.float32)
    ok = np.isfinite(dist)
    src_vals = src_img[valid].ravel()
    out[ok] = src_vals[idx[ok]]
    return out.reshape(dst_lon.shape)

# ===================== 主流程 =====================

def main():
    assert os.path.exists(goci_file), f"文件不存在：{goci_file}"
    obs_time = parse_time_from_name(goci_file)
    d_au = earth_sun_distance_au(obs_time)
    print(f"观测时间(UTC)：{obs_time} | 日地距离 d={d_au:.6f} AU")

    with nc.Dataset(goci_file, "r") as ds:
        # 经纬度
        lat_var = try_get_variable(ds, ["latitude", "lat"])
        lon_var = try_get_variable(ds, ["longitude", "lon"])
        if lat_var is None or lon_var is None:
            raise RuntimeError("未在文件中找到 latitude/longitude 变量。")

        lat = read_scaled(lat_var)
        lon = read_scaled(lon_var)
        if lat.shape != lon.shape:
            raise RuntimeError("latitude 与 longitude 形状不一致。")
        H, W = lat.shape

        # 太阳天顶角 & cos(SZA)
        sza = get_sza_deg(ds)
        if sza is not None and sza.shape == lat.shape:
            cos_sza = np.cos(np.deg2rad(sza))
            cos_sza = np.where(cos_sza <= 0, np.nan, cos_sza)
            has_sza = True
        else:
            print("⚠️ 未找到可用于逐像元的 SZA，或尺寸不匹配，将仅输出 L_TOA（辐亮度）。")
            cos_sza = None
            has_sza = False

        # 读取指定波段的 L_TOA，并尽可能计算反射率
        bands_data = []
        band_names = []
        computed_reflectance = True
        for idx in keep_band_indices:
            wl = INDEX2NM[idx]
            vname = f"L_TOA_{wl}"
            var = try_get_variable(ds, [vname])
            if var is None:
                print(f"⏭️ 跳过：未找到 {vname}")
                continue
            L = read_scaled(var)  # float32 + NaN

            Esun = get_esun_for_band(ds, wl)
            if Esun is None:
                Esun = FALLBACK_ESUN.get(wl, None)

            if has_sza and (Esun is not None):
                rho = (math.pi * L * (d_au ** 2)) / (np.float32(Esun) * cos_sza)
                if CLIP_MIN is not None and CLIP_MAX is not None:
                    rho = np.clip(rho, CLIP_MIN, CLIP_MAX)
                bands_data.append(rho.astype(np.float32))
                band_names.append(f"R_TOA_{wl}")
            else:
                computed_reflectance = False
                bands_data.append(L.astype(np.float32))
                band_names.append(f"L_TOA_{wl}")

        if len(bands_data) == 0:
            raise RuntimeError("未找到任何目标波段（3/4/6/8/12）的 L_TOA 数据。")

        # ====== 投影到 WGS84 ======
        dst_crs = CRS.from_epsg(4326)

        # 优先：若经纬度为规则网格，直接写入（无需重采样）
        transform_est = estimate_lonlat_affine(lon, lat)

        base = os.path.splitext(os.path.basename(goci_file))[0]
        kind = "TOA_REFLECTANCE" if computed_reflectance else "TOA_RADIANCE"
        out_tif = os.path.join(os.path.dirname(goci_file),
                               f"{base}_{kind}_WGS84_keep(3-4-6-8-12).tif")

        if transform_est is not None:
            # 规则经纬网：直接写
            profile = {
                "driver": "GTiff",
                "height": H,
                "width": W,
                "count": len(bands_data),
                "dtype": rasterio.float32,
                "crs": dst_crs,
                "transform": transform_est,
                "compress": "LZW",
                "tiled": True,
                "interleave": "band",
                "nodata": None,  # 使用 NaN
            }
            with rasterio.open(out_tif, "w", **profile) as dst:
                for i, arr in enumerate(bands_data, start=1):
                    dst.write(arr, i)
                    dst.set_band_description(i, band_names[i-1])
                dst.update_tags(
                    SOURCE="GOCI-2 L1B",
                    INPUT_FILE=os.path.basename(goci_file),
                    TIME_UTC=str(obs_time),
                    EARTH_SUN_DISTANCE_AU=f"{d_au:.6f}",
                    OUTPUT_KIND=kind,
                    KEEP_INDICES="3,4,6,8,12",
                    KEEP_WAVELENGTHS=",".join(map(str, keep_wavelengths)),
                    FORMULA="rho=pi*L*d^2/(Esun*cos(SZA))" if computed_reflectance else "L_TOA as-is",
                    INVALID_VALUE="NaN",
                    GRID="regular_lonlat_affine"
                )
            print(f"✅ 已输出（规则经纬网）：{out_tif}")
            return

        # 否则：构建规则经纬网并重采样（若安装 SciPy 用 KDTree 最近邻；否则使用简易插值回退）
        dst_lon, dst_lat, dst_transform, dst_W, dst_H, res_deg = build_regular_lonlat_grid(lon, lat, res_deg=None)
        print(f"使用规则经纬网重采样: {dst_W}x{dst_H}, 分辨率 ~ {res_deg:.6f}°")

        profile = {
            "driver": "GTiff",
            "height": dst_H,
            "width": dst_W,
            "count": len(bands_data),
            "dtype": rasterio.float32,
            "crs": dst_crs,
            "transform": dst_transform,
            "compress": "LZW",
            "tiled": True,
            "interleave": "band",
            "nodata": None,
        }

        with rasterio.open(out_tif, "w", **profile) as dst:
            for i, arr in enumerate(bands_data, start=1):
                if SCIPY_OK:
                    # KDTree 最近邻（最稳妥）
                    reproj = resample_by_kdtree(lon, lat, arr, dst_lon, dst_lat)
                else:
                    # 简易回退：将原像素按最近的整列整行索引映射（近似，仅供无 SciPy 时使用）
                    # 计算每列目标经度在原网格第一行的最近列索引；每行目标纬度在第一列的最近行索引
                    # 仅当网格“近似规则”时效果可接受
                    col_idx = np.abs(lon[0, :][None, :] - dst_lon[0, :, None]).argmin(axis=1)
                    row_idx = np.abs(lat[:, 0][:, None] - dst_lat[:, 0][None, :]).argmin(axis=0)
                    # 组合索引
                    reproj = arr[row_idx[:, None], col_idx[None, :]].astype(np.float32)

                    # 把原始 lon/lat 中的 NaN 影响传播到目标（可选）
                    invalid_src = ~np.isfinite(arr) | ~np.isfinite(lon) | ~np.isfinite(lat)
                    invalid_map = invalid_src[row_idx[:, None], col_idx[None, :]]
                    reproj[invalid_map] = np.nan

                dst.write(reproj.astype(np.float32), i)
                dst.set_band_description(i, band_names[i-1])

            dst.update_tags(
                SOURCE="GOCI-2 L1B",
                INPUT_FILE=os.path.basename(goci_file),
                TIME_UTC=str(obs_time),
                EARTH_SUN_DISTANCE_AU=f"{d_au:.6f}",
                OUTPUT_KIND=kind,
                KEEP_INDICES="3,4,6,8,12",
                KEEP_WAVELENGTHS=",".join(map(str, keep_wavelengths)),
                FORMULA="rho=pi*L*d^2/(Esun*cos(SZA))" if computed_reflectance else "L_TOA as-is",
                INVALID_VALUE="NaN",
                GRID=f"regridded_lonlat_res~{res_deg:.6f}deg",
                RESAMPLER="KDTree-NN" if SCIPY_OK else "Nearest-approx"
            )

        print(f"✅ 已输出（重采样到规则 WGS84 网格）：{out_tif}")

if __name__ == "__main__":
    main()
