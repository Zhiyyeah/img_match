#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按 Landsat 影像范围裁剪 GOCI-2 L1B 的辐亮度数据（保留 3、4、6、8、12 波段）
- 仅裁剪空间范围（不重采样），输出裁剪后的 NC（含 lat/lon 与 L_TOA_*）
"""

import os
import numpy as np
from netCDF4 import Dataset
import rasterio
from rasterio.warp import transform_bounds

# 1-based band index -> GOCI-2 L1B 变量名
BAND_INDEX_TO_NAME = {
    1: "L_TOA_380",
    2: "L_TOA_412",
    3: "L_TOA_443",
    4: "L_TOA_490",
    5: "L_TOA_510",
    6: "L_TOA_555",
    7: "L_TOA_620",
    8: "L_TOA_660",
    9: "L_TOA_680",
    10: "L_TOA_709",
    11: "L_TOA_745",
    12: "L_TOA_865",
}

def read_ref_bounds_wgs84(ref_tif):
    """读取参考影像范围并转换到 WGS84（经纬度）"""
    with rasterio.open(ref_tif) as ds:
        b = ds.bounds  # (left, bottom, right, top) in ds.crs
        src_crs = ds.crs
    minx, miny, maxx, maxy = transform_bounds(
        src_crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21
    )
    return float(minx), float(miny), float(maxx), float(maxy)

def compute_slice_from_bbox(lat, lon, minx, miny, maxx, maxy):
    """
    在 GOCI 的经纬度网格上，根据 bbox 计算最小矩形切片范围
    返回 (r0, r1, c0, c1) —— r1/c1 为开区间
    """
    inside = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
    if not np.any(inside):
        raise RuntimeError("裁剪失败：参考范围与 GOCI 网格不相交。请检查坐标与范围。")
    rows = np.any(inside, axis=1)
    cols = np.any(inside, axis=0)
    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    r0, r1 = int(r_idx.min()), int(r_idx.max()) + 1
    c0, c1 = int(c_idx.min()), int(c_idx.max()) + 1
    return r0, r1, c0, c1

def main():
    # ======== 只需改这几行参数 ========
    goci_nc = r"SR_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"  # GOCI-2 L1B 输入
    ref_tif = r"SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif"  # Landsat 参考范围
    out_nc  = r"./goci_subset_5bands.nc"  # 输出裁剪后的 NC
    keep_indices = [3, 4, 6, 8, 12]       # 必须是 1-based 索引
    # =================================

    # 基本检查
    if not os.path.exists(goci_nc):
        raise FileNotFoundError(f"GOCI 文件不存在：{goci_nc}")
    if not os.path.exists(ref_tif):
        raise FileNotFoundError(f"参考影像不存在：{ref_tif}")
    for k in keep_indices:
        if k not in BAND_INDEX_TO_NAME:
            raise ValueError(f"非法波段索引：{k}")

    # 1) 参考影像的 WGS84 外接范围
    minx, miny, maxx, maxy = read_ref_bounds_wgs84(ref_tif)
    print(f"[INFO] 参考影像 WGS84 范围：lon[{minx:.6f},{maxx:.6f}], lat[{miny:.6f},{maxy:.6f}]")

    # 2) 读取 GOCI 的 lat/lon
    with Dataset(goci_nc, "r") as src_nc:
        lat = src_nc["navigation_data"]["latitude"][:]
        lon = src_nc["navigation_data"]["longitude"][:]
        if lat.shape != lon.shape:
            raise RuntimeError(f"lat/lon 形状不一致：{lat.shape} vs {lon.shape}")

        # 3) 计算切片范围
        r0, r1, c0, c1 = compute_slice_from_bbox(lat, lon, minx, miny, maxx, maxy)
        print(f"[INFO] 裁剪行列范围：rows[{r0}:{r1}] cols[{c0}:{c1}] -> 形状 {(r1-r0, c1-c0)}")

        # 4) 裁剪所需波段
        band_vars = {}
        band_attrs = {}
        fillvals = {}

        geo_grp = src_nc["geophysical_data"]
        for idx in keep_indices:
            vname = BAND_INDEX_TO_NAME[idx]
            if vname not in geo_grp.variables:
                raise KeyError(f"GOCI 文件缺少变量 geophysical_data/{vname}")
            var = geo_grp[vname]
            attrs = {att: getattr(var, att) for att in var.ncattrs()}
            fv = attrs.get("_FillValue", None)
            data_sub = var[r0:r1, c0:c1]
            if fv is not None:
                data_sub = np.where(data_sub == fv, np.nan, data_sub)
            band_vars[vname] = data_sub.astype(np.float32, copy=False)
            band_attrs[vname] = attrs
            fillvals[vname] = fv

        lat_sub = lat[r0:r1, c0:c1].astype(np.float32, copy=False)
        lon_sub = lon[r0:r1, c0:c1].astype(np.float32, copy=False)

    # 5) 写出裁剪 NC
    os.makedirs(os.path.dirname(os.path.abspath(out_nc)), exist_ok=True)
    with Dataset(out_nc, "w", format="NETCDF4") as dst:
        ny, nx = lat_sub.shape
        dst.createDimension("y", ny)
        dst.createDimension("x", nx)

        # 全局属性（可扩展）
        dst.source = "Cropped from GOCI-2 L1B by Landsat extent"
        dst.history = "created by goci_subset_by_landsat_extent (no-arg version)"
        dst.note = "No reprojection/resampling. Rectangle crop only."
        dst.keep_band_indices = ",".join(map(str, keep_indices))
        dst.keep_band_names = ",".join([BAND_INDEX_TO_NAME[i] for i in keep_indices])

        # lat/lon
        v_lat = dst.createVariable("latitude", "f4", ("y", "x"), zlib=True, complevel=4)
        v_lon = dst.createVariable("longitude", "f4", ("y", "x"), zlib=True, complevel=4)
        v_lat.long_name = "latitude"; v_lat.units = "degrees_north"
        v_lon.long_name = "longitude"; v_lon.units = "degrees_east"
        v_lat[:] = lat_sub
        v_lon[:] = lon_sub

        # 各波段
        for vname, data_sub in band_vars.items():
            fv = fillvals[vname]
            if fv is None:
                fv = -999.0
            out = np.nan_to_num(data_sub, nan=fv).astype("f4", copy=False)

            v = dst.createVariable(vname, "f4", ("y", "x"), zlib=True, complevel=4, fill_value=fv)
            # 还原原属性（避免覆盖 _FillValue）
            for k, val in band_attrs[vname].items():
                if k == "_FillValue":
                    continue
                try:
                    setattr(v, k, val)
                except Exception:
                    pass
            if not hasattr(v, "units"):
                v.units = "W m-2 sr-1 um-1"
            v[:] = out

    print(f"[OK] 已输出裁剪后的 NC：{out_nc}")
    print("[TIP] 后续可用 pyresample 将此裁剪结果重采样到 Landsat 网格做严格配准。")

if __name__ == "__main__":
    main()
