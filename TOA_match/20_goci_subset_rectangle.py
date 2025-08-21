#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 Landsat footprint（多边形，EPSG:4326）裁剪 GOCI-2 L1B（保留 3、4、6、8、12 波段）
- 步骤：dataset_mask -> shapes -> transform_geom -> bbox 窗口 -> 多边形掩膜
- 不重采样；多边形外像元写为 _FillValue
"""

import os
import numpy as np
from netCDF4 import Dataset
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom
from matplotlib.path import Path

# ============== 配置区域（按需修改） ==============
GOCI_NC     = r"SR_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
LANDSAT_TIF = r"SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_B1.TIF"
OUT_NC      = r"./goci_subset_5bands_polygon.nc"

# 1-based 波段索引（GOCI-2 L1B 变量名映射见下）
KEEP_INDICES = [3, 4, 6, 8, 12]

# 可选：footprint 顶点抽稀阈值（度），用于加速多边形判定；设为 None 不抽稀
SIMPLIFY_TOL_DEG = 0.00030  # ~33 m；None 表示不抽稀
# 是否生成一个简单的质检图（英文），叠加 footprint 与掩膜轮廓
MAKE_QC_PLOT = True
# ===============================================

# GOCI 变量名映射
BAND_INDEX_TO_NAME = {
    1: "L_TOA_380",  2: "L_TOA_412",  3: "L_TOA_443",  4: "L_TOA_490",
    5: "L_TOA_510",  6: "L_TOA_555",  7: "L_TOA_620",  8: "L_TOA_660",
    9: "L_TOA_680", 10: "L_TOA_709", 11: "L_TOA_745", 12: "L_TOA_865",
}

def extract_footprint_polygons_wgs84(tif_path: str):
    """
    从 Landsat 影像提取 footprint 外多边形（列表，每个为闭合的 [lon,lat] 数组）
    需要影像带有 nodata/alpha；否则得到的将近似为矩形。
    """
    with rasterio.open(tif_path) as src:
        mask = src.dataset_mask()  # 0=无效, 255=有效
        geoms = [
            transform_geom(src.crs, "EPSG:4326", geom, precision=10)
            for geom, val in shapes(mask, mask=(mask > 0), transform=src.transform)
            if val
        ]

    if len(geoms) == 0:
        raise RuntimeError("未从掩膜中提取到 footprint；请确认 TIF 带有 nodata/alpha。")

    polys = []
    for g in geoms:
        ring = np.asarray(g["coordinates"][0], dtype=float)  # exterior
        # 闭合
        if not np.allclose(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[0]])
        # 可选：抽稀，减少顶点数，加速 contains_points
        if SIMPLIFY_TOL_DEG is not None and SIMPLIFY_TOL_DEG > 0:
            ring = _decimate_ring(ring, SIMPLIFY_TOL_DEG)
        polys.append(ring)
    return polys

def _decimate_ring(ring_lonlat: np.ndarray, tol_deg: float) -> np.ndarray:
    """简单抽稀：保留相邻点经纬差超过阈值的点，保持闭合。"""
    lon, lat = ring_lonlat[:, 0], ring_lonlat[:, 1]
    keep = [0]
    for i in range(1, len(lon) - 1):
        if max(abs(lon[i] - lon[keep[-1]]), abs(lat[i] - lat[keep[-1]])) >= tol_deg:
            keep.append(i)
    keep.append(len(lon) - 1)  # 闭合点
    return ring_lonlat[keep]

def compute_window_from_bbox(lat, lon, minx, miny, maxx, maxy):
    """在 GOCI 经纬度网格上，用 bbox 得到最小行列窗口 (r0,r1,c0,c1)。"""
    # 若经度为 0~360，统一到 -180~180
    if lon.max() > 180:
        lon = (lon + 180.0) % 360.0 - 180.0
    inside = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
    if not np.any(inside):
        raise RuntimeError("footprint 的 bbox 与 GOCI 网格不相交。")
    r_idx = np.where(np.any(inside, axis=1))[0]
    c_idx = np.where(np.any(inside, axis=0))[0]
    return int(r_idx.min()), int(r_idx.max()) + 1, int(c_idx.min()), int(c_idx.max()) + 1

def mask_window_by_polygons(lon_win, lat_win, polys):
    """对窗口内经纬度网格做多边形掩膜（任意多边形并集）。"""
    pts = np.column_stack([lon_win.ravel(), lat_win.ravel()])
    mask_all = np.zeros(pts.shape[0], dtype=bool)
    for ring in polys:
        mask_all |= Path(ring).contains_points(pts)
    return mask_all.reshape(lon_win.shape)

def main():
    # —— 基本检查 ——
    if not os.path.exists(GOCI_NC):     raise FileNotFoundError(GOCI_NC)
    if not os.path.exists(LANDSAT_TIF): raise FileNotFoundError(LANDSAT_TIF)
    for k in KEEP_INDICES:
        if k not in BAND_INDEX_TO_NAME:
            raise ValueError(f"非法波段索引：{k}")

    # 1) 提取 Landsat footprint（WGS84）
    polys = extract_footprint_polygons_wgs84(LANDSAT_TIF)
    all_xy = np.vstack(polys)
    minx, maxx = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    miny, maxy = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
    print(f"[INFO] footprint 多边形数：{len(polys)}")
    print(f"[INFO] footprint bbox: lon[{minx:.6f},{maxx:.6f}], lat[{miny:.6f},{maxy:.6f}]")

    # 2) 读取 GOCI lat/lon
    with Dataset(GOCI_NC, "r") as ds:
        lat = ds["navigation_data"]["latitude"][:]
        lon = ds["navigation_data"]["longitude"][:]
        if lat.shape != lon.shape:
            raise RuntimeError(f"lat/lon 形状不一致：{lat.shape} vs {lon.shape}")

        # 3) bbox 窗口 + 多边形掩膜
        r0, r1, c0, c1 = compute_window_from_bbox(lat, lon, minx, miny, maxx, maxy)
        lat_sub = lat[r0:r1, c0:c1].astype(np.float32, copy=False)
        lon_sub = lon[r0:r1, c0:c1].astype(np.float32, copy=False)
        mask_sub = mask_window_by_polygons(lon_sub, lat_sub, polys)

        print(f"[INFO] 裁剪窗口：rows[{r0}:{r1}] cols[{c0}:{c1}] -> 形状={(r1-r0, c1-c0)}")
        print(f"[INFO] 窗口内多边形覆盖率：{mask_sub.mean()*100:.2f}%")

        # 4) 裁剪波段并套掩膜
        geo = ds["geophysical_data"]
        band_vars, band_attrs, fillvals = {}, {}, {}
        for idx in KEEP_INDICES:
            vname = BAND_INDEX_TO_NAME[idx]
            if vname not in geo.variables:
                raise KeyError(f"GOCI 缺少变量 geophysical_data/{vname}")
            var = geo[vname]
            attrs = {a: getattr(var, a) for a in var.ncattrs()}
            fv = attrs.get("_FillValue", None)

            win = var[r0:r1, c0:c1].astype(np.float32, copy=False)
            if fv is not None:
                win = np.where(win == fv, np.nan, win)
            win = np.where(mask_sub, win, np.nan)  # 多边形外设为 NaN

            band_vars[vname] = win
            band_attrs[vname] = attrs
            fillvals[vname] = fv

    # 5) 写出 NetCDF
    os.makedirs(os.path.dirname(os.path.abspath(OUT_NC)), exist_ok=True)
    with Dataset(OUT_NC, "w", format="NETCDF4") as dst:
        ny, nx = lat_sub.shape
        dst.createDimension("y", ny)
        dst.createDimension("x", nx)

        dst.source = "Cropped from GOCI-2 L1B by Landsat footprint polygon"
        dst.history = "created by goci_subset_by_landsat_footprint (no resampling)"
        dst.note = "Windowed by footprint bbox; out-of-polygon pixels set to _FillValue."
        dst.keep_band_indices = ",".join(map(str, KEEP_INDICES))
        dst.keep_band_names = ",".join([BAND_INDEX_TO_NAME[i] for i in KEEP_INDICES])
        dst.footprint_bbox_lon = f"{minx:.10f},{maxx:.10f}"
        dst.footprint_bbox_lat = f"{miny:.10f},{maxy:.10f}"

        # 保存第一个多边形顶点（用于复核；如需全部可自行扩展）
        fp0 = polys[0]
        dst.footprint_vertices_note = "first polygon exterior ring (EPSG:4326)"
        dst.footprint_vertices_lon = ",".join([f"{x:.10f}" for x in fp0[:, 0].tolist()])
        dst.footprint_vertices_lat = ",".join([f"{y:.10f}" for y in fp0[:, 1].tolist()])

        v_lat = dst.createVariable("latitude", "f4", ("y", "x"), zlib=True, complevel=4)
        v_lon = dst.createVariable("longitude", "f4", ("y", "x"), zlib=True, complevel=4)
        v_lat.long_name, v_lat.units = "latitude", "degrees_north"
        v_lon.long_name, v_lon.units = "longitude", "degrees_east"
        v_lat[:] = lat_sub
        v_lon[:] = lon_sub

        # 掩膜变量
        v_mask = dst.createVariable("in_polygon_mask", "i1", ("y", "x"), zlib=True, complevel=4)
        v_mask.long_name = "mask of pixels inside Landsat footprint polygon (1=True, 0=False)"
        v_mask[:] = mask_sub.astype(np.int8)

        # 各波段输出（把 NaN 恢复为 _FillValue）
        for vname, data_sub in band_vars.items():
            fv = fillvals[vname] if fillvals[vname] is not None else -999.0
            out = np.nan_to_num(data_sub, nan=fv).astype("f4", copy=False)
            v = dst.createVariable(vname, "f4", ("y", "x"), zlib=True, complevel=4, fill_value=fv)
            # 恢复原属性（不覆盖 _FillValue）
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

    print(f"[OK] 已输出基于 footprint 多边形裁剪的 NC：{OUT_NC}")

    # ——（可选）快速质检图——
    if MAKE_QC_PLOT:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.title("QC: footprint vs. mask (window)")
        plt.contour(mask_sub.astype(int), levels=[0.5], linewidths=2, colors="r", label="mask")
        for ring in polys:
            plt.plot(ring[:, 0], ring[:, 1], "-", lw=1, label="footprint (lon/lat)")
        plt.xlabel("Longitude (°)")
        plt.ylabel("Latitude (°)")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()