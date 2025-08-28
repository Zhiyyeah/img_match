#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30_batch_calibrate_and_clip_nocli.py
------------------------------------------------------------------
用途：无命令行参数的批处理脚本。在 `main()` 中集中配置参数，
读取 01_discover_and_pair.py 生成的配对清单 `pairs.csv`，对每一对
Landsat C2 L1 与 GOCI-2 L1B 进行如下处理：

处理流水线：
  1) 对 Landsat C2 L1 各波段进行辐射定标，输出多波段 TOA 辐亮度 GeoTIFF；
  2) 根据 Landsat 有效像元的 footprint（矢量多边形，WGS84）裁剪 GOCI L1B（弯曲网格），
     将 footprint 内像元保留，外部像元填充 _FillValue，输出 NetCDF；
  3) 生成示意图：GOCI footprint 掩膜与 Landsat/GOCI 的直方图对比（不做重采样）。

输出示例：
  - `{landsat_dir}/{landsat_scene}_TOA_RAD_B1-2-3-4-5.tif`
  - `{out_root}/{landsat_scene}/{goci_basename}_subset_footprint.nc`
  - `{out_root}/{landsat_scene}/{goci_basename}_compare.png`
  - 批处理结果摘要 CSV（路径在 `main()` 中指定，默认为 `batch_outputs/batch_results.csv`）

注意：
  - 本脚本只进行辐亮度（Radiance）定标，不计算反射率；
  - GOCI 为弯曲网格，使用经纬度点在多边形内测试实现裁剪；
  - 对于 bbox 裁剪也提供了 `clip_goci_bbox()`，但主流程采用 footprint 多边形裁剪，
    能更准确贴合实际覆盖范围，避免外接矩形带来的冗余区域。
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from netCDF4 import Dataset
import re

from rasterio.features import shapes
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
from pyproj import Transformer

# ---------------------------- 可配置常量（函数级别） ----------------------------
LS_BANDS_DEFAULT = [1, 2, 3, 4, 5]
NODATA_VAL = -9999.0
GOCI_BAND_NAMES_DEFAULT = ["L_TOA_443", "L_TOA_490", "L_TOA_555", "L_TOA_660", "L_TOA_865"]
# -----------------------------------------------------------------------------


# ===== 工具函数 =====

def find_mtl_file(landsat_dir: Path) -> Path:
    """在给定的 Landsat 场景目录中查找 MTL 元数据文件。

    参数:
      - landsat_dir: 场景根目录（包含若干 `*_B*.TIF` 与一个 `*_MTL.txt`）。

    返回:
      - 匹配到的 MTL 文件路径（若有多个，取文件名最短的那个）。

    异常:
      - FileNotFoundError: 目录内未找到任何 `*_MTL.txt`。
    """
    # 某些数据目录可能同时包含角度等辅助 MTL，这里按“文件名长度最短”
    # 的启发式来选择主 MTL。也可以按时间或特定前缀进一步筛选。
    cands = list(landsat_dir.glob("*_MTL.txt"))
    if not cands:
        raise FileNotFoundError(f"未找到 MTL 文件: {landsat_dir}")
    return sorted(cands, key=lambda p: len(p.name))[0]


def read_mtl(path: Path) -> Dict[str, str]:
    """读取简单的 MTL 文本为字典。

    说明:
      - 该解析器按 `key = value` 形式逐行解析，不处理嵌套/块结构；
      - 仅用于读取辐射定标相关键值（如 `RADIANCE_MULT_BAND_x`、`RADIANCE_ADD_BAND_x`）；
      - 值两侧可能有引号，已在解析时去除；若某些键缺失，后续访问将触发 KeyError。
    """
    mtl = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if '=' not in line:
                continue
            k, v = line.strip().split('=', 1)
            k = k.strip()
            v = v.strip().strip('"')
            mtl[k] = v
    return mtl


def get_ls_rad_coeffs(mtl: Dict[str, str], band: int) -> Tuple[float, float]:
    """从 MTL 字典中提取指定波段的辐射定标系数。

    返回:
      - 二元组 `(M, A)`，用于 `L = M * DN + A`。

    异常:
      - KeyError: 当 MTL 中不存在对应波段的乘/加系数键时抛出。
    """
    M = float(mtl[f"RADIANCE_MULT_BAND_{band}"])
    A = float(mtl[f"RADIANCE_ADD_BAND_{band}"])
    return M, A


def compute_toa_radiance(dn: np.ndarray, M: float, A: float) -> np.ndarray:
    """线性辐射定标: 将 DN 转为 TOA 辐亮度。

    公式:
      - `L = M * DN + A`

    注意:
      - 本函数不处理无效像元掩膜，掩膜在外部根据 DN 值和输入影像 mask 进行；
      - 输入/输出均为 `float32/float64` 的 Numpy 数组（按调用方决定 dtype）。
    """
    return M * dn + A


def find_band_tif(landsat_dir: Path, band: int) -> Path:
    """在场景目录下查找指定波段号的 GeoTIFF。

    例如 `*_B1.TIF`、`*_B5.TIF`。大小写不敏感。

    异常:
      - FileNotFoundError: 未找到对应波段文件。
    """
    suf = f"_B{band}.TIF"
    for fn in landsat_dir.iterdir():
        if fn.is_file() and fn.name.lower().endswith(suf.lower()):
            return fn
    raise FileNotFoundError(f"未找到波段文件: {landsat_dir} band={band}")


def calibrate_landsat(landsat_tif_path: Path,
                      out_tif_path: Path,
                      bands: List[int] = None) -> Path:
    """对 Landsat C2 L1 单波段 TIF 进行批量辐射定标并合成为多波段 GeoTIFF。

    参数:
      - landsat_tif_path: 场景目录或任意一个波段 TIF（用于定位目录）；
      - out_tif_path: 输出多波段辐亮度 GeoTIFF 路径；
      - bands: 要处理的波段列表（默认 `LS_BANDS_DEFAULT`）。

    行为与实现:
      - 读取参考波段的 profile 作为几何模板（空间分辨率、尺寸、投影等）；
      - 根据 MTL 的 (M, A) 对每个波段执行 `L = M*DN + A`；
      - 将 DN<=0 或内部 mask==0 的像元视为无效并写为 `NODATA_VAL`；
      - 输出数据类型为 `float32`，nodata 为 `NODATA_VAL`；
      - 以逐波段读取/写入的方式控制内存占用。

    返回:
      - 输出多波段 GeoTIFF 的路径（`out_tif_path`）。
    """
    bands = bands or LS_BANDS_DEFAULT
    p = Path(landsat_tif_path)
    landsat_dir = p if p.is_dir() else p.parent
    mtl_path = find_mtl_file(landsat_dir)
    mtl = read_mtl(mtl_path)

    # 读取首波段作为几何模板
    b0_path = find_band_tif(landsat_dir, bands[0])
    with rasterio.open(b0_path) as src0:
        profile = src0.profile.copy()

    profile.update({
        "count": len(bands),
        "dtype": "float32",
        "nodata": NODATA_VAL
    })

    with rasterio.open(out_tif_path, 'w', **profile) as dst:
        for i, b in enumerate(bands, start=1):
            bpath = find_band_tif(landsat_dir, b)
            with rasterio.open(bpath) as srcb:
                dn = srcb.read(1).astype(np.float32)
                # 无效像元判定：DN<=0 或者 内部掩膜为 0（常见为 0/255）
                mask = (dn <= 0) | ~srcb.read_masks(1).astype(bool)

            M, A = get_ls_rad_coeffs(mtl, b)
            arr = compute_toa_radiance(dn, M, A)

            arr = arr.astype(np.float32, copy=False)
            arr[mask] = NODATA_VAL
            dst.write(arr, indexes=i)

    return out_tif_path


def landsat_bounds_wgs84(landsat_like_tif: Path) -> Tuple[float, float, float, float]:
    """获取参考影像的外接矩形在 WGS84 的经纬度范围。

    策略:
      - 优先使用 B1 作为参考；若不可用，则回退到目录下的任意 `*_B*.TIF`；
      - 使用 `rasterio.warp.transform_bounds` 将影像边界从原始投影转换到 WGS84；
      - 返回 `(minx, miny, maxx, maxy)`（经度、纬度）。
    """
    p = Path(landsat_like_tif)
    if p.is_dir():
        # Prefer B1; fallback to any band file *_B*.TIF
        try:
            ref = find_band_tif(p, 1)
        except FileNotFoundError:
            cands = [f for f in p.iterdir() if f.is_file() and re.search(r"_B\d+\.TIF$", f.name, re.IGNORECASE)]
            if not cands:
                raise FileNotFoundError(f"未找到任何 Landsat 波段 TIF 用于计算范围: {p}")
            ref = sorted(cands, key=lambda x: x.name)[0]
        open_path = ref
    else:
        open_path = p
    with rasterio.open(open_path) as src:
        left, bottom, right, top = src.bounds
        src_crs = src.crs
    minx, miny, maxx, maxy = transform_bounds(src_crs, "EPSG:4326", left, bottom, right, top, densify_pts=21)
    return float(minx), float(miny), float(maxx), float(maxy)


def landsat_footprint_polygons_wgs84(landsat_path: Path) -> List[List[Tuple[float, float]]]:
    """
    从 Landsat 场景（目录或某个波段 TIF）计算 footprint 的多边形（WGS84，经纬度）。

    方法:
      - 读取参考波段（优先 B1），以“像元值>0 且内部 mask>0”为有效像元；
      - `rasterio.features.shapes` 基于有效掩膜提取连通区域多边形（初始几何在影像 CRS 下）；
      - 使用 `rasterio.warp.transform_geom` 将几何转换到 WGS84；
      - 仅返回每个（多）多边形的外环点序列；可能返回多个碎片区域。

    注意:
      - 如果影像包含大量无效像元碎片，返回的多边形列表可能较多；
      - 本函数不合并/简化多边形，交由后续点内测试时统一取并集。
    """
    p = Path(landsat_path)
    if p.is_dir():
        try:
            ref = find_band_tif(p, 1)
        except FileNotFoundError:
            # 退而求其次，任意 *_B*.TIF
            cands = [f for f in p.iterdir() if f.is_file() and re.search(r"_B\d+\.TIF$", f.name, re.IGNORECASE)]
            if not cands:
                raise FileNotFoundError(f"未找到任何 Landsat 波段 TIF 用于 footprint: {p}")
            ref = sorted(cands, key=lambda x: x.name)[0]
    else:
        ref = p
    polys = []
    with rasterio.open(ref) as src:
        # 使用 dataset_mask（融合各波段与内部 nodata），避免依赖像元值阈值
        ds_mask = src.dataset_mask().astype(np.uint8)
        valid = ds_mask > 0
        # shapes 提取 True 像元的连通区域（影像 CRS）
        for geom, val in shapes(valid.astype(np.uint8), mask=valid.astype(np.uint8), transform=src.transform):
            if val != 1:
                continue
            # 可选：如需调试 bounds，可用 rasterio.features.bounds 获取几何包络并再投影。
            # 注意：旧版本中误用了不存在的 geometry_bounds，已移除以避免异常中断。
            # minx, miny, maxx, maxy = rasterio.features.bounds(geom)
            # _ = transform_bounds(src.crs, "EPSG:4326", minx, miny, maxx, maxy, densify_pts=0)
            # 使用 transform_geom 将几何从投影坐标系转换到 WGS84，经纬度坐标
            geom84 = rasterio.warp.transform_geom(src.crs, "EPSG:4326", geom, precision=6)
            # 仅取 Polygon / MultiPolygon 的外环
            if geom84["type"] == "Polygon":
                exterior = geom84["coordinates"][0]
                polys.append([(float(x), float(y)) for x, y in exterior])
            elif geom84["type"] == "MultiPolygon":
                for poly in geom84["coordinates"]:
                    exterior = poly[0]
                    polys.append([(float(x), float(y)) for x, y in exterior])
    # 若过多碎片，按照顶点数排序取前若干（此处不过滤，交由 contains_points 逐一判定）
    return polys


def clip_goci_bbox(goci_nc_path: Path,
                   out_nc_path: Path,
                   bbox_wgs84: Tuple[float, float, float, float],
                   keep_vars: List[str] = None) -> Path:
    """按矩形 bbox（WGS84，经纬度）对 GOCI L1B 进行裁剪。

    行为:
      - 读取 `navigation_data/latitude, longitude`（形状 H×W）；
      - 在弯曲网格上基于经纬度逐点做 bbox 包含判定；
      - 基于包含掩膜的最小包络行列范围裁剪；
      - 对保留的 geophysical 变量在 bbox 外的像元写入 `_FillValue`；
      - 输出新 NetCDF：`navigation_data`、`geophysical_data` 和 `mask/inside_bbox`。

    参数:
      - `keep_vars`: 需要保留的 geophysical 变量名列表（默认 `GOCI_BAND_NAMES_DEFAULT`）。

    异常:
      - RuntimeError: 当 bbox 与 GOCI 网格不相交时抛出。
    """
    keep_vars = keep_vars or GOCI_BAND_NAMES_DEFAULT
    minx, miny, maxx, maxy = bbox_wgs84

    with Dataset(goci_nc_path, "r") as ds:
        lat = ds["navigation_data"]["latitude"][:]
        lon = ds["navigation_data"]["longitude"][:]
        if lat.shape != lon.shape:
            raise RuntimeError("GOCI: latitude 与 longitude 形状不一致")

        inside_full = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
        if not np.any(inside_full):
            raise RuntimeError("bbox 与 GOCI 网格不相交")

        rows = np.any(inside_full, axis=1)
        cols = np.any(inside_full, axis=0)
        r_idx = np.where(rows)[0]
        c_idx = np.where(cols)[0]
        r0, r1 = int(r_idx.min()), int(r_idx.max()) + 1
        c0, c1 = int(c_idx.min()), int(c_idx.max()) + 1

        lat_sub = lat[r0:r1, c0:c1]
        lon_sub = lon[r0:r1, c0:c1]
        inside_sub = inside_full[r0:r1, c0:c1]

        with Dataset(out_nc_path, "w") as out:
            out.createDimension("rows", lat_sub.shape[0])
            out.createDimension("cols", lat_sub.shape[1])

            nav = out.createGroup("navigation_data")
            vlat = nav.createVariable("latitude", "f4", ("rows", "cols"), zlib=True, complevel=4)
            vlon = nav.createVariable("longitude","f4", ("rows","cols"), zlib=True, complevel=4)
            vlat[:] = lat_sub.astype(np.float32)
            vlon[:] = lon_sub.astype(np.float32)

            mgrp = out.createGroup("mask")
            vmsk = mgrp.createVariable("inside_bbox", "i1", ("rows","cols"), zlib=True, complevel=4)
            vmsk.setncattr("description", "1=inside Landsat bbox; 0=outside")
            vmsk[:] = inside_sub.astype(np.int8)

            gphy = out.createGroup("geophysical_data")
            for name in keep_vars:
                if "geophysical_data" in ds.groups and name in ds["geophysical_data"].variables:
                    srcv = ds["geophysical_data"][name]
                    data = srcv[:].astype(np.float32)
                    fillv = None
                    if "_FillValue" in srcv.ncattrs():
                        try:
                            fillv = float(srcv.getncattr("_FillValue"))
                        except Exception:
                            fillv = None
                    if fillv is None:
                        fillv = -999.0
                    data = data[r0:r1, c0:c1]
                    data = np.where(inside_sub, data, fillv).astype(np.float32)

                    vo = gphy.createVariable(name, "f4", ("rows","cols"), zlib=True, complevel=4, fill_value=fillv)
                    for att in ("long_name","units","wavelength","bandwidth"):
                        if att in srcv.ncattrs():
                            vo.setncattr(att, srcv.getncattr(att))
                    vo[:] = data
            out.setncattr("title", "GOCI subset by Landsat bbox")
            out.setncattr("history", "subset via rectangular bbox; inside stored in mask/inside_bbox")
    return out_nc_path


def clip_goci_by_polygons(goci_nc_path: Path,
                          out_nc_path: Path,
                          polygons_wgs84: List[List[Tuple[float, float]]],
                          keep_vars: List[str] = None) -> Path:
    """
    使用给定的 footprint 多边形（WGS84，经纬度）在 GOCI 弯曲网格上做点内判定，
    输出裁剪后的 NetCDF（navigation_data / geophysical_data + mask/inside_footprint）。
    多个多边形取并集。

    细节:
      - 将 GOCI 的 (lon,lat) 网格展平为 N×2 点集，使用 Matplotlib Path 的 `contains_points`
        进行点在多边形内判定；对所有多边形逐一求并集；
      - 为降低输出体积，按 inside 掩膜的最小外接行列范围裁剪；
      - geophysical 变量在 footprint 外写入 `_FillValue`（若源变量未定义则使用 -999.0）。

    性能:
      - 对于超大网格或多边形数量很多的情况，点内判定耗时可能较长；可以考虑简化/抽稀多边形
        或分块处理以降低内存峰值与计算时间。本实现适用于中等规模网格。
    """
    keep_vars = keep_vars or GOCI_BAND_NAMES_DEFAULT
    with Dataset(goci_nc_path, "r") as ds:
        lat = ds["navigation_data"]["latitude"][:]
        lon = ds["navigation_data"]["longitude"][:]
        if lat.shape != lon.shape:
            raise RuntimeError("GOCI: latitude 与 longitude 形状不一致")

        H, W = lat.shape

        # ① 计算多边形的联合 bbox 并在网格上做粗筛
        all_x = []
        all_y = []
        for poly in polygons_wgs84:
            xs, ys = zip(*poly)
            all_x.extend(xs)
            all_y.extend(ys)
        minx, maxx = float(np.min(all_x)), float(np.max(all_x))
        miny, maxy = float(np.min(all_y)), float(np.max(all_y))
        bbox_mask = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
        if not np.any(bbox_mask):
            raise RuntimeError("footprint 与 GOCI 网格不相交（bbox 粗筛）")
        r_idx = np.where(np.any(bbox_mask, axis=1))[0]
        c_idx = np.where(np.any(bbox_mask, axis=0))[0]
        r0, r1 = int(r_idx.min()), int(r_idx.max()) + 1
        c0, c1 = int(c_idx.min()), int(c_idx.max()) + 1

        lon_win = lon[r0:r1, c0:c1]
        lat_win = lat[r0:r1, c0:c1]

        # ② 在窗口内做精确点测，边界点视为 inside（radius>0）
        inside_win = np.zeros(lon_win.shape, dtype=bool)
        pts = np.c_[lon_win.ravel(), lat_win.ravel()]
        for poly in polygons_wgs84:
            path = MplPath(poly)
            inside = path.contains_points(pts, radius=1e-9).reshape(lon_win.shape)
            inside_win |= inside

        if not np.any(inside_win):
            raise RuntimeError("footprint 与 GOCI 网格相交为空（精确点测）")

        # ③ 收缩到 inside 的最小包络
        rows_any = np.any(inside_win, axis=1)
        cols_any = np.any(inside_win, axis=0)
        rr = np.where(rows_any)[0]
        cc = np.where(cols_any)[0]
        rr0, rr1 = int(rr.min()), int(rr.max()) + 1
        cc0, cc1 = int(cc.min()), int(cc.max()) + 1

        lon_sub = lon_win[rr0:rr1, cc0:cc1]
        lat_sub = lat_win[rr0:rr1, cc0:cc1]
        inside_sub = inside_win[rr0:rr1, cc0:cc1]

        # ④ 写出子集 NC，遵循与 22_comparision_L_G 一致的约定
        with Dataset(out_nc_path, "w") as out:
            out.createDimension("y", lat_sub.shape[0])
            out.createDimension("x", lat_sub.shape[1])

            nav = out.createGroup("navigation_data")
            vlat = nav.createVariable("latitude", "f4", ("y", "x"), zlib=True, complevel=4)
            vlon = nav.createVariable("longitude","f4", ("y","x"), zlib=True, complevel=4)
            vlat[:] = lat_sub.astype(np.float32)
            vlon[:] = lon_sub.astype(np.float32)
            vlat.setncattr("units", "degrees_north")
            vlon.setncattr("units", "degrees_east")

            mgrp = out.createGroup("mask")
            vmsk = mgrp.createVariable("inside_mask", "u1", ("y","x"), zlib=True, complevel=4)
            vmsk.setncattr("long_name", "1 = inside footprint; 0 = outside")
            vmsk[:] = inside_sub.astype(np.uint8)

            gphy = out.createGroup("geophysical_data")
            if "geophysical_data" in ds.groups:
                in_geo = ds["geophysical_data"]
            else:
                in_geo = ds
            for name in keep_vars:
                if name not in in_geo.variables:
                    # 跳过缺失变量（保持稳健）
                    continue
                srcv = in_geo[name]
                # 只读入粗窗口，减少内存
                data_win = srcv[r0:r1, c0:c1]
                # netCDF4 可能返回 MaskedArray（且已应用 scale_factor/add_offset）
                if np.ma.isMaskedArray(data_win):
                    data_win = data_win.filled(np.nan).astype(np.float32)
                else:
                    data_win = np.array(data_win, dtype=np.float32)
                data_sub = data_win[rr0:rr1, cc0:cc1]

                fillv = None
                if "_FillValue" in srcv.ncattrs():
                    try:
                        fillv = float(np.array(srcv.getncattr("_FillValue")).ravel()[0])
                    except Exception:
                        fillv = None
                if fillv is None:
                    fillv = -999.0

                data_out = data_sub.copy()
                data_out[~inside_sub] = fillv

                vo = gphy.createVariable(name, "f4", ("y","x"), zlib=True, complevel=4, fill_value=float(fillv))
                # 拷贝常用元数据（跳过缩放属性）
                for att in ("long_name","units","wavelength","bandwidth"):
                    if att in srcv.ncattrs():
                        try:
                            vo.setncattr(att, srcv.getncattr(att))
                        except Exception:
                            pass
                vo[:] = data_out.astype(np.float32)

            out.setncattr("title", "GOCI subset by Landsat footprint polygon")
            out.setncattr("history", "subset via footprint mask; inside stored in mask/inside_mask")
    return out_nc_path


def _robust_vmin_vmax(arr_list: List[np.ndarray], q_low=1, q_high=99) -> Tuple[float, float]:
    vals = []
    for a in arr_list:
        if a is None:
            continue
        aa = np.asarray(a)
        aa = aa[np.isfinite(aa)]
        if aa.size:
            vals.append(aa)
    if not vals:
        return 0.0, 1.0
    allv = np.concatenate(vals)
    vmin = np.percentile(allv, q_low)
    vmax = np.percentile(allv, q_high)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmin >= vmax:
        vmin = float(np.nanmin(allv))
        vmax = float(np.nanmax(allv))
    return float(vmin), float(vmax)


def save_comparison_plots(landsat_tif: Path, goci_nc_sub: Path, out_png: Path, max_bins: int = 256):
    """
    生成对比图：
      1) GOCI footprint 内/外掩膜可视化；
      2) 直接显示 Landsat 第1波段 与 GOCI 443nm（或首个可用波段）的影像快视图（统一色标范围）。
    不做配准/重采样，仅做视觉层面对比，便于快速质检。

    参数:
      - `max_bins`: 保留接口（当前未使用在主图中）。
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # 读取 GOCI 子集：lat/lon、mask、选定波段（优先 L_TOA_443）
    with Dataset(goci_nc_sub, "r") as ds:
        lat_g = ds["navigation_data"]["latitude"][:].astype(np.float64)
        lon_g = ds["navigation_data"]["longitude"][:].astype(np.float64)
        inside = None
        if "mask" in ds.groups:
            if "inside_mask" in ds["mask"].variables:
                inside = ds["mask"]["inside_mask"][:].astype(bool)
            elif "inside_footprint" in ds["mask"].variables:
                inside = ds["mask"]["inside_footprint"][:].astype(bool)
        # 选择 GOCI 波段
        g_name = None
        if "geophysical_data" in ds.groups:
            for cand in ("L_TOA_443",) + tuple(n for n in GOCI_BAND_NAMES_DEFAULT if n != "L_TOA_443"):
                if cand in ds["geophysical_data"].variables:
                    g_name = cand
                    break
            if g_name is None:
                raise KeyError("在 geophysical_data 中未找到任何预期的 GOCI 变量")
            v = ds["geophysical_data"][g_name]
            arr_g = v[:]
            if np.ma.isMaskedArray(arr_g):
                arr_g = arr_g.filled(np.nan).astype(np.float32)
            else:
                arr_g = np.array(arr_g, dtype=np.float32)
            if "_FillValue" in v.ncattrs():
                fv = float(np.array(v.getncattr("_FillValue")).ravel()[0])
                arr_g = np.where(arr_g == fv, np.nan, arr_g)
        else:
            raise KeyError("GOCI 子集中缺少 geophysical_data 组")
        if inside is not None:
            arr_g = np.where(inside, arr_g, np.nan)

    # 读取 Landsat B1 与经纬度网格（按 22_comparision_L_G 的逻辑）
    with rasterio.open(landsat_tif) as src:
        L_arr = src.read(1).astype(np.float32)
        if src.nodata is not None:
            L_arr = np.where(L_arr == src.nodata, np.nan, L_arr)
        H, W = src.height, src.width
        T = src.transform
        crs = src.crs
        cols = np.arange(W)
        rows = np.arange(H)
        cgrid, rgrid = np.meshgrid(cols, rows)
        x_proj = T.c + T.a*(cgrid + 0.5) + T.b*(rgrid + 0.5)
        y_proj = T.f + T.d*(cgrid + 0.5) + T.e*(rgrid + 0.5)
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon_L, lat_L = transformer.transform(x_proj, y_proj)

    # 颜色范围采用两者联合的 1–99% 分位
    vmin, vmax = _robust_vmin_vmax([L_arr, arr_g], q_low=1, q_high=99)

    # 并排绘制（两边均按真实经纬度 pcolormesh）
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)

    fig = plt.figure(figsize=(12, 5), dpi=150)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Landsat Band 1 (radiance)")
    im1 = ax1.pcolormesh(lon_L, lat_L, L_arr, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Latitude (°N)")
    ax1.grid(True, alpha=0.3)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).set_label("Radiance")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title(f"GOCI ({g_name})")
    im2 = ax2.pcolormesh(lon_g, lat_g, arr_g, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")
    ax2.grid(True, alpha=0.3)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label("Radiance")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_pair(row: Dict[str, str],
                 bands: List[int],
                 goci_keep_vars: List[str],
                 out_root: Path) -> Dict[str, str]:
    """处理一对配对的 Landsat 与 GOCI 数据。

    步骤:
      1) 对 Landsat 执行辐射定标，输出多波段辐亮度 GeoTIFF；
      2) 计算 Landsat 有效像元 footprint 多边形；
      3) 使用 footprint 对 GOCI L1B 进行裁剪，输出子集 NetCDF；
      4) 生成直方图与掩膜预览图。

    I/O 约定:
      - `row['landsat_tif']`: 可以是场景目录或单个波段路径，内部会自动定位场景目录；
      - `row['goci_nc']`: 指向 GOCI L1B NetCDF；
      - 输出写入 `out_root/{landsat_scene}/` 子目录，文件名包含场景名与 GOCI 基名。

    返回:
      - 包含处理状态、错误原因（如有）与关键输出路径的字典。
    """
    print("[INFO] Start pair:", row.get("landsat_tif"), "|", row.get("goci_nc"))
    ls_tif = Path(row["landsat_tif"]).expanduser()
    goci_nc = Path(row["goci_nc"]).expanduser()

    if not ls_tif.exists():
        return {"status": "skip", "reason": "landsat_tif not found", **row}
    if not goci_nc.exists():
        return {"status": "skip", "reason": "goci_nc not found", **row}

    p = Path(row["landsat_tif"]).expanduser()
    ls_dir = p if p.is_dir() else p.parent
    ls_scene = ls_dir.name
    out_scene = out_root / ls_scene
    out_scene.mkdir(parents=True, exist_ok=True)

    ls_out = out_scene / f"{ls_scene}_TOA_RAD_B{'-'.join(map(str,bands))}.tif"
    goci_out = out_scene / f"{goci_nc.stem}_subset_footprint.nc"
    fig_out = out_scene / f"{goci_nc.stem}_compare.png"

    # 1) 定标
    if not ls_out.exists():
        print("  [STEP] Calibrating Landsat radiance →", ls_out)
        calibrate_landsat(ls_tif, ls_out, bands=bands)
    else:
        print("  [SKIP] Landsat calibrated exists")

    # 2) 计算 footprint 多边形并裁剪 GOCI
    print("  [STEP] Building Landsat footprint polygon(s)…")
    polys = landsat_footprint_polygons_wgs84(ls_out if ls_out.exists() else ls_tif)
    print(f"  [INFO] Footprint polygons: {len(polys)}")

    print("  [STEP] Clipping GOCI by footprint →", goci_out)
    clip_goci_by_polygons(goci_nc, goci_out, polys, keep_vars=goci_keep_vars)
    # Logging: confirm creation and report dimensions
    if goci_out.exists():
        with Dataset(goci_out, 'r') as ds_out:
            # 兼容不同维度命名（rows/cols 或 y/x）
            if 'rows' in ds_out.dimensions and 'cols' in ds_out.dimensions:
                h = ds_out.dimensions['rows'].size
                w = ds_out.dimensions['cols'].size
            else:
                h = ds_out.dimensions['y'].size
                w = ds_out.dimensions['x'].size
            print(f"  [OK] GOCI subset saved: {goci_out}  shape=({h},{w})")
    else:
        print(f"  [WARN] Expected GOCI subset not found: {goci_out}")

    # 3) 生成对比图（不做配准/重采样，仅做分布层面可视化核对）
    print("  [STEP] Saving comparison plot →", fig_out)
    save_comparison_plots(ls_out, goci_out, fig_out)

    print("[OK] Done pair:", ls_scene)
    return {
        "status": "ok",
        "landsat_calibrated": str(ls_out),
        "goci_subset": str(goci_out),
        "comparison_figure": str(fig_out),
        **row
    }


def main():
    """无命令行参数的批处理入口。

    使用 `pairs.csv` 中的配对记录进行批处理：
      - 列名要求至少包含 `landsat_tif` 与 `goci_nc`；
      - 每条记录单独建立输出子目录，保存定标结果、裁剪结果和对比图；
      - 写出 `batch_results.csv` 汇总处理状态（覆盖写）。

    修改指南:
      - 更改 `PAIRS_CSV_PATH` 为你的实际 `pairs.csv` 路径；
      - 调整 `bands`（定标波段）与 `goci_keep_vars`（输出 GOCI 变量）；
      - 设置 `OUTPUT_ROOT` 作为批处理输出根目录。
    """
    # ---------------- 在这里设置所有参数 ----------------
    # 自定义 CSV 路径（请按需修改为你的实际路径）
    PAIRS_CSV_PATH = r"/Users/zy/Python_code/My_Git/img_match/SR_Imagery/Slot_7_2021_2025/pairs.csv"
    pairs_csv = Path(PAIRS_CSV_PATH)

    # Landsat 定标波段（仅 RAD 模式）
    bands = LS_BANDS_DEFAULT[:]        # 例如 [1,2,3,4,5]

    # GOCI 输出中需要保留的 geophysical_data 变量名
    goci_keep_vars = GOCI_BAND_NAMES_DEFAULT[:]

    # 批处理结果摘要 CSV（所有输出统一汇总到该目录下，每个场景单独一个子文件夹）
    OUTPUT_ROOT = Path("/Users/zy/python_code/My_Git/img_match/batch_outputs")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_summary = (OUTPUT_ROOT / "batch_results.csv")
    # ----------------------------------------------------

    if not pairs_csv.exists():
        raise FileNotFoundError(f"未找到 {pairs_csv}")

    print(f"[LOAD] pairs from: {pairs_csv}")
    rows = []
    with open(pairs_csv, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            if "landsat_tif" in r and "goci_nc" in r:
                rows.append(r)
    print(f"[INFO] total pairs: {len(rows)}")

    results = []
    for r in rows:
        try:
            res = process_pair(r, bands=bands, goci_keep_vars=goci_keep_vars, out_root=OUTPUT_ROOT)
        except Exception as e:
            res = {"status": "error", "error": str(e), **r}
        results.append(res)

    fields = sorted({k for d in results for k in d.keys()})
    with open(out_summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for d in results:
            w.writerow(d)
    print(f"[SAVE] summary → {out_summary}")

    print(f"✅ Done. Summary saved: {out_summary}  (total={len(results)}, ok={sum(1 for x in results if x.get('status')=='ok')})")


if __name__ == "__main__":
    main()
