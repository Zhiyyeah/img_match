#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量上采样：将 02_batch_calibrate_and_clip_nocli.py 的输出（Landsat 多波段辐亮度 TIF
与 GOCI 裁剪子集 NC）进行配对，对 GOCI 弯曲网格进行 pyresample 重采样到 Landsat 规则网格，
并按场景输出到 batch_resampled/{scene}/ 目录。

依赖：numpy, rasterio, netCDF4, pyproj, pyresample
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
import rasterio
from rasterio.transform import Affine
from netCDF4 import Dataset
from pyproj import Transformer
from pyresample import geometry, kd_tree


# ---------------- 配置（按需修改） ----------------
# 输入：02 批处理产生的根目录（子目录为各 Landsat 场景）
OUTPUT_ROOT = Path("batch_outputs")

# 输出：重采样结果根目录（按场景分子目录）
RESAMPLED_ROOT = Path("batch_resampled")

# 重采样参数
USE_GAUSSIAN = True   # True: kd_tree.resample_gauss；False: kd_tree.resample_nearest
ROI_METERS   = 800    # 影响半径（米）
SIGMA_METERS = 320    # 高斯核宽度（米）
NEIGHBOURS   = 16
FILL_VALUE   = np.nan
NPROCS       = 1

# 波段映射（与裁剪输出 geophysical_data 保持一致）
GOCI_BANDS = {
    443: "L_TOA_443",
    490: "L_TOA_490",
    555: "L_TOA_555",
    660: "L_TOA_660",
    865: "L_TOA_865",
}


# ---------------- 工具函数 ----------------
def read_goci_subset(nc_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 GOCI 子集 NetCDF，返回 (data, lat, lon)。

    - data: shape=(B,Hs,Ws) float32，外部掩膜/FillValue 已置为 NaN
    - lat/lon: shape=(Hs,Ws) float64
    """
    with Dataset(nc_path, 'r') as ds:
        if 'navigation_data' not in ds.groups:
            raise KeyError(f"缺少 navigation_data 组: {nc_path}")
        nav = ds['navigation_data']
        lat = np.array(nav['latitude'][:], dtype=np.float64)
        lon = np.array(nav['longitude'][:], dtype=np.float64)

        inside = None
        if 'mask' in ds.groups:
            mgrp = ds['mask']
            if 'inside_mask' in mgrp.variables:
                inside = mgrp['inside_mask'][:].astype(bool)
            elif 'inside_footprint' in mgrp.variables:  # 兼容旧字段名
                inside = mgrp['inside_footprint'][:].astype(bool)

        geo = ds['geophysical_data'] if 'geophysical_data' in ds.groups else ds
        bands = []
        for wl, vname in GOCI_BANDS.items():
            if vname not in geo.variables:
                raise KeyError(f"缺少 geophysical_data/{vname}")
            v = geo[vname]
            arr = v[:]
            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan).astype(np.float32)
            else:
                arr = np.array(arr, dtype=np.float32)
            if '_FillValue' in v.ncattrs():
                try:
                    fv = float(np.array(v.getncattr('_FillValue')).ravel()[0])
                    arr = np.where(arr == fv, np.nan, arr)
                except Exception:
                    pass
            if inside is not None:
                arr = np.where(inside, arr, np.nan)
            bands.append(arr)
        data = np.stack(bands, axis=0)  # (B,Hs,Ws)
    return data, lat, lon


def read_landsat_tif(tif_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Affine, object]:
    """读取 Landsat 多波段 TIF，返回 (stack, lon, lat, transform, crs)。
    - stack: (B,H,W) float32，nodata->NaN
    - lon/lat: (H,W) float64，像元中心经纬度
    """
    with rasterio.open(tif_path) as ds:
        stack = ds.read().astype(np.float32)
        if ds.nodata is not None:
            stack = np.where(stack == ds.nodata, np.nan, stack)
        H, W = ds.height, ds.width
        T = ds.transform
        crs = ds.crs
        rows = np.arange(H)
        cols = np.arange(W)
        cgrid, rgrid = np.meshgrid(cols, rows)
        x = T.c + T.a*(cgrid + 0.5) + T.b*(rgrid + 0.5)
        y = T.f + T.d*(cgrid + 0.5) + T.e*(rgrid + 0.5)
        transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        lon, lat = transformer.transform(x, y)
    return stack, lon.astype(np.float64), lat.astype(np.float64), T, crs


def discover_pairs(output_root: Path) -> List[Tuple[str, Path, Path]]:
    """在 OUTPUT_ROOT 下寻找 (scene, landsat_tif, goci_nc) 配对。"""
    pairs = []
    if not output_root.exists():
        return pairs
    for scene_dir in sorted(output_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        ls = None
        goci = None
        for f in scene_dir.iterdir():
            if f.suffix.lower() == '.tif' and '_TOA_RAD_B' in f.name:
                ls = f
            if f.suffix.lower() == '.nc' and '_subset' in f.name:
                goci = f
        if ls and goci:
            pairs.append((scene_dir.name, ls, goci))
    return pairs


def resample_one_pair(ls_tif: Path, goci_nc: Path, out_dir: Path) -> Path:
    """对一对输入执行重采样，输出多波段 GeoTIFF 到 out_dir。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取数据
    g_data, g_lat, g_lon = read_goci_subset(goci_nc)      # (B,Hs,Ws)
    L_stack, L_lon, L_lat, L_T, L_crs = read_landsat_tif(ls_tif)
    B, Ht, Wt = L_stack.shape

    # 源 swath（经纬度）
    src_swath = geometry.SwathDefinition(lons=g_lon, lats=g_lat)

    # 仅对 Landsat 目标网格进行“最小必要窗口”裁剪，显著降低计算量
    g_minx, g_maxx = float(np.nanmin(g_lon)), float(np.nanmax(g_lon))
    g_miny, g_maxy = float(np.nanmin(g_lat)), float(np.nanmax(g_lat))
    mean_lat = float(np.nanmean(g_lat)) if np.isfinite(g_lat).any() else 0.0
    # 米->度 的近似（纬度固定， 经度按 cos(lat) 修正），额外加 1.5x 安全系数
    deg_lat_margin = (ROI_METERS / 111000.0) * 1.5
    coslat = max(0.1, np.cos(np.deg2rad(mean_lat)))
    deg_lon_margin = (ROI_METERS / (111000.0 * coslat)) * 1.5
    lon_min = g_minx - deg_lon_margin
    lon_max = g_maxx + deg_lon_margin
    lat_min = g_miny - deg_lat_margin
    lat_max = g_maxy + deg_lat_margin

    tgt_mask = (L_lon >= lon_min) & (L_lon <= lon_max) & (L_lat >= lat_min) & (L_lat <= lat_max)
    if np.any(tgt_mask):
        rows_any = np.any(tgt_mask, axis=1)
        cols_any = np.any(tgt_mask, axis=0)
        rmin, rmax = int(np.argmax(rows_any)), int(len(rows_any) - np.argmax(rows_any[::-1]) - 1)
        cmin, cmax = int(np.argmax(cols_any)), int(len(cols_any) - np.argmax(cols_any[::-1]) - 1)
        # 目标窗口（裁剪后的 Landsat 经纬度）
        L_lon_win = L_lon[rmin:rmax+1, cmin:cmax+1]
        L_lat_win = L_lat[rmin:rmax+1, cmin:cmax+1]
        tgt_swath = geometry.SwathDefinition(lons=L_lon_win, lats=L_lat_win)
        print(f"  [INFO] 目标窗口裁剪: rows={rmin}:{rmax+1} cols={cmin}:{cmax+1}  尺寸={L_lon_win.shape}")
    else:
        # 回退：使用全幅（可能非常慢）
        rmin, rmax, cmin, cmax = 0, Ht, 0, Wt
        L_lon_win = L_lon
        L_lat_win = L_lat
        tgt_swath = geometry.SwathDefinition(lons=L_lon, lats=L_lat)
        print("  [WARN] 未找到重叠窗口，使用全幅目标网格，计算会很慢")

    # 重采样（仅对窗口内计算，再写回全幅结果）
    resampled = np.full((B, Ht, Wt), np.nan, dtype=np.float32)
    for b in range(B):
        t_band0 = time.time()
        src_b = np.ma.array(g_data[b], mask=~np.isfinite(g_data[b]))
        if USE_GAUSSIAN:
            out_b = kd_tree.resample_gauss(
                src_swath, src_b, tgt_swath,
                radius_of_influence=ROI_METERS,
                sigmas=SIGMA_METERS,
                fill_value=FILL_VALUE,
                neighbours=NEIGHBOURS,
                reduce_data=True,
                nprocs=NPROCS,
            )
        else:
            out_b = kd_tree.resample_nearest(
                src_swath, src_b, tgt_swath,
                radius_of_influence=ROI_METERS,
                fill_value=FILL_VALUE,
            )
        resampled[b, rmin:rmax+1, cmin:cmax+1] = out_b.astype(np.float32)
        print(f"    [BAND {b+1}/{B}] 窗口重采样完成，用时 {time.time()-t_band0:.2f}s")

    # 写多波段 GeoTIFF（沿用 Landsat 几何）
    tag = 'GAUSS' if USE_GAUSSIAN else 'NEAREST'
    out_tif = out_dir / f"GOCI_on_Landsat_{goci_nc.stem}_{tag}.tif"
    profile = {
        'driver': 'GTiff', 'height': Ht, 'width': Wt, 'count': B,
        'dtype': 'float32', 'crs': L_crs, 'transform': L_T,
        'nodata': np.nan, 'compress': 'deflate', 'predictor': 2,
    }
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(resampled)
    return out_tif


def main():
    t0 = time.time()
    pairs = discover_pairs(OUTPUT_ROOT)
    if not pairs:
        print(f"[WARN] 未在 {OUTPUT_ROOT} 下找到可处理的配对目录。")
        return
    print(f"[INFO] 待处理配对数: {len(pairs)}  输出根目录: {RESAMPLED_ROOT}")

    for scene, ls_tif, g_nc in pairs:
        out_dir = RESAMPLED_ROOT / scene
        print(f"\n[PAIR] {scene}\n  LS : {ls_tif}\n  GOCI: {g_nc}")
        try:
            t1 = time.time()
            out_tif = resample_one_pair(ls_tif, g_nc, out_dir)
            dt = time.time() - t1
            print(f"  [OK] 保存 -> {out_tif}  用时 {dt:.2f}s")
        except Exception as e:
            print(f"  [ERR] 处理失败: {e}")

    print(f"\n✅ 全部完成，用时 {time.time() - t0:.1f}s  输出在: {RESAMPLED_ROOT}")


if __name__ == "__main__":
    main()
