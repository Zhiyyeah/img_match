#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Landsat 多波段 GeoTIFF 可视化 (RGB = B4,B3,B2)

功能:
1. 读取 `batch_outputs/<scene>/<scene>_TOA_RAD_B1-2-3-4-5.tif`
2. 使用波段 4/3/2 组成自然色 RGB 合成
3. 做 2–98 百分位线性拉伸 + 可选 gamma
4. 显示坐标轴 (投影坐标). 可选生成经纬度刻度(简易近似, 默认关闭)
5. 保存 PNG

在 main() 中直接修改 scene / tif_path / 输出路径。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform
import matplotlib.pyplot as plt
from pyproj import CRS


def read_landsat_rgb(tif_path: Path, bands=(4,3,2), max_h: int = 3000):
    """读取指定多波段 TIFF 的给定波段 (1-based) -> 返回 (H,W,3) float32 (原始辐亮度)。"""
    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        scale = 1.0
        if H > max_h:
            scale = max_h / float(H)
            out_h = max_h
            out_w = int(round(W * scale))
        else:
            out_h, out_w = H, W
        arr_list = []
        for b in bands:
            if scale != 1.0:
                data = src.read(b, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
            else:
                data = src.read(b).astype(np.float32)
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            arr_list.append(data)
        rgb = np.stack(arr_list, axis=-1)  # (H,W,3)
        transform_aff = src.transform
        crs = src.crs
    return rgb, transform_aff, crs


def percentile_stretch(rgb: np.ndarray, p_low=2, p_high=98, gamma=1.0):
    out = rgb.copy()
    for i in range(3):
        band = out[..., i]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            continue
        lo = np.percentile(finite, p_low)
        hi = np.percentile(finite, p_high)
        if hi <= lo:
            lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        band = np.clip((band - lo) / (hi - lo + 1e-12), 0, 1)
        if gamma != 1.0:
            band = np.power(band, 1.0/gamma)
        out[..., i] = band
    # 掩盖 NaN 为 0
    out[~np.isfinite(out)] = 0
    return out


def extent_from_transform(transform_aff, shape):
    """根据 affine transform 计算 matplotlib extent (left,right,bottom,top)。"""
    H, W = shape
    # 像元左上角 (0,0)
    x0 = transform_aff.c
    y0 = transform_aff.f
    # 像元大小
    px_w = transform_aff.a
    px_h = transform_aff.e  # 通常为负
    left = x0
    top = y0
    right = x0 + px_w * W
    bottom = y0 + px_h * H
    return [left, right, bottom, top]


def add_latlon_ticks(ax, transform_aff, crs, shape, n=5):
    """（可选）添加经纬度刻度 (粗略)：取边界四个角点投影到经纬度并线性插值。
    对较大区域会有形变误差，只做近似展示。"""
    try:
        if crs is None or CRS.from_user_input(crs).is_geographic:
            return  # 已经是经纬度
    except Exception:
        return
    H, W = shape
    extent = extent_from_transform(transform_aff, shape)
    left, right, bottom, top = extent[0], extent[1], extent[2], extent[3]
    xs = np.linspace(left, right, W)
    ys = np.linspace(top, bottom, H)
    # 取四边界点集合（减少数量）
    edge_x = np.concatenate([xs, xs, np.full_like(ys, left), np.full_like(ys, right)])
    edge_y = np.concatenate([np.full_like(xs, top), np.full_like(xs, bottom), ys, ys])
    lon, lat = transform(crs, 'EPSG:4326', edge_x, edge_y)
    # 角点经纬度
    lon_corners = lon[0], lon[W-1], lon[W], lon[W*2-1]  # 近似
    lat_corners = lat[0], lat[W-1], lat[W], lat[W*2-1]
    # 简易线性近似生成刻度
    lon_min, lon_max = min(lon_corners), max(lon_corners)
    lat_min, lat_max = min(lat_corners), max(lat_corners)
    ax.set_xlabel(f'X (proj)  ~ Lon {lon_min:.2f}–{lon_max:.2f}')
    ax.set_ylabel(f'Y (proj)  ~ Lat {lat_min:.2f}–{lat_max:.2f}')


def plot_rgb(rgb_stretched, transform_aff, crs, title, out_png: Path, show_latlon_hint=True):
    H, W, _ = rgb_stretched.shape
    extent = extent_from_transform(transform_aff, (H, W))
    fig, ax = plt.subplots(figsize=(10, 10 * H / W))
    ax.imshow(rgb_stretched, extent=extent, origin='upper')
    ax.set_title(title)
    ax.set_xlabel('Projected X')
    ax.set_ylabel('Projected Y')
    ax.grid(True, alpha=0.3, linestyle='--')
    if show_latlon_hint:
        add_latlon_ticks(ax, transform_aff, crs, (H, W))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f'[OK] 保存: {out_png}')


def main():
    # ---- 用户参数 ----
    scene = 'LC08_L1TP_116036_20210330_20210409_02_T1'
    tif_path = Path('batch_outputs') / scene / f'{scene}_TOA_RAD_B1-2-3-4-5.tif'
    out_png = Path('batch_outputs') / scene / f'{scene}_RGB432.png'
    gamma = 1.0       # 伽马校正 (1.0 为关闭)
    max_h = 3000      # 读取时最大高度（可降低内存）
    # ------------------

    if not tif_path.exists():
        print(f'[ERR] 找不到文件: {tif_path}')
        return

    print(f'[INFO] 读取: {tif_path.name}')
    rgb, transform_aff, crs = read_landsat_rgb(tif_path, bands=(4,3,2), max_h=max_h)
    print(f'[INFO] 原始范围: shape={rgb.shape}, CRS={crs}')

    rgb_disp = percentile_stretch(rgb, 2, 98, gamma=gamma)
    plot_rgb(rgb_disp, transform_aff, crs, f'{scene} RGB(4,3,2)', out_png)


if __name__ == '__main__':
    main()
