#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比 03 步（pyresample 重采样）结果：
- 遍历 batch_resampled/{scene}，同时读取 batch_outputs/{scene} 的 Landsat TIF
- 生成每个场景的快速对比图（HR、LR、差值）按波段展示
- 采用缩略图读取（双线性降采样），内存友好

输出：batch_resampled/{scene}/COMPARE_{scene}.png
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt


OUTPUT_ROOT = Path("batch_outputs")
RESAMPLED_ROOT = Path("batch_resampled")

# 目标缩略图的最大高（像素），宽度按比例缩放
THUMB_MAX_H = 1200

# 波段标签（Landsat B1..B5 对应 GOCI 443/490/555/660/865）
BAND_LABELS = [
    "B1 / 443nm",
    "B2 / 490nm",
    "B3 / 555nm",
    "B4 / 660nm",
    "B5 / 865nm",
]


def discover_pairs():
    pairs = []
    if not RESAMPLED_ROOT.exists():
        return pairs
    for scene_dir in sorted(RESAMPLED_ROOT.iterdir()):
        if not scene_dir.is_dir():
            continue
        # resampled tif
        goci_on_ls = None
        for f in scene_dir.iterdir():
            if f.suffix.lower() == ".tif" and f.name.startswith("GOCI_on_Landsat_"):
                goci_on_ls = f
                break
        # landsat tif
        ls_dir = OUTPUT_ROOT / scene_dir.name
        landsat_tif = None
        if ls_dir.exists():
            for f in ls_dir.iterdir():
                if f.suffix.lower() == ".tif" and "_TOA_RAD_B" in f.name:
                    landsat_tif = f
        if goci_on_ls and landsat_tif:
            pairs.append((scene_dir.name, landsat_tif, goci_on_ls))
    return pairs


def read_thumb(ds: rasterio.DatasetReader, max_h: int) -> np.ndarray:
    count, H, W = ds.count, ds.height, ds.width
    if H <= max_h:
        arr = ds.read().astype(np.float32)
    else:
        scale = max_h / float(H)
        out_h = max_h
        out_w = int(round(W * scale))
        arr = ds.read(
            out_shape=(count, out_h, out_w),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
    # 将 nodata 替换为 NaN
    if ds.nodata is not None:
        arr = np.where(arr == ds.nodata, np.nan, arr)
    return arr


def robust_vmin_vmax(a_list):
    # 基于多个数组的 2-98 分位确定显示范围
    vals = np.concatenate([x[np.isfinite(x)].ravel() for x in a_list if x is not None and np.isfinite(x).any()])
    if vals.size == 0:
        return 0.0, 1.0
    vmin = np.nanpercentile(vals, 2)
    vmax = np.nanpercentile(vals, 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    return float(vmin), float(vmax)


def robust_sym_absmax(a):
    vals = a[np.isfinite(a)]
    if vals.size == 0:
        return 1.0
    q = np.nanpercentile(np.abs(vals), 98)
    if not np.isfinite(q) or q == 0:
        q = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
    return float(q) if q > 0 else 1.0


def main():
    pairs = discover_pairs()
    if not pairs:
        print("[WARN] 未找到可对比的场景（缺少 resampled 或 landsat 文件）")
        return
    print(f"[INFO] 待对比场景数: {len(pairs)}")

    for scene, ls_tif, lr_tif in pairs:
        print(f"\n[SCENE] {scene}\n  HR: {ls_tif}\n  LR: {lr_tif}")
        out_png = RESAMPLED_ROOT / scene / f"COMPARE_{scene}.png"

        with rasterio.open(ls_tif) as ds_hr, rasterio.open(lr_tif) as ds_lr:
            if ds_hr.width != ds_lr.width or ds_hr.height != ds_lr.height:
                print("  [ERR] HR/LR 尺寸不一致，跳过")
                continue
            H, W = ds_hr.height, ds_hr.width
            # 读取缩略图
            hr = read_thumb(ds_hr, THUMB_MAX_H)
            lr = read_thumb(ds_lr, THUMB_MAX_H)

        # 计算差值缩略图与简单指标
        diff = lr - hr

        # 绘图：每个波段三列（HR / LR / LR-HR）
        bands = min(hr.shape[0], lr.shape[0], 5)
        fig, axes = plt.subplots(bands, 3, figsize=(12, 3.2 * bands), constrained_layout=True)
        if bands == 1:
            axes = np.array([axes])

        for b in range(bands):
            hr_b = hr[b]
            lr_b = lr[b]
            df_b = diff[b]
            vmin, vmax = robust_vmin_vmax([hr_b, lr_b])
            amax = robust_sym_absmax(df_b)

            ax = axes[b, 0]
            im = ax.imshow(hr_b, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"HR {BAND_LABELS[b]}")
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

            ax = axes[b, 1]
            im = ax.imshow(lr_b, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"LR {BAND_LABELS[b]}")
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

            ax = axes[b, 2]
            im = ax.imshow(df_b, cmap="coolwarm", vmin=-amax, vmax=amax)
            ax.set_title("LR - HR")
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        fig.suptitle(f"Scene: {scene}  (缩略图高≤{THUMB_MAX_H})", fontsize=12)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  [OK] 保存对比图 -> {out_png}")


if __name__ == "__main__":
    main()

