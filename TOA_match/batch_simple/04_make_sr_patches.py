#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
仅裁剪生成 HR/LR 训练补丁（不进行任何采样/重采样）：
- 遍历 batch_outputs/{scene} 与 batch_resampled/{scene}
- 读取：
  - HR：batch_outputs/{scene}/*_TOA_RAD_B*.tif（辐射定标后的多波段Landsat）
  - LR：batch_resampled/{scene}/GOCI_on_Landsat_*.tif（已对齐到Landsat网格的GOCI）
- 要求 HR 与 LR 尺寸一致；否则跳过该场景
- 按 64x64 非重叠窗口裁剪；任意像元为 NaN 或 nodata 的 patch 直接舍弃（边缘黑边过滤）
- 输出：SR_Imagery/patches/{scene}/HR 与 LR/ 下，保存 .tif 和 .npy

注意：不读取也不使用任何 *.nc 文件。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window


# 目录配置
OUTPUT_ROOT = Path("batch_outputs")
RESAMPLED_ROOT = Path("batch_resampled")
TIF_ROOT = Path("SR_Imagery") / "tif"
NPY_ROOT = Path("SR_Imagery") / "npy"
# 向后兼容的别名（旧日志里引用 PATCH_ROOT）
PATCH_ROOT = TIF_ROOT

# 裁剪参数（默认非重叠，可自行将 STRIDE 调小以产生重叠样本）
PATCH_SIZE = 256
STRIDE = 256


def find_scene_inputs(scene_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """返回 (landsat_tif, goci_on_ls_tif)"""
    landsat_tif: Optional[Path] = None
    goci_on_ls: Optional[Path] = None

    # batch_outputs
    if scene_dir.exists():
        for f in scene_dir.iterdir():
            name = f.name.lower()
            if f.suffix.lower() == ".tif" and "_toa_rad_b" in name:
                landsat_tif = f

    # batch_resampled
    rs_dir = RESAMPLED_ROOT / scene_dir.name
    if rs_dir.exists():
        for f in rs_dir.iterdir():
            if f.suffix.lower() == ".tif" and f.name.startswith("GOCI_on_Landsat_"):
                goci_on_ls = f
                break

    return landsat_tif, goci_on_ls


def ensure_tif_dirs() -> Tuple[Path, Path]:
    hr_dir = TIF_ROOT / "HR"
    lr_dir = TIF_ROOT / "LR"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    return hr_dir, lr_dir


def ensure_npy_dirs() -> Tuple[Path, Path]:
    hr_dir = NPY_ROOT / "HR"
    lr_dir = NPY_ROOT / "LR"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    return hr_dir, lr_dir


# 兼容旧调用名（保留签名但忽略 scene）
def ensure_dirs(scene: str) -> Tuple[Path, Path]:
    return ensure_tif_dirs()


def read_dims(path: Path) -> Tuple[int, int, int]:
    with rasterio.open(path) as ds:
        return ds.count, ds.height, ds.width


def all_finite(a: np.ndarray) -> bool:
    return np.isfinite(a).all()


def save_patch_tif(dst_path: Path, arr: np.ndarray, like_ds: rasterio.DatasetReader, window: Window) -> None:
    # arr: (C, H, W) float32
    profile = like_ds.profile.copy()
    profile.update({
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": arr.shape[0],
        "dtype": "float32",
        "transform": like_ds.window_transform(window),
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 2,
    })
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32))


def main():
    scenes = [d for d in sorted(OUTPUT_ROOT.iterdir()) if d.is_dir()]
    if not scenes:
        print(f"[WARN] 未在 {OUTPUT_ROOT} 发现场景目录")
        return

    print(f"[INFO] 场景数量: {len(scenes)}  输出根目录: {PATCH_ROOT}")

    for sdir in scenes:
        landsat_tif, goci_on_ls = find_scene_inputs(sdir)
        print(f"\n[SCENE] {sdir.name}")
        print(f"  HR(Landsat): {landsat_tif if landsat_tif else '未找到'}")
        print(f"  LR(GOCI_on_Landsat.tif): {goci_on_ls if goci_on_ls else '未找到'}")

        if landsat_tif is None:
            # 兜底到 resampled 目录（通常不会有）
            cand = list((RESAMPLED_ROOT / sdir.name).glob("*_TOA_RAD_B*.tif"))
            if cand:
                landsat_tif = cand[0]
                print(f"  [INFO] HR 改用 resampled 目录: {landsat_tif}")

        if landsat_tif is None:
            print("  [SKIP] 未找到 HR Landsat TIF")
            continue

        if goci_on_ls is None:
            print("  [SKIP] 未找到 LR 对齐影像 (GOCI_on_Landsat_*.tif)，本脚本不使用 NC 文件")
            continue

        with rasterio.open(landsat_tif) as ds_hr:
            H = ds_hr.height
            W = ds_hr.width
            C_hr = ds_hr.count

            # 打开 LR tif（必须与 HR 尺寸一致）
            ds_lr_handle = rasterio.open(goci_on_ls)
            if ds_lr_handle.height != H or ds_lr_handle.width != W:
                print("  [ERR] GOCI_on_Landsat 尺寸与 HR 不一致，跳过场景")
                ds_lr_handle.close()
                continue

            # 输出目录（TIF 按场景分，NPY 全局 HR/LR）
            hr_dir, lr_dir = ensure_dirs(sdir.name)
            hr_npy_dir, lr_npy_dir = ensure_npy_dirs()

            # 遍历窗口
            n_saved = 0
            for r0 in range(0, H - PATCH_SIZE + 1, STRIDE):
                for c0 in range(0, W - PATCH_SIZE + 1, STRIDE):
                    win = Window.from_slices((r0, r0 + PATCH_SIZE), (c0, c0 + PATCH_SIZE))

                    # 读取 HR patch
                    hr_patch = ds_hr.read(window=win).astype(np.float32)  # (C_hr, 64, 64)
                    if ds_hr.nodata is not None:
                        hr_patch = np.where(hr_patch == ds_hr.nodata, np.nan, hr_patch)

                    # 读取 LR patch
                    lr_patch = ds_lr_handle.read(window=win).astype(np.float32)
                    if ds_lr_handle.nodata is not None:
                        lr_patch = np.where(lr_patch == ds_lr_handle.nodata, np.nan, lr_patch)

                    # NaN 过滤（任意位置存在 NaN 则丢弃）
                    if not (all_finite(hr_patch) and all_finite(lr_patch)):
                        continue

                    # 文件名与路径
                    idx = f"r{r0:05d}_c{c0:05d}"
                    hr_tif = hr_dir / f"HR_{sdir.name}_{idx}.tif"
                    lr_tif = lr_dir / f"LR_{sdir.name}_{idx}.tif"
                    hr_npy = hr_npy_dir / f"HR_{sdir.name}_{idx}.npy"
                    lr_npy = lr_npy_dir / f"LR_{sdir.name}_{idx}.npy"

                    # 保存 TIF（使用 HR 的窗口 transform）
                    save_patch_tif(hr_tif, hr_patch, ds_hr, win)
                    # LR 使用其自身窗口 transform
                    save_patch_tif(lr_tif, lr_patch, ds_lr_handle, win)

                    # 保存 NPY（C,H,W）
                    np.save(hr_npy, hr_patch)
                    np.save(lr_npy, lr_patch)

                    n_saved += 1

            if ds_lr_handle is not None:
                ds_lr_handle.close()

            # 总结输出（全局目录）
            print(f"  [OK] 保存 patches: {n_saved} 对 -> TIF:{TIF_ROOT}  NPY:{NPY_ROOT}")

            print(f"  [OK] 保存 patches: {n_saved} 对 -> {PATCH_ROOT / sdir.name}")


if __name__ == "__main__":
    main()
