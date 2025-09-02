#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
031: 上采样结果对比出图（2列×5行）

- 遍历 batch_resampled/{scene}（含 GOCI_on_Landsat_*.tif）
- 同时读取 batch_masked/{scene} 或 batch_outputs/{scene} 下的 HR Landsat TIF
- 生成每个场景的对比图：左列 HR、右列 上采样（共5个波段，行顺序 B1..B5）
- 使用缩略图（双线性降采样）以降低内存占用

输出：batch_resampled/{scene}/COMPARE_2x5_{scene}.png
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt


OUTPUT_ROOT = Path("batch_outputs")
RESAMPLED_ROOT = Path("batch_resampled")
MASKED_ROOT = Path("batch_masked")

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
        # 找到上采样后的 TIF（GOCI_on_Landsat_*.tif）
        goci_on_ls = None
        for f in scene_dir.iterdir():
            if f.suffix.lower() == ".tif" and f.name.startswith("GOCI_on_Landsat_"):
                goci_on_ls = f
                break
        # 找到对应 HR（优先水体掩膜后）
        landsat_tif = None
        masked_scene = MASKED_ROOT / scene_dir.name
        if masked_scene.exists():
            cands = [
                f for f in masked_scene.iterdir()
                if f.suffix.lower() == ".tif" and "_TOA_RAD_B" in f.name and f.name.endswith("_only_water.tif")
            ]
            if cands:
                cand = sorted(cands, key=lambda p: p.name)[0]
                # 验证是否包含有效像素
                try:
                    with rasterio.open(cand) as _ds:
                        scale_h = min(_ds.height, 256)
                        scale_w = int(round(_ds.width * scale_h / max(1, _ds.height)))
                        arr = _ds.read(
                            1,
                            out_shape=(1, scale_h, scale_w),
                            resampling=Resampling.bilinear,
                        ).astype(np.float32)
                        if _ds.nodata is not None:
                            arr = np.where(arr == _ds.nodata, np.nan, arr)
                        if np.isfinite(arr).any():
                            landsat_tif = cand
                except Exception:
                    pass
        # 回退到 batch_outputs/{scene}
        if landsat_tif is None:
            ls_dir = OUTPUT_ROOT / scene_dir.name
            if ls_dir.exists():
                for f in ls_dir.iterdir():
                    if f.suffix.lower() == ".tif" and "_TOA_RAD_B" in f.name:
                        landsat_tif = f
                        break
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
    if ds.nodata is not None:
        arr = np.where(arr == ds.nodata, np.nan, arr)
    return arr


def robust_vmin_vmax(a_list):
    vals_list = []
    for x in a_list:
        if x is None:
            continue
        valid = x[np.isfinite(x)]
        if valid.size:
            vals_list.append(valid.ravel())
    if not vals_list:
        return 0.0, 1.0
    vals = np.concatenate(vals_list)
    vmin = np.nanpercentile(vals, 2)
    vmax = np.nanpercentile(vals, 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if vmin == vmax:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def main():
    pairs = discover_pairs()
    if not pairs:
        print("[WARN] 未找到可对比的场景（缺少 resampled 或 landsat 文件）")
        return
    print(f"[INFO] 待对比场景数: {len(pairs)}")

    for scene, ls_tif, up_tif in pairs:
        print(f"\n[SCENE] {scene}\n  HR: {ls_tif}\n  UP: {up_tif}")
        out_png = RESAMPLED_ROOT / scene / f"COMPARE_2x5_{scene}.png"

        with rasterio.open(ls_tif) as ds_hr, rasterio.open(up_tif) as ds_up:
            if ds_hr.width != ds_up.width or ds_hr.height != ds_up.height:
                print("  [ERR] HR/UP 尺寸不一致，跳过")
                continue
            # 缩略图读取
            hr = read_thumb(ds_hr, THUMB_MAX_H)
            up = read_thumb(ds_up, THUMB_MAX_H)

        bands = min(hr.shape[0], up.shape[0], 5)
        fig, axes = plt.subplots(5, 2, figsize=(10, 16), constrained_layout=True)

        # 统一配色范围（用每行的 HR/UP 自适应范围）
        for b in range(5):
            ax_left = axes[b, 0]
            ax_right = axes[b, 1]
            if b < bands:
                hr_b = hr[b]
                up_b = up[b]
                vmin, vmax = robust_vmin_vmax([hr_b, up_b])

                im = ax_left.imshow(hr_b, cmap="viridis", vmin=vmin, vmax=vmax)
                ax_left.set_title(f"HR {BAND_LABELS[b]}")
                ax_left.axis("off")
                fig.colorbar(im, ax=ax_left, fraction=0.046, pad=0.02)

                im = ax_right.imshow(up_b, cmap="viridis", vmin=vmin, vmax=vmax)
                ax_right.set_title(f"UP {BAND_LABELS[b]}")
                ax_right.axis("off")
                fig.colorbar(im, ax=ax_right, fraction=0.046, pad=0.02)
            else:
                ax_left.axis("off")
                ax_right.axis("off")

        fig.suptitle(f"Scene: {scene}  (2x5 对比，缩略图高≤{THUMB_MAX_H})", fontsize=12)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  [OK] 保存对比图 -> {out_png}")


if __name__ == "__main__":
    main()

