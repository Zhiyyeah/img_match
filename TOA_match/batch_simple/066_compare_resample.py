#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比配准后（同网格） Landsat 与 GOCI 重采样影像的每个波段影像 & 直方图。
输出：batch_resampled/{scene}/HIST_COMPARE_B{idx}_{scene}.png

假设：
- Landsat 文件位于 batch_outputs/{scene}/ 目录，名称含 '_TOA_RAD_B'（多波段）
- GOCI 重采样文件位于 batch_resampled/{scene}/，名称以 'GOCI_on_Landsat_' 开头
- 两者具有相同空间尺寸，并且波段顺序一一对应（例如 5 个波段）

依赖：numpy, rasterio, matplotlib
"""
from __future__ import annotations
import os
from pathlib import Path
from tkinter import font
import numpy as np
import rasterio
from rasterio.transform import xy as transform_xy
from rasterio.warp import transform as warp_transform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# 全局字体配置 —— 把图中所有字体整体调大一点
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 14,
    'figure.titlesize': 18,
})

BATCH_OUTPUTS = Path("batch_outputs")
BATCH_RESAMPLED = Path("batch_resampled")

# 如果需要标注波长，可按重采样脚本中的顺序填写；若不确定可留空或修改
# 与 GOCI_BANDS (443,490,555,660,865) 对应时：
WAVELENGTHS = ["443nm","490nm","555nm","660nm","865nm"]  # 若数量不符会自动忽略

PCT_LOW, PCT_HIGH = 2, 98   # 影像显示拉伸百分位
MAX_SCENES = None            # 设为整数可限制处理场景数；None 处理全部
COLORBAR_UNIT = "W m$^{-2}$ sr$^{-1}$ μm$^{-1}$"  # Radiance 单位（可按实际修改）

def find_landsat_tif(scene_dir: Path) -> Path | None:
    for f in scene_dir.iterdir():
        if f.suffix.lower()==".tif" and "_TOA_RAD_B" in f.name:
            return f
    return None

def find_goci_tif(resampled_dir: Path) -> Path | None:
    cands = sorted(resampled_dir.glob("GOCI_on_Landsat_*.tif"))
    return cands[0] if cands else None

def read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        arr = ds.read().astype(np.float32)
        if ds.nodata is not None:
            arr = np.where(arr == ds.nodata, np.nan, arr)
    return arr  # (B,H,W)

def stretch(img: np.ndarray, p_low=2, p_high=98):
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return np.zeros_like(img)
    lo, hi = np.percentile(finite, [p_low, p_high])
    if hi <= lo:
        return np.clip((img - lo), 0, 1)
    return np.clip((img - lo)/(hi - lo), 0, 1)

def plot_band(
    scene: str,
    b_idx: int,
    ls_band: np.ndarray,
    g_band: np.ndarray,
    out_png: Path,
    wavelength: str | None,
    extent: tuple[float, float, float, float] | None,
):
    """两幅影像上排（各自右侧竖直色标），下排直方图。两幅影像都显示 X/Y 轴。"""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # 布局：2 行 2 列；直方图跨下方整行；上排两幅图之间留一定 wspace 以便显示 Y 轴与色标
    # 图幅尺寸（宽, 高，英寸）——稍矮更紧凑
    fig = plt.figure(figsize=(13.5, 9))
    gs = gridspec.GridSpec(
        nrows=2, ncols=2,
        width_ratios=[1, 1],
        # 增大直方图的纵向高度；缩小上排两图的水平间距
        height_ratios=[1.0, 0.85],
        wspace=0.12, hspace=0.28
    )
    ax_ls_img = fig.add_subplot(gs[0, 0])
    ax_g_img  = fig.add_subplot(gs[0, 1])
    ax_hist   = fig.add_subplot(gs[1, :])

    # —— 统一显示范围（按两图合并百分位）
    finite_all = np.concatenate([
        ls_band[np.isfinite(ls_band)],
        g_band[np.isfinite(g_band)]
    ]) if (np.isfinite(ls_band).any() or np.isfinite(g_band).any()) else np.array([])

    if finite_all.size:
        lo, hi = np.percentile(finite_all, [PCT_LOW, PCT_HIGH])
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
            lo, hi = float(np.nanmin(finite_all)), float(np.nanmax(finite_all))
            if lo == hi:
                hi = lo + 1e-6
    else:
        lo, hi = 0.0, 1.0

    im_kwargs = dict(cmap="viridis", origin="upper", vmin=lo, vmax=hi)
    if extent is not None:
        im_kwargs["extent"] = extent

    # —— 遥感图（更大可视区域）
    im0 = ax_ls_img.imshow(ls_band, **im_kwargs)
    im1 = ax_g_img.imshow(g_band, **im_kwargs)

    band_title = f"B{b_idx+1}" + (f" ({wavelength})" if wavelength else "")
    ax_ls_img.set_title(f"Landsat {band_title}", fontsize=12.5, pad=6)
    ax_g_img.set_title(f"GOCI {band_title}", fontsize=12.5, pad=6)

    # 两幅图均显示 X/Y 轴标签与刻度
    if extent is not None:
        for ax in (ax_ls_img, ax_g_img):
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
    else:
        ax_ls_img.axis("off")
        ax_g_img.axis("off")

    # 竖直色标放置于右侧（各自独立），通过较小 size 与 pad 控制紧凑
    for ax, im in [(ax_ls_img, im0), (ax_g_img, im1)]:
        divider = make_axes_locatable(ax)
        # 缩小色标并让其更贴近图像，避免浪费横向空间
        cax = divider.append_axes("right", size="3.0%", pad=0.04)
        cb = fig.colorbar(im, cax=cax, orientation="vertical")
        cb.set_label(f"Radiance ({COLORBAR_UNIT})", fontsize=14, labelpad=8, rotation=90)
        cb.ax.tick_params(labelsize=8, pad=2)

    # —— 直方图（密度）
    ax_hist.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    for s in ["top", "right"]:
        ax_hist.spines[s].set_visible(False)

    finite_ls = ls_band[np.isfinite(ls_band)]
    finite_go = g_band[np.isfinite(g_band)]
    if finite_ls.size == 0 and finite_go.size == 0:
        ax_hist.text(0.5, 0.5, "两影像均全为 NaN", ha='center', va='center')
    else:
        def clip_arr(a):
            if a.size == 0:
                return a
            lo_, hi_ = np.percentile(a, [PCT_LOW, PCT_HIGH])
            if hi_ <= lo_:
                return a
            sel = a[(a >= lo_) & (a <= hi_)]
            return sel if sel.size else a

        ls_clip = clip_arr(finite_ls)
        go_clip = clip_arr(finite_go)
        merged  = np.concatenate([ls_clip, go_clip]) if (ls_clip.size and go_clip.size) else (ls_clip if ls_clip.size else go_clip)
        bins    = np.linspace(np.min(merged), np.max(merged), 80) if merged.size else 80

        if ls_clip.size:
            ax_hist.hist(ls_clip, bins=bins, density=True, alpha=0.55, color='#1f77b4', label='Landsat-8/9')
        if go_clip.size:
            ax_hist.hist(go_clip, bins=bins, density=True, alpha=0.55, color='#ff7f0e', label='GOCI-2')
        ax_hist.legend(frameon=False, fontsize=12)

    ax_hist.set_xlabel('Radiance (W m$^{-2}$ sr$^{-1}$ μm$^{-1}$)')
    ax_hist.set_ylabel('Density')
    # ax_hist.set_title('Histogram (Density)', fontsize=12)

    # —— 总标题与边距
    # 提高总标题的位置，同时把子图整体下移，避免与子图标题重叠
    fig.suptitle(
        f"{scene}  Band {b_idx+1}" + (f" / {wavelength}" if wavelength else ""),
        fontsize=13, y=0.985
    )
    # 收紧左右边距以增大有效绘图宽度；降低 top 使上排标题与总标题不重合
    fig.subplots_adjust(left=0.055, right=0.99, top=0.90, bottom=0.085)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def main():
    scenes = [d.name for d in BATCH_RESAMPLED.iterdir() if d.is_dir()]
    scenes.sort()
    if MAX_SCENES is not None:
        scenes = scenes[:MAX_SCENES]
    if not scenes:
        print("未找到任何场景。")
        return

    print(f"场景数量: {len(scenes)}")
    for scene in scenes:
        res_dir = BATCH_RESAMPLED / scene
        out_dir = res_dir  # 输出在同目录
        goci_tif = find_goci_tif(res_dir)
        if goci_tif is None:
            print(f"[SKIP] {scene} 缺少 GOCI 重采样 TIF")
            continue
        ls_dir = BATCH_OUTPUTS / scene
        ls_tif = find_landsat_tif(ls_dir)
        if ls_tif is None:
            print(f"[SKIP] {scene} 缺少 Landsat TIF")
            continue 

        print(f"[SCENE] {scene}")
        # 读取并保留空间参考用于计算经纬度 extent
        try:
            with rasterio.open(ls_tif) as ds_ls:
                ls_arr = ds_ls.read().astype(np.float32)
                if ds_ls.nodata is not None:
                    ls_arr = np.where(ls_arr == ds_ls.nodata, np.nan, ls_arr)
                ls_transform = ds_ls.transform
                ls_crs = ds_ls.crs
                height, width = ds_ls.height, ds_ls.width
            with rasterio.open(goci_tif) as ds_go:
                g_arr = ds_go.read().astype(np.float32)
                if ds_go.nodata is not None:
                    g_arr = np.where(g_arr == ds_go.nodata, np.nan, g_arr)
        except Exception as e:
            print(f"  读取失败: {e}")
            continue

        # 计算四角经纬度 extent
        extent = None
        try:
            # 像素角点 (行,列)
            corners = [
                (0, 0),
                (0, width - 1),
                (height - 1, 0),
                (height - 1, width - 1),
            ]
            xs, ys = zip(*[transform_xy(ls_transform, r, c, offset='ul') for r, c in corners])
            if ls_crs is not None:
                lon, lat = warp_transform(ls_crs, 'EPSG:4326', xs, ys)
            else:  # 无 CRS 情况直接按原值
                lon, lat = xs, ys
            lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
            lat_min, lat_max = float(np.min(lat)), float(np.max(lat))
            extent = (lon_min, lon_max, lat_max, lat_min)  # 注意 origin='upper'
        except Exception as e:
            print(f"  [WARN] 计算经纬度 extent 失败: {e}")
            extent = None

        if ls_arr.shape != g_arr.shape:
            print(f"  尺寸不匹配 Landsat{ls_arr.shape} vs GOCI{g_arr.shape}，尝试按最小波段数对齐。")
        B = min(ls_arr.shape[0], g_arr.shape[0])
        for b in range(B):
            wav = WAVELENGTHS[b] if b < len(WAVELENGTHS) else None
            out_png = out_dir / f"HIST_COMPARE_B{b+1}_{scene}.png"
            plot_band(scene, b, ls_arr[b], g_arr[b], out_png, wav, extent)
        print(f"  完成 {B} 个波段。")

if __name__ == "__main__":
    main()
