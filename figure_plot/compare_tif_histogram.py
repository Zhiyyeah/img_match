#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比较两个 GeoTIFF：生成一行三列图像 —— 左: LR, 中: HR, 右: 叠加直方图。

示例：在 main() 中设置输入路径或使用命令行参数。
"""
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt


def read_band(tif_path: Path, band: int = 1, max_h: int = 800):
    """读取 GeoTIFF 指定波段并按高度缩放到 max_h（若需要），返回 array(float32) 和元数据。"""
    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        if H > max_h:
            scale = max_h / float(H)
            out_h = max_h
            out_w = int(round(W * scale))
            arr = src.read(band, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
        else:
            arr = src.read(band).astype(np.float32)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        meta = dict(width=arr.shape[1], height=arr.shape[0])
    return arr, meta


def robust_vmin_vmax(a_list):
    vals = np.concatenate([x[np.isfinite(x)].ravel() for x in a_list if x is not None and np.isfinite(x).any()])
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(vals, 2))
    vmax = float(np.nanpercentile(vals, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    return vmin, vmax


def make_compare_figure(lr_arr, hr_arr, lr_label: str, hr_label: str, out_png: Path, cmap='viridis'):
    vmin, vmax = robust_vmin_vmax([lr_arr, hr_arr])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    im0 = axes[0].imshow(lr_arr, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(lr_label)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(False)
    c0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    c0.set_label('Radiance')

    im1 = axes[1].imshow(hr_arr, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(hr_label)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(False)
    c1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    c1.set_label('Radiance')

    # Histogram overlay
    a0 = lr_arr[np.isfinite(lr_arr)].ravel()
    a1 = hr_arr[np.isfinite(hr_arr)].ravel()
    axh = axes[2]
    bins = 256
    # use log scale for counts if distributions very skewed
    axh.hist(a0, bins=bins, alpha=0.6, density=True, label=lr_label, color='C0')
    axh.hist(a1, bins=bins, alpha=0.6, density=True, label=hr_label, color='C1')
    axh.set_title('Histogram (normalized)')
    axh.set_xlabel('Radiance')
    axh.set_ylabel('Density')
    axh.legend()

    # add simple statistics text
    def stats_text(arr):
        arrf = arr[np.isfinite(arr)]
        return f'n={arrf.size}\nmean={arrf.mean():.3g}\nstd={arrf.std():.3g}' if arrf.size>0 else 'n=0'

    txt = f"LR:\n{stats_text(lr_arr)}\n\nHR:\n{stats_text(hr_arr)}"
    axes[2].text(0.98, 0.95, txt, transform=axes[2].transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle('Compare LR vs HR and Histogram', fontsize=14)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def main():
    # 在此处直接定义输入路径和参数（按要求：在 main 中定义）
    ID="LC08_L1TP_116036_20210330_20210409_02_T1_r02304_c03840"
    lr = Path(rf"D:\Py_Code\img_match\SR_Imagery\tif\LR\LR_{ID}.tif")
    hr = Path(rf"D:\Py_Code\img_match\SR_Imagery\tif\HR\HR_{ID}.tif")
    band = 1  # 需要的波段索引（1-based）
    out_png = Path(r"d:\Py_Code\img_match\outputs\compare_example.png")

    if not lr.exists():
        print(f"LR 文件不存在: {lr}")
        return
    if not hr.exists():
        print(f"HR 文件不存在: {hr}")
        return

    lr_arr, _ = read_band(lr, band=band)
    hr_arr, _ = read_band(hr, band=band)

    res = make_compare_figure(lr_arr, hr_arr, lr_label=lr.name, hr_label=hr.name, out_png=out_png)
    print(f"Saved comparison -> {res}")


if __name__ == '__main__':
    main()
