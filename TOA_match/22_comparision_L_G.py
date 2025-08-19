#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GOCI vs Landsat（不重采样）对比与直方图
- 输入：
    1) goci_subset_5bands.nc  （裁剪到 Landsat 范围的 GOCI，含 lat/lon 和 L_TOA_443/490/555/660/865）
    2) SR_Imagery/..._TOA_RAD_B1-2-3-4-5.tif  （Landsat 5 波段辐亮度，目标参考）
- 处理：
    * 不做重采样：分别读取两者对应波段
    * 生成每个波段的并排可视化（各自网格）
    * 用相同的动态范围（1-99%分位）叠加直方图比较两者分布
- 输出：
    figs/compare_band_{λ}nm.png
    figs/hist_band_{λ}nm.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import rasterio

# ======== 路径（按需修改） ========
GOCI_NC = "goci_subset_5bands.nc"
LANDSAT_TIF = "SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif"
OUT_DIR_FIG = "figs"
# =================================

# GOCI 波段名映射（裁剪NC里保留的是3/4/6/8/12）
GOCI_BANDS = {
    443: "L_TOA_443",  # idx 3
    490: "L_TOA_490",  # idx 4
    555: "L_TOA_555",  # idx 6
    660: "L_TOA_660",  # idx 8
    865: "L_TOA_865",  # idx 12
}
# Landsat 5波段顺序（B1..B5）对应大致中心波长
LANDSAT_WAVELENGTHS = [443, 483, 561, 655, 865]
# 配对（用最接近的GOCI中心波长与之比较）
PAIRING = {
    1: 443,  # Landsat B1 ~443nm
    2: 490,  # B2 ~483nm ≈ GOCI 490nm
    3: 555,  # B3 ~561nm ≈ GOCI 555nm
    4: 660,  # B4 ~655nm ≈ GOCI 660nm
    5: 865,  # B5 ~865nm
}

def robust_vmin_vmax(a_list, q=(1, 99)):
    """
    从多个数组联合估计用于可视化/直方图的共同范围（忽略NaN）
    a_list: [arr1, arr2, ...]
    """
    vals = []
    for a in a_list:
        if a is None:
            continue
        aa = np.asarray(a)
        aa = aa[np.isfinite(aa)]
        if aa.size:
            vals.append(aa)
    if not vals:
        return 0.0, 1.0
    allv = np.concatenate(vals)
    vmin = np.percentile(allv, q[0])
    vmax = np.percentile(allv, q[1])
    if vmin >= vmax:
        vmin = float(np.nanmin(allv))
        vmax = float(np.nanmax(allv))
    return float(vmin), float(vmax)

def read_goci_nc(nc_path):
    with Dataset(nc_path, "r") as nc:
        # lat/lon不用于坐标配准，仅保留参考
        lat = nc["latitude"][:]
        lon = nc["longitude"][:]
        bands = {}
        for wl, name in GOCI_BANDS.items():
            if name not in nc.variables:
                raise KeyError(f"缺少变量 {name}")
            var = nc.variables[name]
            fv = getattr(var, "_FillValue", None)
            arr = var[:].astype(np.float32)
            if fv is not None:
                arr = np.where(arr == fv, np.nan, arr)
            bands[wl] = arr
    return lat, lon, bands

def read_landsat_tif(tif_path):
    with rasterio.open(tif_path) as ds:
        stack = ds.read().astype(np.float32)  # (5, H, W)
        # 将 nodata -> NaN
        nodata = ds.nodata
        if nodata is not None:
            stack = np.where(stack == nodata, np.nan, stack)
    return stack

def plot_side_by_side(goci_arr, landsat_arr, wl_goci, wl_landsat, out_path):
    """
    并排对比（各自网格，不重采样）
    - 动态范围：两者联合的1-99%分位，保证颜色尺度一致
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vmin, vmax = robust_vmin_vmax([goci_arr, landsat_arr], q=(1, 99))

    plt.figure(figsize=(10, 4.2), dpi=150)

    plt.subplot(1, 2, 1)
    plt.title(f"Landsat ~{wl_landsat} nm (radiance)")
    im1 = plt.imshow(landsat_arr, vmin=vmin, vmax=vmax)
    plt.axis("off")
    cbar = plt.colorbar(im1, fraction=0.046, pad=0.04)
    cbar.set_label("W m$^{-2}$ sr$^{-1}$ µm$^{-1}$")

    plt.subplot(1, 2, 2)
    plt.title(f"GOCI ~{wl_goci} nm (radiance)")
    im2 = plt.imshow(goci_arr, vmin=vmin, vmax=vmax)
    plt.axis("off")
    cbar2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
    cbar2.set_label("W m$^{-2}$ sr$^{-1}$ µm$^{-1}$")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_hist_compare(goci_arr, landsat_arr, wl_goci, out_path):
    """
    直方图对比：相同范围（两者联合的1-99%分位），density=True
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    l = landsat_arr[np.isfinite(landsat_arr)]
    g = goci_arr[np.isfinite(goci_arr)]
    if l.size == 0 or g.size == 0:
        print(f"[WARN] ~{wl_goci}nm 有效像元为空，跳过直方图。")
        return

    vmin, vmax = robust_vmin_vmax([l, g], q=(1, 99))
    bins = 60

    plt.figure(figsize=(6.4, 4.2), dpi=150)
    plt.hist(l, bins=bins, range=(vmin, vmax), alpha=0.6, label="Landsat", density=True)
    plt.hist(g, bins=bins, range=(vmin, vmax), alpha=0.6, label="GOCI", density=True)
    plt.title(f"Histogram ~{wl_goci} nm (radiance)")
    plt.xlabel("Radiance (W m$^{-2}$ sr$^{-1}$ µm$^{-1}$)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def quick_stats(arr):
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "n": 0}
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "n": int(a.size),
    }

def main():
    if not os.path.exists(GOCI_NC):
        raise FileNotFoundError(f"找不到 GOCI NC：{GOCI_NC}")
    if not os.path.exists(LANDSAT_TIF):
        raise FileNotFoundError(f"找不到 Landsat TIF：{LANDSAT_TIF}")

    os.makedirs(OUT_DIR_FIG, exist_ok=True)

    # 读取数据
    print("[INFO] 读取 GOCI NC ...")
    _, _, goci_bands = read_goci_nc(GOCI_NC)
    print("[INFO] 读取 Landsat TIF ...")
    landsat_stack = read_landsat_tif(LANDSAT_TIF)  # shape: (5,H,W)

    # 每个波段：并排 + 直方图
    for bidx in range(1, 6):
        wl_goci = PAIRING[bidx]
        wl_landsat = [443, 483, 561, 655, 865][bidx-1]

        g_arr = goci_bands[wl_goci]
        L_arr = landsat_stack[bidx-1]

        # 统计信息
        sL = quick_stats(L_arr)
        sG = quick_stats(g_arr)
        print(f"[STATS] ~{wl_goci}nm | Landsat: n={sL['n']}, min={sL['min']:.6g}, max={sL['max']:.6g}, mean={sL['mean']:.6g} | "
              f"GOCI: n={sG['n']}, min={sG['min']:.6g}, max={sG['max']:.6g}, mean={sG['mean']:.6g}")

        # 并排
        side_path = os.path.join(OUT_DIR_FIG, f"compare_band_{wl_goci}nm.png")
        plot_side_by_side(g_arr, L_arr, wl_goci, wl_landsat, side_path)

        # 直方图
        hist_path = os.path.join(OUT_DIR_FIG, f"hist_band_{wl_goci}nm.png")
        plot_hist_compare(g_arr, L_arr, wl_goci, hist_path)

    print("[DONE] 已输出所有对比图与直方图到：", OUT_DIR_FIG)

if __name__ == "__main__":
    main()
