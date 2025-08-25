#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GOCI vs Landsat（不重采样）对比与直方图
- 读取裁剪后的 GOCI 子集（含 navigation_data、geophysical_data、可选 mask/inside_mask）
- 读取 Landsat 5 波段 TOA radiance 多波段 TIF
- 可视化：
    * 两边都用经纬度网格 pcolormesh（地理形状不扁）
    * 颜色范围：两者联合的 1–99% 分位
- 直方图（严格按你给的旧代码风格）：
    * 两者联合 1–99% 分位为范围
    * bins=60、density=True、两组柱叠加显示
- 输出：
    figs_compare/compare_geo_{λ}nm.png
    figs_hist/hist_band_{λ}nm.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import rasterio
from pyproj import Transformer

# ======== 路径（按需修改） ========
GOCI_NC = "./goci_subset_5bands.nc"
LANDSAT_TIF = "SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif"
OUT_DIR_COMPARE = "figs_compare"
OUT_DIR_HIST = "figs_hist"
os.makedirs(OUT_DIR_COMPARE, exist_ok=True)
os.makedirs(OUT_DIR_HIST, exist_ok=True)
# =================================

# GOCI 波段名映射（与裁剪保持一致）
GOCI_BANDS = {
    443: "L_TOA_443",
    490: "L_TOA_490",
    555: "L_TOA_555",
    660: "L_TOA_660",
    865: "L_TOA_865",
}
LANDSAT_WAVELENGTHS = [443, 483, 561, 655, 865]
PAIRING = {1: 443, 2: 490, 3: 555, 4: 660, 5: 865}

def robust_vmin_vmax(a_list, q=(1, 99)):
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
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmin >= vmax:
        vmin = float(np.nanmin(allv))
        vmax = float(np.nanmax(allv))
    return float(vmin), float(vmax)

# -------- 读取 GOCI 子集（保持掩膜，兜底用 _FillValue -> NaN） --------
def read_goci_nc(nc_path):
    with Dataset(nc_path, "r") as ds:
        nav = ds["navigation_data"] if "navigation_data" in ds.groups else ds
        geo = ds["geophysical_data"] if "geophysical_data" in ds.groups else ds

        lat = np.array(nav["latitude"][:], dtype=np.float32)
        lon = np.array(nav["longitude"][:], dtype=np.float32)

        inside_mask = None
        if "mask" in ds.groups and "inside_mask" in ds["mask"].variables:
            inside_mask = ds["mask"]["inside_mask"][:].astype(bool)

        bands = {}
        for wl, name in GOCI_BANDS.items():
            var = geo[name]
            data = var[:]  # 可能是 MaskedArray（且已应用 scale_factor）
            if np.ma.isMaskedArray(data):
                arr = data.filled(np.nan).astype(np.float32)
            else:
                arr = np.array(data, dtype=np.float32)
            if "_FillValue" in var.ncattrs():
                fv = float(np.array(var.getncattr("_FillValue")).ravel()[0])
                arr = np.where(arr == fv, np.nan, arr)
            bands[wl] = arr

    return lat, lon, bands, inside_mask

# -------- 读取 Landsat 并生成经纬度像元中心网格（不重采样） --------
def read_landsat_lonlat_grid(tif_path):
    with rasterio.open(tif_path) as ds:
        stack = ds.read().astype(np.float32)  # (5,H,W)
        if ds.nodata is not None:
            stack = np.where(stack == ds.nodata, np.nan, stack)

        H, W = ds.height, ds.width
        T = ds.transform
        crs = ds.crs

        cols = np.arange(W)
        rows = np.arange(H)
        cgrid, rgrid = np.meshgrid(cols, rows)

        # 计算像元中心投影坐标（适配任意仿射变换，不假设北向上）
        x_proj = T.c + T.a*(cgrid + 0.5) + T.b*(rgrid + 0.5)
        y_proj = T.f + T.d*(cgrid + 0.5) + T.e*(rgrid + 0.5)

        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x_proj, y_proj)

    return stack, lon.astype(np.float64), lat.astype(np.float64)

# -------- 并排绘图（两边都用 pcolormesh） --------
def plot_side_by_side_geo(L_lon, L_lat, L_arr, G_lon, G_lat, G_arr, wl_landsat, wl_goci, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    vmin, vmax = robust_vmin_vmax([L_arr, G_arr], q=(1, 99))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)

    plt.figure(figsize=(12, 5), dpi=150)

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title(f"Landsat ~{wl_landsat} nm (radiance)")
    im1 = ax1.pcolormesh(L_lon, L_lat, L_arr, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Latitude (°N)")
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("W m$^{-2}$ sr$^{-1}$ µm$^{-1}$")

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title(f"GOCI ~{wl_goci} nm (radiance)")
    im2 = ax2.pcolormesh(G_lon, G_lat, G_arr, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("W m$^{-2}$ sr$^{-1}$ µm$^{-1}$")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

# -------- 直方图（严格按你给的旧代码实现） --------
def plot_hist_compare(goci_arr, landsat_arr, wl_goci, out_path):
    """
    直方图对比：相同范围（两者联合的1-99%分位），density=True
    —— 完全照你给的旧版本实现（只在这里参考旧代码，其他部分不参考）
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
    return {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()), "n": int(a.size)}

def main():
    if not os.path.exists(GOCI_NC):
        raise FileNotFoundError(f"找不到 GOCI NC：{GOCI_NC}")
    if not os.path.exists(LANDSAT_TIF):
        raise FileNotFoundError(f"找不到 Landsat TIF：{LANDSAT_TIF}")

    print("[INFO] 读取 GOCI 子集 ...")
    g_lat, g_lon, g_bands, g_inside = read_goci_nc(GOCI_NC)

    print("[INFO] 读取 Landsat 并生成经纬度网格 ...")
    L_stack, L_lon, L_lat = read_landsat_lonlat_grid(LANDSAT_TIF)  # (5,H,W), (H,W), (H,W)

    for bidx in range(1, 6):
        wl_goci = PAIRING[bidx]
        wl_landsat = LANDSAT_WAVELENGTHS[bidx - 1]

        g_arr = g_bands[wl_goci]
        # 统计/直方图：只用多边形内部（若有 mask）
        if g_inside is not None:
            g_arr_stats = np.where(g_inside, g_arr, np.nan)
            g_show = np.where(g_inside, g_arr, np.nan)
        else:
            g_arr_stats = g_arr
            g_show = g_arr

        L_arr = L_stack[bidx - 1]

        # 并排可视化（两边都按真实经纬度网格）
        out_compare = os.path.join(OUT_DIR_COMPARE, f"compare_geo_{wl_goci}nm.png")
        plot_side_by_side_geo(L_lon, L_lat, L_arr, g_lon, g_lat, g_show, wl_landsat, wl_goci, out_compare)

        # 直方图（严格沿用你旧代码的画法）
        out_hist = os.path.join(OUT_DIR_HIST, f"hist_band_{wl_goci}nm.png")
        plot_hist_compare(g_arr_stats, L_arr, wl_goci, out_hist)

        # 打印统计
        sL = quick_stats(L_arr)
        sG = quick_stats(g_arr_stats)
        print(f"[STATS] ~{wl_goci}nm | Landsat: n={sL['n']}, min={sL['min']:.6g}, max={sL['max']:.6g}, mean={sL['mean']:.6g} | "
              f"GOCI(in polygon): n={sG['n']}, min={sG['min']:.6g}, max={sG['max']:.6g}, mean={sG['mean']:.6g}")

    print("[DONE] 对比图输出目录：", OUT_DIR_COMPARE)
    print("[DONE] 直方图输出目录：", OUT_DIR_HIST)

if __name__ == "__main__":
    main()
