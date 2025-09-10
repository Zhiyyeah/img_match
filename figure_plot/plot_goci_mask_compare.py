#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 GOCI 去陆地掩膜前 (original) 与去陆地后 (water-only) 的单波段对比图。

参考：根目录下的 Landsat 对比脚本 `plot_landsat_mask_compare.py` 思路，
但本脚本面向弯曲网格的 GOCI NetCDF。

使用说明（简单模式）：
  1. 在 main() 里修改 `scene` 与 `band` (可填波长 443/490/555/660/865 *或* 变量名如 L_TOA_443)；
  2. 运行脚本；
  3. 程序在以下位置自动寻找：
       原始(或被覆盖的) subset NC : batch_outputs/{scene}/*_subset_footprint.nc
       备份的原始 NC (.bak)       : 若存在 *.nc.bak 则视作 pre-mask 原始文件
       去陆地后的 water-only NC  : batch_masked/{scene}/*_subset_footprint_only_water.nc
  4. 输出对比图: batch_masked/{scene}/COMPARE_GOCI_band_<label>_{scene}.png

依赖：netCDF4, numpy, matplotlib
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# ------------------ 可调参数（也可通过 main 直接修改） ------------------
BAND_WAVELENGTH_MAP = {
    412: "L_TOA_412", 443: "L_TOA_443", 490: "L_TOA_490", 555: "L_TOA_555",
    561: "L_TOA_555", 660: "L_TOA_660", 665: "L_TOA_660", 680: "L_TOA_680",  # 680 如不存在会回退
    865: "L_TOA_865"
}
# 候选顺序（当用户不指定时）
DEFAULT_CANDIDATES = ("L_TOA_443", "L_TOA_490", "L_TOA_555", "L_TOA_660", "L_TOA_865")
# ---------------------------------------------------------------------


def _find_original_and_masked(scene: str):
    """返回 (orig_nc, masked_nc, orig_label)。

    优先使用 batch_outputs/{scene} 下 *_subset_footprint.nc.bak 作为原始；
    若存在 .bak 则当前 *_subset_footprint.nc 可能已被覆盖成 water-only；
    若无 .bak，则直接把 *_subset_footprint.nc 当作原始。
    masked 版本在 batch_masked/{scene}/*_subset_footprint_only_water.nc。
    """
    out_dir = Path("batch_outputs") / scene
    mask_dir = Path("batch_masked") / scene

    orig_nc = None
    bak_label = False
    if out_dir.exists():
        for f in out_dir.iterdir():
            if f.suffix == ".nc" and f.name.endswith("_subset_footprint.nc"):
                # 检查 .bak
                bak = f.with_suffix(f.suffix + ".bak")
                if bak.exists():
                    orig_nc = bak
                    bak_label = True
                else:
                    orig_nc = f
                break
    masked_nc = None
    if mask_dir.exists():
        for f in mask_dir.iterdir():
            if f.suffix == ".nc" and f.name.endswith("_subset_footprint_only_water.nc"):
                masked_nc = f
                break

    return orig_nc, masked_nc, ("original resampled" if bak_label else "original")


def _select_variable(ds: Dataset, band: str | int | None):
    """根据 band（波长或变量名）选择一个 GOCI geophysical_data 变量，返回 (var_name, ndarray)。

    - 若 band 为 int：映射到 BAND_WAVELENGTH_MAP；
    - 若 band 为 str 且在变量中存在：直接使用；
    - 若 band 为 None：按照 DEFAULT_CANDIDATES 依次寻找；
    - 返回数组为 float32，并将 FillValue / MaskedArray -> np.nan。
    """
    if "geophysical_data" in ds.groups:
        g = ds["geophysical_data"]
    else:
        g = ds

    target_name = None
    if isinstance(band, int):
        target_name = BAND_WAVELENGTH_MAP.get(band)
        if target_name is None:
            raise ValueError(f"未识别的波长: {band}")
    elif isinstance(band, str) and band:
        if band in g.variables:
            target_name = band
        else:
            raise KeyError(f"变量 {band} 不存在于 geophysical_data")
    else:  # band is None
        for cand in DEFAULT_CANDIDATES:
            if cand in g.variables:
                target_name = cand
                break
    if target_name is None:
        raise KeyError("未找到任何可用的 GOCI 波段变量")

    var = g[target_name]
    arr = var[:]
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    else:
        arr = np.array(arr)
    arr = arr.astype(np.float32)
    # 处理 FillValue
    if "_FillValue" in var.ncattrs():
        try:
            fv = float(np.array(var.getncattr("_FillValue")).ravel()[0])
            arr = np.where(arr == fv, np.nan, arr)
        except Exception:
            pass
    return target_name, arr


def _read_latlon(ds: Dataset):
    """返回 (lat, lon) 2D 数组 (float64)。兼容 navigation_data 分组或根变量。"""
    if "navigation_data" in ds.groups:
        nav = ds["navigation_data"]
        lat = np.array(nav["latitude"][:], dtype=np.float64)
        lon = np.array(nav["longitude"][:], dtype=np.float64)
    else:  # 兼容无分组结构
        lat = np.array(ds["latitude"][:], dtype=np.float64)
        lon = np.array(ds["longitude"][:], dtype=np.float64)
    return lat, lon


def _robust_limits(arr_list):
    vals = []
    for a in arr_list:
        if a is None:
            continue
        v = a[np.isfinite(a)]
        if v.size:
            vals.append(v.ravel())
    if not vals:
        return 0.0, 1.0
    allv = np.concatenate(vals)
    vmin = np.nanpercentile(allv, 2)
    vmax = np.nanpercentile(allv, 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(allv)), float(np.nanmax(allv))
        if vmin == vmax:
            vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def plot_goci_mask_compare(scene: str, band: int | str | None = 490, out_dir: Path | None = None):
    orig_nc, masked_nc, orig_label = _find_original_and_masked(scene)
    if orig_nc is None:
        raise FileNotFoundError(f"未找到原始 subset NC: batch_outputs/{scene}/*_subset_footprint.nc(.bak)")
    if masked_nc is None:
        raise FileNotFoundError(f"未找到 masked NC: batch_masked/{scene}/*_subset_footprint_only_water.nc")

    with Dataset(orig_nc, 'r') as d0, Dataset(masked_nc, 'r') as d1:
        lat0, lon0 = _read_latlon(d0)
        lat1, lon1 = _read_latlon(d1)
        var_name0, a0 = _select_variable(d0, band)
        # masked 文件中的变量名应与原始一致（可再次查找以防缺失）
        try:
            var_name1, a1 = _select_variable(d1, var_name0)
        except Exception:
            var_name1, a1 = _select_variable(d1, band)

    vmin, vmax = _robust_limits([a0, a1])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)
    vmin, vmax = 55, 100  # 490nm typical range

    fig = plt.figure(figsize=(12, 5), dpi=150, constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.pcolormesh(lon0, lat0, a0, shading='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    ax1.set_title(f"GOCI {var_name0} {orig_label}")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.grid(True, alpha=0.3)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
    cb1.set_label('Radiance (W m$^{-2}$ sr$^{-1}$ $\u03bcm^{-1}$)', fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.pcolormesh(lon1, lat1, a1, shading='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    ax2.set_title(f"GOCI {var_name1} land masked")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.grid(True, alpha=0.3)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cb2.set_label('Radiance (W m$^{-2}$ sr$^{-1}$ $\u03bcm^{-1}$)', fontsize=9)

    label = var_name0 if isinstance(band, str) else (str(band) if band is not None else var_name0)
    if out_dir is None:
        out_dir = Path("batch_masked") / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"COMPARE_GOCI_band_{label}_{scene}.png"
    fig.suptitle(f"GOCI landmask compare ({scene})", fontsize=11)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"保存: {out_png}")


def main():
    # 在此设置场景与波段：band 可为 443 / 490 / 555 / 660 / 865 (波长) 或 'L_TOA_443' 等变量名；
    scene = "LC08_L1TP_116036_20210330_20210409_02_T1"  # 修改为你的场景名
    band = 490  # 可改为波长或变量名（None 表示自动挑选默认候选）
    try:
        plot_goci_mask_compare(scene, band)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()
