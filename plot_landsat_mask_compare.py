#!/usr/bin/env python3
"""生成 Landsat 原始 vs 去陆地（water-only）单波段对比图。

用法：在 main() 中设置 scene 和 band_wavelength（例如 490 或 443），运行脚本。
脚本会在 `batch_outputs/{scene}` 查找含 `_TOA_RAD_B` 的多波段 TIFF，和 `batch_masked/{scene}` 下的 *_only_water.tif，
然后读取指定波段并生成左右对比图，保存到 `batch_masked/{scene}/COMPARE_band_{band}_{scene}.png`。
"""
from pathlib import Path
import rasterio
from rasterio.transform import Affine
from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt
import sys


def find_landsat_tif(scene: str) -> Path | None:
    p = Path("batch_outputs") / scene
    if not p.exists():
        return None
    for f in p.iterdir():
        if f.suffix.lower() == ".tif" and "_TOA_RAD_B" in f.name:
            return f
    return None


def find_masked_tif(scene: str) -> Path | None:
    p = Path("batch_masked") / scene
    if not p.exists():
        return None
    for f in p.iterdir():
        if f.suffix.lower() == ".tif" and f.name.endswith("_only_water.tif"):
            return f
    return None


def read_band_from_multiband_tif(tif: Path, band_idx: int, max_h: int = 1200) -> tuple[np.ndarray, dict]:
    """读取多波段 tif 中的单波段（1-based 索引），并根据 max_h 缩放到合适大小（保持比例）。
    返回 (array2d, meta) 。array 为 float32，nodata -> np.nan。
    """
    with rasterio.open(tif) as src:
        H, W = src.height, src.width
        count = src.count
        if band_idx < 1 or band_idx > count:
            raise IndexError(f"band_idx {band_idx} out of range (1..{count})")
        if H > max_h:
            scale = max_h / float(H)
            out_h = max_h
            out_w = int(round(W * scale))
            arr = src.read(band_idx, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
        else:
            arr = src.read(band_idx).astype(np.float32)
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        meta = dict(width=arr.shape[1], height=arr.shape[0])
    return arr, meta


def lonlat_grid_for_tif(tif: Path, out_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """生成与 downsampled 图像一致的 lon, lat 网格（像元中心），返回 (lon, lat) 2D 数组。"""
    out_h, out_w = out_shape
    with rasterio.open(tif) as src:
        H, W = src.height, src.width
        T: Affine = src.transform
        src_crs = src.crs
        # prepare target rows/cols (像元中心)
        if H > out_h:
            scale = out_h / float(H)
            # pick rows/cols of the downsampled grid
            rows = (np.arange(out_h) / scale).astype(np.float64)
            cols = (np.arange(out_w) / scale).astype(np.float64)
        else:
            rows = np.arange(out_h, dtype=np.float64)
            cols = np.arange(out_w, dtype=np.float64)
        cgrid, rgrid = np.meshgrid(cols, rows)
        # compute x,y of pixel centers in source CRS
        x = T.c + T.a * (cgrid + 0.5) + T.b * (rgrid + 0.5)
        y = T.f + T.d * (cgrid + 0.5) + T.e * (rgrid + 0.5)
        # transform to lon/lat
        try:
            if src_crs is None:
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
            else:
                transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x, y)
        except Exception:
            # fallback: assume already lon/lat
            lon, lat = x, y
    return lon, lat


def robust_vmin_vmax(a_list):
    vals = np.concatenate([x[np.isfinite(x)].ravel() for x in a_list if x is not None and np.isfinite(x).any()])
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(vals, 2))
    vmax = float(np.nanpercentile(vals, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    return vmin, vmax


def plot_compare(scene: str, band_wavelength: int, out_dir: Path | None = None):
    # find files
    ls = find_landsat_tif(scene)
    masked = find_masked_tif(scene)
    if ls is None:
        raise FileNotFoundError(f"未在 batch_outputs/{scene} 找到 Landsat 多波段 TIFF")
    if masked is None:
        raise FileNotFoundError(f"未在 batch_masked/{scene} 找到 *_only_water.tif")

    # map wavelength to band index for 5-band TOA_B1-2-3-4-5: typical mapping
    # B1=Blue(1), B2=Green(2), B3=Red(3), B4=NIR(4), B5=swir1(5)
    wavelength_map = {443:1, 485:1, 490:1, 561:2, 560:2, 655:3, 660:3, 865:4}
    band_idx = wavelength_map.get(band_wavelength, None)
    if band_idx is None:
        # fallback: allow user to pass band index directly
        try:
            bi = int(band_wavelength)
            band_idx = bi
        except Exception:
            raise ValueError("无法识别波段，请传入已知波长（如443,490,561,660,865）或波段索引。")

    a0, meta0 = read_band_from_multiband_tif(ls, band_idx)
    a1, meta1 = read_band_from_multiband_tif(masked, band_idx)

    vmin, vmax = robust_vmin_vmax([a0, a1])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)

    # compute lon/lat grids matching the downsampled arrays
    lon0, lat0 = lonlat_grid_for_tif(ls, a0.shape)
    lon1, lat1 = lonlat_grid_for_tif(masked, a1.shape)

    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.pcolormesh(lon0, lat0, a0, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(f"{scene} B{band_idx} ({band_wavelength}nm) (original)")
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)
    cb.set_label('Radiance (W m$^{-2}$ sr$^{-1}$ $\u03bcm^{-1}$)', fontsize=10, rotation=90, labelpad=12)

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.pcolormesh(lon1, lat1, a1, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_title(f"{scene} B{band_idx} ({band_wavelength}nm) (land masked)")
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cb2.set_label('Radiance (W m$^{-2}$ sr$^{-1}$ $\u03bcm^{-1}$)', fontsize=10, rotation=90, labelpad=12)

    if out_dir is None:
        out_dir = Path("batch_masked") / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"COMPARE_band_{band_wavelength}_{scene}.png"
    fig.suptitle(f"Landsat band {band_wavelength}nm compare", fontsize=12)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"保存 -> {out_png}")


def main():
    # 用户在此设置输入
    scene = "LC08_L1TP_116036_20210330_20210409_02_T1"
    band = 490  # 可改为 443/490/561/660/865 或直接传入波段索引
    try:
        plot_compare(scene, band)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()
