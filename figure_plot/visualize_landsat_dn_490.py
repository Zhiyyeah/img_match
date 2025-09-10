#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化指定 Landsat 场景 490nm (B2) 原始 DN。

更新改动：
    - 使用 viridis 配色；
    - NaN 像元显示为白色（cmap.set_bad('white')）；
    - 将 DN<=0 或 掩膜为 0 的像元视为无效并置为 NaN；
    - 可选强制经纬度等比例显示（避免图像被拉伸）。

使用：在 main() 中设置 `scene_dir`，如需强制等比例设置 force_equal_aspect=True。
输出：PNG 图像（分位 2–98 拉伸，避免极端值影响）。
"""
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from pyproj import Transformer


def find_band2_file(scene_dir: Path):
    """返回场景目录中 B2 文件路径。"""
    if not scene_dir.exists():
        raise FileNotFoundError(f"目录不存在: {scene_dir}")
    cand = None
    for f in scene_dir.iterdir():
        name_low = f.name.lower()
        if f.is_file() and name_low.endswith('_b2.tif'):
            cand = f
            break
    if cand is None:
        raise FileNotFoundError("未找到 *_B2.TIF 文件")
    return cand


def read_dn_band(tif_path: Path, max_h: int = 4000):
    """读取 B2 DN, 若高度超过 max_h 进行等比例最近邻下采样。

    返回: (dn_float32, transform, crs, scale, validity_mask)
      - dn_float32: float32 数组
      - validity_mask: True=原始有效（DN>0 且内部掩膜>0）
    """
    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        if H > max_h:
            scale = max_h / float(H)
            out_h = max_h
            out_w = int(round(W * scale))
            dn = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
            m = src.read_masks(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
        else:
            scale = 1.0
            dn = src.read(1)
            m = src.read_masks(1)
        dn = dn.astype(np.float32)
        valid = (dn > 0) & (m > 0)
        dn = np.where(valid, dn, np.nan)  # 无效置 NaN
        transform_aff = src.transform
        crs = src.crs
    return dn, transform_aff, crs, scale, valid


def lonlat_grid(tif_path: Path, out_shape: tuple[int, int], scale: float) -> tuple[np.ndarray, np.ndarray]:
    """生成经纬度网格 (lon, lat) 对应下采样后的像元中心。

    参数:
        tif_path: 原始 GeoTIFF 路径
        out_shape: 下采样后数组形状 (H, W)
        scale: 下采样比例 (down_h / orig_h)
    返回:
        (lon2d, lat2d)
    """
    out_h, out_w = out_shape
    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        T = src.transform
        src_crs = src.crs
        if scale < 1.0:  # 需要把下采样网格映射回原像素索引
            rows = (np.arange(out_h) / scale).astype(np.float64)
            cols = (np.arange(out_w) / scale).astype(np.float64)
        else:
            rows = np.arange(out_h, dtype=np.float64)
            cols = np.arange(out_w, dtype=np.float64)
        cgrid, rgrid = np.meshgrid(cols, rows)
        # 像元中心坐标
        x = T.c + T.a * (cgrid + 0.5) + T.b * (rgrid + 0.5)
        y = T.f + T.d * (cgrid + 0.5) + T.e * (rgrid + 0.5)
        try:
            if src_crs is None:
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
            else:
                transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(x, y)
        except Exception:
            lon, lat = x, y
    return lon, lat


def robust_min_max(arr, p1=2, p2=98):
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return 0, 1
    # 额外预截尾，减少极端高值影响
    if a.size > 5000:
        low_clip = np.percentile(a, 0.1)
        high_clip = np.percentile(a, 99.9)
        a = a[(a >= low_clip) & (a <= high_clip)]
    lo = np.percentile(a, p1)
    hi = np.percentile(a, p2)
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if lo == hi:
        hi = lo + 1e-6
    return float(lo), float(hi)


def extent_from_transform(transform_aff, shape):
    H, W = shape
    x0 = transform_aff.c
    y0 = transform_aff.f
    px_w = transform_aff.a
    px_h = transform_aff.e
    left = x0
    top = y0
    right = x0 + px_w * W
    bottom = y0 + px_h * H
    return [left, right, bottom, top]


def plot_dn(dn, lon, lat, title, out_png: Path, force_equal_aspect: bool = True):
    """用 pcolormesh 绘制 DN。

    force_equal_aspect: True 时使用 ax.set_aspect('equal') 使经纬度 1°:1°，避免比例失真。
    """
    vmin, vmax = robust_min_max(dn, 2, 98)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='white')  # NaN -> 白色
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.pcolormesh(lon, lat, dn, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3, linestyle='--')
    if force_equal_aspect:
        ax.set_aspect('equal', adjustable='box')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('DN')
    # txt = f"vmin={vmin:.2f} vmax={vmax:.2f} (2-98 pct)"  # 添加简单注释
    # ax.text(0.01, 0.01, txt, transform=ax.transAxes, fontsize=8, color='#333', ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[OK] 保存: {out_png}")


def main():
    # 直接在此定义场景目录
    scene_dir = Path(r"D:\Py_Code\img_match\SR_Imagery\Slot_7_2021_2025\LC08_L1TP_116036_20210330_20210409_02_T1")
    out_png = scene_dir / "B2_490nm_DN.png"

    b2_file = find_band2_file(scene_dir)
    print(f"[INFO] 读取 B2 文件: {b2_file.name}")
    dn, transform_aff, crs, scale, valid_mask = read_dn_band(b2_file)
    print(f"[INFO] shape={dn.shape}, dtype={dn.dtype}, CRS={crs}  valid_ratio={np.isfinite(dn).mean():.3f}")
    lon, lat = lonlat_grid(b2_file, dn.shape, scale)
    plot_dn(dn, lon, lat, f"{scene_dir.name} B2 (~490nm) DN", out_png, force_equal_aspect=True)


if __name__ == '__main__':
    main()
