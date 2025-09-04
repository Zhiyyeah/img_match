#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_c6: 基于 Landsat 865 nm 辐亮度阈值的云掩膜与裁剪（输出到单独文件夹）

- 输入：
  - 优先使用 batch_masked/{scene}/*_only_water.tif（已去陆地）；
    回退 batch_outputs/{scene}/*_TOA_RAD_B*.tif。
  - 使用 batch_resampled/{scene}/GOCI_on_Landsat_*.tif（与 Landsat 同网格）。

- 处理：
  - 按第5波段（865 nm）阈值 >120 判定云，保留 <=120。
  - 取保留像元的最小外接窗口，对 Landsat 与 GOCI_on_Landsat 同窗裁剪；
    窗口内云像元置为 NaN。

- 输出（单独目录）：
  - batch_cloudmasked/{scene}/Landsat_cloudmasked_cropped.tif
  - batch_cloudmasked/{scene}/GOCI_on_Landsat_cloudmasked_cropped.tif
  - batch_cloudmasked/{scene}/COMPARE_cloudmask.png（可配置）
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window


OUTPUT_ROOT = Path("batch_outputs")
MASKED_ROOT = Path("batch_masked")
RESAMPLED_ROOT = Path("batch_resampled")
CLOUDMASK_ROOT = Path("batch_cloudmasked")  # 独立输出根目录

CLOUD_L865_THRESH = 120.0
WRITE_QUICKLOOK = True


def _find_ls_tif(scene: str) -> Path | None:
    d_mask = MASKED_ROOT / scene
    if d_mask.exists():
        cands = sorted([p for p in d_mask.iterdir() if p.suffix.lower()=='.tif' and p.name.endswith('_only_water.tif')])
        if cands:
            return cands[0]
    d_out = OUTPUT_ROOT / scene
    if d_out.exists():
        for p in d_out.iterdir():
            if p.suffix.lower()=='.tif' and '_TOA_RAD_B' in p.name:
                return p
    return None


def _find_goci_on_ls(scene: str) -> Path | None:
    d = RESAMPLED_ROOT / scene
    if not d.exists():
        return None
    for p in d.iterdir():
        if p.suffix.lower()=='.tif' and p.name.startswith('GOCI_on_Landsat_'):
            return p
    return None


def _compute_valid_window(mask_keep: np.ndarray) -> tuple[int,int,int,int] | None:
    ys, xs = np.where(mask_keep)
    if ys.size == 0:
        return None
    rmin, rmax = int(ys.min()), int(ys.max())
    cmin, cmax = int(xs.min()), int(xs.max())
    return rmin, cmin, (rmax - rmin + 1), (cmax - cmin + 1)


def _read_thumb(ds: rasterio.DatasetReader, max_h: int = 1200) -> np.ndarray:
    if ds.height <= max_h:
        arr = ds.read().astype(np.float32)
    else:
        scale = max_h / float(ds.height)
        arr = ds.read(
            out_shape=(ds.count, max_h, int(round(ds.width * scale))),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
    if ds.nodata is not None:
        arr = np.where(arr == ds.nodata, np.nan, arr)
    return arr


def _save_quicklook(ls_tif: Path, go_tif: Path, out_png: Path, max_h: int = 1200):
    import matplotlib.pyplot as plt
    with rasterio.open(ls_tif) as d_ls, rasterio.open(go_tif) as d_go:
        hr = _read_thumb(d_ls, max_h)
        lr = _read_thumb(d_go, max_h)
    bands = min(3, hr.shape[0], lr.shape[0])
    fig, axes = plt.subplots(bands, 2, figsize=(10, 3.2 * bands), constrained_layout=True)
    if bands == 1:
        import numpy as _np
        axes = _np.array([axes])
    def vminmax(a_list):
        vals = []
        for x in a_list:
            if x is None:
                continue
            v = x[_np.isfinite(x)]
            if v.size:
                vals.append(v.ravel())
        if not vals:
            return 0.0, 1.0
        vals = _np.concatenate(vals)
        vmin = _np.nanpercentile(vals, 2)
        vmax = _np.nanpercentile(vals, 98)
        if not _np.isfinite(vmin) or not _np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = float(_np.nanmin(vals)), float(_np.nanmax(vals))
        if vmin == vmax:
            vmax = vmin + 1e-6
        return vmin, vmax
    for b in range(bands):
        vmin, vmax = vminmax([hr[b], lr[b]])
        im = axes[b, 0].imshow(hr[b], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[b, 0].set_title(f'Landsat B{b+1}')
        axes[b, 0].axis('off')
        fig.colorbar(im, ax=axes[b, 0], fraction=0.046, pad=0.02)
        im = axes[b, 1].imshow(lr[b], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[b, 1].set_title('GOCI_on_Landsat')
        axes[b, 1].axis('off')
        fig.colorbar(im, ax=axes[b, 1], fraction=0.046, pad=0.02)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _process_scene(ls_tif: Path, go_tif: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ls_tif) as d_ls, rasterio.open(go_tif) as d_go:
        if d_ls.width != d_go.width or d_ls.height != d_go.height:
            raise RuntimeError("GOCI_on_Landsat 与 Landsat 尺寸不一致")
        # 云掩膜依据：第5波段（865nm）
        if d_ls.count < 5:
            raise RuntimeError("Landsat 波段不足 5")
        b865 = d_ls.read(5).astype(np.float32)
        if d_ls.nodata is not None:
            b865 = np.where(b865 == d_ls.nodata, np.nan, b865)
        keep = np.isfinite(b865) & (b865 <= CLOUD_L865_THRESH)
        win_tup = _compute_valid_window(keep)
        if win_tup is None:
            raise RuntimeError("阈值过严，无有效区域")
        r0, c0, h, w = win_tup
        win = Window.from_slices((r0, r0 + h), (c0, c0 + w))

        # 写出裁剪后的 Landsat
        ls_prof = d_ls.profile.copy()
        ls_prof.update({
            'height': h,
            'width': w,
            'transform': rasterio.windows.transform(win, d_ls.transform),
            # 对 float 写 NaN 时不强制 nodata；若原为整型，这里保持 dtype，视需要改为 float32
        })
        # 若原 dtype 为整型，改为 float32 以容纳 NaN
        if np.issubdtype(np.dtype(ls_prof['dtype']), np.integer):
            ls_prof['dtype'] = 'float32'
        ls_out = out_dir / 'Landsat_cloudmasked_cropped.tif'
        with rasterio.open(ls_out, 'w', **ls_prof) as dst:
            for i in range(1, d_ls.count + 1):
                band = d_ls.read(i, window=win).astype(np.float32)
                if d_ls.nodata is not None:
                    band = np.where(band == d_ls.nodata, np.nan, band)
                band_mask = ~keep[r0:r0+h, c0:c0+w]
                band = np.where(band_mask, np.nan, band)
                dst.write(band, i)

        # 写出裁剪后的 GOCI_on_Landsat（应用相同云掩膜）
        go_prof = d_go.profile.copy()
        go_prof.update({
            'height': h,
            'width': w,
            'transform': rasterio.windows.transform(win, d_go.transform),
        })
        if np.issubdtype(np.dtype(go_prof['dtype']), np.integer):
            go_prof['dtype'] = 'float32'
        go_out = out_dir / 'GOCI_on_Landsat_cloudmasked_cropped.tif'
        with rasterio.open(go_out, 'w', **go_prof) as dst:
            for i in range(1, d_go.count + 1):
                band = d_go.read(i, window=win).astype(np.float32)
                band = np.where(~keep[r0:r0+h, c0:c0+w], np.nan, band)
                dst.write(band, i)

    if WRITE_QUICKLOOK:
        try:
            _save_quicklook(ls_out, go_out, out_dir / 'COMPARE_cloudmask.png')
        except Exception as e:
            print(f"  [WARN] 快速对比图失败: {e}")


def main():
    if not RESAMPLED_ROOT.exists():
        print(f"[WARN] 未找到 {RESAMPLED_ROOT}")
        return
    scenes = [d.name for d in sorted(RESAMPLED_ROOT.iterdir()) if d.is_dir()]
    if not scenes:
        print("[WARN] 未发现场景目录")
        return
    print(f"[INFO] 场景数：{len(scenes)}  | 输出目录：{CLOUDMASK_ROOT}")
    for scene in scenes:
        print(f"\n[SCENE] {scene}")
        ls_tif = _find_ls_tif(scene)
        go_tif = _find_goci_on_ls(scene)
        if not ls_tif or not go_tif:
            print("  [WARN] 缺少输入（Landsat 或 GOCI_on_Landsat），跳过")
            continue
        print(f"  HR : {ls_tif}")
        print(f"  GOCI: {go_tif}")
        out_dir = CLOUDMASK_ROOT / scene
        try:
            _process_scene(ls_tif, go_tif, out_dir)
            print(f"  [OK] 输出 -> {out_dir}")
        except Exception as e:
            print(f"  [ERR] 处理失败: {e}")


if __name__ == '__main__':
    main()

