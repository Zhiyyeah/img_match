#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
同步去陆地（Landsat + GOCI-2）
- 对每个场景：基于 Natural Earth 的 ne_10m_land.shp 生成陆地掩膜
- 对 Landsat 多波段 TIF：陆地 -> NoData/0，输出 *_only_water.tif，并保存 *_landmask.tif
- 对 GOCI 子集 NC：陆地 -> NaN，输出 *_only_water.nc

依赖：geopandas, shapely, rasterio, netCDF4, numpy, matplotlib
pip install geopandas shapely rasterio netCDF4 matplotlib
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio
import geopandas as gpd
from netCDF4 import Dataset
import shutil
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from pyproj import Transformer

# ----------------- 配置（按需修改） -----------------
OUTPUT_ROOT   = Path("batch_cal_clip")    # 你的 02 脚本输出根目录
MASKED_ROOT   = Path("batch_masked")     # 去陆地后的输出根目录
NE_LAND_SHP   = Path("TOA_match/batch_simple/ne_10m_land/ne_10m_land.shp")  # 或解压到此处
WRITE_MASK_GT = True   # 是否另存一张 Landsat 网格的二值陆地掩膜 GeoTIFF
# ---------------------------------------------------


def discover_pairs(root: Path):
    """在 batch_outputs/{scene}/ 里找 Landsat 多波段 TIF 与 GOCI 子集 NC 的配对。"""
    pairs = []
    if not root.exists():
        return pairs
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        ls = None
        goci = None
        for f in scene_dir.iterdir():
            if f.suffix.lower() == ".tif" and "_TOA_RAD_B" in f.name:
                ls = f
            if f.suffix.lower() == ".nc" and "_subset" in f.name:
                goci = f
        if ls and goci:
            pairs.append((scene_dir.name, ls, goci))
    return pairs


    


def landmask_for_landsat_grid_points(ls_tif: Path, land_shp: Path, tile_rows: int = 1024) -> np.ndarray:
    """基于点测（像元中心在经纬度上的点是否落在陆地多边形内）生成 Landsat 掩膜。

    更换思路：
    - 不再做矢量栅格化，直接在像元中心进行点在面内测试（shapely contains）；
    - 先将矢量统一到 EPSG:4326，再把 Landsat 网格分块转换到经纬度；
    - True 表示陆地。
    """
    if not Path(land_shp).exists():
        raise FileNotFoundError(f"未找到陆地矢量: {land_shp}")

    gdf = gpd.read_file(land_shp)
    # 统一到 EPSG:4326
    if gdf.crs is None or str(gdf.crs).strip() == "":
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # 修复无效几何、仅保留面
    from shapely.validation import make_valid as _mk
    def _fix(g):
        if g is None or g.is_empty:
            return None
        if not g.is_valid:
            try:
                g = _mk(g)
            except Exception:
                try:
                    g = g.buffer(0)
                except Exception:
                    return None
        return g
    gdf["geometry"] = gdf.geometry.apply(_fix)
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)]
    # 合并为一个多边形（含洞）
    try:
        land_geom = gdf.union_all()
    except Exception:
        land_geom = gdf.unary_union
    if land_geom.is_empty:
        raise RuntimeError("陆地矢量 union 为空")

    mask: np.ndarray
    with rasterio.open(ls_tif) as src:
        H, W = src.height, src.width
        T = src.transform
        crs = src.crs
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        mask = np.zeros((H, W), dtype=bool)
        # 逐块转换行到经纬度并测试
        col_idx = np.arange(W, dtype=np.float64)
        for r0 in range(0, H, tile_rows):
            r1 = min(H, r0 + tile_rows)
            rows = np.arange(r0, r1, dtype=np.float64)
            cgrid, rgrid = np.meshgrid(col_idx, rows)
            # 像元中心
            x = T.c + T.a * (cgrid + 0.5) + T.b * (rgrid + 0.5)
            y = T.f + T.d * (cgrid + 0.5) + T.e * (rgrid + 0.5)
            lon, lat = transformer.transform(x, y)
            # contains 测试（Shapely 2: contains_xy；老版本回退 vectorized.contains）
            try:
                from shapely import contains_xy
                inside = contains_xy(land_geom, lon, lat)
            except Exception:
                from shapely import vectorized
                inside = vectorized.contains(land_geom, lon, lat)
            # NaN 安全
            valid = np.isfinite(lon) & np.isfinite(lat)
            mask[r0:r1, :] = np.where(valid, inside, False)
    return mask

def apply_landmask_to_landsat(ls_tif: Path, land_mask: np.ndarray, out_tif: Path):
    """对 Landsat 多波段 TIF 应用陆地掩膜：陆地 -> NoData/0。"""
    with rasterio.open(ls_tif) as src:
        data = src.read().astype(np.float32)     # (B,H,W)
        prof = src.profile
        nod = src.nodata if src.nodata is not None else 0
        for i in range(data.shape[0]):
            band = data[i]
            band[land_mask] = nod
            data[i] = band
        prof.update(compress="lzw", tiled=True, predictor=2)
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, "w", **prof) as dst:
            dst.write(data)


def _read_thumb(ds: rasterio.DatasetReader, max_h: int) -> np.ndarray:
    """Read thumbnail with bilinear downsampling and map nodata to NaN."""
    count, H, W = ds.count, ds.height, ds.width
    if H <= max_h:
        arr = ds.read().astype(np.float32)
    else:
        scale = max_h / float(H)
        out_h = max_h
        out_w = int(round(W * scale))
        arr = ds.read(out_shape=(count, out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
    if ds.nodata is not None:
        arr = np.where(arr == ds.nodata, np.nan, arr)
    return arr


def _robust_vmin_vmax(a_list):
    vals = np.concatenate([x[np.isfinite(x)].ravel() for x in a_list if x is not None and np.isfinite(x).any()])
    if vals.size == 0:
        return 0.0, 1.0
    vmin = np.nanpercentile(vals, 2)
    vmax = np.nanpercentile(vals, 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    return float(vmin), float(vmax)


def save_landsat_mask_compare(ls_src: Path, ls_masked: Path, out_png: Path, max_h: int = 1200):
    """Save side-by-side quicklook of original vs masked Landsat (per band)."""
    with rasterio.open(ls_src) as ds_a, rasterio.open(ls_masked) as ds_b:
        # ensure same size
        if ds_a.width != ds_b.width or ds_a.height != ds_b.height or ds_a.count != ds_b.count:
            print("  [WARN] 原始与去陆地后的 Landsat 维度不一致，跳过对比图")
            return
        A = _read_thumb(ds_a, max_h)
        B = _read_thumb(ds_b, max_h)

    bands = min(A.shape[0], 5)
    fig, axes = plt.subplots(bands, 2, figsize=(10, 3.2 * bands), constrained_layout=True)
    if bands == 1:
        axes = np.array([axes])
    for b in range(bands):
        a = A[b]
        c = B[b]
        vmin, vmax = _robust_vmin_vmax([a, c])

        ax = axes[b, 0]
        im = ax.imshow(a, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Original B{b+1}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        ax = axes[b, 1]
        im = ax.imshow(c, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Water-only B{b+1}")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Landsat Landmask Compare ", fontsize=12)
    fig.savefig(out_png, dpi=400)
    plt.close(fig)
    print(f"  [OK] 保存对比图 -> {out_png.name}")


def save_goci_mask_compare(goci_src_nc: Path, goci_masked_nc: Path, out_png: Path, band_candidates=None):
    """生成 GOCI 原始 vs 去陆地 的快速对比图（按一条代表性波段）。

    - 优先使用 geophysical_data/L_TOA_443；若缺失，则按候选列表依次回退。
    - 显示相同色标范围（基于 2–98 分位）。
    """
    if band_candidates is None:
        band_candidates = ("L_TOA_443", "L_TOA_490", "L_TOA_555", "L_TOA_660", "L_TOA_865")

    def _read_nc_one(nc_path: Path):
        with Dataset(nc_path, "r") as ds:
            nav = ds["navigation_data"] if "navigation_data" in ds.groups else ds
            lat = np.array(nav["latitude"][:], dtype=np.float64)
            lon = np.array(nav["longitude"][:], dtype=np.float64)
            g = ds["geophysical_data"] if "geophysical_data" in ds.groups else ds
            vname = None
            for cand in band_candidates:
                if cand in g.variables:
                    vname = cand; break
            if vname is None:
                raise KeyError("GOCI 对比图未找到任何候选波段")
            var = g[vname]
            arr = var[:]
            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan).astype(np.float32)
            else:
                arr = np.array(arr, dtype=np.float32)
            if "_FillValue" in var.ncattrs():
                try:
                    fv = float(np.array(var.getncattr("_FillValue")).ravel()[0])
                    arr = np.where(arr == fv, np.nan, arr)
                except Exception:
                    pass
        return lat, lon, arr, vname

    lat0, lon0, a0, vname = _read_nc_one(goci_src_nc)
    lat1, lon1, a1, _ = _read_nc_one(goci_masked_nc)

    vmin, vmax = _robust_vmin_vmax([a0, a1])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)

    fig = plt.figure(figsize=(12, 5), dpi=150)
    ax1 = fig.add_subplot(1, 2, 1)
    im1 = ax1.pcolormesh(lon0, lat0, a0, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax1.set_title(f"GOCI {vname} (original)")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.grid(True, alpha=0.3)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.pcolormesh(lon1, lat1, a1, shading="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax2.set_title(f"GOCI {vname} (water-only)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.grid(True, alpha=0.3)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [OK] 保存 GOCI 对比图 -> {out_png.name}")


def save_binary_mask_gt(ls_tif: Path, land_mask: np.ndarray, out_tif: Path):
    """另存一张与 Landsat 同网格的二值掩膜（1=陆地, 0=水）。"""
    with rasterio.open(ls_tif) as src:
        prof = src.profile
        prof.update(count=1, dtype="uint8", compress="lzw", nodata=0)
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, "w", **prof) as dst:
            dst.write(land_mask.astype("uint8")[None, ...])


def landmask_for_goci_grid(nc_path: Path, land_shp: Path, buffer_deg=0.0) -> np.ndarray:
    """
    在 GOCI 弯曲网格（lat/lon）上构建陆地掩膜（True=陆地）。
    注意：Natural Earth 默认为 EPSG:4326，经纬度。GOCI 的 lat/lon 也是经纬度。
    """
    gdf = gpd.read_file(land_shp)
    # 投影到经纬度
    try:
        gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        pass
    # 修复无效几何，避免 union/buffer 失败
    def _make_valid(geom):
        try:
            from shapely.validation import make_valid
            return make_valid(geom)
        except Exception:
            try:
                return geom.buffer(0)
            except Exception:
                return geom
    gdf["geometry"] = gdf.geometry.apply(lambda g: _make_valid(g) if g is not None and not g.is_empty and not g.is_valid else g)
    gdf = gdf[(~gdf.geometry.is_empty) & gdf.geometry.notna()]
    # union: 兼容新旧 API（union_all 优先）
    try:
        geom = gdf.union_all()
    except Exception:
        geom = gdf.unary_union
    if buffer_deg and buffer_deg != 0.0:
        # 如果想用角度缓冲（不常用；通常建议在 Landsat 栅格化时用米级缓冲）
        try:
            geom = geom.buffer(buffer_deg)
        except Exception:
            geom = _make_valid(geom).buffer(buffer_deg)

    with Dataset(nc_path, "r") as ds:
        lat = np.array(ds["navigation_data"]["latitude"][:], dtype=np.float64)
        lon = np.array(ds["navigation_data"]["longitude"][:], dtype=np.float64)

    # shapely.vectorized.contains：点是否在多边形内部（True=陆地）
    # contains：Shapely 2.0 推荐 contains_xy；旧版回退到 vectorized.contains
    try:
        from shapely import contains_xy
        land = contains_xy(geom, lon, lat)
    except Exception:
        from shapely import vectorized
        land = vectorized.contains(geom, lon, lat)
    return land


def apply_landmask_to_goci_nc(nc_in: Path, land_mask: np.ndarray, nc_out: Path):
    """把 GOCI 子集 NC 的 geophysical_data/* 陆地像元置为无效（写 FillValue），其余保持；保留分组结构。

    关键点：
    - 维持 geophysical_data 变量的原 dtype/属性（含 scale_factor/add_offset），写入 MaskedArray，
      将陆地像元置为掩膜，交由 netCDF4 用 _FillValue 写出，避免数值缩放/类型不一致问题。
    - 其他组（navigation_data、mask 等）原样复制。
    """
    def _copy_attrs(src_obj, dst_obj, skip: set[str] | None = None):
        skip = skip or set()
        for k in getattr(src_obj, 'ncattrs', lambda: [])():
            if k in skip:
                continue
            try:
                dst_obj.setncattr(k, src_obj.getncattr(k))
            except Exception:
                pass

    def _copy_group(src_grp, dst_grp, in_geo_group: bool = False):
        # 复制组属性
        _copy_attrs(src_grp, dst_grp)
        # 复制变量
        for vname, var in src_grp.variables.items():
            # 创建变量（沿用原 dtype/维度/压缩与 FillValue）
            fillv = None
            if "_FillValue" in var.ncattrs():
                try:
                    fillv = np.array(var.getncattr("_FillValue")).ravel()[0]
                    if hasattr(fillv, 'item'):
                        fillv = fillv.item()
                except Exception:
                    fillv = None
            out = dst_grp.createVariable(vname, var.datatype, var.dimensions, zlib=True, fill_value=fillv)
            # 拷贝属性（跳过 _FillValue，已在创建时处理）
            _copy_attrs(var, out, skip={"_FillValue"})

            data = var[:]
            if in_geo_group and hasattr(data, 'ndim') and data.ndim == 2 and data.shape == land_mask.shape:
                # 合并掩膜，在陆地置为掩膜（由 _FillValue 写出）
                marr = np.ma.array(data, copy=False)
                old_mask = np.ma.getmaskarray(marr)
                new_mask = np.logical_or(old_mask, land_mask)
                out[:] = np.ma.array(marr, mask=new_mask)
            else:
                out[:] = data
        # 递归复制子组
        for gname, sub in src_grp.groups.items():
            sub_out = dst_grp.createGroup(gname)
            _copy_group(sub, sub_out, in_geo_group=(in_geo_group or gname == 'geophysical_data'))

    with Dataset(nc_in, "r") as src, Dataset(nc_out, "w") as dst:
        # 维度
        for name, dim in src.dimensions.items():
            dst.createDimension(name, (len(dim) if not dim.isunlimited() else None))
        # 根属性
        _copy_attrs(src, dst)
        # 复制根下变量与各组
        _copy_group(src, dst, in_geo_group=False)


def main():
    pairs = discover_pairs(OUTPUT_ROOT)
    if not pairs:
        print(f"[WARN] 未在 {OUTPUT_ROOT} 发现配对数据。")
        return
    if not NE_LAND_SHP.exists():
        raise FileNotFoundError(f"未找到陆地 shp：{NE_LAND_SHP}")

    # print(f"[INFO] 配对数量: {len(pairs)}  | 使用陆地矢量: {NE_LAND_SHP}")
    for scene, ls_tif, goci_nc in pairs:
        out_dir = MASKED_ROOT / scene
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[SCENE] {scene}")
        print(f"  Landsat: {ls_tif.name}")
        print(f"  GOCI   : {goci_nc.name}")

        # 1) 在 Landsat 网格栅格化陆地 -> land_mask(True=land)
        land_mask = landmask_for_landsat_grid_points(ls_tif, NE_LAND_SHP, tile_rows=1024)
        # 可按需输出占比：ratio = float(land_mask.mean())
        # 2) 应用到 Landsat
        ls_out = out_dir / (ls_tif.stem + "_only_water.tif")
        apply_landmask_to_landsat(ls_tif, land_mask, ls_out)
        print(f"  [OK] Landsat 去陆地 -> {ls_out.name}")
        # 保存对比图（原始 vs 去陆地）
        cmp_png = out_dir / f"COMPARE_mask_{scene}.png"
        try:
            save_landsat_mask_compare(ls_tif, ls_out, cmp_png, max_h=1200)
        except Exception as e:
            print(f"  [WARN] 生成对比图失败: {e}")

        # 2.1) 可选：输出二值掩膜（同 Landsat 网格）
        if WRITE_MASK_GT:
            msk_out = out_dir / (ls_tif.stem + "_landmask.tif")
            save_binary_mask_gt(ls_tif, land_mask, msk_out)
            print(f"  [OK] 掩膜保存 -> {msk_out.name}")

        # 3) 在 GOCI 网格构建陆地掩膜（同一套矢量，保持一致）
        goci_land = landmask_for_goci_grid(goci_nc, NE_LAND_SHP, buffer_deg=0.0)
        goci_out = out_dir / (goci_nc.stem + "_only_water.nc")
        apply_landmask_to_goci_nc(goci_nc, goci_land, goci_out)
        print(f"  [OK] GOCI 去陆地 -> {goci_out.name}")

        # GOCI 对比图（原始 vs 去陆地）
        try:
            goci_cmp_png = out_dir / f"COMPARE_GOCI_mask_{scene}.png"
            save_goci_mask_compare(goci_nc, goci_out, goci_cmp_png)
        except Exception as e:
            print(f"  [WARN] 生成 GOCI 对比图失败: {e}")

        # 4) 为与后续 03 重采样脚本无缝衔接：用 "仅水体" NC 覆盖 batch_outputs 中的原始 subset NC
        #    先做一次备份（*.bak），然后复制覆盖同名文件。
        try:
            orig_nc = goci_nc
            backup_nc = orig_nc.with_suffix(orig_nc.suffix + ".bak")
            if not backup_nc.exists():
                shutil.copy2(orig_nc, backup_nc)
            shutil.copy2(goci_out, orig_nc)
            print(f"  [OK] 覆盖原 subset NC 以供 03 使用（已备份为 {backup_nc.name}）")
        except Exception as e:
            print(f"  [WARN] 覆盖原 subset NC 失败: {e}")

    print(f"\n✅ 全部完成。输出目录：{MASKED_ROOT}")


if __name__ == "__main__":
    main()
