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
import warnings
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
import geopandas as gpd
from netCDF4 import Dataset
import shutil
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

# ----------------- 配置（按需修改） -----------------
OUTPUT_ROOT   = Path("batch_outputs")    # 你的 02 脚本输出根目录
MASKED_ROOT   = Path("batch_masked")     # 去陆地后的输出根目录
NE_LAND_SHP   = Path("TOA_match/batch_simple/ne_10m_land/ne_10m_land.shp")  # 或解压到此处
ALL_TOUCHED   = False  # rasterize 时是否认为任何接触到像元边界的多边形都算覆盖
BUFFER_M      = 150    # 岸线缓冲（米），>0 可以“向海扩一点”，减少潮滩/岸线误差；0 表示不用
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


def landmask_for_landsat_grid(ls_tif: Path, land_shp: Path, all_touched=False, buffer_m=0) -> np.ndarray:
    """把 ne_10m_land.shp 栅格化到 Landsat 规则网格，返回布尔掩膜（True=陆地）。

    更稳健的处理：
    - 先投影到影像 CRS
    - 修复无效几何（make_valid 或 buffer(0)）
    - 先裁剪到影像 bbox，降低复杂度
    - 再进行米级缓冲（如启用）
    """
    def _make_valid(geom):
        try:
            from shapely.validation import make_valid
            return make_valid(geom)
        except Exception:
            try:
                return geom.buffer(0)
            except Exception:
                return geom

    def _safe_buffer(geom, dist):
        try:
            return geom.buffer(dist)
        except Exception:
            try:
                return _make_valid(geom).buffer(dist)
            except Exception:
                return geom

    gdf = gpd.read_file(land_shp)
    with rasterio.open(ls_tif) as src:
        # 将矢量投影到影像 CRS
        gdf = gdf.to_crs(src.crs)
        # 修复无效几何并剔除空
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
            gdf["geometry"] = gdf.geometry.apply(lambda g: _make_valid(g) if g is not None and not g.is_empty and not g.is_valid else g)
            gdf = gdf[(~gdf.geometry.is_empty) & gdf.geometry.notna()]
        # 先裁剪到影像外接矩形，避免处理与影像相距很远的陆地
        bbox_poly = box(*src.bounds)
        # 先用 bounds 做快速过滤再做精确相交，减少无意义运算与告警
        bxmin, bymin, bxmax, bymax = src.bounds
        b = gdf.bounds
        sel = (b["maxx"] >= bxmin) & (b["minx"] <= bxmax) & (b["maxy"] >= bymin) & (b["miny"] <= bymax)
        gdf = gdf[sel]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely.*')
            try:
                gdf["geometry"] = gdf.geometry.intersection(bbox_poly)
            except Exception:
                # 保底：如果 intersection 异常，跳过裁剪
                pass
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
            gdf = gdf[(~gdf.geometry.is_empty) & gdf.geometry.notna()]
        # 可选：岸线缓冲（米）
        if buffer_m and buffer_m != 0:
            gdf["geometry"] = gdf.geometry.apply(lambda g: _safe_buffer(g, buffer_m))
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
        mask = rasterize(
            shapes=shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8"
        )
    return mask.astype(bool)  # True=land


def landmask_for_landsat_grid_v2(ls_tif: Path, land_shp: Path, all_touched: bool = False, buffer_m: float = 0.0) -> np.ndarray:
    """更稳健的陆地掩膜构建（True=陆地）。

    改进：
    - 若影像 CRS 为经纬度，自动将米级缓冲转换为角度缓冲；
    - 栅格化后做占比检查，极端时去掉缓冲重算；仍异常则返回全 False，避免整幅被掩掉。
    """
    def _make_valid(geom):
        try:
            from shapely.validation import make_valid
            return make_valid(geom)
        except Exception:
            try:
                return geom.buffer(0)
            except Exception:
                return geom

    def _safe_buffer(geom, dist):
        try:
            return geom.buffer(dist)
        except Exception:
            try:
                return _make_valid(geom).buffer(dist)
            except Exception:
                return geom

    if not Path(land_shp).exists():
        raise FileNotFoundError(f"未找到陆地矢量: {land_shp}")
    gdf = gpd.read_file(land_shp)
    with rasterio.open(ls_tif) as src:
        # 投影到影像 CRS
        gdf = gdf.to_crs(src.crs)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
            gdf["geometry"] = gdf.geometry.apply(lambda g: _make_valid(g) if g is not None and not g.is_empty and not g.is_valid else g)
            gdf = gdf[(~gdf.geometry.is_empty) & gdf.geometry.notna()]

        # 裁剪到影像 bbox
        from shapely.geometry import box as _box
        bxmin, bymin, bxmax, bymax = src.bounds
        bbox_poly = _box(bxmin, bymin, bxmax, bymax)
        b = gdf.bounds
        sel = (b["maxx"] >= bxmin) & (b["minx"] <= bxmax) & (b["maxy"] >= bymin) & (b["miny"] <= bymax)
        gdf = gdf[sel]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely.*')
            try:
                gdf["geometry"] = gdf.geometry.intersection(bbox_poly)
            except Exception:
                pass
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
            gdf = gdf[(~gdf.geometry.is_empty) & gdf.geometry.notna()]

        # 计算缓冲（自动选择米或度）
        eff_buffer = 0.0
        if buffer_m and buffer_m != 0:
            try:
                is_geographic = getattr(src.crs, "is_geographic", False)
            except Exception:
                is_geographic = False
            eff_buffer = float(buffer_m)
            if is_geographic:
                # 转角度（取图心近似）
                Hc, Wc = src.height // 2, src.width // 2
                x_c = src.transform.c + src.transform.a * (Wc + 0.5)
                y_c = src.transform.f + src.transform.e * (Hc + 0.5)
                try:
                    from pyproj import Transformer
                    tf = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                    lon_c, lat_c = tf.transform(x_c, y_c)
                except Exception:
                    lon_c, lat_c = 0.0, 30.0
                import numpy as _np
                deg_lat = eff_buffer / 111000.0
                coslat = max(0.1, _np.cos(_np.deg2rad(lat_c)))
                deg_lon = eff_buffer / (111000.0 * coslat)
                eff_buffer = max(deg_lat, deg_lon)
            # 记录缓冲信息
            try:
                print(f"  [INFO] landmask: buffer={eff_buffer:.6f} ({'deg' if is_geographic else 'm'}), all_touched={all_touched}")
            except Exception:
                pass
            gdf["geometry"] = gdf.geometry.apply(lambda g: _safe_buffer(g, eff_buffer))
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]

        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]

        def _rasterize(shps):
            return rasterize(
                shapes=shps,
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0,
                all_touched=all_touched,
                dtype="uint8",
            )

        if not shapes:
            try:
                print("  [WARN] landmask: shapes 为空，返回全 False 掩膜（无陆地）")
            except Exception:
                pass
            return np.zeros((src.height, src.width), dtype=bool)

        mask = _rasterize(shapes)
        ratio = float(mask.mean()) if mask.size else 0.0
        try:
            print(f"  [INFO] landmask: 初始陆地占比={ratio*100:.2f}%  (H={src.height}, W={src.width})")
        except Exception:
            pass
        if ratio > 0.95 and (buffer_m and buffer_m != 0):
            # 去掉缓冲重算一次
            gdf2 = gpd.read_file(land_shp).to_crs(src.crs)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
                gdf2["geometry"] = gdf2.geometry.apply(lambda g: _make_valid(g) if g is not None and not g.is_empty and not g.is_valid else g)
                gdf2 = gdf2[(~gdf2.geometry.is_empty) & gdf2.geometry.notna()]
            b2 = gdf2.bounds
            sel2 = (b2["maxx"] >= bxmin) & (b2["minx"] <= bxmax) & (b2["maxy"] >= bymin) & (b2["miny"] <= bymax)
            gdf2 = gdf2[sel2]
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely.*')
                try:
                    gdf2["geometry"] = gdf2.geometry.intersection(bbox_poly)
                except Exception:
                    pass
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
                gdf2 = gdf2[(~gdf2.geometry.is_empty) & gdf2.geometry.notna()]
            shapes2 = [(geom, 1) for geom in gdf2.geometry if geom is not None and not geom.is_empty]
            if shapes2:
                mask2 = _rasterize(shapes2)
                ratio2 = float(mask2.mean()) if mask2.size else 0.0
                try:
                    print(f"  [INFO] landmask: 去缓冲重算占比={ratio2*100:.2f}%")
                except Exception:
                    pass
                if ratio2 <= 0.95:
                    return mask2.astype(bool)

        # 兜底：若依然异常（几乎全是陆地），返回全 False，避免把水体也清空
        if ratio > 0.95:
            try:
                print("  [WARN] landmask: 占比异常(>95%)，兜底返回全 False 掩膜（无陆地）")
            except Exception:
                pass
            return np.zeros((src.height, src.width), dtype=bool)

    return mask.astype(bool)

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
    fig.suptitle("Landsat Landmask Compare (缩略图)", fontsize=12)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  [OK] 保存对比图 -> {out_png.name}")


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
    gdf = gpd.read_file(land_shp)  # EPSG:4326
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
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)
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

    print(f"[INFO] 配对数量: {len(pairs)}  | 使用陆地矢量: {NE_LAND_SHP}")
    for scene, ls_tif, goci_nc in pairs:
        out_dir = MASKED_ROOT / scene
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[SCENE] {scene}")
        print(f"  Landsat: {ls_tif.name}")
        print(f"  GOCI   : {goci_nc.name}")

        # 1) 在 Landsat 网格栅格化陆地 -> land_mask(True=land)
        land_mask = landmask_for_landsat_grid_v2(
            ls_tif, NE_LAND_SHP, all_touched=ALL_TOUCHED, buffer_m=BUFFER_M
        )
        # 简要汇报掩膜占比
        try:
            ratio = float(land_mask.mean()) if land_mask.size else 0.0
            print(f"  [INFO] 最终陆地掩膜占比: {ratio*100:.2f}%")
        except Exception:
            pass
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
