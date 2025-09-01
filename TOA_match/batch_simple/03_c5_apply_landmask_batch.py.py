#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
同步去陆地（Landsat + GOCI-2）
- 对每个场景：基于 Natural Earth 的 ne_10m_land.shp 生成陆地掩膜
- 对 Landsat 多波段 TIF：陆地 -> NoData/0，输出 *_only_water.tif，并保存 *_landmask.tif
- 对 GOCI 子集 NC：陆地 -> NaN，输出 *_only_water.nc

依赖：geopandas, shapely, rasterio, netCDF4, numpy
pip install geopandas shapely rasterio netCDF4
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from netCDF4 import Dataset

# ----------------- 配置（按需修改） -----------------
OUTPUT_ROOT   = Path("batch_outputs")    # 你的 02 脚本输出根目录
MASKED_ROOT   = Path("batch_masked")     # 去陆地后的输出根目录
NE_LAND_SHP   = Path("natural_earth_vector/10m_physical/ne_10m_land.shp")  # 或解压到此处
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
    """把 ne_10m_land.shp 栅格化到 Landsat 规则网格，返回布尔掩膜（True=陆地）。"""
    gdf = gpd.read_file(land_shp)
    with rasterio.open(ls_tif) as src:
        # 将矢量投影到影像 CRS
        gdf = gdf.to_crs(src.crs)
        if buffer_m and buffer_m != 0:
            gdf["geometry"] = gdf.geometry.buffer(buffer_m)
        shapes = [(geom, 1) for geom in gdf.geometry if not geom.is_empty]
        mask = rasterize(
            shapes=shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            all_touched=all_touched,
            dtype="uint8"
        )
    return mask.astype(bool)  # True=land


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
    geom = gdf.unary_union
    if buffer_deg and buffer_deg != 0.0:
        # 如果想用角度缓冲（不常用；通常建议在 Landsat 栅格化时用米级缓冲）
        geom = geom.buffer(buffer_deg)

    with Dataset(nc_path, "r") as ds:
        lat = np.array(ds["navigation_data"]["latitude"][:], dtype=np.float64)
        lon = np.array(ds["navigation_data"]["longitude"][:], dtype=np.float64)

    # shapely.vectorized.contains：点是否在多边形内部（True=陆地）
    from shapely import vectorized
    land = vectorized.contains(geom, lon, lat)
    return land


def apply_landmask_to_goci_nc(nc_in: Path, land_mask: np.ndarray, nc_out: Path):
    """把 GOCI NC 的 geophysical_data/* 陆地像元置为 NaN，其他变量原样复制。"""
    with Dataset(nc_in, "r") as src, Dataset(nc_out, "w") as dst:
        # 复制维度
        for name, dim in src.dimensions.items():
            dst.createDimension(name, (len(dim) if not dim.isunlimited() else None))
        # 逐变量处理
        for name, var in src.variables.items():
            out = dst.createVariable(name, var.datatype, var.dimensions, zlib=True)
            out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
            data = var[:]
            # 只对 geophysical_data/*（光学变量）的 2D 栅格应用掩膜；其他如 lat/lon 保持
            is_geo = name.startswith("geophysical_data/")
            if is_geo and data.ndim == 2 and data.shape == land_mask.shape:
                arr = data.filled(np.nan) if np.ma.isMaskedArray(data) else data.astype(np.float32)
                arr[land_mask] = np.nan
                out[:] = arr
            else:
                out[:] = data


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
        land_mask = landmask_for_landsat_grid(
            ls_tif, NE_LAND_SHP, all_touched=ALL_TOUCHED, buffer_m=BUFFER_M
        )
        # 2) 应用到 Landsat
        ls_out = out_dir / (ls_tif.stem + "_only_water.tif")
        apply_landmask_to_landsat(ls_tif, land_mask, ls_out)
        print(f"  [OK] Landsat 去陆地 -> {ls_out.name}")

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

    print(f"\n✅ 全部完成。输出目录：{MASKED_ROOT}")


if __name__ == "__main__":
    main()