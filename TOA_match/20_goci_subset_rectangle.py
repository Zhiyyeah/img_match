#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按 Landsat footprint 多边形精确裁剪 GOCI-2 L1B（弯曲网格）
- 像元级点测：按 (lon,lat) 判断是否落入多边形
- 输出为：紧凑包络（仅包含 inside=True 的最小行列范围）+ 多边形外写为 _FillValue
- 删除 scale_factor/add_offset 等缩放属性，避免读出时“二次缩放”
- 附带写出 mask/inside_mask 供后续统计与匹配
- 可选：显示叠加多边形与 inside 掩膜分布（SHOW_PLOT）
"""

import os
import numpy as np
from netCDF4 import Dataset
from matplotlib.path import Path
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom
import matplotlib.pyplot as plt
import numpy.ma as ma

# -------- 常量（波段选择与命名映射） --------
KEEP_INDICES = [3, 4, 6, 8, 12]  # 443, 490, 555, 660, 865 nm
BAND_INDEX_TO_NAME = {
    1: "L_TOA_380",  2: "L_TOA_412",  3: "L_TOA_443",  4: "L_TOA_490",
    5: "L_TOA_510",  6: "L_TOA_555",  7: "L_TOA_620",  8: "L_TOA_660",
    9: "L_TOA_680", 10: "L_TOA_709", 11: "L_TOA_745", 12: "L_TOA_865",
}

def safe_copy_ncattrs_without_scaling(src_var, dst_var):
    """
    复制 netCDF 变量属性到新变量，跳过保留属性和缩放相关属性
    （我们写入的是已解包的 float 真值，不能再携带 scale_factor/add_offset）
    """
    drop = {
        "name", "_FillValue",
        "scale_factor", "add_offset",
        "valid_min", "valid_max", "valid_range", "_Unsigned",
        # 这些是 Variable 对象属性或由库管理的，不应设置
        "dimensions", "dtype", "data_model", "disk_format", "path", "parent",
        "ndim", "mask", "scale", "cmptypes", "vltypes", "enumtypes",
        "file_format", "always_mask",
    }
    for att in src_var.ncattrs():
        if att in drop:
            continue
        try:
            dst_var.setncattr(att, src_var.getncattr(att))
        except Exception:
            pass

def extract_footprint_lonlat(tif_path: str, to_crs: str = "EPSG:4326"):
    """
    从 Landsat TIF 提取 footprint（外环），转换到 WGS84，经闭合返回 (lon, lat)。
    兼容旧版 rasterio（不使用 densify_pts）。
    """
    with rasterio.open(tif_path) as src:
        m = src.dataset_mask().astype(np.uint8)
        geoms = []
        for geom, val in shapes(m, mask=(m > 0), transform=src.transform):
            if val:
                geoms.append(transform_geom(src.crs, to_crs, geom, precision=10))
    if not geoms:
        raise RuntimeError("未从掩膜中提取到有效 footprint。")

    def outer_ring(g):
        if g["type"] == "Polygon":
            return np.asarray(g["coordinates"][0])
        elif g["type"] == "MultiPolygon":
            rings = [np.asarray(poly[0]) for poly in g["coordinates"]]
            return rings[np.argmax([r.shape[0] for r in rings])]
        else:
            raise ValueError(f"不支持的几何类型: {g['type']}")

    rings = [outer_ring(g) for g in geoms]
    ring = rings[np.argmax([r.shape[0] for r in rings])]
    lon, lat = ring[:, 0], ring[:, 1]
    if lon[0] != lon[-1] or lat[0] != lat[-1]:
        lon = np.r_[lon, lon[0]]
        lat = np.r_[lat, lat[0]]
    return lon, lat

def clip_goci_by_polygon(
    goci_nc_path: str,
    out_nc_path: str,
    lon_ring: np.ndarray,
    lat_ring: np.ndarray,
    keep_indices=KEEP_INDICES,
    band_map=BAND_INDEX_TO_NAME,
    fill_default=-999.0,
    return_inside=False
):
    """
    用闭合多边形裁剪 GOCI L1B 并输出 5 个波段到新 NC
    - bbox 粗筛 -> 像元级点测 -> inside 最小包络收缩 -> 写出
    返回 (out_nc_path, (r0,r1,c0,c1), (rr0,rr1,cc0,cc1))，若 return_inside=True 还返回 inside 掩膜
    """
    assert len(lon_ring) == len(lat_ring), "lon/lat 顶点数不一致"
    if lon_ring[0] != lon_ring[-1] or lat_ring[0] != lat_ring[-1]:
        lon_ring = np.r_[lon_ring, lon_ring[0]]
        lat_ring = np.r_[lat_ring, lat_ring[0]]

    poly = Path(np.c_[lon_ring, lat_ring])

    with Dataset(goci_nc_path, "r") as ds:
        # 经纬度（弯曲网格）
        lat = ds["navigation_data"]["latitude"][:]
        lon = ds["navigation_data"]["longitude"][:]
        if lat.shape != lon.shape:
            raise RuntimeError("latitude 与 longitude 形状不一致。")

        # ① bbox 粗窗口
        minx, maxx = float(lon_ring.min()), float(lon_ring.max())
        miny, maxy = float(lat_ring.min()), float(lat_ring.max())
        bbox_mask = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
        if not np.any(bbox_mask):
            raise RuntimeError("多边形与 GOCI 网格不相交（bbox）。")
        r_idx = np.where(np.any(bbox_mask, axis=1))[0]
        c_idx = np.where(np.any(bbox_mask, axis=0))[0]
        r0, r1 = int(r_idx.min()), int(r_idx.max()) + 1
        c0, c1 = int(c_idx.min()), int(c_idx.max()) + 1

        # ② 像元级点测（边界算 inside）
        lon_win = lon[r0:r1, c0:c1]
        lat_win = lat[r0:r1, c0:c1]
        inside_full = poly.contains_points(
            np.c_[lon_win.ravel(), lat_win.ravel()],
            radius=1e-9
        ).reshape(lon_win.shape)
        if not np.any(inside_full):
            raise RuntimeError("相交区域为空（精确点测）。")

        # ③ 收缩到 inside=True 的最小行列包络
        rows_any = np.any(inside_full, axis=1)
        cols_any = np.any(inside_full, axis=0)
        rr = np.where(rows_any)[0]
        cc = np.where(cols_any)[0]
        rr0, rr1 = int(rr.min()), int(rr.max()) + 1
        cc0, cc1 = int(cc.min()), int(cc.max()) + 1

        # 最终窗口
        lon_fin = lon_win[rr0:rr1, cc0:cc1]
        lat_fin = lat_win[rr0:rr1, cc0:cc1]
        inside = inside_full[rr0:rr1, cc0:cc1]

        # ④ 写出
        if os.path.exists(out_nc_path):
            os.remove(out_nc_path)
        out = Dataset(out_nc_path, "w", format="NETCDF4")

        try:
            H, W = inside.shape
            out.createDimension("y", H)
            out.createDimension("x", W)

            # navigation_data
            nav_grp = out.createGroup("navigation_data")
            lat_var = nav_grp.createVariable("latitude", "f4", ("y", "x"), zlib=True, complevel=1)
            lon_var = nav_grp.createVariable("longitude", "f4", ("y", "x"), zlib=True, complevel=1)
            lat_var[:, :] = lat_fin.astype(np.float32)
            lon_var[:, :] = lon_fin.astype(np.float32)
            lat_var.units = "degrees_north"
            lon_var.units = "degrees_east"

            # mask
            mask_grp = out.createGroup("mask")
            mvar = mask_grp.createVariable("inside_mask", "u1", ("y", "x"), zlib=True, complevel=1)
            mvar[:, :] = inside.astype(np.uint8)
            mvar.long_name = "1 = inside polygon; 0 = outside"

            # geophysical_data
            in_geo  = ds["geophysical_data"]
            out_geo = out.createGroup("geophysical_data")

            for bi in keep_indices:
                vname = band_map[bi]
                if vname not in in_geo.variables:
                    raise KeyError(f"缺少变量 geophysical_data/{vname}")
                src_var = in_geo[vname]

                # 只读入粗窗口，再切到最终窗口
                data_win = src_var[r0:r1, c0:c1]
                # 注意：netCDF4 会自动按 scale_factor/add_offset 返回实值（MaskedArray）
                if np.ma.isMaskedArray(data_win):
                    data_win = data_win.filled(np.nan).astype(np.float32)
                else:
                    data_win = np.array(data_win, dtype=np.float32)
                data_fin = data_win[rr0:rr1, cc0:cc1]

                # 多边形外写 FillValue
                fill_value = getattr(src_var, "_FillValue", -999.0)
                data_out = data_fin.copy()
                data_out[~inside] = fill_value

                out_var = out_geo.createVariable(
                    vname, "f4", ("y", "x"),
                    zlib=True, complevel=1,
                    fill_value=float(fill_value)
                )
                out_var[:, :] = data_out

                # 复制元数据，但**不**复制缩放属性
                safe_copy_ncattrs_without_scaling(src_var, out_var)

            # 可选：复制少量全局属性
            for att in ds.ncattrs():
                if att.lower() in ("title", "summary", "history", "product_name", "product_version"):
                    out.setncattr(att, getattr(ds, att))
        finally:
            out.close()

    print(f"[OK] 已输出裁剪后的 NC：{out_nc_path}")
    print(f"     ① 初始bbox窗口：r={r0}:{r1}, c={c0}:{c1}")
    print(f"     ② inside收缩后：r={rr0}:{rr1}, c={cc0}:{cc1}  (输出尺寸={inside.shape})")
    if return_inside:
        return out_nc_path, (r0, r1, c0, c1), (rr0, rr1, cc0, cc1), inside
    return out_nc_path, (r0, r1, c0, c1), (rr0, rr1, cc0, cc1)

def show_subset_with_polygon(nc_path: str, lon_ring: np.ndarray, lat_ring: np.ndarray, out_png: str = "./goci_subset_shape.png"):
    """
    可视化：叠加多边形边界 + inside 掩膜热度，按弯曲网格正确绘制
    """
    with Dataset(nc_path, "r") as ds:
        lat = ds["navigation_data"]["latitude"][:]
        lon = ds["navigation_data"]["longitude"][:]
        inside = ds["mask"]["inside_mask"][:].astype(bool)

    plt.figure(figsize=(7, 6), dpi=130)
    plt.title("GOCI subset (curvilinear) with polygon overlay")

    # NaN 透明
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)

    show_arr = ma.masked_where(~inside, inside)
    plt.pcolormesh(lon, lat, show_arr, shading="auto", alpha=0.5, cmap=cmap)

    plt.plot(lon_ring, lat_ring, "r-", lw=1.5, label="Footprint polygon")
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Latitude (°N)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_png, bbox_inches="tight")
    plt.show()
    print(f"[INFO] 已保存叠加示意图 -> {out_png}")

# -------------------- main --------------------
if __name__ == "__main__":
    # ==== 在这里定义路径与是否显示范围 ====
    GOCI_NC = r"SR_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
    LANDSAT_TIF = r"SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_B1.TIF"
    OUT_NC = r"./goci_subset_5bands.nc"
    SHOW_PLOT = True
    OUT_PNG = "./goci_subset_shape.png"

    if not os.path.exists(GOCI_NC):
        raise FileNotFoundError(f"GOCI 文件不存在：{GOCI_NC}")
    if not os.path.exists(LANDSAT_TIF):
        raise FileNotFoundError(f"Landsat TIF 不存在：{LANDSAT_TIF}")

    # 1) 得到 footprint 多边形（WGS84）
    lon_ring, lat_ring = extract_footprint_lonlat(LANDSAT_TIF, to_crs="EPSG:4326")

    # 2) 像元级裁剪 + inside 收缩窗口（并删除缩放属性）
    out_nc, bbox_win, tight_win = clip_goci_by_polygon(
        GOCI_NC, OUT_NC, lon_ring, lat_ring,
        keep_indices=KEEP_INDICES,
        band_map=BAND_INDEX_TO_NAME,
        fill_default=-999.0
    )

    # 3) 可视化（叠加多边形边界）
    if SHOW_PLOT:
        show_subset_with_polygon(out_nc, lon_ring, lat_ring, OUT_PNG)
