#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom


def extract_footprint_lonlat(tif_path: str, to_crs: str = "EPSG:4326"):
    """
    最小流程：dataset_mask -> shapes -> transform_geom
    返回：闭合的 (lon, lat)，用于可视化或作为裁剪多边形
    """
    with rasterio.open(tif_path) as src:
        # 1) 取有效像元掩膜（0=无效，255=有效）
        m = src.dataset_mask()

        # 2) 将掩膜多边形化（只保留有效区域），并转换到目标坐标系（默认 EPSG:4326）
        geoms = [
            transform_geom(src.crs, to_crs, geom, precision=10)
            for geom, val in shapes(m, mask=(m > 0), transform=src.transform)
            if val
        ]

    # 3) 取外环顶点最多的一个多边形，作为 footprint
    rings = [np.asarray(g["coordinates"][0]) for g in geoms]
    lon, lat = rings[np.argmax([r.shape[0] for r in rings])].T

    # 4) 闭合多边形（首尾相同）
    lon = np.r_[lon, lon[0]]
    lat = np.r_[lat, lat[0]]
    return lon, lat

if __name__ == "__main__":
    # 输入：任意一个带有效掩膜（alpha 或 nodata）的 Landsat 波段/影像
    TIF_PATH = "SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif"

    lon, lat = extract_footprint_lonlat(TIF_PATH, to_crs="EPSG:4326")
    print(f"[INFO] footprint 点数：{len(lon)}")
    print(f"[INFO] 经度范围：[{lon.min():.6f}, {lon.max():.6f}]，纬度范围：[{lat.min():.6f}, {lat.max():.6f}]")

    # 输出前几个顶点坐标
    for i in range(min(10, len(lon))):  # 只显示前 10 个
        print(f"顶点 {i+1}: ({lon[i]:.6f}, {lat[i]:.6f})")

    # 仅出图时用英文
    plt.figure(figsize=(6, 6))
    plt.plot(lon, lat, "-", linewidth=2, label="Landsat footprint")
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.title("Landsat Footprint (EPSG:4326)")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()