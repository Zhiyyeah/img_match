#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landsat 8/9 C2 L1 顶层大气反射率（TOA reflectance）计算脚本
- 从 MTL.txt 读取反射率定标系数
- 计算 TOA 反射率，使用 np.nan 作为无效值（避免 SNAP 拉伸问题）
- 仅处理指定波段，输出多波段 GeoTIFF
"""

import os
import math
import platform
import numpy as np
import rasterio

# ====== 系统路径设置 ======
system_type = platform.system()
if system_type == "Windows":
    root = r"D:\Py_Code\SR_Imagery\LC09_L1TP_116035_20250504_20250504_02_T1"
elif system_type == "Darwin":
    root = "/Users/zy/Python_code/My_Git/SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1"
else:
    root = "/public/home/zyye/SR/Image_match_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1"

print(f"当前系统: {system_type}")
print(f"数据目录: {root}")

bands = [1, 2, 3, 4, 5]  # 处理的波段
CLIP_MIN, CLIP_MAX = None, None  # 可视化裁剪范围，可设为 None 关闭


def find_mtl_path(folder):
    for fn in os.listdir(folder):
        if fn.upper().endswith("_MTL.TXT"):
            return os.path.join(folder, fn)
    raise FileNotFoundError("未找到 MTL 文件")


def parse_mtl(mtl_path):
    kv = {}
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if " = " in line:
                k, v = line.strip().split(" = ", 1)
                kv[k.strip()] = v.strip().strip('"')
    return kv


def get_coeffs(mtl, band):
    M = float(mtl[f"REFLECTANCE_MULT_BAND_{band}"])
    A = float(mtl[f"REFLECTANCE_ADD_BAND_{band}"])
    return M, A


def find_band_path(folder, band):
    suffix = f"_B{band}.TIF"
    for fn in os.listdir(folder):
        if fn.endswith(suffix) or fn.lower().endswith(suffix.lower()):
            return os.path.join(folder, fn)
    raise FileNotFoundError(f"未找到波段 {band} 文件")


def compute_toa(dn, M, A, sun_elev):
    rho_prime = M * dn.astype(np.float32) + A
    return rho_prime / math.sin(math.radians(sun_elev))


def main():
    mtl_path = find_mtl_path(root)
    mtl = parse_mtl(mtl_path)
    sun_elev = float(mtl["SUN_ELEVATION"])
    product_id = mtl.get("LANDSAT_PRODUCT_ID", "Landsat_C2_L1")

    # 读取第一波段确定输出格式
    first_band = find_band_path(root, bands[0])
    with rasterio.open(first_band) as src0:
        profile = src0.profile.copy()
        height, width = src0.height, src0.width
        transform, crs = src0.transform, src0.crs

    out_path = os.path.join(root, f"{product_id}_TOA_B{'-'.join(map(str,bands))}.tif")
    profile.update(
        dtype=rasterio.float32,
        count=len(bands),
        nodata=None,  # 不设 nodata，保留 NaN
        compress="LZW"
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, b in enumerate(bands, start=1):
            band_path = find_band_path(root, b)
            with rasterio.open(band_path) as src:
                dn = src.read(1)
                mask_invalid = (dn == 0)

                M, A = get_coeffs(mtl, b)
                rho = compute_toa(dn, M, A, sun_elev)

                if CLIP_MIN is not None and CLIP_MAX is not None:
                    rho = np.clip(rho, CLIP_MIN, CLIP_MAX)

                rho = rho.astype(np.float32)
                rho[mask_invalid] = np.nan  # 用 NaN 表示无效值

                dst.write(rho, indexes=i)

    print(f"✅ 输出完成: {out_path}")


if __name__ == "__main__":
    main()
