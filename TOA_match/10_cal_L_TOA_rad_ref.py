#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landsat 8/9 C2 L1 顶层大气：辐亮度/反射率 计算脚本（显式 NoData 版）
- 模式选择：
  - mode = "ref"  -> 计算 TOA 反射率（rho）
  - mode = "rad"  -> 计算 TOA 辐亮度（L）
- 从 MTL.txt 读取相应定标系数
- 无效像元统一写入 -9999.0，并在 GeoTIFF 元数据中声明 nodata=-9999.0（便于 footprint 提取）
- 仅处理指定波段，输出多波段 GeoTIFF（保持与原始 B1.TIF 一致的 CRS/transform/分辨率/尺寸）
"""

import os
import math
import numpy as np
import rasterio


NODATA_VAL = -9999.0  # 显式 NoData 数值（float32 范围内）

def find_mtl_path(folder):
    for fn in os.listdir(folder):
        if fn.upper().endswith("_MTL.TXT"):
            return os.path.join(folder, fn)
    raise FileNotFoundError("未找到 MTL 文件（*_MTL.txt）")

def parse_mtl(mtl_path):
    kv = {}
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if " = " in line:
                k, v = line.strip().split(" = ", 1)
                kv[k.strip()] = v.strip().strip('"')
    return kv

def get_coeffs(mtl, band, coeff_type):
    """
    获取指定波段的定标系数
    coeff_type: "ref"（反射率）或 "rad"（辐亮度）
    """
    if coeff_type == "ref":
        M = float(mtl[f"REFLECTANCE_MULT_BAND_{band}"])
        A = float(mtl[f"REFLECTANCE_ADD_BAND_{band}"])
    elif coeff_type == "rad":
        M = float(mtl[f"RADIANCE_MULT_BAND_{band}"])
        A = float(mtl[f"RADIANCE_ADD_BAND_{band}"])
    else:
        raise ValueError(f"不支持的系数类型: {coeff_type}（仅 'ref' 或 'rad'）")
    return M, A

def find_band_path(folder, band):
    suffix = f"_B{band}.TIF"
    for fn in os.listdir(folder):
        if fn.endswith(suffix) or fn.lower().endswith(suffix.lower()):
            return os.path.join(folder, fn)
    raise FileNotFoundError(f"未找到波段 {band} 文件（*{suffix}）")

def compute_toa_reflectance(dn, M, A, sun_elev_deg):
    # ρ' = M * DN + A ; ρ = ρ' / sin(太阳高度角)
    rho_prime = M * dn.astype(np.float32) + A
    sin_elev = math.sin(math.radians(sun_elev_deg))
    # 避免极端情况下除 0（几乎不会发生）
    if sin_elev <= 0:
        sin_elev = 1e-6
    return rho_prime / sin_elev

def compute_toa_radiance(dn, M, A):
    # L = M * DN + A
    return M * dn.astype(np.float32) + A

def main():
    # ====== 用户可选：计算模式 ======
    mode = "rad"   # "ref" -> 反射率；"rad" -> 辐亮度

    # ====== 数据目录（按需修改） ======
    root = "SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1"
    print(f"数据目录: {root}")
    print(f"计算模式: {mode}")

    # ====== 需要处理的波段（可按需修改） ======
    bands = [1, 2, 3, 4, 5]

    # 读 MTL
    mtl_path = find_mtl_path(root)
    mtl = parse_mtl(mtl_path)
    sun_elev = float(mtl["SUN_ELEVATION"])
    product_id = mtl.get("LANDSAT_PRODUCT_ID", "Landsat_C2_L1")

    # 读取第一波段确定输出空间信息（CRS/transform/size/res）
    first_band_path = find_band_path(root, bands[0])
    with rasterio.open(first_band_path) as src0:
        profile = src0.profile.copy()

    # 输出路径
    if mode == "ref":
        out_path = os.path.join(root, f"{product_id}_TOA_REF_B{'-'.join(map(str,bands))}.tif")
    else:
        out_path = os.path.join(root, f"{product_id}_TOA_RAD_B{'-'.join(map(str,bands))}.tif")

    # 更新输出 profile：显式 nodata，float32，多波段，压缩
    profile.update(
        dtype=rasterio.float32,
        count=len(bands),
        nodata=NODATA_VAL,   # ✨ 显式 NoData
        compress="LZW",
        tiled=True,           # 可选：块金字（提升 IO）
        blockxsize=512,       # 可选：常见的 tile 大小
        blockysize=512
    )

    # 写出
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, b in enumerate(bands, start=1):
            band_path = find_band_path(root, b)
            with rasterio.open(band_path) as src:
                dn = src.read(1)

                # C2 L1 常见：DN==0 为无效（含黑边/填充值等）
                mask_invalid = (dn == 0)

                if mode == "ref":
                    M, A = get_coeffs(mtl, b, "ref")
                    arr = compute_toa_reflectance(dn, M, A, sun_elev)
                else:  # "rad"
                    M, A = get_coeffs(mtl, b, "rad")
                    arr = compute_toa_radiance(dn, M, A)

                # 转 float32，并把无效像元写成显式 NoData 值
                arr = arr.astype(np.float32, copy=False)
                arr[mask_invalid] = NODATA_VAL

                dst.write(arr, indexes=i)

    print(f"✅ 输出完成: {out_path}")
    print(f"   - nodata = {NODATA_VAL}（已写入元数据）")
    print("   - CRS/transform/尺寸与 B1.TIF 一致（从首波段继承）")

if __name__ == "__main__":
    main()