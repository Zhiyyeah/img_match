#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landsat 8/9 C2 L1 顶层大气：辐亮度/反射率 计算脚本
- 模式选择：
  - mode = "ref"  -> 计算 TOA 反射率（rho）
  - mode = "rad"  -> 计算 TOA 辐亮度（L）
- 从 MTL.txt 读取相应定标系数
- 使用 np.nan 作为无效值（避免可视化拉伸问题）
- 仅处理指定波段，输出多波段 GeoTIFF
"""

import os
import math
import platform
import numpy as np
import rasterio



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

def get_coeffs(mtl, band, coeff_type):
    """
    获取指定波段的定标系数
    Args:
        mtl: MTL解析字典
        band: 波段号
        coeff_type: 系数类型，"ref"表示反射率，"rad"表示辐亮度
    Returns:
        (M, A): 乘法系数和加法系数
    """
    if coeff_type == "ref":
        M = float(mtl[f"REFLECTANCE_MULT_BAND_{band}"])
        A = float(mtl[f"REFLECTANCE_ADD_BAND_{band}"])
    elif coeff_type == "rad":
        M = float(mtl[f"RADIANCE_MULT_BAND_{band}"])
        A = float(mtl[f"RADIANCE_ADD_BAND_{band}"])
    else:
        raise ValueError(f"不支持的系数类型: {coeff_type}，只能是 'ref' 或 'rad'")
    return M, A

def find_band_path(folder, band):
    suffix = f"_B{band}.TIF"
    for fn in os.listdir(folder):
        if fn.endswith(suffix) or fn.lower().endswith(suffix.lower()):
            return os.path.join(folder, fn)
    raise FileNotFoundError(f"未找到波段 {band} 文件")

def compute_toa_reflectance(dn, M, A, sun_elev_deg):
    # ρ' = M * DN + A ; ρ = ρ' / sin(太阳高度角)
    rho_prime = M * dn.astype(np.float32) + A
    return rho_prime / math.sin(math.radians(sun_elev_deg))

def compute_toa_radiance(dn, M, A):
    # L = M * DN + A
    return M * dn.astype(np.float32) + A

def main():
    # ====== 用户可选：计算模式 ======
    mode = "rad"   # "ref" -> 反射率；"rad" -> 辐亮度

    # ====== 系统路径设置 ======
    root = "SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1"
    print(f"数据目录: {root}")
    print(f"计算模式: {mode}")

    bands = [1, 2, 3, 4, 5]  # 处理的波段
    CLIP_MIN, CLIP_MAX = None, None  # 可视化裁剪范围，可设为 None 关闭
    assert mode in ("ref", "rad"), "mode 只能为 'ref' 或 'rad'"

    mtl_path = find_mtl_path(root)
    mtl = parse_mtl(mtl_path)
    sun_elev = float(mtl["SUN_ELEVATION"])
    product_id = mtl.get("LANDSAT_PRODUCT_ID", "Landsat_C2_L1")

    # 读取第一波段确定输出格式
    first_band = find_band_path(root, bands[0])
    with rasterio.open(first_band) as src0:
        profile = src0.profile.copy()

    if mode == "ref":
        out_path = os.path.join(root, f"{product_id}_TOA_REF_B{'-'.join(map(str,bands))}.tif")
    else:
        out_path = os.path.join(root, f"{product_id}_TOA_RAD_B{'-'.join(map(str,bands))}.tif")

    profile.update(
        dtype=rasterio.float32,
        count=len(bands),
        nodata=None,      # 不设 nodata，保留 NaN
        compress="LZW"
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, b in enumerate(bands, start=1):
            band_path = find_band_path(root, b)
            with rasterio.open(band_path) as src:
                dn = src.read(1)
                # DN==0 通常为无效值；C2 L1也可能有饱和等标记位，必要时可叠加QA判断
                mask_invalid = (dn == 0)

                if mode == "ref":
                    M, A = get_coeffs(mtl, b, "ref")
                    arr = compute_toa_reflectance(dn, M, A, sun_elev)
                else:  # "rad"
                    M, A = get_coeffs(mtl, b, "rad")
                    arr = compute_toa_radiance(dn, M, A)

                if CLIP_MIN is not None and CLIP_MAX is not None:
                    arr = np.clip(arr, CLIP_MIN, CLIP_MAX)

                arr = arr.astype(np.float32)
                arr[mask_invalid] = np.nan
                dst.write(arr, indexes=i)

    print(f"✅ 输出完成: {out_path}")

if __name__ == "__main__":
    main()
