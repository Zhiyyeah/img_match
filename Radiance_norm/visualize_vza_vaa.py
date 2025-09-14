#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landsat 传感器角度影像可视化（VZA/VAA）
依赖：rasterio, numpy, matplotlib
"""
from pathlib import Path
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# 固定 MTL.txt 路径
MTL_PATH = Path("/Users/zy/Python_code/My_Git/img_match/SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_MTL.txt")

# 解析 MTL.txt 获取VZA/VAA文件名
def parse_mtl_vza_vaa(mtl_path):
    txt = mtl_path.read_text(encoding="utf-8", errors="ignore")
    pats = {
        "VZA": r'FILE_NAME_ANGLE_SENSOR_ZENITH_BAND_4\s*=\s*"?([A-Za-z0-9_\-\.]+)"?',
        "VAA": r'FILE_NAME_ANGLE_SENSOR_AZIMUTH_BAND_4\s*=\s*"?([A-Za-z0-9_\-\.]+)"?',
    }
    base = mtl_path.parent
    return {
        k: (base / m.group(1)).resolve() if (m := re.search(pat, txt)) else None
        for k, pat in pats.items()
    }

# 读取角度TIF并自动缩放到度
def load_angle(tif_path):
    with rasterio.open(tif_path) as ds:
        arr = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            arr[arr == ds.nodata] = np.nan
        vmax = (
            np.nanmax(np.abs(arr[np.isfinite(arr)]))
            if np.isfinite(arr).any()
            else 0
        )
        if vmax > 360.0:
            arr /= 100.0
        return arr

# 可视化
def show(arr, title):
    plt.figure(figsize=(7, 6))
    im = plt.imshow(arr, origin="upper")
    plt.colorbar(im, label="degrees")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# 主流程
if __name__ == "__main__":
    if not MTL_PATH.exists():
        print(f"未找到 MTL 文件: {MTL_PATH}")
        exit(1)
    angle_files = parse_mtl_vza_vaa(MTL_PATH)
    print("角度文件:")
    for k, p in angle_files.items():
        print(f"  {k}: {p if p and p.exists() else '未找到'}")
    for k, p in angle_files.items():
        if p and p.exists():
            arr = load_angle(p)
            show(arr, f"{k}: {p.name}")
