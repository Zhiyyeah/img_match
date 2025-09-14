#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landsat-8/9 L1TP：辐亮度 L 与“天顶辐亮度”归一化 L_norm（仅天顶辐亮度相关）
输入：场景目录（含 *_MTL.txt 与 *_B*.TIF）
输出：
  - <scene>_TOA_RAD_30m.tif     （B1-7,9 的 TOA 辐亮度 L）
  - <scene>_ZNADIR_RAD_30m.tif  （B1-7,9 的 归一化“天顶辐亮度” L_norm）
  - quicklook_B4_L_vs_Lnorm.png （B4 的影像与直方图对比）
"""

import os
import math
from typing import Dict, Tuple, List

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============== 用户配置（按需修改） ===============
SCENE_DIR = r"/Users/zy/Python_code/My_Git/img_match/SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1"
BANDS_30M = [1, 2, 3, 4, 5, 6, 7, 9]  # 统一 30 m，排除 B8(15 m) 与 B10/11(TIRS)
# ==================================================


def find_mtl(scene_dir: str) -> str:
    base = os.path.basename(scene_dir.rstrip(os.sep))
    cand = os.path.join(scene_dir, f"{base}_MTL.txt")
    if os.path.exists(cand):
        return cand
    for name in os.listdir(scene_dir):  # 不用 glob
        if name.endswith("_MTL.txt"):
            return os.path.join(scene_dir, name)
    raise FileNotFoundError("未找到 MTL 文件 (*_MTL.txt)。")

def parse_mtl(mtl_path: str) -> Dict[str, str]:
    kv = {}
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "=" not in line: 
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip().strip('"')
    return kv

def band_path(scene_dir: str, band: int) -> str:
    base = os.path.basename(scene_dir.rstrip(os.sep))
    p = os.path.join(scene_dir, f"{base}_B{band}.TIF")
    if not os.path.exists(p):
        raise FileNotFoundError(f"缺少波段文件：{os.path.basename(p)}")
    return p

def read_dn_and_profile(path: str):
    with rasterio.open(path) as src:
        dn = src.read(1).astype(np.float32)
        mask = src.read_masks(1) > 0
        profile = src.profile
    dn[~mask] = np.nan
    return dn, profile, mask

def out_profile(template: dict, count: int):
    prof = template.copy()
    prof.update(dtype="float32", count=count, nodata=np.nan)
    return prof

def compute_L_Lnorm(dn: np.ndarray, band: int, mtl: Dict[str, str], cos_sza: float, d_es: float):
    ML = float(mtl[f"RADIANCE_MULT_BAND_{band}"])
    AL = float(mtl[f"RADIANCE_ADD_BAND_{band}"])
    L = ML * dn + AL                                  # TOA 辐亮度
    L_norm = L * (d_es ** 2) / max(cos_sza, 1e-6)     # 归一化天顶辐亮度
    return L, L_norm

def write_stack(path: str, profile: dict, arrays: List[np.ndarray], band_ids: List[int]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        for i, arr in enumerate(arrays, start=1):
            dst.write(arr.astype(np.float32), i)
            dst.set_band_description(i, f"B{band_ids[i-1]}")
    print(f"[保存] {path}")

def quicklook_compare(scene_dir: str, band_ids: List[int], Ls: List[np.ndarray], Lns: List[np.ndarray], pick_band=4):
    if pick_band not in band_ids:
        pick_band = band_ids[0]
    idx = band_ids.index(pick_band)
    L, Ln = Ls[idx], Lns[idx]

    def stretch(a):
        vmin, vmax = np.nanpercentile(a, [5, 95])
        return np.clip((a - vmin) / (vmax - vmin + 1e-6), 0, 1)

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(2, 2, 1); ax1.imshow(stretch(L));  ax1.set_title(f"B{pick_band}  L");      ax1.axis("off")
    ax2 = fig.add_subplot(2, 2, 2); ax2.imshow(stretch(Ln)); ax2.set_title(f"B{pick_band}  L_norm"); ax2.axis("off")
    ax3 = fig.add_subplot(2, 2, 3); ax3.hist(L[np.isfinite(L)].ravel(), bins=100);  ax3.set_title("L  histogram")
    ax4 = fig.add_subplot(2, 2, 4); ax4.hist(Ln[np.isfinite(Ln)].ravel(), bins=100); ax4.set_title("L_norm histogram")
    fig.suptitle("B4: Radiance L vs Radiance L_norm")
    fig.tight_layout()
    out_png = os.path.join(scene_dir, "quicklook_B4_L_vs_Lnorm.png")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[保存] {out_png}")

def main():
    scene_dir = SCENE_DIR
    base = os.path.basename(scene_dir.rstrip(os.sep))

    # 读取 MTL，获取太阳几何与日地距离
    mtl = parse_mtl(find_mtl(scene_dir))
    sun_elev = float(mtl["SUN_ELEVATION"])
    sza = 90.0 - sun_elev
    cos_sza = math.cos(math.radians(sza))
    d_es = float(mtl["EARTH_SUN_DISTANCE"])
    print("归一化因子 k =", (d_es**2)/cos_sza)

    # 用首个波段作为模板
    first_band_path = band_path(scene_dir, BANDS_30M[0])
    _, tpl_profile, _ = read_dn_and_profile(first_band_path)
    prof = out_profile(tpl_profile, count=len(BANDS_30M))

    L_stack, Ln_stack, bands_ok = [], [], []
    for b in BANDS_30M:
        dn, _, _ = read_dn_and_profile(band_path(scene_dir, b))
        L, L_norm = compute_L_Lnorm(dn, b, mtl, cos_sza, d_es)
        L_stack.append(L); Ln_stack.append(L_norm); bands_ok.append(b)

    out_L  = os.path.join(scene_dir, f"{base}_TOA_RAD_30m.tif")
    out_Ln = os.path.join(scene_dir, f"{base}_ZNADIR_RAD_30m.tif")
    write_stack(out_L,  prof, L_stack,  bands_ok)
    write_stack(out_Ln, prof, Ln_stack, bands_ok)

    # 快速对比图（B4）
    quicklook_compare(scene_dir, bands_ok, L_stack, Ln_stack, pick_band=4)

    print("\n完成。产物：")
    print("  -", out_L)
    print("  -", out_Ln)
    print("  -", os.path.join(scene_dir, "quicklook_B4_L_vs_Lnorm.png"))

if __name__ == "__main__":
    main()
