#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查裁剪后的 GOCI NC 文件是否正确：
1) 必要变量是否存在（latitude/longitude 与 3,4,6,8,12 波段）
2) 形状是否一致
3) 各波段数值统计（min/max/mean、NaN比例、_FillValue）
4) （可选）与参考 Landsat TIF 的经纬度范围是否合理相交
"""

import os
import numpy as np
from netCDF4 import Dataset
import rasterio
from rasterio.warp import transform_bounds

# 期望保留波段（1-based 索引）
KEEP_INDICES = [3, 4, 6, 8, 12]
BAND_INDEX_TO_NAME = {
    1: "L_TOA_380",
    2: "L_TOA_412",
    3: "L_TOA_443",
    4: "L_TOA_490",
    5: "L_TOA_510",
    6: "L_TOA_555",
    7: "L_TOA_620",
    8: "L_TOA_660",
    9: "L_TOA_680",
    10: "L_TOA_709",
    11: "L_TOA_745",
    12: "L_TOA_865",
}
KEEP_NAMES = [BAND_INDEX_TO_NAME[i] for i in KEEP_INDICES]

def ref_bounds_wgs84(ref_tif):
    with rasterio.open(ref_tif) as ds:
        b = ds.bounds
        crs = ds.crs
    # 将参考范围转换到 WGS84 经纬度
    minx, miny, maxx, maxy = transform_bounds(crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21)
    return float(minx), float(miny), float(maxx), float(maxy)

def stats_ignore_fill(arr, fill_value=None):
    arr = np.array(arr)
    if fill_value is not None:
        arr = np.where(arr == fill_value, np.nan, arr)
    finite = np.isfinite(arr)
    n_total = arr.size
    n_finite = np.count_nonzero(finite)
    n_nan = n_total - n_finite
    if n_finite > 0:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
        vmean = float(np.nanmean(arr))
    else:
        vmin = vmax = vmean = np.nan
    return {
        "total": int(n_total),
        "finite": int(n_finite),
        "nan": int(n_nan),
        "nan_ratio": float(n_nan / n_total),
        "min": vmin,
        "max": vmax,
        "mean": vmean,
    }

def check_nc(nc_path, ref_tif=None):
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"找不到 NC 文件：{nc_path}")
    if ref_tif and not os.path.exists(ref_tif):
        raise FileNotFoundError(f"找不到参考 TIF：{ref_tif}")

    report = {"file": nc_path, "ok": True, "messages": []}

    with Dataset(nc_path, "r") as nc:
        # 1) 必要变量存在性检查
        try:
            lat = nc["latitude"][:]
            lon = nc["longitude"][:]
        except Exception as e:
            report["ok"] = False
            report["messages"].append(f"[ERROR] 缺少 latitude/longitude：{e}")
            return report

        # 2) 形状一致性
        ny, nx = lat.shape
        if lon.shape != (ny, nx):
            report["ok"] = False
            report["messages"].append(f"[ERROR] lat/lon 形状不一致：lat={lat.shape}, lon={lon.shape}")
            return report
        report["messages"].append(f"[OK] 形状：ny={ny}, nx={nx}")

        # 3) 波段存在性与形状检查
        missing = [n for n in KEEP_NAMES if n not in nc.variables]
        if missing:
            report["ok"] = False
            report["messages"].append(f"[ERROR] 缺少波段变量：{missing}")
            return report

        band_stats = {}
        for name in KEEP_NAMES:
            var = nc.variables[name]
            if var.shape != (ny, nx):
                report["ok"] = False
                report["messages"].append(f"[ERROR] {name} 形状不一致：{var.shape} != {(ny, nx)}")
                continue

            fv = getattr(var, "_FillValue", None)
            units = getattr(var, "units", "NA")
            s = stats_ignore_fill(var[:], fv)
            band_stats[name] = {"units": units, "_FillValue": fv, **s}

        if not report["ok"]:
            return report

        # 4) lat/lon 数值范围合理性
        lat_s = stats_ignore_fill(lat)
        lon_s = stats_ignore_fill(lon)
        report["messages"].append(
            f"[OK] 经纬度范围：lon[{lon_s['min']:.6f},{lon_s['max']:.6f}], lat[{lat_s['min']:.6f},{lat_s['max']:.6f}]"
        )

        # 5) （可选）与参考范围的相交性检查
        if ref_tif:
            rminx, rminy, rmaxx, rmaxy = ref_bounds_wgs84(ref_tif)
            # 以数据本身的经纬度范围作为裁剪结果 bbox
            dminx, dmaxx = lon_s["min"], lon_s["max"]
            dminy, dmaxy = lat_s["min"], lat_s["max"]

            # 判断相交
            intersects = not (dmaxx < rminx or dminx > rmaxx or dmaxy < rminy or dminy > rmaxy)
            if intersects:
                report["messages"].append(
                    "[OK] 与参考范围相交："
                    f"参考 lon[{rminx:.6f},{rmaxx:.6f}], lat[{rminy:.6f},{rmaxy:.6f}]"
                )
            else:
                report["ok"] = False
                report["messages"].append(
                    "[ERROR] 裁剪结果经纬度范围与参考范围不相交，可能裁剪出错或坐标异常。"
                )

        # 6) 输出每个波段的统计
        report["messages"].append("\n[波段统计] (忽略 _FillValue / NaN)")
        for name in KEEP_NAMES:
            s = band_stats[name]
            report["messages"].append(
                f"  - {name:>10s} | units={s['units']}, _FillValue={s['_FillValue']}"
                f" | min={s['min']:.6g}, max={s['max']:.6g}, mean={s['mean']:.6g}"
                f" | NaN比例={s['nan_ratio']:.4f} (finite={s['finite']}/{s['total']})"
            )

    return report

def main():
    # ===== 修改这里 =====
    nc_path = r"./goci_subset_5bands.nc"  # 你的裁剪后 NC 路径
    ref_tif = r"SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif"
    # ref_tif = ""  # 若不想做范围核对，可设为空字符串
    # ===================

    if not ref_tif:
        ref_tif = None

    report = check_nc(nc_path, ref_tif)

    print("=" * 80)
    print(f"文件：{report['file']}")
    print(f"结果：{'通过' if report['ok'] else '有问题'}")
    print("-" * 80)
    for msg in report["messages"]:
        print(msg)
    print("=" * 80)

    # 若希望脚本在失败时返回非零退出码，可取消下行注释
    # import sys; sys.exit(0 if report["ok"] else 1)

if __name__ == "__main__":
    main()
