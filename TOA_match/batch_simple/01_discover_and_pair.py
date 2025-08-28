#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版 01_discover_and_pair.py
功能：在固定根目录下发现 GOCI .nc 和 Landsat TIFF（或波段 TIFF），
从文件（NetCDF/MTL/TIFF tags）读取时间（若不可用回退到文件名解析），
按时间容差或同日匹配，并输出包含两端时间与时间差的 pairs.csv。
"""

from pathlib import Path
import re, csv, datetime as dt
import rasterio
import numpy as np
from netCDF4 import Dataset, num2date

# 配置：固定根目录与时间容差（分钟）
ROOT = Path("/Users/zy/Python_code/My_Git/img_match/SR_Imagery/Slot_7_2021_2025")
TIME_TOL_MIN = 40

# 正则用于从文件名回退解析日期
GOCI_TIME_RE = re.compile(r"(\d{8})_(\d{6})")
DATE8_RE = re.compile(r"(\d{8})")
LANDSAT_ID_RE = re.compile(r"(L[COTEM]0[89]_L1\w+_\d{6}_\d{8}_\d{8}_\d{2}_T\d)")

# ----- 时间读取函数（优先从文件/MTL/NetCDF读取） -----

def parse_goci_time_from_name(p: Path):
    """从 GOCI 文件名回退解析时间"""
    m = GOCI_TIME_RE.search(p.name)
    if not m: return None
    try:
        return dt.datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S")
    except Exception:
        return None


def get_goci_datetime(nc_path: Path):
    """从 NetCDF 全局属性直接读取时间。
    假设 GOCI L1B 具有 `time_synchro_utc` 且格式固定为 "YYYYMMDD_HHMMSS"。
    例如: 20250504_022655 -> datetime(2025-05-04 02:26:55)
    """
    with Dataset(nc_path, 'r') as ds:
        s = ds.time_synchro_utc  # e.g. "20250504_022655"
        if isinstance(s, bytes):
            s = s.decode('utf-8')
        s = s.strip()
        return dt.datetime.strptime(s, '%Y%m%d_%H%M%S')


def parse_landsat_time_from_name(p: Path):
    """从 Landsat 文件名退回解析日期（只返回 date 部分）"""
    m = LANDSAT_ID_RE.search(p.name)
    if m:
        parts = m.group(1).split('_')
        if len(parts) >= 5:
            try:
                return dt.datetime.strptime(parts[3], "%Y%m%d")
            except Exception:
                pass
    m2 = DATE8_RE.search(p.name)
    if m2:
        try:
            return dt.datetime.strptime(m2.group(1), "%Y%m%d")
        except Exception:
            pass
    return None

def get_landsat_datetime(path: Path):
    """精简版：优先从**场景目录**（或给定文件的同目录）读取 MTL 的 DATE_ACQUIRED/SCENE_CENTER_TIME。
    - 支持传入目录或单个 TIF；内部统一在 search_dir 中查找 MTL
    - DATE_ACQUIRED: YYYY-MM-DD；SCENE_CENTER_TIME: HH:MM:SS 或 HH:MM:SSZ
    - 若缺少时间，仅返回当天 00:00:00
    - 如无 MTL，回退到文件名解析
    """
    search_dir = path if path.is_dir() else path.parent
    mtl_candidates = list(search_dir.glob('*MTL*.txt')) + list(search_dir.glob('*_MTL.txt'))
    for mtl in mtl_candidates:
        try:
            txt = mtl.read_text(encoding='utf-8', errors='ignore')
            m_date = re.search(r"DATE_ACQUIRED\s*=\s*(\d{4}-\d{2}-\d{2})", txt)
            if not m_date:
                continue
            date = m_date.group(1)
            m_time = re.search(r'SCENE_CENTER_TIME\s*=\s*"?([0-2]\d:[0-5]\d:[0-5]\d)Z?"?', txt)
            if m_time:
                return dt.datetime.fromisoformat(f"{date}T{m_time.group(1)}")
            return dt.datetime.strptime(date, '%Y-%m-%d')
        except Exception:
            continue
    return parse_landsat_time_from_name(path)

# ----- bbox 读取（用于粗略空间过滤） -----

def goci_bbox(nc_path: Path):
    """从 GOCI NetCDF 的全局属性中读取经纬度边界，如果不可用则回退到读取 navigation_data arrays。
    返回 (lon_min, lon_max, lat_min, lat_max) 或 None。
    """
    try:
        with Dataset(nc_path, 'r') as ds:
            # 优先使用全局属性
            attrs = ds.ncattrs()
            keys_lon_min = ['geospatial_lon_min', 'geospatial_lonitude_min', 'image_upperleft_longitude', 'image_lowerright_longitude']
            keys_lat_min = ['geospatial_lat_min', 'geospatial_latitude_min', 'image_lowerright_latitude', 'image_upperleft_latitude']
            # try standard names
            if 'geospatial_lon_min' in attrs and 'geospatial_lon_max' in attrs and 'geospatial_lat_min' in attrs and 'geospatial_lat_max' in attrs:
                return (float(getattr(ds, 'geospatial_lon_min')),
                        float(getattr(ds, 'geospatial_lon_max')),
                        float(getattr(ds, 'geospatial_lat_min')),
                        float(getattr(ds, 'geospatial_lat_max')))
            # fallback: try image corner attrs
            if 'image_upperleft_longitude' in attrs and 'image_lowerright_longitude' in attrs and 'image_upperleft_latitude' in attrs and 'image_lowerright_latitude' in attrs:
                lon_min = float(getattr(ds, 'image_upperleft_longitude'))
                lon_max = float(getattr(ds, 'image_lowerright_longitude'))
                lat_max = float(getattr(ds, 'image_upperleft_latitude'))
                lat_min = float(getattr(ds, 'image_lowerright_latitude'))
                # ensure ordering
                return (min(lon_min, lon_max), max(lon_min, lon_max), min(lat_min, lat_max), max(lat_min, lat_max))
            # 最后回退到读取 navigation_data 变量
            if 'navigation_data' in ds.groups:
                grp = ds.groups['navigation_data']
                if 'longitude' in grp.variables and 'latitude' in grp.variables:
                    lon = grp.variables['longitude'][:]
                    lat = grp.variables['latitude'][:]
                    return (float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat)))
    except Exception:
        pass
    return None


def landsat_bbox(path: Path):
    try:
        tif_path = path
        if path.is_dir():
            # 优先常见波段；找不到则取任意 TIF
            candidates = list(path.glob('*_B1*.tif')) + list(path.glob('*.tif')) + list(path.glob('*.TIF'))
            if not candidates:
                return None
            tif_path = candidates[0]
        with rasterio.open(tif_path) as ds:
            T = ds.transform
            w, h = ds.width, ds.height
            xs = [0, w]; ys = [0, h]
            lon_list=[]; lat_list=[]
            from pyproj import Transformer
            transformer = None
            if ds.crs:
                transformer = Transformer.from_crs(ds.crs, 'EPSG:4326', always_xy=True)
            for X in xs:
                for Y in ys:
                    x = T.c + T.a*X + T.b*Y
                    y = T.f + T.d*X + T.e*Y
                    if transformer:
                        lo, la = transformer.transform(x, y)
                    else:
                        lo, la = x, y
                    lon_list.append(lo); lat_list.append(la)
            return float(min(lon_list)), float(max(lon_list)), float(min(lat_list)), float(max(lat_list))
    except Exception:
        return None


def bbox_overlap(b1, b2):
    """简单 bbox 相交判断"""
    if b1 is None or b2 is None: return 'unknown'
    a1,b1x,c1,d1 = b1; a2,b2x,c2,d2 = b2
    inter_w = max(0, min(b1x,b2x) - max(a1,a2))
    inter_h = max(0, min(d1,d2) - max(c1,c2))
    return 'yes' if inter_w>0 and inter_h>0 else 'no'

# ----- 文件发现与配对 -----

def discover(root: Path):
    """递归发现：
    - GOCI: 直接收集 .nc
    - Landsat: 返回**场景目录**（解压后的 LC08/LC09/LE07 等文件夹）。对 .tar 归档忽略。
    """
    goci_files = []
    scene_dirs_set = set()
    for p in root.rglob('*'):
        if p.is_file():
            low = p.suffix.lower()
            if low == '.nc' and 'GOCI' in p.name:
                goci_files.append(p)
            # 忽略 .tar 压缩包
            continue
        # 目录：判断是否为 Landsat 场景
        if p.is_dir():
            name = p.name.upper()
            if name.startswith(('LC08','LC09','LE07','LT08','LT09')):
                has_mtl = any(p.glob('*MTL*.txt')) or any(p.glob('*_MTL.txt'))
                has_tif = any(p.glob('*.tif')) or any(p.glob('*.TIF'))
                if has_mtl or has_tif:
                    scene_dirs_set.add(p)
    return goci_files, sorted(scene_dirs_set)


def build_pairs(goci_files, landsat_files, time_tol_min=40):
    """从文件读取时间并配对，返回记录字典列表"""
    glist = [(g, get_goci_datetime(g), goci_bbox(g)) for g in goci_files]
    llist = [(l, get_landsat_datetime(l), landsat_bbox(l)) for l in landsat_files]
    recs = []
    for lpath, ldt, lbb in llist:
        if ldt is None: continue
        for gpath, gdt, gbb in glist:
            if gdt is None: continue
            # Landsat 若为日期(00:00)则按同日匹配；否则按分钟差
            is_date_only = (ldt.hour==0 and ldt.minute==0 and ldt.second==0)
            if is_date_only:
                delta = 0.0 if ldt.date()==gdt.date() else 1e9
            else:
                delta = abs((ldt - gdt).total_seconds())/60.0
            if delta <= time_tol_min:
                recs.append({
                    'landsat_tif': str(lpath if lpath.is_dir() else lpath.parent),
                    'landsat_time': ldt.isoformat(),
                    'goci_nc': str(gpath),
                    'goci_time': gdt.isoformat(),
                    'delta_min': round(delta,1),
                    'overlap': bbox_overlap(lbb, gbb)
                })
    return recs

# ----- 主入口 -----

def main():
    if not ROOT.exists():
        print('根目录不存在：', ROOT); return
    goci_files, landsat_files = discover(ROOT)
    print(f'Found GOCI={len(goci_files)} Landsat_DIR={len(landsat_files)}')
    pairs = build_pairs(goci_files, landsat_files, TIME_TOL_MIN)
    print('Pairs=', len(pairs))
    if not pairs:
        print('No pairs.'); return
    out = ROOT / 'pairs.csv'
    fields = ['landsat_tif','landsat_time','goci_nc','goci_time','delta_min','overlap']
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(pairs)
    print('Saved', out)

if __name__ == '__main__':
    main()
