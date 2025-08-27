#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from datetime import datetime, timezone

ROOT = "/Users/zy/Downloads/Slot_7_2021_2025"

def normalize_time_component(t: str) -> str:
    """
    规范化时间串：去掉Z，裁剪/补齐小数到6位，返回 'HH:MM:SS.ffffff'
    兼容形如 '02:11:19Z', '02:11:19.59Z', '02:11:19.5900480Z' 等
    """
    t = t.strip().strip('"').rstrip("Z").strip()
    # 匹配 HH:MM:SS[.fraction]
    m = re.match(r"^(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?$", t)
    if not m:
        raise ValueError(f"非法时间格式: {t}")
    hh, mm, ss, frac = m.group(1), m.group(2), m.group(3), m.group(4)
    if frac is None or len(frac) == 0:
        frac = "000000"
    else:
        # 裁剪/补齐到6位（Python 的 microseconds 上限）
        if len(frac) > 6:
            frac = frac[:6]
        else:
            frac = frac.ljust(6, "0")
    return f"{hh}:{mm}:{ss}.{frac}"

def parse_mtl_datetime(mtl_path):
    """从 MTL 中读取 DATE_ACQUIRED 与 SCENE_CENTER_TIME，返回 UTC datetime。"""
    date_str = None
    time_raw = None
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 形如：KEY = VALUE（VALUE 可能带引号）
            m = re.match(r"\s*([A-Z0-9_]+)\s=\s(.*?)\s*$", line)
            if not m:
                continue
            k, v = m.group(1), m.group(2).strip()
            if k == "DATE_ACQUIRED":
                date_str = v.strip().strip('"')
            elif k == "SCENE_CENTER_TIME":
                time_raw = v.strip()
    if not date_str or not time_raw:
        raise ValueError("缺少 DATE_ACQUIRED 或 SCENE_CENTER_TIME")

    # 规范化时间
    time_norm = normalize_time_component(time_raw)

    # 组合 ISO 串并解析为 UTC
    iso = f"{date_str}T{time_norm}"
    dt = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)
    return dt

def find_mtl_in_dir(d):
    """在子目录 d 中寻找 *_MTL.txt；找到则返回绝对路径，否则返回 None。"""
    try:
        for name in os.listdir(d):
            if name.endswith("_MTL.txt"):
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    return p
    except Exception:
        pass
    return None

def main():
    if not os.path.isdir(ROOT):
        raise SystemExit(f"目录不存在: {ROOT}")

    items = []
    for name in sorted(os.listdir(ROOT)):
        sub = os.path.join(ROOT, name)
        if not os.path.isdir(sub):
            continue
        mtl = find_mtl_in_dir(sub)
        if not mtl:
            print(f"- {name}: 跳过（未找到 *_MTL.txt）")
            continue
        try:
            dt_utc = parse_mtl_datetime(mtl)
            items.append((dt_utc, name))
        except Exception as e:
            print(f"- {name}: 解析失败 -> {e}")

    items.sort(key=lambda x: x[0])
    if items:
        print("Scene acquisition times (UTC):")
        for dt, folder in items:
            # 统一打印为 ISO8601 + Z
            print(f"- {folder}: {dt.isoformat().replace('+00:00','Z')}")
    else:
        print("未解析到任何时间。")

if __name__ == "__main__":
    main()