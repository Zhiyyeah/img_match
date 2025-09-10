"""批量生成 HR / LR TIF 配对对比图（无需命令行参数，直接在 main() 中改配置）。

目录结构：
	SR_Imagery/tif/HR/HR_<ID>.tif
	SR_Imagery/tif/LR/LR_<ID>.tif

匹配规则：文件名去掉前缀 HR_/LR_ 后（以及 .tif）剩余部分一致。

当前版本输出双联图：
	1. LR 影像  2. HR 影像
每幅图分别拥有独立 colorbar，并关闭插值 (interpolation='nearest') 以保留原始像素，不再显示差值与直方图。

输出： <root>/compare/compare_<ID>.png

使用方法：直接运行 `python compare_hr_lr_tifs.py`，需要修改参数就在 main() 最上方 CONFIG 里编辑。
依赖：rasterio, numpy, matplotlib
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

try:
	import rasterio
except ImportError:  # 提示用户安装 rasterio
	print("[ERR] 需要安装 rasterio 库: pip install rasterio", file=sys.stderr)
	raise


def discover_pairs(root: Path) -> List[Tuple[str, Path, Path]]:
	"""发现 HR / LR 交集配对。

	Returns: list[(id, hr_path, lr_path)] (按 id 排序)
	"""
	hr_dir = root / "HR"
	lr_dir = root / "LR"
	if not hr_dir.is_dir() or not lr_dir.is_dir():
		raise FileNotFoundError(f"未找到 HR 或 LR 目录: {hr_dir} | {lr_dir}")

	def collect(d: Path, prefix: str) -> Dict[str, Path]:
		m: Dict[str, Path] = {}
		for f in d.iterdir():
			if f.is_file() and f.suffix.lower() == ".tif" and f.name.startswith(prefix):
				key = f.name[len(prefix):-4]  # 去掉前缀与 .tif
				m[key] = f
		return m

	hr_map = collect(hr_dir, "HR_")
	lr_map = collect(lr_dir, "LR_")
	ids = sorted(set(hr_map.keys()) & set(lr_map.keys()))
	pairs = [(i, hr_map[i], lr_map[i]) for i in ids]
	return pairs


def read_band(path: Path, band: int) -> np.ndarray:
	"""读取指定 1-based 波段 -> float32 (保持原值)。"""
	with rasterio.open(path) as ds:
		if band < 1 or band > ds.count:
			raise ValueError(f"{path.name} 只有 {ds.count} 个波段，无法读取 band={band}")
		arr = ds.read(band).astype(np.float32)
	return arr


def compute_display_range(hr: np.ndarray, lr: np.ndarray, vmin: float | None, vmax: float | None) -> Tuple[float, float]:
	if vmin is not None and vmax is not None:
		return vmin, vmax
	both = np.concatenate([hr.ravel(), lr.ravel()])
	finite = both[np.isfinite(both)]
	if finite.size == 0:
		return 0.0, 1.0
	if vmin is None:
		vmin = float(np.percentile(finite, 2))
	if vmax is None:
		vmax = float(np.percentile(finite, 98))
	if vmin == vmax:  # 避免色阶崩塌
		vmax = vmin + 1e-6
	return vmin, vmax


def plot_one(id_: str, hr: np.ndarray, lr: np.ndarray, out_png: Path, vmin: float, vmax: float, unit_label: str, dpi: int = 120) -> None:
	"""生成双联图并保存（仅 LR / HR，各自独立 colorbar）。"""
	fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
	cm = 'viridis'
	im0 = axes[0].imshow(lr, vmin=vmin, vmax=vmax, cmap=cm, interpolation='nearest')
	axes[0].set_title('LR')
	axes[0].axis('off')
	cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
	cbar0.set_label(unit_label)

	im1 = axes[1].imshow(hr, vmin=vmin, vmax=vmax, cmap=cm, interpolation='nearest')
	axes[1].set_title('HR')
	axes[1].axis('off')
	cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
	cbar1.set_label(unit_label)

	fig.suptitle(f"ID: {id_}", fontsize=14)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=dpi)
	plt.close(fig)


def main():  # 用户只需修改 CONFIG
	# ========= 可编辑区域 (配置) =========
	CONFIG = {
		"root": Path(r"D:\Py_Code\img_match\SR_Imagery\tif"),  # 根目录（含 HR / LR）
		"band": 2,                 # 读取的 1-based 波段序号
		"max_pairs": None,         # 限制最多处理多少对 (None=全部)
		"overwrite": False,        # 已存在 PNG 是否覆盖
		"vmin": None,              # 显示下限 (None=2% 分位)
		"vmax": None,              # 显示上限 (None=98% 分位)
		"dpi": 120,                # 输出 DPI
		"out_dir": None,           # 自定义输出目录 (None= root/compare)
		"unit_label": "Radiance (W·m⁻²·sr⁻¹·µm⁻¹)",  # 颜色条单位标签，可按需要修改
	}
	# ====================================

	root = CONFIG["root"]
	band = CONFIG["band"]
	out_dir = CONFIG["out_dir"] or (root / "compare")
	overwrite = CONFIG["overwrite"]
	max_pairs = CONFIG["max_pairs"]
	vmin_cfg = CONFIG["vmin"]
	vmax_cfg = CONFIG["vmax"]
	dpi = CONFIG["dpi"]
	unit_label = CONFIG["unit_label"]

	print(f"[CONFIG] root={root}  out={out_dir}  band={band} overwrite={overwrite}")
	pairs = discover_pairs(root)
	if not pairs:
		print("[WARN] 未找到 HR/LR 匹配对。")
		return
	print(f"[INFO] 共发现匹配对: {len(pairs)}")
	if max_pairs:
		pairs = pairs[:max_pairs]
		print(f"[INFO] 仅处理前 {len(pairs)} 个 (受 max_pairs 限制)")

	for idx, (id_, hr_path, lr_path) in enumerate(pairs, 1):
		out_png = out_dir / f"compare_{id_}.png"
		if out_png.exists() and not overwrite:
			print(f"[{idx}/{len(pairs)}] 跳过 (已存在): {out_png.name}")
			continue
		try:
			hr = read_band(hr_path, band)
			lr = read_band(lr_path, band)
		except Exception as e:  # noqa: BLE001
			print(f"[{idx}/{len(pairs)}] 读取失败 {id_}: {e}")
			continue
		if hr.shape != lr.shape:
			print(f"[{idx}/{len(pairs)}] 尺寸不一致 {id_} HR{hr.shape} LR{lr.shape} -> 跳过")
			continue
		vmin, vmax = compute_display_range(hr, lr, vmin_cfg, vmax_cfg)
		try:
			plot_one(id_, hr, lr, out_png, vmin, vmax, unit_label=unit_label, dpi=dpi)
			print(f"[{idx}/{len(pairs)}] OK -> {out_png.name} (range {vmin:.4g}~{vmax:.4g})")
		except Exception as e:  # noqa: BLE001
			print(f"[{idx}/{len(pairs)}] 生成图失败 {id_}: {e}")


if __name__ == "__main__":
	main()

