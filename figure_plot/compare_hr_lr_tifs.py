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

from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import rasterio


# 导入 SSIM（scikit-image）
from skimage.metrics import structural_similarity as _ssim


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


def read_bands(path: Path, bands: List[int]) -> np.ndarray:
	"""读取多波段并按最后一维堆叠为 (H, W, C)。忽略越界波段。"""
	with rasterio.open(path) as ds:
		valid_idx = [b for b in bands if 1 <= b <= ds.count]
		if not valid_idx:
			raise ValueError(f"{path.name} 不包含请求的波段: {bands}; 实际共有 {ds.count} 个波段")
		stack = [ds.read(b).astype(np.float32) for b in valid_idx]
		arr = np.stack(stack, axis=-1)  # (H, W, C)
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


def compute_metrics(hr: np.ndarray, lr: np.ndarray) -> Tuple[float, float]:
	"""计算图像对的 (SSIM, R²)。

	- 开头检查：若 HR 或 LR 包含 NaN，则输出 NaN 数量并返回 (NaN, NaN)。
	- 若无 NaN，直接计算 SSIM 和 R²，不再进行其他判断。
	"""
	# 开头检查 NaN
	nan_count_hr = np.sum(np.isnan(hr))
	nan_count_lr = np.sum(np.isnan(lr))
	total_nan = nan_count_hr + nan_count_lr
	if total_nan > 0:
		print(f"[WARN] 发现 NaN: HR 中 {nan_count_hr} 个, LR 中 {nan_count_lr} 个 -> 返回 NaN")
		return float("nan"), float("nan")

	# 无 NaN，直接计算
	xv = hr.ravel().astype(np.float64, copy=False)
	yv = lr.ravel().astype(np.float64, copy=False)

	# R²（相关系数平方）
	r = np.corrcoef(xv, yv)[0, 1]
	r2 = float(r * r)

	# SSIM
	dr = float(hr.max() - hr.min()) if hr.size > 0 else 1e-6
	try:
		ssim_val = float(_ssim(hr, lr, data_range=dr, gaussian_weights=True))
	except Exception:
		ssim_val = float("nan")

	return ssim_val, r2


def compute_metrics_multiband(hr_stack: np.ndarray, lr_stack: np.ndarray) -> Tuple[List[float], List[float], float, float, float, float]:
	"""逐波段计算 (SSIM, R²)，并返回每个波段结果与均值。

	返回: (ssim_list, r2_list, ssim_mean, r2_mean, r2_flat_all, ssim_channel_all)
	规则:
	- 开头检查：若 HR 或 LR 含 NaN，则打印数量并返回各通道 NaN 与均值 NaN。
	- 若无 NaN：对每个通道 c 分别计算：
		* R²：展平后皮尔逊相关系数平方（保留“展开”的计算方式）；
		* SSIM：对该通道 2D 数组，data_range=hr_band.max()-hr_band.min()，gaussian_weights=True。
	- 均值使用 np.nanmean。
	"""
	nan_count_hr = int(np.sum(np.isnan(hr_stack)))
	nan_count_lr = int(np.sum(np.isnan(lr_stack)))
	if (nan_count_hr + nan_count_lr) > 0:
		print(f"[WARN] 多波段发现 NaN: HR 中 {nan_count_hr} 个, LR 中 {nan_count_lr} 个 -> 返回 NaN")
		C = hr_stack.shape[-1] if hr_stack.ndim == 3 else 0
		return [float("nan")] * C, [float("nan")] * C, float("nan"), float("nan"), float("nan"), float("nan")

	if hr_stack.ndim != 3 or lr_stack.ndim != 3 or hr_stack.shape != lr_stack.shape:
		C = hr_stack.shape[-1] if hr_stack.ndim == 3 else 0
		return [float("nan")] * C, [float("nan")] * C, float("nan"), float("nan"), float("nan"), float("nan")

	H, W, C = hr_stack.shape
	if H * W < 9 or C < 1:
		return [float("nan")] * C, [float("nan")] * C, float("nan"), float("nan"), float("nan"), float("nan")

	ssim_list: List[float] = []
	r2_list: List[float] = []
	for c in range(C):
		x = hr_stack[..., c]
		y = lr_stack[..., c]
		# R²（逐波段展开）
		xv = x.ravel().astype(np.float64, copy=False)
		yv = y.ravel().astype(np.float64, copy=False)
		try:
			r = np.corrcoef(xv, yv)[0, 1]
			r2 = float(r * r)
		except Exception:
			r2 = float("nan")
		r2_list.append(r2)
		# SSIM（逐波段）
		try:
			dr = float(x.max() - x.min()) if x.size > 0 else 1e-6
			ssim_val = float(_ssim(x, y, data_range=dr, gaussian_weights=True))
		except Exception:
			ssim_val = float("nan")
		ssim_list.append(ssim_val)

	ssim_mean = float(np.nanmean(ssim_list)) if ssim_list else float("nan")
	r2_mean = float(np.nanmean(r2_list)) if r2_list else float("nan")

	# 整体“展开”R²（将所有像素与波段扁平化为一维向量）
	try:
		xv_all = hr_stack.ravel().astype(np.float64, copy=False)
		yv_all = lr_stack.ravel().astype(np.float64, copy=False)
		r_all = np.corrcoef(xv_all, yv_all)[0, 1]
		r2_flat_all = float(r_all * r_all)
	except Exception:
		r2_flat_all = float("nan")

	# 多通道 SSIM（将波段作为通道）
	try:
		dr_all = float(hr_stack.max() - hr_stack.min()) if hr_stack.size > 0 else 1e-6
		ssim_channel_all = float(_ssim(hr_stack, lr_stack, data_range=dr_all, channel_axis=-1, gaussian_weights=True))
	except Exception:
		ssim_channel_all = float("nan")

	return ssim_list, r2_list, ssim_mean, r2_mean, r2_flat_all, ssim_channel_all


def plot_one(id_: str, hr: np.ndarray, lr: np.ndarray, out_png: Path, vmin: float, vmax: float, unit_label: str, dpi: int = 120, hr_all: Optional[np.ndarray] = None, lr_all: Optional[np.ndarray] = None, band_index: Optional[int] = None) -> None:
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

	# 计算并展示指标：当前波段 + 逐波段 + 均值
	ssim_val, r2 = compute_metrics(hr, lr)
	def _fmt(v: float) -> str:
		return "NaN" if (not isinstance(v, (int, float)) or not np.isfinite(v)) else f"{v:.4f}"
	metrics_band = f"Band{band_index if band_index is not None else ''}: SSIM={_fmt(ssim_val)}  R²={_fmt(r2)}"
	lines = [f"ID: {id_}", metrics_band]
	if hr_all is not None and lr_all is not None:
		ssim_list, r2_list, ssim_mean, r2_mean, r2_flat_all, ssim_channel_all = compute_metrics_multiband(hr_all, lr_all)
		def _fmt3(v: float) -> str:
			return "NaN" if (not isinstance(v, (int, float)) or not np.isfinite(v)) else f"{v:.3f}"
		parts = [f"B{i}({_fmt3(s)},{_fmt3(r_)})" for i, (s, r_) in enumerate(zip(ssim_list, r2_list), start=1)]
		lines.append(" | ".join(parts))
		lines.append(f"Mean: SSIM={_fmt(ssim_mean)}  R²={_fmt(r2_mean)}")
		lines.append(f"Flattened All: R²={_fmt(r2_flat_all)}  Channel SSIM={_fmt(ssim_channel_all)}")
	else:
		lines.append("Bands: NaN  Mean: SSIM=NaN  R²=NaN")
	fig.suptitle("\n".join(lines), fontsize=12)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=dpi)
	plt.close(fig)


def main():  # 用户只需修改 CONFIG
	# ========= 可编辑区域 (配置) =========
	CONFIG = {
		"root": Path(r"D:\Py_Code\img_match\SR_Imagery\tif"),  # 根目录（含 HR / LR）
		"band": 2,                 # 读取的 1-based 波段序号
		"overall_bands": [1, 2, 3, 4, 5],  # 用于整体指标的波段集合（默认 5 波段）
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
	overall_bands: List[int] = CONFIG.get("overall_bands", [1, 2, 3, 4, 5])
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

		# 读取整体多波段
		hr_all = None
		lr_all = None
		try:
			hr_all = read_bands(hr_path, overall_bands)
			lr_all = read_bands(lr_path, overall_bands)
			if hr_all.shape != lr_all.shape:
				print(f"[{idx}/{len(pairs)}] 多波段尺寸不一致 {id_} HR{hr_all.shape} LR{lr_all.shape} -> 仅输出单波段指标")
				hr_all = None
				lr_all = None
		except Exception as e:
			print(f"[{idx}/{len(pairs)}] 多波段读取警告 {id_}: {e} -> 仅输出单波段指标")
		vmin, vmax = compute_display_range(hr, lr, vmin_cfg, vmax_cfg)
		try:
			plot_one(id_, hr, lr, out_png, vmin, vmax, unit_label=unit_label, dpi=dpi, hr_all=hr_all, lr_all=lr_all, band_index=band)
			print(f"[{idx}/{len(pairs)}] OK -> {out_png.name} (range {vmin:.4g}~{vmax:.4g})")
		except Exception as e:  # noqa: BLE001
			print(f"[{idx}/{len(pairs)}] 生成图失败 {id_}: {e}")


if __name__ == "__main__":
	main()

