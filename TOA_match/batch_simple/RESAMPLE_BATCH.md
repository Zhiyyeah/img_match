# 批量上采样说明：将裁剪后的 GOCI 子集重采样到 Landsat 网格

本文档给出一个标准流程：批量读取 `02_batch_calibrate_and_clip_nocli.py` 生成的结果（每对配对包含一个 Landsat 多波段辐亮度 GeoTIFF 和一个 GOCI 子集 NetCDF），使用 `pyresample` 将 GOCI 弯曲网格数据上采样/重采样到 Landsat 的规则网格，并把结果统一输出到一个文件夹中。

## 输入

- 批处理输出根目录 `OUTPUT_ROOT`（来自 `02_batch_calibrate_and_clip_nocli.py`）：
  - 结构：`{OUTPUT_ROOT}/{landsat_scene}/`
  - 每个子目录包含：
    - `{landsat_scene}_TOA_RAD_B1-2-3-4-5.tif`（Landsat 多波段 TOA 辐亮度 GeoTIFF）
    - `{goci_basename}_subset_footprint.nc`（或等价命名；GOCI 按 footprint 精确裁剪的子集 NetCDF）

注意：较新的裁剪脚本会输出维度名 `y/x`，掩膜变量为 `mask/inside_mask`；早期版本可能是 `rows/cols` 与 `inside_footprint`。下面的示例代码做了兼容。

## 输出

- 统一输出到 `RESAMPLED_ROOT`（示例设为 `batch_resampled/`）：
  - `{RESAMPLED_ROOT}/{landsat_scene}/GOCI_on_Landsat_{goci_basename}_GAUSS.tif`（或 `NEAREST.tif`）
  - 每个 GeoTIFF 含 5 个波段（443, 490, 555, 660, 865 nm，对应输入 GOCI 子集）
  - 可选：保存逐波段 `.npy` 与记录参数的 `.json`

## 依赖

- Python 3.9+
- numpy, rasterio, netCDF4, pyproj, pyresample, matplotlib（可选，仅用于调试可视化）

安装（如需要）：

```bash
pip install numpy rasterio netCDF4 pyproj pyresample matplotlib
```

## 参数建议

- 方法：
  - 高斯加权（推荐）：`kd_tree.resample_gauss`（更平滑稳健）
  - 最近邻：`kd_tree.resample_nearest`（速度快，可能有块状伪影）
- 半径（米）：`ROI_METERS = 800`（对 GOCI ~500m，放宽到 ~800m 更稳健）
- 高斯核宽度（米）：`SIGMA_METERS = 320`（约为 ROI 的 0.4）
- 邻居数：`NEIGHBOURS = 16`
- 填充值：`FILL_VALUE = NaN`（写 GeoTIFF 时可选择 nodata）

可按场景/区域适当调整（海岸复杂区域可适度增大 ROI）。

## 批量流程概述

1. 遍历 `OUTPUT_ROOT` 的子目录，寻找成对文件：`*_TOA_RAD_B*.tif` 与 `*_subset_footprint.nc`。
2. 读取 GOCI 子集：`navigation_data/latitude, longitude`（弯曲网格）、`geophysical_data` 下 5 个波段；用 `_FillValue` 和 `mask/inside_mask` 将外部置为 NaN。
3. 读取 Landsat 多波段 TIF，计算像元中心经纬度（根据仿射矩阵 + `pyproj.Transformer`）。
4. 使用 `pyresample.geometry.SwathDefinition` 构建源/目标网格，逐波段执行 KD-tree 重采样（高斯或最近邻）。
5. 将 5 个重采样结果栈回多波段数组，按 Landsat 的空间参考写出 GeoTIFF（`float32` + `nodata`）。
6. 可选：写出 `.npy` 与参数 `.json`，记录方法与时间耗时。

## 参考实现（示例脚本片段）

```python
import os, json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine
from netCDF4 import Dataset
from pyproj import Transformer
from pyresample import geometry, kd_tree

OUTPUT_ROOT = Path("/Users/zy/python_code/My_Git/img_match/batch_outputs")  # 你的批处理输出根目录
RESAMPLED_ROOT = Path("batch_resampled")
RESAMPLED_ROOT.mkdir(parents=True, exist_ok=True)

GOCI_BANDS = {
    443: "L_TOA_443",
    490: "L_TOA_490",
    555: "L_TOA_555",
    660: "L_TOA_660",
    865: "L_TOA_865",
}

# 重采样参数
USE_GAUSSIAN = True
ROI_METERS   = 800
SIGMA_METERS = 320
NEIGHBOURS   = 16
FILL_VALUE   = np.nan
NPROCS       = 1

def read_goci_subset(nc_path: Path):
    with Dataset(nc_path, 'r') as ds:
        nav = ds['navigation_data']
        lat = np.array(nav['latitude'][:], dtype=np.float32)
        lon = np.array(nav['longitude'][:], dtype=np.float32)

        inside = None
        if 'mask' in ds.groups:
            gmask = ds['mask']
            if 'inside_mask' in gmask.variables:
                inside = gmask['inside_mask'][:].astype(bool)
            elif 'inside_footprint' in gmask.variables:
                inside = gmask['inside_footprint'][:].astype(bool)

        geo = ds['geophysical_data'] if 'geophysical_data' in ds.groups else ds
        bands = []
        for wl, name in GOCI_BANDS.items():
            if name not in geo.variables:
                raise KeyError(f"缺少 {name}")
            v = geo[name]
            a = v[:]
            if np.ma.isMaskedArray(a):
                arr = a.filled(np.nan).astype(np.float32)
            else:
                arr = np.array(a, dtype=np.float32)
            if '_FillValue' in v.ncattrs():
                fv = float(np.array(v.getncattr('_FillValue')).ravel()[0])
                arr = np.where(arr == fv, np.nan, arr)
            if inside is not None:
                arr = np.where(inside, arr, np.nan)
            bands.append(arr)
        data = np.stack(bands, axis=0)  # (5, Hs, Ws)
    return data, lat, lon

def read_landsat_tif(tif_path: Path):
    with rasterio.open(tif_path) as ds:
        stack = ds.read().astype(np.float32)  # (5, Ht, Wt)
        if ds.nodata is not None:
            stack = np.where(stack == ds.nodata, np.nan, stack)
        H, W = ds.height, ds.width
        T = ds.transform
        crs = ds.crs
        rows = np.arange(H)
        cols = np.arange(W)
        cgrid, rgrid = np.meshgrid(cols, rows)
        x = T.c + T.a*(cgrid + 0.5) + T.b*(rgrid + 0.5)
        y = T.f + T.d*(cgrid + 0.5) + T.e*(rgrid + 0.5)
        transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        lon, lat = transformer.transform(x, y)
    return stack, lon.astype(np.float64), lat.astype(np.float64), T, crs

def resample_one_pair(ls_tif: Path, goci_nc: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # 读取
    g_arr, g_lon, g_lat = read_goci_subset(goci_nc)
    L_stack, L_lon, L_lat, L_T, L_crs = read_landsat_tif(ls_tif)
    B, Ht, Wt = L_stack.shape

    # 源/目标 swath
    src_swath = geometry.SwathDefinition(lons=g_lon, lats=g_lat)
    tgt_swath = geometry.SwathDefinition(lons=L_lon, lats=L_lat)

    # 逐波段重采样
    resampled = np.full((B, Ht, Wt), np.nan, dtype=np.float32)
    for b in range(B):
        src_b = np.ma.array(g_arr[b], mask=~np.isfinite(g_arr[b]))
        if USE_GAUSSIAN:
            out_b = kd_tree.resample_gauss(
                src_swath, src_b, tgt_swath,
                radius_of_influence=ROI_METERS,
                sigmas=SIGMA_METERS,
                fill_value=FILL_VALUE,
                neighbours=NEIGHBOURS,
                reduce_data=True,
                nprocs=NPROCS
            )
        else:
            out_b = kd_tree.resample_nearest(
                src_swath, src_b, tgt_swath,
                radius_of_influence=ROI_METERS,
                fill_value=FILL_VALUE
            )
        resampled[b] = out_b.astype(np.float32)

    # 写多波段 GeoTIFF（沿用 Landsat 几何）
    out_tif = out_dir / f"GOCI_on_Landsat_{goci_nc.stem}_{'GAUSS' if USE_GAUSSIAN else 'NEAREST'}.tif"
    profile = {
        'driver': 'GTiff', 'height': Ht, 'width': Wt, 'count': B,
        'dtype': 'float32', 'crs': L_crs, 'transform': L_T,
        'nodata': np.nan, 'compress': 'deflate', 'predictor': 2
    }
    with rasterio.open(out_tif, 'w', **profile) as dst:
        dst.write(resampled)
    print('[OK] 保存重采样 TIF ->', out_tif)
    return out_tif

def discover_pairs(output_root: Path):
    pairs = []
    for scene_dir in sorted(output_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        ls = None
        goci = None
        for f in scene_dir.iterdir():
            if f.suffix.lower() == '.tif' and '_TOA_RAD_B' in f.name:
                ls = f
            if f.suffix.lower() == '.nc' and '_subset' in f.name:
                goci = f
        if ls and goci:
            pairs.append((scene_dir.name, ls, goci))
    return pairs

def run_batch():
    pairs = discover_pairs(OUTPUT_ROOT)
    print('[INFO] 待处理对数:', len(pairs))
    for scene, ls_tif, g_nc in pairs:
        out_dir = RESAMPLED_ROOT / scene
        try:
            resample_one_pair(ls_tif, g_nc, out_dir)
        except Exception as e:
            print('[ERR]', scene, e)

if __name__ == '__main__':
    run_batch()
```

> 将上述代码保存为 `run_batch_resample.py`（或放到你习惯的模块中），执行 `python run_batch_resample.py` 即可把所有配对批量上采样，并把结果放到 `batch_resampled/` 下按场景分文件夹存放。

## 命名/匹配约定

- 扫描 `OUTPUT_ROOT/{scene}/`，按文件名模式自动匹配一对：
  - Landsat: `*_TOA_RAD_B*.tif`
  - GOCI: `*_subset*.nc`
- 若同一目录存在多对（极少见），可在 `discover_pairs` 中加更严格规则，例如按最短文件名或按日期优先等。

## 性能与内存建议

- 先裁剪再重采样：确保 GOCI 已按 footprint 精确裁剪（inside 外已置 NaN），减少无效邻居搜索，明显加速。
- 合理设置 ROI/SIGMA：过小会导致空洞（找不到邻居），过大导致过平滑；建议从 800/320 起步。
- `reduce_data=True`：开启以降低 KD-tree 规模（pyresample 推荐做法）。
- 多进程 `nprocs`：根据 CPU 核心数尝试增大，但内存占用也会增加。

## 常见问题

- 输出全是 NaN：
  - 检查 ROI 是否过小，或 GOCI 子集是否与 Landsat 覆盖区不重叠（坐标系/时间不匹配）。
  - 确认 GOCI `_FillValue` 与 `inside_mask` 已转换为 NaN（源数据不应把填充值参与插值）。
- 结果错位：
  - 确认 Landsat 像元中心经纬度计算无误（仿射矩阵 + Transformer）；
  - 检查是否给错 TIF（必须是定标后与裁剪时一致的几何/场景）。
- 速度慢：
  - 先在小区域试参；或只处理 1–2 对样例；确认后再跑全量。

