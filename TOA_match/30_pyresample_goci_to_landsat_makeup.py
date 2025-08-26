# -*- coding: utf-8 -*-
# =============================================================================
# Script Name : 30_pyresample_goci_to_landsat.py
# Generated   : 2025-08-26 05:24:05
# Source      : 30_pyresample_goci_to_landsat.ipynb
# Purpose     : Converted from notebook. Functionality preserved.
# Notes       : Markdown cells are preserved as comments for readability.
# =============================================================================

#!/usr/bin/env python
# coding: utf-8

# # 使用 pyresample 将弯曲网格 GOCI 子集重采样到 Landsat 规则网格
# 
# 本 Notebook 以教学方式演示：
# 1. 读取已经按 Landsat footprint 精确裁剪后的 GOCI 子集（弯曲/曲线网格: curvilinear lat/lon）。
# 2. 读取 Landsat 多波段 TOA 辐亮度 GeoTIFF（规则栅格）。
# 3. 利用 `pyresample` 将 GOCI 五个波段插值/匹配到 Landsat 像元中心网格（两种算法：最近邻 / 高斯加权）。
# 4. 输出结果：多波段 GeoTIFF（主输出）与可选的逐波段 `.npy` 数组及元数据 JSON。
# 5. 做统计与可视化：差值统计、直方图、示例空间图。
# 6. 给出性能与内存优化提示，并封装 `run_all()` 供一键复现。
# 
# > 建议按顺序逐节运行与阅读。你可以先了解参数，再尝试修改重采样算法与半径。

# ## 1. 安装与导入库
# 如果你的环境还没有安装 `pyresample`, `rasterio`, `netCDF4`，运行下一单元进行安装。已安装可重复执行（pip 会跳过）。

# 安装可能缺失的库（静默）

import os, json, time, math
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from pyproj import Transformer
from pyresample import geometry, kd_tree

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 130
print('库导入完成。')


# ## 2. 定义输入输出路径与参数
# 可以根据自己文件布局调整。重采样方法可以切换：`USE_GAUSSIAN=False` 表示最近邻 (KD-tree)；True 则使用高斯加权。半径参数需要根据 GOCI 原始分辨率（约 500 m）和 Landsat 30 m 调整。

# ---- 路径与参数可修改 ----
GOCI_NC = 'goci_subset_5bands.nc'  # 由裁剪脚本 20_goci_subset_rectangle.py 生成
LANDSAT_TIF = 'SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif'
OUT_TIF = '/goci_resampled_to_landsat.tif'
SAVE_NPY = True
NPY_DIR = 'resampled_npy'

# 波长映射 (GOCI 波段名与已裁剪 NC 中 geophysical_data 变量名保持一致)
GOCI_BANDS = {443:'L_TOA_443', 490:'L_TOA_490', 555:'L_TOA_555', 660:'L_TOA_660', 865:'L_TOA_865'}
LANDSAT_WAVELENGTHS = [443, 483, 561, 655, 865]  # Landsat TIF 中五个波段中心波长（近似）
PAIR_L2G = {0:443, 1:490, 2:555, 3:660, 4:865}  # Landsat 索引 -> 对应 GOCI 波长

os.makedirs(NPY_DIR, exist_ok=True)


# ## 3. 读取 GOCI 子集 NC（弯曲网格）
# 读取 latitude/longitude（curvilinear），以及 5 个波段数据；外部 FillValue 和掩膜 outside 区域替换为 `np.nan`，以便后续重采样忽略。

def read_goci_subset(nc_path: str):
    if not os.path.exists(nc_path):
        raise FileNotFoundError(nc_path)

    with Dataset(nc_path, 'r') as ds:
        nav = ds['navigation_data']
        geo = ds['geophysical_data']

        lat = np.array(nav['latitude'][:], dtype=np.float32)
        lon = np.array(nav['longitude'][:], dtype=np.float32)

        band_list = []
        for wl, vname in GOCI_BANDS.items():
            var = geo[vname]
            data = var[:]

            # 1. 如果是 MaskedArray，直接填充 NaN
            if np.ma.isMaskedArray(data):
                arr = data.filled(np.nan).astype(np.float32)
            else:
                arr = np.array(data, dtype=np.float32)

            # 2. 处理 _FillValue（兜底，防止没有被自动识别）
            fv = None
            if '_FillValue' in var.ncattrs():
                try:
                    fv = float(var.getncattr('_FillValue'))
                except Exception:
                    fv = None
            if fv is not None:
                arr = np.where(arr == fv, np.nan, arr)

            band_list.append(arr)

        # 拼接成一个 (H, W, C) 的数组
        data_array = np.stack(band_list, axis=-1)  # shape=(H, W, 5)

    print('GOCI 子集读取完成: shape=', lat.shape, '波段有效像元统计:')
    for i, wl in enumerate(sorted(GOCI_BANDS.keys())):
        valid = np.isfinite(data_array[:, :, i]).sum()
        print(f'  wl {wl}nm: valid={valid}')
    print('最终 data 类型:', type(data_array), 'shape=', data_array.shape)

    return {'data': data_array, 'lat': lat, 'lon': lon}


# 调用
goci_data = read_goci_subset(GOCI_NC)


type(goci_data['data'])


# ## 4. 读取 Landsat 多波段 TIF 并构建目标网格
# 读取 5 波段数组及其仿射变换，计算像元中心经纬度（WGS84）。

def read_landsat_tif(tif_path: str):
    """
    读取 Landsat 多波段 GeoTIFF，并生成对应的经纬度网格。
    返回:
        dict:
            {
                'data':    多波段影像数据 (bands, H, W)，float32，NaN 表示无效值
                'lon':     每个像元的经度 (H, W)
                'lat':     每个像元的纬度 (H, W)
                'transform': 仿射变换矩阵 (Affine)
                'crs':     投影坐标系 (CRS 对象)
                'dtype':   数据类型 (一般为 float32)
            }
    """
    if not os.path.exists(tif_path):
        raise FileNotFoundError(tif_path)

    # 打开 GeoTIFF 文件
    with rasterio.open(tif_path) as ds:
        # 读取所有波段数据，转为 float32
        # 输出 shape = (波段数, H, W)，例如 (5, 8000, 8000)
        stack = ds.read().astype(np.float32)

        # 将 nodata 值替换为 NaN，方便后续处理
        if ds.nodata is not None:
            stack = np.where(stack == ds.nodata, np.nan, stack)

        # 影像尺寸
        H, W = ds.height, ds.width

        # 仿射变换矩阵 (row/col -> 投影坐标)
        T = ds.transform
        # 投影坐标系 (例如 UTM)
        crs = ds.crs

        # 构建行列索引网格
        rows = np.arange(H)
        cols = np.arange(W)
        cgrid, rgrid = np.meshgrid(cols, rows)

        # 利用仿射矩阵计算像元中心点的投影坐标 (x, y)
        # 公式: x = c + a*(col+0.5) + b*(row+0.5)
        #       y = f + d*(col+0.5) + e*(row+0.5)
        x = T.c + T.a * (cgrid + 0.5) + T.b * (rgrid + 0.5)
        y = T.f + T.d * (cgrid + 0.5) + T.e * (rgrid + 0.5)

        # 将投影坐标转换为经纬度 (EPSG:4326)
        transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
        lon, lat = transformer.transform(x, y)

    # 打印一些基本信息
    print(
        'Landsat 读取完成: shape=', stack.shape,
        'lon范围=(', np.nanmin(lon), ',', np.nanmax(lon), '),',
        'lat范围=(', np.nanmin(lat), ',', np.nanmax(lat), ')'
    )

    # 返回结果字典
    return {
        'data': stack,   # 影像数据
        'lon': lon,      # 每个像元的经度
        'lat': lat,      # 每个像元的纬度
        'transform': T,  # 仿射矩阵
        'crs': crs,      # 投影坐标系
        'dtype': stack.dtype
    }

# 调用示例
landsat_data = read_landsat_tif(LANDSAT_TIF)


print(type(landsat_data['data']))


# ## 5. 构建 pyresample 几何对象 (SwathDefinition)
# 源：GOCI 弯曲网格 (lat, lon)；目标：Landsat 像元中心 (lat, lon)。本例直接用两个 SwathDefinition；如果需要投影规则网格，可改用 AreaDefinition。

# 取出数据并统一形状
#    支持 HxW 或 BxHxW；内部统一本为 (B,H,W)
# ----------------------------

# 1) 可调参数（更稳妥的起步值：以源像元为尺度，放宽到 ~3×）
ROI_METERS   = 800   # 搜索半径（米）——对 GOCI~250–500m，先放宽，确保能找到邻居
SIGMA_METERS = 320   # 高斯核宽度（米）≈ ROI 的 0.4
NEIGHBOURS   = 16
FILL_VALUE   = np.nan
NPROCS       = 1

# 2) 读取你已有的数据（当前已是 Hs, Ws, B）
goci_arr = goci_data['data']          # 形状 (Hs, Ws, B) = (669, 901, 5)
g_lat    = goci_data['lat']           # 形状 (Hs, Ws)
g_lon    = goci_data['lon']           # 形状 (Hs, Ws)
L_lat    = landsat_data['lat']        # 形状 (Ht, Wt) = (7961, 7841)
L_lon    = landsat_data['lon']        # 形状 (Ht, Wt)

print(goci_arr.shape)
gH, gW, B = goci_arr.shape
lH, lW = L_lat.shape  # 目标尺寸与 Landsat 一致
print(B, gH, gW, lH, lW)

# 3) 构建源/目标 SwathDefinition（仅一次性构建）
source_swath = geometry.SwathDefinition(lons=g_lon, lats=g_lat)
target_swath = geometry.SwathDefinition(lons=L_lon, lats=L_lat)

print("Swath 已构建：source=", g_lat.shape, " target=", L_lat.shape)

# 4) 选择一个波段做演示（先从第0个波段开始）
b = 0
src_band = goci_arr[:, :, b]          # 形状 (Hs, Ws)

# 5) 将无效值转为掩膜（MaskedArray）；如果有固定填充值可在这里叠加
src_band_masked = np.ma.array(src_band, mask=~np.isfinite(src_band))
print(f"准备完成：band={b}, src_band 形状={src_band.shape}, 掩膜比例={(src_band_masked.mask.mean()*100):.2f}%")


# ===== 第2步：执行重采样（单波段） =====
import time

t0 = time.time()

resampled_band = kd_tree.resample_gauss(
    source_swath,              # 源 swath（GOCI 经纬度）
    src_band_masked,           # 源数据（MaskedArray；已把 NaN 当作无效）
    target_swath,              # 目标 swath（Landsat 经纬度网格）
    radius_of_influence=ROI_METERS,
    sigmas=SIGMA_METERS,
    fill_value=FILL_VALUE,
    neighbours=NEIGHBOURS,
    reduce_data=True,          # <<< 显式开启剪枝，避免无效邻居导致全 NaN
    with_uncert=False,
    nprocs=NPROCS
)

t1 = time.time()
print(f"✅ 重采样完成：输出形状 = {resampled_band.shape}，用时 {t1 - t0:.2f} s")

# ---- 质量检查（统计信息）----
finite = np.isfinite(resampled_band)
ratio_finite = finite.mean() * 100.0
n_nan = np.size(resampled_band) - finite.sum()

if np.any(finite):
    vmin = np.nanmin(resampled_band)
    vmax = np.nanmax(resampled_band)
    vmean = np.nanmean(resampled_band)
else:
    vmin = vmax = vmean = np.nan

print(f"有效像元比例: {ratio_finite:.2f}%  (NaN个数: {n_nan})")
print(f"值域: min={vmin:.6g}, max={vmax:.6g}, mean={vmean:.6g}")

# ---- 最近邻基线（几何/ROI 自检）----
try:
    resampled_nn = kd_tree.resample_nearest(
        source_swath, src_band_masked, target_swath,
        radius_of_influence=ROI_METERS,
        fill_value=np.nan
    )
    print("nearest finite ratio =", float(np.isfinite(resampled_nn).mean()))
except Exception as _e:
    print("nearest baseline failed:", _e)

# ## 6. 生成对比图：GOCI 原始 vs 重采样到 Landsat vs Landsat 原始
# 说明：
#  - 仅针对当前处理的波段 b。
#  - 为避免显示过慢，对超大 Landsat 栅格做自适应抽稀 (quicklook)。
#  - 统一颜色尺度（使用 2~98 百分位裁剪，减少极端值影响）。

try:
    wl_g_list = sorted(GOCI_BANDS.keys())
    wl_g = wl_g_list[b]
    # 找到最接近的 Landsat 波段索引
    landsat_idx = int(np.argmin(np.abs(np.array(LANDSAT_WAVELENGTHS) - wl_g)))
    wl_l = LANDSAT_WAVELENGTHS[landsat_idx]

    goci_native = goci_arr[:, :, b]
    goci_resampled = resampled_band  # 已与 Landsat 尺寸一致
    landsat_native = landsat_data['data'][landsat_idx, :, :]

    # 用户指定的空间范围 (lon_min, lon_max, lat_min, lat_max)
    lon_min, lon_max = 126.3, 126.5
    lat_min, lat_max = 35.35, 35.5

    # 裁剪函数：基于 lon/lat 2D 数组计算 bounding box 并裁剪数组
    def crop_to_bounds(arr, lon, lat, lon_min, lon_max, lat_min, lat_max):
        # lon/lat are 2D arrays with same shape as arr (or target grid)
        mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        if not np.any(mask):
            return None, None, None
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]) - 1)
        cmin, cmax = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]) - 1)
        # slice and return
        arr_sub = arr[rmin:rmax+1, cmin:cmax+1]
        lon_sub = lon[rmin:rmax+1, cmin:cmax+1]
        lat_sub = lat[rmin:rmax+1, cmin:cmax+1]
        return arr_sub, lon_sub, lat_sub

    # 对 GOCI 原始（curvilinear grid）进行裁剪
    g_crop, g_lon_crop, g_lat_crop = crop_to_bounds(goci_native, g_lon, g_lat, lon_min, lon_max, lat_min, lat_max)
    if g_crop is not None:
        goci_native_plot = g_crop
        g_lon_plot = g_lon_crop
        g_lat_plot = g_lat_crop
    else:
        print('⚠️ No GOCI pixels inside requested bounds, using full GOCI extent for plotting')
        goci_native_plot = goci_native
        g_lon_plot = g_lon
        g_lat_plot = g_lat

    # 对重采样结果与 Landsat 使用 Landsat 的 lon/lat 做裁剪（它们共享网格）
    r_crop, r_lon_crop, r_lat_crop = crop_to_bounds(goci_resampled, L_lon, L_lat, lon_min, lon_max, lat_min, lat_max)
    if r_crop is not None:
        goci_resampled_plot = r_crop
        res_lon_plot = r_lon_crop
        res_lat_plot = r_lat_crop
    else:
        print('⚠️ No resampled pixels inside requested bounds, using full resampled extent for plotting')
        goci_resampled_plot = goci_resampled
        res_lon_plot = L_lon
        res_lat_plot = L_lat

    l_crop, l_lon_crop, l_lat_crop = crop_to_bounds(landsat_native, L_lon, L_lat, lon_min, lon_max, lat_min, lat_max)
    if l_crop is not None:
        landsat_native_plot = l_crop
        landsat_lon_plot = l_lon_crop
        landsat_lat_plot = l_lat_crop
    else:
        print('⚠️ No Landsat pixels inside requested bounds, using full Landsat extent for plotting')
        landsat_native_plot = landsat_native
        landsat_lon_plot = L_lon
        landsat_lat_plot = L_lat

    # 计算颜色范围（排除 NaN）
    def robust_min_max(*arrays, pmin=2, pmax=98):
        vals = np.concatenate([a[np.isfinite(a)] for a in arrays if np.isfinite(a).any()])
        if vals.size == 0:
            return 0, 1
        v1, v2 = np.percentile(vals, [pmin, pmax])
        if not np.isfinite(v1) or not np.isfinite(v2) or v1 == v2:
            v1, v2 = float(np.nanmin(vals)), float(np.nanmax(vals))
            if v1 == v2:
                v2 = v1 + 1e-6
        return v1, v2

    v1, v2 = robust_min_max(goci_native_plot, goci_resampled_plot, landsat_native_plot)

    # 自适应抽稀函数（同时返回截取后的经纬度范围 extent）
    def quicklook(arr, lon=None, lat=None, max_pixels=2_000_000):
        h, w = arr.shape
        total = h * w
        if total <= max_pixels:
            stride = 1
            arr_sub = arr
            lon_sub = lon
            lat_sub = lat
        else:
            stride = int(math.ceil(math.sqrt(total / max_pixels)))
            arr_sub = arr[::stride, ::stride]
            lon_sub = lon[::stride, ::stride] if lon is not None else None
            lat_sub = lat[::stride, ::stride] if lat is not None else None

        # 计算 extent = [minlon, maxlon, minlat, maxlat]
        if lon_sub is not None and lat_sub is not None:
            lon_vals = lon_sub[np.isfinite(lon_sub)]
            lat_vals = lat_sub[np.isfinite(lat_sub)]
            if lon_vals.size and lat_vals.size:
                extent = [float(np.nanmin(lon_vals)), float(np.nanmax(lon_vals)),
                          float(np.nanmin(lat_vals)), float(np.nanmax(lat_vals))]
            else:
                extent = None
        else:
            extent = None

        return arr_sub, stride, extent

    g_native_show, s_g, ext_g = quicklook(goci_native_plot, lon=g_lon_plot, lat=g_lat_plot)
    g_resamp_show, s_r, ext_r = quicklook(goci_resampled_plot, lon=res_lon_plot, lat=res_lat_plot)
    l_native_show, s_l, ext_l = quicklook(landsat_native_plot, lon=landsat_lon_plot, lat=landsat_lat_plot)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im0 = axes[0].imshow(g_native_show, vmin=v1, vmax=v2, cmap='viridis', extent=ext_g, origin='upper')
    axes[0].set_title(f'GOCI original {wl_g} nm\n{goci_native.shape[1]}x{goci_native.shape[0]} (stride {s_g})')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].grid(False)

    im1 = axes[1].imshow(g_resamp_show, vmin=v1, vmax=v2, cmap='viridis', extent=ext_r, origin='upper')
    axes[1].set_title(f'GOCI->Landsat resampled {wl_g} nm\n{goci_resampled.shape[1]}x{goci_resampled.shape[0]} (stride {s_r})')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].grid(False)

    im2 = axes[2].imshow(l_native_show, vmin=v1, vmax=v2, cmap='viridis', extent=ext_l, origin='upper')
    axes[2].set_title(f'Landsat original ~{wl_l} nm\n{landsat_native.shape[1]}x{landsat_native.shape[0]} (stride {s_l})')
    axes[2].set_xlabel('Longitude')
    axes[2].set_ylabel('Latitude')
    axes[2].grid(False)

    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
    cbar.set_label('Radiance (TOA)')
    fig.suptitle(f'Band {wl_g} nm comparison (GOCI original / resampled / Landsat)')
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    # 保存
    out_dir = 'figs_compare'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'compare_pyresample_band_{wl_g}nm.png')
    fig.savefig(out_path, dpi=600)
    print(f'✅ Comparison figure saved: {out_path}')
    plt.close(fig)
except Exception as e:
    print('❌ Failed to generate comparison figure:', e)

print("全图 finite ratio =", np.isfinite(resampled_band).mean())  # 全景有效比例
print("AOI finite ratio =", np.isfinite(r_crop).mean() if r_crop is not None else "AOI无交集")

plt.figure(figsize=(5,4))
plt.imshow(np.isfinite(resampled_band), origin='upper')
plt.title('Finite mask of resampled (True=valid)')
plt.gca().grid(False)
plt.tight_layout(); plt.show()

