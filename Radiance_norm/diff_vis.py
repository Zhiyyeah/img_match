import numpy as np
import rasterio
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# 路径
nc_path = "/Users/zy/Python_code/My_Git/img_match/batch_outputs/LC08_L1TP_116036_20240509_20240514_02_T1/GK2_GOCI2_L1B_20240509_021530_LA_S007_subset_footprint.nc"
tif_path = "/Users/zy/Python_code/My_Git/img_match/batch_outputs/LC08_L1TP_116036_20240509_20240514_02_T1/LC08_L1TP_116036_20240509_20240514_02_T1_TOA_RAD_B1-2-3-4-5.tif"

# 读取 NetCDF（假设变量名为 'Radiance'，请根据实际变量名修改）
with Dataset(nc_path, 'r') as nc:
    nc_var = nc.variables['Radiance'][:]  # 或其他变量名
    nc_img = np.squeeze(nc_var)

# 读取 GeoTIFF（取第一个波段为例）
with rasterio.open(tif_path) as src:
    tif_img = src.read(1).astype(np.float32)
    tif_img[tif_img == src.nodata] = np.nan

# 可视化
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(nc_img, cmap='viridis')
plt.title('GOCI2 NC 影像')
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(tif_img, cmap='viridis')
plt.title('Landsat TOA RAD')
plt.colorbar()

plt.subplot(1,3,3)
diff = nc_img - tif_img
plt.imshow(diff, cmap='bwr', vmin=-np.nanmax(np.abs(diff)), vmax=np.nanmax(np.abs(diff)))
plt.title('差值 (GOCI2 - Landsat)')
plt.colorbar()

plt.tight_layout()
plt.show()