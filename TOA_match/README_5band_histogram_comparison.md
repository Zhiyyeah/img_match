# GOCI2和Landsat 5波段辐亮度直方图对比脚本

## 概述

这个脚本用于对比GOCI2和Landsat卫星的5个对应波段的TOA辐亮度数据，生成直方图对比图。

## 功能特点

- **多波段处理**：自动处理5个对应波段 [443, 490, 555, 660, 865] nm
- **智能裁剪**：使用Landsat范围对GOCI2数据进行裁剪
- **直方图对比**：生成重叠区域的辐亮度直方图对比
- **统计信息**：提供详细的统计信息（均值、标准差、分位数等）
- **跨平台支持**：自动适配Windows、macOS和Linux系统

## 输入文件要求

### GOCI2文件
- 格式：NetCDF (.nc)
- 来源：GOCI2 L1B原始文件（官网下载）
- 数据结构：
  ```
  geophysical_data/
    ├── L_TOA_443
    ├── L_TOA_490
    ├── L_TOA_555
    ├── L_TOA_660
    └── L_TOA_865
  navigation_data/
    ├── latitude
    └── longitude
  ```

### Landsat文件
- 格式：GeoTIFF (.tif)
- 来源：`cal_L_TOA_rad_ref.py`脚本的输出文件
- 要求：多波段文件，包含5个波段的TOA辐亮度数据
- 波段顺序：B1(443nm), B2(490nm), B3(555nm), B4(660nm), B5(865nm)

## 使用方法

### 1. 准备输入文件

首先确保您有以下文件：
- GOCI2 L1B原始文件（如：`GK2_GOCI2_L1B_20250504_021530_LA_S007.nc`）
- Landsat TOA辐亮度文件（如：`LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif`）

### 2. 修改文件路径

根据您的系统，修改脚本中的文件路径：

```python
# Windows系统
goci_file = r"D:\Py_Code\SR_Imagery\GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
landsat_file = r"D:\Py_Code\SR_Imagery\LC09_L1TP_116035_20250504_20250504_02_T1\LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif"

# macOS系统
goci_file = "/Users/zy/python_code/My_Git/SR_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
landsat_file = "/Users/zy/python_code/My_Git/SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif"

# Linux服务器
goci_file = "/public/home/zyye/SR/Image_match_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
landsat_file = "/public/home/zyye/SR/Image_match_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif"
```

### 3. 运行脚本

```bash
cd TOA_match
python goci_landsat_5band_histogram_comparison.py
```

### 4. 查看输出结果

脚本会在指定目录下生成5个直方图对比文件：
- `hist_443nm_comparison.png`
- `hist_490nm_comparison.png`
- `hist_555nm_comparison.png`
- `hist_660nm_comparison.png`
- `hist_865nm_comparison.png`

## 输出说明

每个直方图包含：

1. **双直方图对比**：
   - 蓝色：GOCI2裁剪后的数据
   - 橙色：Landsat重叠区域数据

2. **统计信息**：
   - 像素数量
   - 均值
   - 标准差
   - 1%和99%分位数

3. **图表元素**：
   - 清晰的标题和标签
   - 图例说明
   - 网格线
   - 统计信息文本框

## 波段对应关系

| GOCI2波长 | Landsat波段 | 中心波长 |
|-----------|-------------|----------|
| 443nm     | B1          | 443nm    |
| 490nm     | B2          | 490nm    |
| 555nm     | B3          | 555nm    |
| 660nm     | B4          | 660nm    |
| 865nm     | B5          | 865nm    |

## 依赖库

确保安装以下Python库：
```bash
pip install numpy matplotlib netCDF4 rasterio
```

## 注意事项

1. **文件路径**：确保输入文件路径正确且文件存在
2. **内存使用**：处理大文件时可能需要较多内存
3. **坐标系**：脚本自动处理坐标系转换（Landsat投影坐标系 → WGS84）
4. **数据质量**：脚本会自动处理无效值（NaN、nodata等）
5. **中文字体**：如果遇到中文显示问题，请安装相应的中文字体

## 错误处理

脚本包含完善的错误处理机制：
- 文件不存在检查
- 数据读取错误处理
- 裁剪失败处理
- 内存不足处理

如果遇到问题，请检查：
1. 文件路径是否正确
2. 文件格式是否符合要求
3. 依赖库是否正确安装
4. 系统内存是否充足

## 示例输出

成功运行后，您将看到类似以下的输出：

```
当前系统: Darwin
GOCI2文件: /Users/zy/python_code/My_Git/SR_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc
Landsat文件: /Users/zy/python_code/My_Git/SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif
输出目录: /Users/zy/python_code/My_Git/SR_Imagery/histogram_comparison_output

=== 读取Landsat元数据 ===
读取Landsat文件: /Users/zy/python_code/My_Git/SR_Imagery/LC09_L1TP_116035_20250504_20250504_02_T1/LC09_L1TP_116035_20250504_20250504_02_T1_TOA_RAD_B1-2-3-4-5.tif
读取波段: 1
Landsat波段1数据形状: (8001, 8001)

==================================================
处理波段: 443nm
==================================================
读取GOCI2文件: /Users/zy/python_code/My_Git/SR_Imagery/GK2_GOCI2_L1B_20250504_021530_LA_S007.nc
读取波段: L_TOA_443
在geophysical_data组中找到L_TOA_443
GOCI2 443nm波段数据形状: (2500, 2500)
经纬度数据形状: lat=(2500, 2500), lon=(2500, 2500)
Landsat WGS84范围: lon[116.123456, 116.234567], lat[39.123456, 39.234567]
裁剪后GOCI形状: (156, 234)
创建443nm波段直方图对比...
GOCI统计: {'n': 36504, 'mean': 67.2345, 'std': 12.3456, 'p1': 45.1234, 'p99': 89.5678}
Landsat统计: {'n': 64008001, 'mean': 65.7890, 'std': 11.2345, 'p1': 44.5678, 'p99': 87.1234}
统一直方图范围: [44.567800, 89.567800]
直方图已保存: /Users/zy/python_code/My_Git/SR_Imagery/histogram_comparison_output/hist_443nm_comparison.png
✅ 443nm波段处理完成

...

==================================================
所有波段处理完成！
输出目录: /Users/zy/python_code/My_Git/SR_Imagery/histogram_comparison_output
==================================================
``` 