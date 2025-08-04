import netCDF4 as nc

def extract_band_wavelengths(nc_file_path):
    """
    从Landsat L2W NetCDF文件中提取所有波段波长数字
    
    参数:
        nc_file_path (str): NetCDF文件路径
        
    返回:
        list: 包含所有波段波长数字的列表（如 [443, 482, ...]）
    """
    try:
        with nc.Dataset(nc_file_path, 'r') as ds:
            # 找出所有Rrs波段并提取波长数字
            wavelengths = [
                int(var.split('_')[1])  # 从"Rrs_443"中提取443
                for var in ds.variables 
                if var.startswith('Rrs_')
            ]
            
            if not wavelengths:
                raise ValueError("未找到任何Rrs波段数据")
                
            return sorted(wavelengths)  # 按波长排序
    
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {nc_file_path}")
    except Exception as e:
        raise RuntimeError(f"处理文件时出错: {str(e)}")

# 使用示例
if __name__ == "__main__":
    file_path = r"D:\Py_Code\SR_Imagery\LC09_L1TP_116035_20250504_20250504_02_T1\output\L9_OLI_2025_05_04_02_11_09_116035_L2W.nc"
    
    try:
        wavelengths = extract_band_wavelengths(file_path)
        print(wavelengths)  # 直接打印波长列表
    
    except Exception as e:
        print(f"错误: {str(e)}")