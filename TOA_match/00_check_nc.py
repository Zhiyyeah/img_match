import netCDF4 as nc
import numpy as np
import platform

def analyze_goci2_file(file_path):
    """
    分析GOCI2 L1B NetCDF文件的结构和变量信息
    """
    try:
        # 打开NetCDF文件
        dataset = nc.Dataset(file_path, 'r')
        
        print("=" * 80)
        print(f"文件路径: {file_path}")
        print("=" * 80)
        
        # 1. 显示全局属性
        print("\n📋 全局属性:")
        print("-" * 50)
        for attr_name in dataset.ncattrs():
            attr_value = dataset.getncattr(attr_name)
            print(f"{attr_name}: {attr_value}")
        
        # 2. 显示维度信息
        print("\n📏 维度信息:")
        print("-" * 50)
        for dim_name, dim in dataset.dimensions.items():
            size = len(dim) if not dim.isunlimited() else "无限制"
            print(f"{dim_name}: {size}")
        
        # 3. 显示所有变量名及其基本信息
        print("\n📊 根级别变量列表:")
        print("-" * 50)
        if dataset.variables:
            for var_name, var in dataset.variables.items():
                # 获取变量的维度、数据类型和形状
                dimensions = var.dimensions
                dtype = var.dtype
                shape = var.shape
                
                print(f"\n变量名: {var_name}")
                print(f"  - 数据类型: {dtype}")
                print(f"  - 维度: {dimensions}")
                print(f"  - 形状: {shape}")
                
                # 显示变量属性
                if var.ncattrs():
                    print("  - 属性:")
                    for attr_name in var.ncattrs():
                        attr_value = var.getncattr(attr_name)
                        # 如果属性值太长，只显示前100个字符
                        if isinstance(attr_value, str) and len(attr_value) > 100:
                            attr_value = attr_value[:100] + "..."
                        print(f"    {attr_name}: {attr_value}")
        else:
            print("根级别没有变量，变量可能存储在组中")
        
        # 4. 检查并显示组结构中的变量（HDF5格式特有）
        total_variables = len(dataset.variables)
        if hasattr(dataset, 'groups') and dataset.groups:
            print("\n🗂️ 组结构及其变量:")
            print("-" * 50)
            for group_name, group in dataset.groups.items():
                print(f"\n📁 组名: {group_name}")
                print(f"  维度数量: {len(group.dimensions)}")
                print(f"  变量数量: {len(group.variables)}")
                
                # 显示组的维度
                if group.dimensions:
                    print("  📏 组维度:")
                    for dim_name, dim in group.dimensions.items():
                        size = len(dim) if not dim.isunlimited() else "无限制"
                        print(f"    {dim_name}: {size}")
                
                # 显示组中的变量
                if group.variables:
                    print("  📊 组变量:")
                    for var_name, var in group.variables.items():
                        dimensions = var.dimensions
                        dtype = var.dtype
                        shape = var.shape
                        total_variables += 1
                        
                        print(f"\n    变量名: {var_name}")
                        print(f"      - 数据类型: {dtype}")
                        print(f"      - 维度: {dimensions}")
                        print(f"      - 形状: {shape}")
                        
                        # 显示变量属性
                        if var.ncattrs():
                            print("      - 属性:")
                            for attr_name in var.ncattrs():
                                attr_value = var.getncattr(attr_name)
                                # 如果属性值太长，只显示前100个字符
                                if isinstance(attr_value, str) and len(attr_value) > 100:
                                    attr_value = attr_value[:100] + "..."
                                print(f"        {attr_name}: {attr_value}")
                
                # 显示组的全局属性
                if group.ncattrs():
                    print("  📋 组属性:")
                    for attr_name in group.ncattrs():
                        attr_value = group.getncattr(attr_name)
                        if isinstance(attr_value, str) and len(attr_value) > 100:
                            attr_value = attr_value[:100] + "..."
                        print(f"    {attr_name}: {attr_value}")
        
        # 5. 显示文件结构摘要
        print("\n" + "=" * 80)
        print("📈 文件结构摘要:")
        print("=" * 80)
        print(f"维度数量: {len(dataset.dimensions)}")
        print(f"根级别变量数量: {len(dataset.variables)}")
        print(f"总变量数量: {total_variables}")
        print(f"组数量: {len(dataset.groups) if hasattr(dataset, 'groups') else 0}")
        print(f"全局属性数量: {len(dataset.ncattrs())}")
        
        # 关闭文件
        dataset.close()
        
        print("\n✅ 文件分析完成!")
        
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {file_path}")
        print("请检查文件路径是否正确。")
    except Exception as e:
        print(f"❌ 读取文件时出错: {str(e)}")
        print("请确保:")
        print("1. 文件是有效的NetCDF格式")
        print("2. 已安装netCDF4库: pip install netCDF4")
        print("3. 文件没有被其他程序占用")

def list_variables_only(file_path):
    """
    仅列出变量名（简化版本）- 包括组中的变量
    """
    try:
        dataset = nc.Dataset(file_path, 'r')
        print(f"\n📋 文件中的所有变量名:")
        print("-" * 50)
        
        var_count = 0
        
        # 根级别变量
        if dataset.variables:
            print("🔸 根级别变量:")
            for var_name in dataset.variables.keys():
                var_count += 1
                print(f"{var_count:2d}. {var_name}")
        
        # 组中的变量
        if hasattr(dataset, 'groups') and dataset.groups:
            for group_name, group in dataset.groups.items():
                if group.variables:
                    print(f"\n🔸 组 '{group_name}' 中的变量:")
                    for var_name in group.variables.keys():
                        var_count += 1
                        print(f"{var_count:2d}. {group_name}/{var_name}")
        
        print(f"\n总共找到 {var_count} 个变量")
        dataset.close()
    except Exception as e:
        print(f"❌ 错误: {str(e)}")

def list_variables_by_group(file_path):
    """
    按组分类列出所有变量名
    """
    try:
        dataset = nc.Dataset(file_path, 'r')
        print(f"\n📋 按组分类的变量列表:")
        print("=" * 60)
        
        # 根级别变量
        if dataset.variables:
            print("🔸 根级别变量:")
            for i, var_name in enumerate(dataset.variables.keys(), 1):
                print(f"  {i:2d}. {var_name}")
        
        # 组中的变量
        if hasattr(dataset, 'groups') and dataset.groups:
            for group_name, group in dataset.groups.items():
                if group.variables:
                    print(f"\n🔸 组: {group_name}")
                    for i, var_name in enumerate(group.variables.keys(), 1):
                        print(f"  {i:2d}. {var_name}")
        
        dataset.close()
    except Exception as e:
        print(f"❌ 错误: {str(e)}")

if __name__ == "__main__":

    file_path = r"/Users/zy/Downloads/GK2_GOCI2_L1B_20250429_021530_LA_S007.nc"

    
    print("🌊 GOCI2 L1B NetCDF文件分析工具")
    print("=" * 80)
    
    # 执行详细分析
    analyze_goci2_file(file_path)
    
    # 快速查看变量名列表
    print("\n" + "=" * 80)
    list_variables_by_group(file_path)
    
    # 也可以使用简化版本只显示变量名
    # list_variables_only(file_path)