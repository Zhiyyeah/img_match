#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查GOCI2文件结构的脚本
"""

import netCDF4 as nc

def explore_nc_structure(file_path):
    """
    探索NetCDF文件的结构
    """
    print(f"检查文件: {file_path}")
    print("="*60)
    
    try:
        with nc.Dataset(file_path, 'r') as dataset:
            print(f"文件存在，开始分析结构...")
            
            # 显示所有组
            print(f"\n顶级组:")
            for group_name in dataset.groups:
                print(f"  {group_name}")
            
            # 显示所有变量
            print(f"\n顶级变量:")
            for var_name in dataset.variables:
                print(f"  {var_name}")
            
            # 检查geophysical_data组
            if 'geophysical_data' in dataset.groups:
                print(f"\ngeophysical_data组的内容:")
                geo_group = dataset.groups['geophysical_data']
                
                # 显示geophysical_data的子组
                print(f"  子组:")
                for sub_group_name in geo_group.groups:
                    print(f"    {sub_group_name}")
                
                # 显示geophysical_data的变量
                print(f"  变量:")
                for var_name in geo_group.variables:
                    print(f"    {var_name}")
                
                # 检查是否有Rrs组
                if 'Rrs' in geo_group.groups:
                    print(f"\nRrs组的内容:")
                    rrs_group = geo_group.groups['Rrs']
                    
                    # 显示Rrs组的变量
                    print(f"  变量:")
                    for var_name in rrs_group.variables:
                        print(f"    {var_name}")
                else:
                    print(f"\n警告：geophysical_data组中没有Rrs子组！")
                    
                    # 检查geophysical_data组中是否有Rrs相关的变量
                    print(f"检查geophysical_data组中是否有Rrs相关变量:")
                    for var_name in geo_group.variables:
                        if 'Rrs' in var_name:
                            print(f"    ✓ 找到: {var_name}")
            
            # 检查navigation_data组
            if 'navigation_data' in dataset.groups:
                print(f"\nnavigation_data组的内容:")
                nav_group = dataset.groups['navigation_data']
                
                print(f"  变量:")
                for var_name in nav_group.variables:
                    print(f"    {var_name}")
    
    except FileNotFoundError:
        print(f"错误：文件不存在 - {file_path}")
    except Exception as e:
        print(f"错误：{str(e)}")

if __name__ == "__main__":
    # 检查当前使用的GOCI2文件
    goci_file = r"D:\Py_Code\SR_Imagery\GK2B_GOCI2_L2_20250309_021530_LA_S007_AC.nc"
    explore_nc_structure(goci_file) 