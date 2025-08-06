#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试输出修改的脚本
验证日期提取和文件夹创建功能
"""

import os
from datetime import datetime

def test_date_extraction():
    """测试日期提取功能"""
    print("测试日期提取功能...")
    
    # 测试用例
    test_files = [
        "GK2B_GOCI2_L2_20250309_021530_LA_S007_AC.nc",
        "GK2B_GOCI2_L2_20250323_021530_LA_S009_AC.nc",
        "test_file.nc",  # 没有日期的文件
        "goci2_upsampled_multiband_20250315.nc"  # nc文件
    ]
    
    for filename in test_files:
        date_match = None
        for part in filename.split('_'):
            if len(part) == 8 and part.isdigit():
                date_match = part
                break
        
        if date_match is None:
            date_match = datetime.now().strftime('%Y%m%d')
            print(f"  文件: {filename} -> 使用当前日期: {date_match}")
        else:
            print(f"  文件: {filename} -> 提取日期: {date_match}")
        
        # 创建输出目录名
        output_dir = f"{date_match}_match"
        print(f"    输出目录: {output_dir}")

def test_directory_creation():
    """测试目录创建功能"""
    print("\n测试目录创建功能...")
    
    test_date = "20250101"
    output_dir = f"{test_date}_match"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  创建目录: {output_dir}")
    else:
        print(f"  目录已存在: {output_dir}")
    
    # 清理测试目录
    if os.path.exists(output_dir):
        os.rmdir(output_dir)
        print(f"  清理测试目录: {output_dir}")

def test_filename_generation():
    """测试文件名生成功能"""
    print("\n测试文件名生成功能...")
    
    output_dir = "20250101_match"
    date_str = "20250101"
    
    # 测试图片文件名
    g_band, l_band = 443, 443
    image_filename = f'goci2_upsampling_{g_band}to{l_band}nm_{date_str}_memory_optimized.png'
    full_image_path = os.path.join(output_dir, image_filename)
    print(f"  图片文件: {full_image_path}")
    
    # 测试nc文件名
    nc_filename = f'goci2_upsampled_multiband_{date_str}.nc'
    full_nc_path = os.path.join(output_dir, nc_filename)
    print(f"  NC文件: {full_nc_path}")

if __name__ == "__main__":
    print("="*60)
    print("测试输出修改功能")
    print("="*60)
    
    test_date_extraction()
    test_directory_creation()
    test_filename_generation()
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60) 