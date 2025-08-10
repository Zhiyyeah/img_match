import h5py
import os

def check_with_h5py(file_path):
    """
    使用h5py检查NETCDF4文件
    """
    print(f"🔍 使用h5py检查文件: {file_path}")
    print("=" * 80)
    
    if not os.path.exists(file_path):
        print("❌ 文件不存在!")
        return
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"✅ 成功打开HDF5文件")
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"📊 数据集: {name}")
                    print(f"    形状: {obj.shape}")
                    print(f"    类型: {obj.dtype}")
                    print(f"    属性: {dict(obj.attrs)}")
                    print()
                elif isinstance(obj, h5py.Group):
                    print(f"📁 组: {name}")
                    print(f"    属性: {dict(obj.attrs)}")
                    print()
            
            print("📋 文件结构:")
            print("-" * 40)
            f.visititems(print_structure)
            
            # 尝试访问根级别的数据集
            print("\n🔍 根级别数据集:")
            print("-" * 40)
            for key in f.keys():
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    print(f"📊 {key}: {item.shape} ({item.dtype})")
                elif isinstance(item, h5py.Group):
                    print(f"📁 {key} (组)")
            
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    file_path = r"D:\Py_Code\SR_Imagery\GK2B_GOCI2_L2_20250504_021530_LA_S007_AC.nc"
    check_with_h5py(file_path) 