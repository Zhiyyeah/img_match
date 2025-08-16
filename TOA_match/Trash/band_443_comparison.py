#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple 443 band comparison
Compare GOCI 443 band with Landsat B2 band radiance
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def read_goci_443_band(goci_file):
    """Read GOCI 443 band data"""
    print(f"Reading GOCI file: {goci_file}")
    
    with nc.Dataset(goci_file, 'r') as dataset:
        # Find L_TOA_443 variable
        if 'L_TOA_443' in dataset.variables:
            data = dataset.variables['L_TOA_443'][:]
        else:
            # Search in groups
            for group in dataset.groups.values():
                if 'L_TOA_443' in group.variables:
                    data = group.variables['L_TOA_443'][:]
                    break
            else:
                print("L_TOA_443 variable not found")
                return None, None, None
        
        # Read coordinate information
        lat = None
        lon = None
        
        # Find latitude variable
        for var_name in ['latitude', 'lat', 'Latitude', 'Lat']:
            if var_name in dataset.variables:
                lat = dataset.variables[var_name][:]
                break
        
        # Find longitude variable
        for var_name in ['longitude', 'lon', 'Longitude', 'Lon']:
            if var_name in dataset.variables:
                lon = dataset.variables[var_name][:]
                break
        
        # If not found, try searching in groups
        if lat is None or lon is None:
            for group in dataset.groups.values():
                for var_name in ['latitude', 'lat', 'Latitude', 'Lat']:
                    if var_name in group.variables:
                        lat = group.variables[var_name][:]
                        break
                for var_name in ['longitude', 'lon', 'Longitude', 'Lon']:
                    if var_name in group.variables:
                        lon = group.variables[var_name][:]
                        break
                if lat is not None and lon is not None:
                    break
        
        print(f"GOCI 443 band data shape: {data.shape}")
        return data, lat, lon

def read_landsat_b2_radiance(landsat_file):
    """Read Landsat B2 band radiance data"""
    print(f"Reading Landsat file: {landsat_file}")
    
    with rasterio.open(landsat_file) as src:
        data = src.read(1)  # Read first band
        print(f"Landsat B2 band data shape: {data.shape}")
        return data, src

def crop_goci_to_landsat_extent(goci_data, goci_lat, goci_lon, landsat_src):
    """Crop GOCI data to Landsat extent using coordinate ranges"""
    print("Cropping GOCI data to Landsat extent...")
    
    # Get Landsat bounds in UTM coordinates
    landsat_bounds = landsat_src.bounds
    print(f"Landsat bounds (UTM): {landsat_bounds}")
    
    # Convert Landsat UTM bounds to geographic coordinates
    from rasterio.warp import transform_bounds
    landsat_crs = landsat_src.crs
    print(f"Landsat CRS: {landsat_crs}")
    
    # Convert UTM bounds to WGS84 (geographic coordinates)
    if landsat_crs.is_projected:
        # Convert from UTM to WGS84
        wgs84_crs = 'EPSG:4326'
        left, bottom, right, top = transform_bounds(landsat_crs, wgs84_crs, 
                                                   landsat_bounds.left, landsat_bounds.bottom,
                                                   landsat_bounds.right, landsat_bounds.top)
        print(f"Landsat bounds (WGS84): left={left:.6f}, bottom={bottom:.6f}, right={right:.6f}, top={top:.6f}")
    else:
        # Already in geographic coordinates
        left, bottom, right, top = landsat_bounds.left, landsat_bounds.bottom, landsat_bounds.right, landsat_bounds.top
        print(f"Landsat bounds (geographic): left={left:.6f}, bottom={bottom:.6f}, right={right:.6f}, top={top:.6f}")
    
    # Check coordinate array shapes and ranges
    print(f"GOCI lat shape: {goci_lat.shape}")
    print(f"GOCI lon shape: {goci_lon.shape}")
    print(f"GOCI data shape: {goci_data.shape}")
    print(f"GOCI lat range: {goci_lat.min():.6f} to {goci_lat.max():.6f}")
    print(f"GOCI lon range: {goci_lon.min():.6f} to {goci_lon.max():.6f}")
    
    # Find the valid coordinate range for cropping
    # We'll crop to the intersection of GOCI and Landsat bounds
    crop_left = max(left, goci_lon.min())
    crop_right = min(right, goci_lon.max())
    crop_bottom = max(bottom, goci_lat.min())
    crop_top = min(top, goci_lat.max())
    
    print(f"Crop bounds: left={crop_left:.6f}, bottom={crop_bottom:.6f}, right={crop_right:.6f}, top={crop_top:.6f}")
    
    # Check if there's any overlap
    if crop_left >= crop_right or crop_bottom >= crop_top:
        print("Warning: No overlap between GOCI and Landsat bounds!")
        return goci_data, goci_lat, goci_lon, None
    
    # Create masks for the crop region
    # Note: goci_lat and goci_lon are 2D arrays, so we need to create 2D masks
    lat_mask = (goci_lat >= crop_bottom) & (goci_lat <= crop_top)
    lon_mask = (goci_lon >= crop_left) & (goci_lon <= crop_right)
    
    # Combine masks to get the crop region
    crop_mask = lat_mask & lon_mask
    
    print(f"Crop mask sum: {np.sum(crop_mask)}")
    
    if np.sum(crop_mask) == 0:
        print("Warning: No pixels found in crop region!")
        return goci_data, goci_lat, goci_lon, None
    
    # Find the bounding box of the crop region
    # Get row and column indices where crop_mask is True
    crop_rows = np.where(np.any(crop_mask, axis=1))[0]
    crop_cols = np.where(np.any(crop_mask, axis=0))[0]
    
    if len(crop_rows) == 0 or len(crop_cols) == 0:
        print("Warning: No valid crop region found!")
        return goci_data, goci_lat, goci_lon, None
    
    # Get the actual crop bounds
    row_start, row_end = crop_rows[0], crop_rows[-1] + 1
    col_start, col_end = crop_cols[0], crop_cols[-1] + 1
    
    print(f"Crop region: rows {row_start} to {row_end}, cols {col_start} to {col_end}")
    
    # Crop the data using slice indexing
    cropped_data = goci_data[row_start:row_end, col_start:col_end]
    cropped_lat = goci_lat[row_start:row_end, col_start:col_end]
    cropped_lon = goci_lon[row_start:row_end, col_start:col_end]
    
    print(f"Cropped GOCI data shape: {cropped_data.shape}")
    print(f"Cropped GOCI lat range: {cropped_lat.min():.6f} to {cropped_lat.max():.6f}")
    print(f"Cropped GOCI lon range: {cropped_lon.min():.6f} to {cropped_lon.max():.6f}")
    
    return cropped_data, cropped_lat, cropped_lon, (crop_left, crop_bottom, crop_right, crop_top)

def create_simple_comparison(goci_data, landsat_data, lat=None, lon=None, landsat_src=None):
    """Create comparison plot of cropped GOCI vs Landsat"""
    print("Creating comparison plot...")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('443 Band Comparison: GOCI vs Landsat (Cropped)', fontsize=16, fontweight='bold')
    
    # Handle invalid values
    goci_plot = np.where(np.isfinite(goci_data), goci_data, np.nan)
    landsat_plot = np.where(landsat_data != -9999.0, landsat_data, np.nan)
    
    # Use unified range 40-100 for both plots
    vmin_unified = 40
    vmax_unified = 100
    print(f"Unified colorbar range: {vmin_unified} to {vmax_unified}")
    
    # Left plot: GOCI 443 band
    ax1 = axes[0]
    ax1.set_title('GOCI L_TOA_443 (Cropped)')
    
    # If coordinate information is available, use geographic coordinates
    if lat is not None and lon is not None:
        # Use the actual coordinate range of the data
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
        print(f"GOCI coordinate range for visualization: {extent}")
        
        im1 = ax1.imshow(goci_plot, cmap='viridis', vmin=vmin_unified, vmax=vmax_unified, 
                         extent=extent, aspect='auto', origin='upper')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
    else:
        # If no coordinates, use pixel indices
        im1 = ax1.imshow(goci_plot, cmap='viridis', vmin=vmin_unified, vmax=vmax_unified, origin='upper')
        ax1.set_xlabel('Column Index')
        ax1.set_ylabel('Row Index')
    
    # Add colorbar for GOCI plot
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, aspect=20)
    cbar1.set_label('Value', fontsize=10)
    
    # Right plot: Landsat B2 band
    ax2 = axes[1]
    ax2.set_title('Landsat B2 Radiance (W/(m²·sr·μm))')
    
    # For Landsat, use the source bounds to get proper geographic coordinates
    if landsat_src is not None:
        # Get Landsat bounds and convert to WGS84 if needed
        landsat_bounds = landsat_src.bounds
        landsat_crs = landsat_src.crs
        
        if landsat_crs.is_projected:
            # Convert from projected coordinates to WGS84
            from rasterio.warp import transform_bounds
            wgs84_crs = 'EPSG:4326'
            left, bottom, right, top = transform_bounds(landsat_crs, wgs84_crs, 
                                                       landsat_bounds.left, landsat_bounds.bottom,
                                                       landsat_bounds.right, landsat_bounds.top)
            landsat_extent = [left, right, bottom, top]
        else:
            # Already in geographic coordinates
            landsat_extent = [landsat_bounds.left, landsat_bounds.right, 
                             landsat_bounds.bottom, landsat_bounds.top]
        
        print(f"Landsat coordinate range for visualization: {landsat_extent}")
        
        im2 = ax2.imshow(landsat_plot, cmap='viridis', vmin=vmin_unified, vmax=vmax_unified, 
                         extent=landsat_extent, aspect='auto', origin='upper')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
    else:
        # If no source info, use pixel indices
        im2 = ax2.imshow(landsat_plot, cmap='viridis', vmin=vmin_unified, vmax=vmax_unified, origin='upper')
        ax2.set_xlabel('Column Index')
        ax2.set_ylabel('Row Index')
    
    # Add colorbar for Landsat plot
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=20)
    cbar2.set_label('Value', fontsize=10)
    
    # Add statistics for both plots
    valid_goci = goci_data[np.isfinite(goci_data)]
    valid_landsat = landsat_data[landsat_data != -9999.0]
    
    if len(valid_goci) > 0:
        stats_text_goci = f'Min: {valid_goci.min():.4f}\nMax: {valid_goci.max():.4f}\nMean: {valid_goci.mean():.4f}'
        ax1.text(0.02, 0.98, stats_text_goci, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if len(valid_landsat) > 0:
        stats_text_landsat = f'Min: {valid_landsat.min():.4f}\nMax: {valid_landsat.max():.4f}\nMean: {valid_landsat.mean():.4f}'
        ax2.text(0.02, 0.98, stats_text_landsat, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Ensure both plots have the same aspect ratio and consistent sizing
    for ax in axes:
        ax.set_aspect('equal')
    
    plt.tight_layout()
    
    return fig

def main():
    """Main function"""
    # File paths
    goci_file = r"D:\Py_Code\SR_Imagery\GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
    landsat_file = r"D:\Py_Code\SR_Imagery\LC09_L1TP_116035_20250504_20250504_02_T1\radiance_calibrated\LC09_L1TP_116035_20250504_20250504_02_T1_B2_radiance.tif"
    
    # Check if files exist
    if not os.path.exists(goci_file):
        print(f"Error: GOCI file does not exist: {goci_file}")
        return
    
    if not os.path.exists(landsat_file):
        print(f"Error: Landsat file does not exist: {landsat_file}")
        return
    
    try:
        # Read data
        goci_data, goci_lat, goci_lon = read_goci_443_band(goci_file)
        if goci_data is None:
            print("Unable to read GOCI data")
            return
            
        landsat_data, landsat_src = read_landsat_b2_radiance(landsat_file)
        if landsat_data is None:
            print("Unable to read Landsat data")
            return
        
        # Crop GOCI to Landsat extent
        cropped_goci_data, cropped_lat, cropped_lon, crop_bounds = crop_goci_to_landsat_extent(
            goci_data, goci_lat, goci_lon, landsat_src)
        
        # Check if cropping was successful
        if crop_bounds is None:
            print("Warning: Cropping failed, using original data for comparison")
            cropped_goci_data, cropped_lat, cropped_lon = goci_data, goci_lat, goci_lon
        
        # Create cropped comparison plot
        print("\n--- Creating Cropped Comparison Plot ---")
        fig_cropped = create_simple_comparison(cropped_goci_data, landsat_data, cropped_lat, cropped_lon, landsat_src)
        
        # Save cropped image
        output_file_cropped = "band_443_comparison_cropped.png"
        fig_cropped.savefig(output_file_cropped, dpi=300, bbox_inches='tight')
        print(f"Cropped comparison plot saved: {output_file_cropped}")
        
        # Display plot
        plt.show()
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 