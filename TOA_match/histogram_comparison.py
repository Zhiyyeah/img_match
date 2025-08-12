#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple band comparison (with cropping & histogram)
- Crop GOCI-2 band to Landsat extent (auto handling CRS to WGS84)
- Make side-by-side image comparison
- Make histogram comparison over the overlapping area with unified value range
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

# ----------------------------
# Reading functions
# ----------------------------

def read_goci_band(goci_file, band):
    """Read GOCI band data and lat/lon"""
    print(f"Reading GOCI file: {goci_file}")
    print(f"Reading band: L_TOA_{band}")

    with nc.Dataset(goci_file, 'r') as dataset:
        data = None
        band_var = f'L_TOA_{band}'
        if band_var in dataset.variables:
            data = dataset.variables[band_var][:]
        else:
            for group in dataset.groups.values():
                if band_var in group.variables:
                    data = group.variables[band_var][:]
                    break
        if data is None:
            print(f"{band_var} variable not found")
            return None, None, None

        lat = None; lon = None
        for var_name in ['latitude', 'lat', 'Latitude', 'Lat']:
            if var_name in dataset.variables:
                lat = dataset.variables[var_name][:]; break
        for var_name in ['longitude', 'lon', 'Longitude', 'Lon']:
            if var_name in dataset.variables:
                lon = dataset.variables[var_name][:]; break

        if lat is None or lon is None:
            for group in dataset.groups.values():
                if lat is None:
                    for var_name in ['latitude', 'lat', 'Latitude', 'Lat']:
                        if var_name in group.variables:
                            lat = group.variables[var_name][:]; break
                if lon is None:
                    for var_name in ['longitude', 'lon', 'Longitude', 'Lon']:
                        if var_name in group.variables:
                            lon = group.variables[var_name][:]; break
                if lat is not None and lon is not None:
                    break

        print(f"GOCI {band} band data shape: {data.shape}")
        return data, lat, lon


def read_landsat_b2_radiance_with_meta(landsat_file):
    """Read Landsat B2 radiance band and basic metadata needed for cropping."""
    print(f"Reading Landsat file: {landsat_file}")
    with rasterio.open(landsat_file) as src:
        data = src.read(1)
        meta = {
            "crs": src.crs,
            "bounds": src.bounds,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata
        }
        print(f"Landsat B2 band data shape: {data.shape}")
    return data, meta


# ----------------------------
# Cropping utilities
# ----------------------------

def get_landsat_lonlat_bounds(landsat_meta):
    """
    Convert Landsat outer bounds to WGS84 lon/lat.
    landsat_meta: dict with keys 'crs', 'bounds'
    """
    b = landsat_meta["bounds"]
    crs = landsat_meta["crs"]
    lonmin, latmin, lonmax, latmax = transform_bounds(
        crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21
    )
    lonmin, lonmax = (min(lonmin, lonmax), max(lonmin, lonmax))
    latmin, latmax = (min(latmin, latmax), max(latmin, latmax))
    return lonmin, latmin, lonmax, latmax


def crop_goci_to_landsat_extent(goci_data, goci_lat, goci_lon, landsat_meta):
    """
    Crop GOCI by Landsat WGS84 extent using 2D lat/lon masks.
    Returns: cropped_data, cropped_lat, cropped_lon, (left,bottom,right,top in WGS84)
    """
    if goci_data is None or goci_lat is None or goci_lon is None:
        return None, None, None, None

    lonmin, latmin, lonmax, latmax = get_landsat_lonlat_bounds(landsat_meta)
    print(f"Landsat WGS84 extent: lon[{lonmin:.6f}, {lonmax:.6f}], lat[{latmin:.6f}, {latmax:.6f}]")

    # Ensure 2D lat/lon grids
    if goci_lat.ndim == 1 and goci_lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(goci_lon, goci_lat)
    else:
        lat2d, lon2d = goci_lat, goci_lon

    inside = (lon2d >= lonmin) & (lon2d <= lonmax) & (lat2d >= latmin) & (lat2d <= latmax)
    if not np.any(inside):
        print("Warning: No overlap between GOCI and Landsat bounds.")
        return None, None, None, None

    rows = np.any(inside, axis=1)
    cols = np.any(inside, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # optional padding to keep boundary pixels
    pad = 1
    rmin = max(0, rmin - pad); rmax = min(goci_data.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad); cmax = min(goci_data.shape[1] - 1, cmax + pad)

    goci_cut = goci_data[rmin:rmax+1, cmin:cmax+1].astype(np.float64, copy=False)
    lat_cut  = lat2d[rmin:rmax+1, cmin:cmax+1]
    lon_cut  = lon2d[rmin:rmax+1, cmin:cmax+1]

    # Mask strictly outside the extent as NaN
    inside_cut = (lon_cut >= lonmin) & (lon_cut <= lonmax) & (lat_cut >= latmin) & (lat_cut <= latmax)
    goci_cut = np.where(inside_cut, goci_cut, np.nan)

    print(f"Cropped GOCI shape: {goci_cut.shape}")
    crop_bounds = (lonmin, latmin, lonmax, latmax)
    return goci_cut, lat_cut, lon_cut, crop_bounds


# ----------------------------
# Plotting: image comparison
# ----------------------------

def create_simple_comparison(goci_data, landsat_data, lat=None, lon=None, landsat_meta=None, band="443"):
    """Create side-by-side comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # GOCI plot (left)
    if lat is not None and lon is not None:
        if lat.ndim == 1 and lon.ndim == 1:
            extent = [lon.min(), lon.max(), lat.min(), lat.max()]
        else:
            extent = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
        im1 = ax1.imshow(goci_data, extent=extent, origin='lower', aspect='equal')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
    else:
        im1 = ax1.imshow(goci_data, origin='lower', aspect='equal')
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')
    ax1.set_title(f'GOCI-2 L_TOA_{band} (cropped)')

    # Add statistics
    goci_valid = goci_data[np.isfinite(goci_data)]
    if goci_valid.size > 0:
        stats_text = f"Min: {np.nanmin(goci_valid):.2f}\nMax: {np.nanmax(goci_valid):.2f}\nMean: {np.nanmean(goci_valid):.2f}"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, ha="left", va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Landsat plot (right)
    if landsat_meta is not None:
        lonmin, latmin, lonmax, latmax = get_landsat_lonlat_bounds(landsat_meta)
        extent = [lonmin, lonmax, latmin, latmax]
        im2 = ax2.imshow(landsat_data, extent=extent, origin='lower', aspect='equal')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
    else:
        im2 = ax2.imshow(landsat_data, origin='lower', aspect='equal')
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Row')
    ax2.set_title('Landsat B2 Radiance')

    # Add statistics
    nodata = landsat_meta.get("nodata", -9999.0) if landsat_meta else -9999.0
    landsat_valid = landsat_data[np.isfinite(landsat_data) & (landsat_data != nodata)]
    if landsat_valid.size > 0:
        stats_text = f"Min: {np.nanmin(landsat_valid):.2f}\nMax: {np.nanmax(landsat_valid):.2f}\nMean: {np.nanmean(landsat_valid):.2f}"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, ha="left", va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Unified color range (customize if needed)
    vmin_unified = 40
    vmax_unified = 100
    print(f"Unified colorbar range: {vmin_unified} to {vmax_unified}")
    im1.set_clim(vmin_unified, vmax_unified)
    im2.set_clim(vmin_unified, vmax_unified)

    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8); cbar1.set_label('Value')
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8); cbar2.set_label('Value')

    plt.tight_layout()
    return fig


# ----------------------------
# Histogram utilities
# ----------------------------

def read_landsat_window_in_wgs84(landsat_path, bounds_wgs84):
    """
    Read only Landsat pixels within the given WGS84 bounds.
    Returns (window_array, window_bounds_wgs84, nodata) or (None, None, None) if no overlap.
    """
    with rasterio.open(landsat_path) as src:
        # project requested WGS84 bounds into Landsat CRS to build window
        l, b, r, t = transform_bounds("EPSG:4326", src.crs, *bounds_wgs84, densify_pts=21)
        win = from_bounds(l, b, r, t, transform=src.transform)
        # intersect with full image window
        full = rasterio.windows.Window(0, 0, src.width, src.height)
        win = win.intersection(full)
        if win.width <= 0 or win.height <= 0:
            return None, None, None

        arr = src.read(1, window=win)
        nodata = src.nodata

        # compute the actual window bounds in WGS84 for reference
        win_bounds_proj = rasterio.windows.bounds(win, src.transform)
        win_bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326",
                                            win_bounds_proj[0], win_bounds_proj[1],
                                            win_bounds_proj[2], win_bounds_proj[3],
                                            densify_pts=21)
        return arr, win_bounds_wgs84, nodata


def plot_histogram_comparison(goci_cropped, landsat_path, bounds_wgs84, band="443",
                              bins=100, clip_percent=(1, 99), logy=False,
                              outfile="hist_cropped.png"):
    """Plot histogram comparison for cropped GOCI vs Landsat overlap"""
    print(f"Creating histogram comparison for band {band}...")

    # 读取 Landsat 重叠窗口
    landsat_arr, _, ls_nodata = read_landsat_window_in_wgs84(landsat_path, bounds_wgs84)
    if landsat_arr is None:
        print("No Landsat overlap window found; using full Landsat for histogram.")
        with rasterio.open(landsat_path) as src:
            landsat_arr = src.read(1)
            ls_nodata = src.nodata

    # 准备有效像素并“展平”为一维
    goci_vals = goci_cropped[np.isfinite(goci_cropped)].ravel()
    if ls_nodata is None:
        landsat_mask = np.isfinite(landsat_arr)
    else:
        landsat_mask = np.isfinite(landsat_arr) & (landsat_arr != ls_nodata)
    landsat_vals = landsat_arr[landsat_mask].ravel()

    if goci_vals.size == 0 or landsat_vals.size == 0:
        print("Warning: empty valid pixels; skip histogram.")
        return None

    # 统一分位数范围（更稳健）
    lo_g, hi_g = np.nanpercentile(goci_vals, clip_percent)
    lo_l, hi_l = np.nanpercentile(landsat_vals, clip_percent)
    lo = min(lo_g, lo_l); hi = max(hi_g, hi_l)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.nanmin([goci_vals.min(), landsat_vals.min()]))
        hi = float(np.nanmax([goci_vals.max(), landsat_vals.max()]))

    def stats(arr):
        return dict(n=arr.size, mean=float(np.nanmean(arr)), std=float(np.nanstd(arr)),
                    p1=float(np.nanpercentile(arr, 1)), p99=float(np.nanpercentile(arr, 99)))
    s_g = stats(goci_vals); s_l = stats(landsat_vals)
    print(f"GOCI stats: {s_g}")
    print(f"Landsat stats: {s_l}")
    print(f"Unified histogram range: [{lo:.6f}, {hi:.6f}]")

    # 绘制直方图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(goci_vals, bins=bins, range=(lo, hi), alpha=0.6, density=True, label=f"GOCI-2 L_TOA_{band} (cropped)")
    ax.hist(landsat_vals, bins=bins, range=(lo, hi), alpha=0.6, density=True, label="Landsat B2 (overlap)")
    ax.set_xlabel("Value"); ax.set_ylabel("Density")
    ax.set_title(f"Histogram Comparison: GOCI L_TOA_{band} (cropped) vs Landsat B2 (overlap)")
    ax.legend()
    if logy:
        ax.set_yscale("log")

    txt = (f"GOCI  n={s_g['n']}, mean={s_g['mean']:.4f}, std={s_g['std']:.4f}\n"
           f"       p1={s_g['p1']:.4f}, p99={s_g['p99']:.4f}\n"
           f"Landsat n={s_l['n']}, mean={s_l['mean']:.4f}, std={s_l['std']:.4f}\n"
           f"       p1={s_l['p1']:.4f}, p99={s_l['p99']:.4f}")
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Histogram saved: {outfile}")
    return fig


# ----------------------------
# Main
# ----------------------------

def main():
    """Main function"""
    # Define parameters directly in code
    goci_file = r"D:\Py_Code\SR_Imagery\GK2_GOCI2_L1B_20250504_021530_LA_S007.nc"
    landsat_file = r"D:\Py_Code\SR_Imagery\LC09_L1TP_116035_20250504_20250504_02_T1\radiance_calibrated\LC09_L1TP_116035_20250504_20250504_02_T1_B1_radiance.tif"
    goci_band = "443"  # Change this to analyze different bands

    # Validate band parameter
    try:
        band_num = int(goci_band)
        if band_num <= 0:
            print("Error: Band number must be a positive integer")
            return
    except ValueError:
        print("Error: Band number must be a valid integer")
        return

    if not os.path.exists(goci_file):
        print(f"Error: GOCI file does not exist: {goci_file}")
        return
    if not os.path.exists(landsat_file):
        print(f"Error: Landsat file does not exist: {landsat_file}")
        return

    try:
        # 1) Read data
        goci_data, goci_lat, goci_lon = read_goci_band(goci_file, goci_band)
        if goci_data is None:
            print("Unable to read GOCI data")
            return

        landsat_data, landsat_meta = read_landsat_b2_radiance_with_meta(landsat_file)
        if landsat_data is None:
            print("Unable to read Landsat data")
            return

        # 2) Crop GOCI to Landsat extent (WGS84)
        cropped_goci_data, cropped_lat, cropped_lon, crop_bounds = crop_goci_to_landsat_extent(
            goci_data, goci_lat, goci_lon, landsat_meta
        )
        if cropped_goci_data is None:
            print("Warning: Cropping failed; fall back to original GOCI for plotting.")
            cropped_goci_data, cropped_lat, cropped_lon = goci_data, goci_lat, goci_lon
            if cropped_lat is not None and cropped_lon is not None:
                bounds_for_hist = (float(np.nanmin(cropped_lon)), float(np.nanmin(cropped_lat)),
                                   float(np.nanmax(cropped_lon)), float(np.nanmax(cropped_lat)))
            else:
                bounds_for_hist = get_landsat_lonlat_bounds(landsat_meta)
        else:
            bounds_for_hist = crop_bounds

        # 3) Side-by-side comparison plot
        print("\n--- Creating Cropped Comparison Plot ---")
        fig_cmp = create_simple_comparison(cropped_goci_data, landsat_data,
                                           cropped_lat, cropped_lon, landsat_meta, goci_band)
        out_cmp = f"band_{goci_band}_comparison_cropped.png"
        fig_cmp.savefig(out_cmp, dpi=300, bbox_inches='tight')
        print(f"Cropped comparison plot saved: {out_cmp}")
        plt.close(fig_cmp)

        # 4) Histogram comparison over overlap area
        print("\n--- Creating Histogram Comparison ---")
        fig_hist = plot_histogram_comparison(
            goci_cropped=cropped_goci_data,
            landsat_path=landsat_file,
            bounds_wgs84=bounds_for_hist,
            band=goci_band,
            bins=100,
            clip_percent=(1, 99),
            logy=False,
            outfile=f"hist_{goci_band}_cropped.png"
        )
        if fig_hist is not None:
            plt.close(fig_hist)

        print("\nAll done.")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
