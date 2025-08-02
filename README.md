# GOCI2-Landsat Image Matching and Upsampling

This project provides tools for processing and upsampling GOCI2 satellite imagery to match Landsat spatial resolution, with memory-optimized data handling using float16 precision.

## Features

- **GOCI2 Data Processing**: Read and process GOCI2 Level 2 satellite data
- **Landsat Data Processing**: Read and process Landsat satellite data  
- **Automatic Band Matching**: Automatically match similar wavelength bands between GOCI2 and Landsat
- **Memory Optimized**: Uses float16 precision to reduce memory usage
- **Upsampling**: Resample GOCI2 data to Landsat spatial resolution using Gaussian weighted resampling
- **Visualization**: Generate comparison plots showing original, reference, and upsampled imagery

## Requirements

```bash
pip install numpy netCDF4 matplotlib pyresample
```

## Usage

### Main Processing Script

```python
python upresample_vis_lonlatlim.py
```

This script will:
1. Load GOCI2 and Landsat data
2. Automatically match similar wavelength bands
3. Upsample GOCI2 data to Landsat resolution
4. Generate comparison visualizations
5. Save results as PNG files

### Individual Components

- `read_GOCI2_nc.py`: Read GOCI2 NetCDF files
- `read_Landsat_nc.py`: Read Landsat NetCDF files  
- `upresample.py`: Core upsampling functionality
- `resample.py`: Resampling utilities
- `upresample_vis.py`: Visualization tools

## File Structure

```
img_match/
├── upresample_vis_lonlatlim.py  # Main processing script
├── upresample_vis.py            # Visualization script
├── upresample.py                # Upsampling core
├── resample.py                  # Resampling utilities
├── read_GOCI2_nc.py            # GOCI2 data reader
├── read_Landsat_nc.py          # Landsat data reader
├── read_vis_GOCI_nc.py         # GOCI2 visualization reader
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Band Matching

The system automatically matches GOCI2 and Landsat bands with wavelength differences ≤ 20nm:

| GOCI2 (nm) | Landsat (nm) | Difference |
|------------|--------------|------------|
| 443        | 443          | 0          |
| 490        | 483          | 7          |
| 555        | 561          | 6          |
| 620        | 613          | 7          |
| 660        | 655          | 5          |
| 865        | 865          | 0          |

## Memory Optimization

- All large arrays are converted to float16 precision
- Reduces memory usage by ~50% compared to float64
- Suitable for processing large satellite datasets

## Output

The script generates comparison plots for each matched band pair:
- `goci2_upsampling_comparison_XXXtoYYYnm_lonlat_lim.png`

Each plot shows:
1. Original GOCI2 imagery (250m resolution)
2. Reference Landsat imagery (30m resolution)  
3. Upsampled GOCI2 imagery (30m resolution)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
