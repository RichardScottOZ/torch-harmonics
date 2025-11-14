# Raster Preprocessing for Spherical Training

This guide covers tools for transforming arbitrary raster data into formats suitable for spherical processing.

## Overview

The spherical pipeline requires data in specific formats:
- **Equirectangular**: EPSG:4326 with uniform pixel spacing in degrees
- **HEALPix**: Equal-area pixelization of the sphere

This preprocessing toolset provides:
1. **Python API**: Functions for custom preprocessing workflows
2. **Command-line tool**: Easy-to-use script for common tasks
3. **Validation utilities**: Check if data is properly formatted

## Quick Start

### Command-Line Preprocessing

```bash
# Full preprocessing pipeline (raster + shapefile)
python preprocess_for_spherical.py \
    --raster input.tif \
    --shapefile labels.shp \
    --output_dir processed/ \
    --format equirectangular \
    --resolution 0.1

# Result:
# processed/
#   input_equirect.tif       # Raster in equirectangular
#   labels_epsg4326.shp      # Shapefile in EPSG:4326
```

### Validate Existing Data

```bash
# Check if raster is properly formatted
python preprocess_for_spherical.py --validate my_raster.tif

# Output:
# ✓ CRS is geographic: EPSG:4326
# ✓ Uniform pixel spacing: 0.100000° x 0.100000°
# ✓ VALIDATION PASSED
```

### Python API

```python
from torch_harmonics.examples.raster_transforms import preprocess_for_spherical_training

# Automated preprocessing
outputs = preprocess_for_spherical_training(
    raster_path='utm_raster.tif',
    shapefile_path='labels.shp',
    output_dir='processed/',
    target_resolution=0.05,  # 0.05° ≈ 5.5km at equator
)

# Use outputs with train_spherical.py
print(f"Raster: {outputs['raster']}")
print(f"Shapefile: {outputs['shapefile']}")
```

## Features

### 1. Transform to Equirectangular

Convert any raster to equirectangular projection with uniform spacing.

**Python:**
```python
from torch_harmonics.examples.raster_transforms import transform_to_equirectangular

transform_to_equirectangular(
    input_raster_path='utm_raster.tif',
    output_raster_path='equirect_raster.tif',
    target_resolution=0.1,  # degrees
    extent=(-180, -90, 180, 90),  # global
    resampling_method='bilinear',
)
```

**Command-line:**
```bash
python preprocess_for_spherical.py \
    --raster utm_raster.tif \
    --output_dir processed/ \
    --resolution 0.1 \
    --extent -180 -90 180 90
```

**Options:**
- `target_resolution`: Pixel size in degrees
  - 0.01° ≈ 1.1 km at equator (high resolution)
  - 0.05° ≈ 5.5 km at equator (medium)
  - 0.1° ≈ 11 km at equator (standard)
  - 0.5° ≈ 55 km at equator (coarse)
- `extent`: (lon_min, lat_min, lon_max, lat_max)
  - Global: (-180, -90, 180, 90)
  - Regional examples below
- `resampling_method`: 'nearest', 'bilinear', 'cubic', 'average'

### 2. HEALPix Conversion

Convert raster to HEALPix format for equal-area pixelization.

**Python:**
```python
from torch_harmonics.examples.raster_transforms import raster_to_healpix

raster_to_healpix(
    input_raster_path='equirect.tif',  # Must be EPSG:4326
    output_path='healpix.npz',
    nside=128,  # Resolution parameter
    nest=False,  # Pixel ordering: False=ring, True=nested
)
```

**Command-line:**
```bash
python preprocess_for_spherical.py \
    --raster input.tif \
    --output_dir processed/ \
    --format healpix \
    --nside 128
```

**HEALPix Parameters:**
- `nside`: Resolution (power of 2)
  - 16: 3,072 pixels, ~3.7° resolution
  - 32: 12,288 pixels, ~1.8° resolution
  - 64: 49,152 pixels, ~0.9° resolution
  - 128: 196,608 pixels, ~0.5° resolution
  - 256: 786,432 pixels, ~0.2° resolution
  - 512: 3,145,728 pixels, ~0.1° resolution

**Note:** HEALPix support requires `healpy`:
```bash
pip install healpy
```

### 3. Shapefile Reprojection

Transform shapefiles to EPSG:4326.

**Python:**
```python
from torch_harmonics.examples.raster_transforms import transform_shapefile_to_epsg4326

transform_shapefile_to_epsg4326(
    input_shapefile_path='utm_labels.shp',
    output_shapefile_path='geographic_labels.shp',
)
```

**Command-line:**
```bash
# Included in full pipeline
python preprocess_for_spherical.py \
    --raster input.tif \
    --shapefile labels.shp \
    --output_dir processed/
```

### 4. Format Validation

Check if data is properly formatted for spherical processing.

**Python:**
```python
from torch_harmonics.examples.raster_transforms import validate_equirectangular_format

results = validate_equirectangular_format('raster.tif')

if results['valid']:
    print("✓ Raster is properly formatted")
else:
    print("✗ Validation failed")
    for rec in results['recommendations']:
        print(f"  • {rec}")
```

**Command-line:**
```bash
python preprocess_for_spherical.py --validate raster.tif
```

**Checks:**
- ✓ CRS is geographic (EPSG:4326 or similar)
- ✓ Uniform pixel spacing (latitude = longitude in degrees)
- ✓ Provides recommendations if validation fails

## Common Workflows

### Global Coverage

```bash
# Preprocess for global training
python preprocess_for_spherical.py \
    --raster global_data.tif \
    --shapefile global_labels.shp \
    --output_dir processed_global/ \
    --resolution 0.1 \
    --extent -180 -90 180 90

# Train
python train_spherical.py \
    --raster_path processed_global/global_data_equirect.tif \
    --shapefile_path processed_global/global_labels_epsg4326.shp \
    --require_global \
    --exclude_polar_fraction 0.05
```

### Regional Coverage

**Western US:**
```bash
python preprocess_for_spherical.py \
    --raster landsat_utm.tif \
    --shapefile mineral_deposits.shp \
    --output_dir processed_western_us/ \
    --resolution 0.01 \
    --extent -125 30 -100 50
```

**Australia:**
```bash
python preprocess_for_spherical.py \
    --raster sentinel_data.tif \
    --shapefile labels.shp \
    --output_dir processed_australia/ \
    --resolution 0.02 \
    --extent 110 -45 155 -10
```

**Europe:**
```bash
python preprocess_for_spherical.py \
    --raster europe_data.tif \
    --shapefile labels.shp \
    --output_dir processed_europe/ \
    --resolution 0.05 \
    --extent -10 35 40 70
```

### High-Resolution Processing

```bash
# Very high resolution (0.001° ≈ 110m)
# Warning: Large file sizes!
python preprocess_for_spherical.py \
    --raster input.tif \
    --shapefile labels.shp \
    --output_dir processed_hires/ \
    --resolution 0.001 \
    --extent -122 37 -121 38  # Small region
```

### HEALPix Workflow

```bash
# Convert to HEALPix for global analysis
python preprocess_for_spherical.py \
    --raster global_climate.tif \
    --shapefile zones.shp \
    --output_dir processed_healpix/ \
    --format healpix \
    --nside 256

# Output: processed_healpix/global_climate_healpix.npz
```

**Note:** HEALPix format requires custom dataset loader. Use equirectangular for standard training pipeline.

### Batch Processing

```python
# Process multiple rasters with Python API
from torch_harmonics.examples.raster_transforms import preprocess_for_spherical_training

datasets = [
    ('raster1.tif', 'labels1.shp', 'dataset1'),
    ('raster2.tif', 'labels2.shp', 'dataset2'),
    ('raster3.tif', 'labels3.shp', 'dataset3'),
]

for raster, shapefile, name in datasets:
    print(f"\nProcessing {name}...")
    outputs = preprocess_for_spherical_training(
        raster_path=raster,
        shapefile_path=shapefile,
        output_dir=f'processed/{name}/',
        target_resolution=0.05,
    )
    print(f"✓ {name} complete")
```

## Resolution Guidelines

Choose resolution based on:
- **Data source**: Match source resolution or coarsen
- **Coverage area**: Larger areas → coarser resolution
- **Memory constraints**: Higher resolution → more memory
- **Application**: Classification may use coarser than detection

| Resolution | km at equator | Use Case | File Size* |
|-----------|---------------|----------|-----------|
| 0.001° | ~110 m | Urban mapping, very local | Very Large |
| 0.01° | ~1.1 km | Local surveys, cities | Large |
| 0.05° | ~5.5 km | Regional mapping | Medium |
| 0.1° | ~11 km | Country-scale, standard | Moderate |
| 0.25° | ~28 km | Continental | Small |
| 0.5° | ~55 km | Global coarse | Very Small |
| 1.0° | ~111 km | Global very coarse | Tiny |

*For 99-band raster, global coverage

**Memory estimates for global coverage (99 bands, float32):**
- 0.01°: 36000×18000 = ~25 GB
- 0.05°: 7200×3600 = ~1 GB  
- 0.1°: 3600×1800 = ~256 MB
- 0.5°: 720×360 = ~10 MB

## Integration with Training Pipeline

### Equirectangular → train_spherical.py

```bash
# 1. Preprocess
python preprocess_for_spherical.py \
    --raster raw_data.tif \
    --shapefile labels.shp \
    --output_dir preprocessed/ \
    --resolution 0.1

# 2. Train
python train_spherical.py \
    --raster_path preprocessed/raw_data_equirect.tif \
    --shapefile_path preprocessed/labels_epsg4326.shp \
    --label_column label \
    --num_epochs 100 \
    --validate_spacing

# 3. Predict
python predict_spherical.py \
    --checkpoint checkpoints_spherical/model/checkpoint_spherical.pt \
    --model_name unet_sc2_layers4_e32 \
    --raster_path new_data_equirect.tif \
    --output_path predictions.tif
```

### Already-Formatted Data

If your data is already in EPSG:4326 with uniform spacing:

```bash
# Validate first
python preprocess_for_spherical.py --validate my_raster.tif

# If valid, use directly
python train_spherical.py \
    --raster_path my_raster.tif \
    --shapefile_path my_labels.shp \
    ...
```

## Troubleshooting

### "Raster must be in geographic coordinates"

**Problem:** Input raster is in projected CRS (UTM, etc.)

**Solution:**
```bash
python preprocess_for_spherical.py \
    --raster input.tif \
    --output_dir processed/
```

### "Non-uniform pixel spacing"

**Problem:** Pixels are not square in degrees

**Solution:** Use preprocessing to resample:
```bash
python preprocess_for_spherical.py \
    --raster input.tif \
    --output_dir processed/ \
    --resolution 0.1
```

### "No CRS defined"

**Problem:** Raster has no coordinate system

**Solution:** Define CRS with GDAL first:
```bash
gdal_edit.py -a_srs EPSG:4326 input.tif
```

Or if source CRS is known:
```bash
gdalwarp -s_srs EPSG:32610 -t_srs EPSG:4326 input.tif output.tif
```

### Memory Issues with Large Rasters

**Problem:** Out of memory during processing

**Solutions:**
1. **Use coarser resolution:**
   ```bash
   --resolution 0.5  # Instead of 0.01
   ```

2. **Process regional extents:**
   ```bash
   --extent -120 35 -110 45  # Smaller area
   ```

3. **Process tiles separately** (manual workflow)

### HEALPix Import Error

**Problem:** `ImportError: No module named 'healpy'`

**Solution:**
```bash
pip install healpy
```

## Advanced Usage

### Custom Resampling

Different resampling methods for different data types:

```python
# Categorical data (land cover) - use nearest neighbor
transform_to_equirectangular(
    'landcover.tif',
    'landcover_equirect.tif',
    resampling_method='nearest'
)

# Continuous data (elevation) - use cubic
transform_to_equirectangular(
    'elevation.tif',
    'elevation_equirect.tif',
    resampling_method='cubic'
)

# Multi-spectral imagery - use bilinear
transform_to_equirectangular(
    'multispectral.tif',
    'multispectral_equirect.tif',
    resampling_method='bilinear'
)
```

### Extent Optimization

For regional data, optimize extent to reduce file size:

```python
import rasterio
from shapely.geometry import box
import geopandas as gpd

# Get raster bounds
with rasterio.open('input.tif') as src:
    bounds = src.bounds

# Get shapefile bounds
gdf = gpd.read_file('labels.shp')
shp_bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)

# Use intersection for optimal extent
extent = (
    max(bounds.left, shp_bounds[0]),
    max(bounds.bottom, shp_bounds[1]),
    min(bounds.right, shp_bounds[2]),
    min(bounds.top, shp_bounds[3]),
)

# Preprocess with optimized extent
preprocess_for_spherical_training(
    'input.tif',
    'labels.shp',
    'processed/',
    extent=extent
)
```

### Parallel Processing

Process multiple regions in parallel:

```python
from multiprocessing import Pool
from torch_harmonics.examples.raster_transforms import preprocess_for_spherical_training

def process_region(args):
    raster, shapefile, output_dir, extent = args
    return preprocess_for_spherical_training(
        raster, shapefile, output_dir, extent=extent
    )

regions = [
    ('data.tif', 'labels.shp', 'region1/', (-130, 30, -120, 40)),
    ('data.tif', 'labels.shp', 'region2/', (-120, 30, -110, 40)),
    ('data.tif', 'labels.shp', 'region3/', (-110, 30, -100, 40)),
]

with Pool(3) as pool:
    results = pool.map(process_region, regions)
```

## API Reference

See module documentation:
```python
from torch_harmonics.examples import raster_transforms
help(raster_transforms)
```

Key functions:
- `transform_to_equirectangular()` - Convert to equirectangular
- `transform_shapefile_to_epsg4326()` - Reproject shapefile
- `raster_to_healpix()` - Convert to HEALPix
- `preprocess_for_spherical_training()` - Automated pipeline
- `validate_equirectangular_format()` - Validate format

## Performance Notes

**Processing time** depends on:
- Input raster size
- Output resolution
- Number of bands
- Resampling method (nearest fastest, cubic slowest)

**Typical times** (single-core, approximate):
- Small region, 10 bands, 0.1°: ~10 seconds
- Large region, 50 bands, 0.05°: ~2 minutes
- Global, 99 bands, 0.1°: ~10 minutes

**Tips for faster processing:**
- Use `resampling_method='nearest'` for categorical data
- Process regional extents instead of global
- Use coarser resolution when appropriate
- Consider parallel processing for multiple datasets

## See Also

- `README_spherical.md` - Spherical training pipeline
- `README_raster.md` - Planar training pipeline
- `train_spherical.py` - Training script
- `predict_spherical.py` - Prediction script
