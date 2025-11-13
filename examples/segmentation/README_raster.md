# Raster Stack Binary Segmentation Training

This directory contains an adapted version of the semantic segmentation training script specifically designed for working with raster stacks and shapefile labels for binary classification tasks.

## Overview

The `train_raster.py` script is adapted from the original `train.py` to handle:
- Large raster stacks (e.g., 99 bands, 2000x4000 pixels)
- Binary classification using shapefile labels
- Geospatial data formats (GeoTIFF, shapefiles)
- Optional tiling for large rasters

## Requirements

In addition to the standard torch-harmonics dependencies, you'll need:

```bash
pip install rasterio geopandas
```

## Usage

### Basic Usage

For a raster stack that fits in memory:

```bash
python train_raster.py \
    --raster_path /path/to/your/raster.tif \
    --shapefile_path /path/to/sedex_mineral_deposits.shp \
    --label_column label \
    --num_epochs 100 \
    --batch_size 4
```

### With Tiling (for large rasters)

If your raster is too large to fit in memory, you can tile it:

```bash
python train_raster.py \
    --raster_path /path/to/your/raster.tif \
    --shapefile_path /path/to/sedex_mineral_deposits.shp \
    --label_column label \
    --tile_size 512 512 \
    --stride 256 256 \
    --num_epochs 100 \
    --batch_size 8
```

### Advanced Options

```bash
python train_raster.py \
    --raster_path /path/to/your/raster.tif \
    --shapefile_path /path/to/sedex_mineral_deposits.shp \
    --label_column label \
    --models unet_sc2_layers4_e32 transformer_sc2_layers4_e128 \
    --num_epochs 100 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --amp_mode bf16 \
    --enable_data_augmentation \
    --output_path ./checkpoints_custom
```

## Arguments

### Required Arguments

- `--raster_path`: Path to the input raster file (GeoTIFF or similar)
- `--shapefile_path`: Path to the shapefile containing polygon labels

### Optional Arguments

- `--label_column`: Column name in shapefile containing labels (default: "label")
- `--tile_size`: Height and width for tiling (e.g., `--tile_size 512 512`)
- `--stride`: Stride for tile extraction (e.g., `--stride 256 256`)
- `--models`: List of models to train (default: `unet_sc2_layers4_e32 transformer_sc2_layers4_e128`)
- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 4)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--max_grad_norm`: Max gradient norm for clipping (default: 1.0)
- `--label_smoothing_factor`: Label smoothing factor [0, 1] (default: 0.0)
- `--amp_mode`: Automatic mixed precision mode: `none`, `fp16`, or `bf16` (default: none)
- `--enable_data_augmentation`: Enable data augmentation
- `--enable_ddp`: Enable distributed data parallel training
- `--resume`: Resume from checkpoint
- `--output_path`: Directory for checkpoints and outputs

## Input Data Format

### Raster Data
- Format: GeoTIFF or any format supported by rasterio
- Expected shape: (bands, height, width) - e.g., (99, 2000, 4000)
- All bands will be used as input channels to the model
- **For spherical predictions**: Data should be in geographic coordinates (EPSG:4326 or similar)

### Shapefile Data
- Format: Shapefile (.shp) with associated files (.shx, .dbf, .prj)
- **Supported geometries**: Points, MultiPoints, Polygons, MultiPolygons
  - **Point geometries** (e.g., mineral deposit locations): Automatically buffered to be visible at raster resolution
  - **Polygon geometries**: Directly rasterized
- Must have a column with label values (specified by `--label_column`)
- Should be in the same CRS as the raster, or will be automatically reprojected

## Output

The script creates the following outputs:

- `checkpoints_raster/<model_name>/checkpoint.pt`: Saved model weights
- `checkpoints_raster/<model_name>/figures/`: Validation visualizations
- `checkpoints_raster/<model_name>/output_data/metrics.pkl`: Training metrics

## Example Workflow

1. Prepare your data:
   - Raster stack: `data/raster_stack.tif` (99 bands, 2000x4000)
   - Shapefile: `data/sedex_mineral_deposits.shp` with a "label" column

2. Run training:
   ```bash
   python train_raster.py \
       --raster_path data/raster_stack.tif \
       --shapefile_path data/sedex_mineral_deposits.shp \
       --label_column label \
       --num_epochs 50 \
       --batch_size 4
   ```

3. Monitor training progress (if wandb is configured)

4. Check outputs in `checkpoints_raster/`

## Making Predictions

After training, use the prediction script to make predictions on new rasters:

```bash
# Basic prediction (all data processed as rectangular arrays)
python predict_raster.py \
    --checkpoint checkpoints_raster/model_name/checkpoint.pt \
    --model_name unet_sc2_layers4_e32 \
    --raster_path /path/to/new_raster.tif \
    --output_path predictions.tif \
    --in_channels 99 \
    --num_classes 2

# With --spherical flag (adds CRS validation only - processing identical)
python predict_raster.py \
    --checkpoint checkpoints_raster/model_name/checkpoint.pt \
    --model_name unet_sc2_layers4_e32 \
    --raster_path /path/to/raster.tif \
    --output_path predictions.tif \
    --in_channels 99 \
    --num_classes 2 \
    --spherical  # Only validates CRS - no special processing
```

**IMPORTANT**: The `--spherical` flag does NOT perform any data transformations or special 
spherical processing. Both modes treat rasters identically as rectangular arrays.

## Important Notes

### Data Processing: Planar Rectangular Arrays Only

**CRITICAL LIMITATION**: This implementation treats ALL rasters as planar rectangular arrays.
There is NO special handling for spherical/equirectangular data despite using spherical models.

#### What Actually Happens

**Both "planar" and "spherical" modes:**
1. Read raster as rectangular numpy array (bands, height, width)
2. Pass through model with spherical convolutions applied to this rectangular data
3. Save output as rectangular raster

**The models from torch-harmonics apply spherical operations**, but the data pipeline does
not transform or validate data for spherical topology. This means:

- ✅ Works for regional/local data (approximate but functional)
- ❌ Does NOT properly handle spherical topology (no periodic boundaries, pole treatment, etc.)
- ❌ Does NOT validate equiangular spacing or other spherical grid requirements
- ❌ Does NOT differ between planar and spherical modes (only CRS warning differs)

#### Comparison with Original Stanford 2D3DS

**Stanford 2D3DS (what models were designed for):**
- 360° panoramic images in equirectangular projection
- True spherical topology with equiangular latitude spacing
- Models operate correctly on this spherical representation

**This Implementation:**
- Standard GIS rasters treated as planar rectangles
- No spherical topology handling
- Models apply spherical operations to rectangular data (approximate)

#### When to Use This Implementation

✅ **Suitable for:**
- Regional mineral deposit mapping
- Standard satellite imagery (Landsat, Sentinel)
- Local surveys where approximation acceptable

❌ **Not suitable for:**
- True spherical/panoramic data requiring proper topology
- Applications requiring geometric precision
- Global coverage with proper pole/periodic boundary handling

**For proper spherical data processing**, additional preprocessing and model adaptations
would be required (not implemented in this script).

### Point Geometries
- The dataset now supports **point geometries** (e.g., mineral deposit locations)
- Points are automatically buffered to ~0.5 pixel radius to be visible in rasterization
- Both point and polygon labels can be used in the same shapefile

### Platform Compatibility
- **Windows**: Fully supported with automatic multiprocessing configuration
- **Linux/macOS**: Uses forkserver for better process isolation

### Other Notes
- The script automatically computes normalization statistics from the training data
- Binary classification is assumed (2 classes: 0 and 1)
- For very large rasters, use tiling to reduce memory usage
- Adjust batch size based on your GPU memory availability
