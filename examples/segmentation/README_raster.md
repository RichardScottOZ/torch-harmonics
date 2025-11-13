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
# Basic prediction
python predict_raster.py \
    --checkpoint checkpoints_raster/model_name/checkpoint.pt \
    --model_name unet_sc2_layers4_e32 \
    --raster_path /path/to/new_raster.tif \
    --output_path predictions.tif \
    --in_channels 99 \
    --num_classes 2

# Spherical prediction for global/geographic data
python predict_raster.py \
    --checkpoint checkpoints_raster/model_name/checkpoint.pt \
    --model_name unet_sc2_layers4_e32 \
    --raster_path /path/to/global_raster.tif \
    --output_path predictions.tif \
    --in_channels 99 \
    --num_classes 2 \
    --spherical
```

## Important Notes

### Spherical vs Planar Data
**torch-harmonics is a spherical library** designed for data on a sphere (e.g., global climate, Earth observation). The models use spherical convolutions and harmonics.

- **For planar/local raster data**: The implementation works but uses spherical operations on planar data. This is geometrically approximate but functional.
- **For global/spherical data**: 
  - Input rasters should be in geographic coordinates (lat/lon, EPSG:4326)
  - Use `--spherical` flag in predict_raster.py for proper spherical inference
  - The equiangular grid assumption from the library is appropriate

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
