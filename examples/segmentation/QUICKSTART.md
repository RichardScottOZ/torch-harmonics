# Quick Start Guide: Raster Stack Binary Classification

This guide will help you get started with training binary classification models on raster stacks with shapefile labels.

## Prerequisites

### 1. Install torch-harmonics with raster support

```bash
# First, make sure torch-harmonics is built (required for custom kernels)
cd torch-harmonics
pip install --no-build-isolation -e .

# Then install raster dependencies
pip install torch-harmonics[raster]
# Or manually: pip install rasterio geopandas shapely
```

### 2. Prepare Your Data

You need two files:
- **Raster file** (GeoTIFF format recommended): Your multi-band raster stack
  - Example: `raster_stack.tif` with shape (99, 2000, 4000)
  - For global data: Should be in geographic coordinates (EPSG:4326)
- **Shapefile**: Vector data with labels
  - Example: `sedex_mineral_deposits.shp` with a "label" column
  - **Supports**: Points (mineral deposits), MultiPoints, Polygons, MultiPolygons

Make sure both files:
- Use the same or compatible coordinate reference system (CRS)
- Cover the same geographic area
- Have valid geospatial metadata

### ⚠️ Important: Spherical vs Planar Data

**torch-harmonics is a spherical library** using spherical harmonics and convolutions:
- **For local/planar data**: Works but uses spherical operations on planar data (approximate)
- **For global/spherical data**: Proper use case - ensure data is in lat/lon coordinates
- Consider standard CNNs for purely planar applications if geometric precision is critical

## Quick Start Examples

### Example 1: Train with Full Raster (Small Files)

If your raster fits in memory:

```bash
python train_raster.py \
    --raster_path /path/to/your/raster.tif \
    --shapefile_path /path/to/your/labels.shp \
    --label_column label \
    --num_epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4
```

### Example 2: Train with Tiling (Large Rasters)

For large rasters (e.g., 2000x4000 pixels):

```bash
python train_raster.py \
    --raster_path /path/to/large_raster.tif \
    --shapefile_path /path/to/labels.shp \
    --label_column label \
    --tile_size 512 512 \
    --stride 256 256 \
    --num_epochs 100 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --amp_mode bf16
```

**Tiling Parameters:**
- `--tile_size 512 512`: Extract 512x512 pixel tiles
- `--stride 256 256`: Move by 256 pixels (50% overlap)
- Overlap helps with edge effects

### Example 3: Production Training with All Options

```bash
python train_raster.py \
    --raster_path /data/raster_stack_99bands.tif \
    --shapefile_path /data/sedex_mineral_deposits.shp \
    --label_column label \
    --tile_size 512 512 \
    --stride 384 384 \
    --models unet_sc2_layers4_e32 transformer_sc2_layers4_e128 \
    --num_epochs 200 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --max_grad_norm 1.0 \
    --label_smoothing_factor 0.1 \
    --amp_mode bf16 \
    --enable_data_augmentation \
    --output_path ./checkpoints_production \
    --resume  # Resume from checkpoint if exists
```

## Test with Synthetic Data

Don't have real data yet? Generate synthetic data for testing:

```bash
# Generate synthetic data
python example_usage.py \
    --bands 99 \
    --height 512 \
    --width 512 \
    --output_dir /tmp/synthetic_test

# The script will print a training command you can run
```

## Understanding the Output

The training script creates:

```
output_path/
├── model_name/
│   ├── checkpoint.pt              # Saved model weights
│   ├── figures/                   # Validation visualizations
│   │   ├── validation_0.png
│   │   ├── validation_1.png
│   │   └── ...
│   └── output_data/
│       └── metrics.pkl            # Training metrics
```

## Common Issues and Solutions

### Issue 1: Out of Memory

**Solution:** Use tiling with smaller tile_size and/or reduce batch_size

```bash
--tile_size 256 256 --batch_size 2
```

### Issue 2: CRS Mismatch

**Error:** "Shapefile CRS doesn't match raster CRS"

**Solution:** The dataset automatically reprojects. If issues persist, manually reproject:

```bash
# Using gdalwarp for raster
gdalwarp -t_srs EPSG:4326 input.tif output.tif

# Using ogr2ogr for shapefile
ogr2ogr -t_srs EPSG:4326 output.shp input.shp
```

### Issue 3: No Labels in Output

**Check:**
1. Shapefile has a "label" column (or specify correct column with `--label_column`)
2. Polygons overlap with raster extent
3. Label values are valid (non-null)

### Issue 4: ModuleNotFoundError: disco_helpers

This means torch-harmonics needs to be built:

```bash
cd torch-harmonics
pip install --no-build-isolation -e .
```

## Monitoring Training

### With Weights & Biases (wandb)

If you have wandb configured:

```bash
wandb login
python train_raster.py ...  # wandb logging automatic
```

### Without wandb

Check console output:
- Training loss decreases each epoch
- Validation metrics (IoU, Accuracy) improve
- Check `figures/` folder for visual progress

## Next Steps

1. **Tune Hyperparameters:**
   - Learning rate: Try 1e-4, 5e-4, 1e-3
   - Batch size: Depends on GPU memory
   - Number of epochs: 100-200 typical

2. **Try Different Models:**
   ```bash
   --models unet_sc2_layers4_e32 transformer_sc2_layers4_e128 segformer_sc2_layers4_e128
   ```

3. **Enable Augmentation:**
   ```bash
   --enable_data_augmentation
   ```

4. **Use Mixed Precision:**
   ```bash
   --amp_mode bf16  # or fp16
   ```

## Advanced Usage

### Multi-GPU Training

```bash
# Launch with torchrun
torchrun --nproc_per_node=4 train_raster.py \
    --enable_ddp \
    ... # other args
```

### Resume from Checkpoint

```bash
python train_raster.py \
    --resume \
    ... # same args as before
```

### Custom Output Path

```bash
python train_raster.py \
    --output_path /custom/path/checkpoints \
    ...
```

## Getting Help

1. Check the full documentation:
   - `README_raster.md` - Complete usage guide
   - `IMPLEMENTATION_SUMMARY.md` - Technical details

2. Run verification:
   ```bash
   python verify_implementation.py
   ```

3. View script help:
   ```bash
   python train_raster.py --help
   ```

## Example: Complete Workflow

```bash
# 1. Verify installation
python verify_implementation.py

# 2. Test with synthetic data
python example_usage.py --bands 99 --height 512 --width 512

# 3. Train on synthetic data (quick test)
python train_raster.py \
    --raster_path /tmp/torch_harmonics_demo_*/synthetic_raster.tif \
    --shapefile_path /tmp/torch_harmonics_demo_*/synthetic_labels.shp \
    --label_column label \
    --num_epochs 5 \
    --batch_size 2

# 4. Train on your real data
python train_raster.py \
    --raster_path /your/data/raster.tif \
    --shapefile_path /your/data/labels.shp \
    --label_column label \
    --tile_size 512 512 \
    --stride 256 256 \
    --num_epochs 100 \
    --batch_size 8 \
    --amp_mode bf16 \
    --enable_data_augmentation
```

## Performance Tips

1. **Use AMP:** `--amp_mode bf16` or `--amp_mode fp16`
2. **Optimize batch size:** Maximize GPU utilization
3. **Use tiling for large rasters:** Reduces memory usage
4. **Enable augmentation:** Often improves generalization
5. **Multi-GPU:** Use `--enable_ddp` with torchrun

Happy training! 🚀
