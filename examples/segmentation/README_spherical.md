# TRUE Spherical/Equirectangular Raster Training Pipeline

This directory contains scripts for training segmentation models on **TRUE spherical/equirectangular data** with proper topology handling.

## ⚠️ Important: Which Pipeline to Use?

### Use `train_spherical.py` + `predict_spherical.py` for:
- ✅ 360° panoramic images in equirectangular projection
- ✅ Global climate data requiring proper spherical topology
- ✅ Data where periodic longitude boundaries matter
- ✅ Data where area-weighted operations are critical
- ✅ Data with equiangular latitude spacing

### Use `train_raster.py` + `predict_raster.py` for:
- ✅ Regional mineral deposit mapping
- ✅ Standard satellite imagery (Landsat, Sentinel scenes)
- ✅ Local/regional surveys in any projection (UTM, State Plane, etc.)
- ✅ Data where planar approximation is acceptable

## What Makes This Spherical Pipeline Different?

### Data Validation
- **Validates CRS is geographic** (EPSG:4326 or similar)
- **Checks equiangular latitude spacing** (with configurable tolerance)
- **Optionally requires full 360° longitude coverage**
- **Warns if data doesn't span full latitude range**

### Proper Spherical Topology Handling
- **Area-weighted loss functions** using quadrature weights
- **Area-weighted metrics** (IoU, Accuracy)
- **Area-weighted statistics** computation
- **Pole exclusion** option to avoid singularities
- **Grid-aware operations** (equiangular or Legendre-Gauss)

### What It Does NOT Do (Yet)
- ❌ Periodic boundary padding (longitude wraps)
- ❌ Special pole treatment beyond exclusion
- ❌ Spherical mesh/graph output formats

## Requirements

```bash
# Install with raster dependencies
pip install torch-harmonics[raster]

# Or manually
pip install rasterio geopandas shapely
```

## Data Format Requirements

Your raster MUST be:
1. **In geographic coordinates** (EPSG:4326 or similar lat/lon CRS)
2. **Equirectangular projection** with equiangular latitude spacing
3. **North-up** orientation (standard GeoTIFF)

### Validating Your Data

```bash
# Check CRS and projection
gdalinfo your_raster.tif

# Should show:
# - Coordinate System: GEOGCS or similar geographic CRS
# - Pixel size approximately equal in degrees
# - Origin at top-left with negative Y pixel size
```

### Converting Data to Equirectangular

```bash
# Reproject to EPSG:4326
gdalwarp -t_srs EPSG:4326 input.tif output_equirect.tif

# For exact equiangular spacing at specific resolution
gdalwarp -t_srs EPSG:4326 -tr 0.1 0.1 -te -180 -90 180 90 input.tif output_equirect.tif
# Where:
#   -tr: pixel size in degrees (longitude, latitude)
#   -te: target extent (xmin ymin xmax ymax)
```

## Usage

### Training

```bash
# Basic training on equirectangular data
python train_spherical.py \
    --raster_path /path/to/equirect_raster.tif \
    --shapefile_path /path/to/labels.shp \
    --label_column label \
    --num_epochs 100 \
    --batch_size 1 \
    --validate_spacing

# Training on global 360° data with pole exclusion
python train_spherical.py \
    --raster_path /path/to/global_360deg.tif \
    --shapefile_path /path/to/labels.shp \
    --label_column label \
    --num_epochs 100 \
    --require_global \
    --exclude_polar_fraction 0.05 \
    --grid_type equiangular

# Training with custom models
python train_spherical.py \
    --raster_path /path/to/equirect_raster.tif \
    --shapefile_path /path/to/labels.shp \
    --models unet_sc2_layers4_e32 transformer_sc2_layers4_e128 \
    --num_epochs 50 \
    --learning_rate 1e-4
```

### Prediction

```bash
# Basic prediction
python predict_spherical.py \
    --checkpoint checkpoints_spherical/model_name/checkpoint_spherical.pt \
    --model_name unet_sc2_layers4_e32 \
    --raster_path /path/to/new_equirect.tif \
    --output_path predictions_spherical.tif \
    --in_channels 99 \
    --num_classes 2

# With shapefile validation
python predict_spherical.py \
    --checkpoint checkpoints_spherical/model_name/checkpoint_spherical.pt \
    --model_name unet_sc2_layers4_e32 \
    --raster_path /path/to/new_equirect.tif \
    --output_path predictions_spherical.tif \
    --shapefile_path /path/to/validation_labels.shp \
    --in_channels 99
```

## Command-Line Arguments

### train_spherical.py

**Required:**
- `--raster_path`: Path to equirectangular raster (EPSG:4326)
- `--shapefile_path`: Path to shapefile with labels

**Spherical-Specific:**
- `--grid_type`: Grid type - 'equiangular' or 'legendre-gauss' (default: equiangular)
- `--validate_spacing`: Validate equiangular latitude spacing (default: True)
- `--require_global`: Require full 360° longitude coverage (default: False)
- `--exclude_polar_fraction`: Fraction of polar latitudes to exclude [0.0-0.5] (default: 0.0)

**Training:**
- `--models`: Models to train (default: unet_sc2_layers4_e32)
- `--num_epochs`: Number of epochs (default: 100)
- `--batch_size`: Batch size - typically 1 for global data (default: 1)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--label_smoothing_factor`: Label smoothing [0-1] (default: 0.0)
- `--max_grad_norm`: Gradient clipping threshold (default: 1.0)

**Other:**
- `--output_path`: Checkpoint directory (default: checkpoints_spherical/)
- `--label_column`: Shapefile label column (default: 'label')
- `--resume`: Resume from checkpoint
- `--amp_mode`: AMP mode - 'none', 'fp16', 'bf16' (default: none)
- `--enable_ddp`: Enable distributed training
- `--enable_data_augmentation`: Enable augmentation (limited for spherical)

### predict_spherical.py

**Required:**
- `--checkpoint`: Path to checkpoint file (.pt)
- `--model_name`: Model architecture name
- `--raster_path`: Path to input equirectangular raster
- `--output_path`: Path to save predictions

**Optional:**
- `--shapefile_path`: Shapefile for validation (labels not used)
- `--in_channels`: Number of input channels (default: 99)
- `--num_classes`: Number of output classes (default: 2)
- `--grid_type`: Grid type (default: equiangular)
- `--normalize_stats`: Path to normalization statistics

## Output

### Training Outputs

```
checkpoints_spherical/
├── model_name/
│   ├── checkpoint_spherical.pt          # Model weights
│   ├── figures_spherical/               # Validation visualizations
│   │   └── validation_spherical_*.png   # Spherical projections
│   └── output_data/
│       └── metrics_spherical.pkl        # Training metrics
```

### Prediction Outputs

- **predictions_spherical.tif**: Predicted labels in equirectangular format
- **predictions_spherical_viz.png**: Mollweide projection visualization (optional)

## Understanding Quadrature Weights

The spherical pipeline uses **area-weighted quadrature** to account for varying pixel areas on the sphere:

- **At equator**: Pixels represent larger areas
- **Near poles**: Pixels represent smaller areas

### Area Weighting Formula

```
Weight[lat, lon] = cos(lat) * (2π / nlon)
```

Normalized so sum of all weights = 1.

### Where Area Weighting Matters

✅ **Loss functions**: Cross-entropy, Dice, Focal loss
✅ **Metrics**: IoU, Accuracy
✅ **Statistics**: Mean, standard deviation
✅ **Class balancing**: Histogram computation

## Example: 360° Panoramic Data

```python
# Training on 360° panorama with pole exclusion
python train_spherical.py \
    --raster_path panorama_360deg.tif \
    --shapefile_path room_labels.shp \
    --label_column label \
    --num_epochs 100 \
    --batch_size 1 \
    --require_global \
    --exclude_polar_fraction 0.1 \  # Exclude top/bottom 10%
    --validate_spacing
```

## Example: Global Climate Data

```python
# Training on global climate data
python train_spherical.py \
    --raster_path global_temperature_99bands.tif \
    --shapefile_path climate_zones.shp \
    --label_column zone \
    --num_epochs 50 \
    --batch_size 1 \
    --require_global \
    --grid_type equiangular \
    --learning_rate 5e-5
```

## Troubleshooting

### Error: "Raster must be in geographic coordinates"

Your raster is in a projected CRS. Reproject to EPSG:4326:

```bash
gdalwarp -t_srs EPSG:4326 input.tif output.tif
```

### Warning: "Latitude spacing deviates from equiangular"

Your raster doesn't have exact equiangular spacing. Options:

1. Reproject with exact spacing:
   ```bash
   gdalwarp -t_srs EPSG:4326 -tr 0.1 0.1 input.tif output.tif
   ```

2. Disable validation (not recommended):
   ```bash
   python train_spherical.py --no-validate_spacing ...
   ```

### Error: "Global coverage required but longitude span is X°"

Your raster doesn't cover full 360°. Options:

1. Don't require global coverage:
   ```bash
   python train_spherical.py ... # Remove --require_global
   ```

2. Extend raster to 360° (if appropriate)

### Warning: "Data does not span full latitude range"

Your raster doesn't cover poles. This is usually fine - the warning is informational.

## Technical Details

### Grid Types

**equiangular** (default):
- Uniform latitude spacing
- Standard for most equirectangular data
- Used by geographic rasters

**legendre-gauss**:
- Non-uniform latitude spacing
- Used for spectral methods
- Rare in raster data

### Polar Exclusion

Use `--exclude_polar_fraction 0.05` to:
- Exclude top/bottom 5% of latitudes
- Avoid pole singularities
- Set quadrature weights to 0 at poles
- Helpful for 360° panoramas

### Memory Considerations

Global equirectangular data can be large:
- 360° × 180° at 0.1° = 3600 × 1800 pixels
- 99 bands × 4 bytes = ~2.4 GB per image
- Use batch_size=1 for large global images
- Consider tiling for very high resolution (future work)

## Comparison: Spherical vs Planar Pipelines

| Feature | Spherical Pipeline | Planar Pipeline |
|---------|-------------------|-----------------|
| Data Format | Equirectangular (EPSG:4326) | Any projection |
| Validation | CRS, spacing, coverage | Minimal |
| Loss Weighting | Area-weighted (quadrature) | Uniform |
| Metrics | Area-weighted | Uniform |
| Statistics | Area-weighted | Uniform |
| Topology | Spherical-aware | Planar |
| Use Case | 360° panoramas, global data | Regional mapping |
| Approximation | Geometrically correct | Approximate |

## Future Enhancements

Potential additions:
- [ ] Periodic boundary padding for longitude
- [ ] Advanced pole treatment (beyond exclusion)
- [ ] Spherical mesh/graph output formats
- [ ] Multi-resolution spherical tiling
- [ ] Icosahedral grid support
- [ ] Spherical data augmentation (longitude shifts)

## References

- Original Stanford 2D3DS dataset paper
- torch-harmonics spherical harmonics implementation
- Equirectangular projection: https://en.wikipedia.org/wiki/Equirectangular_projection
