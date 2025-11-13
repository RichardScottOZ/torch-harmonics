# Implementation Summary: Raster Stack Binary Classification

## Overview

This implementation adapts the semantic segmentation training pipeline from `train.py` to work with geospatial raster stacks and shapefile labels for binary classification tasks.

## Problem Statement

The original requirement was to:
> Take the train.py script in examples/semantic-segmentation and adapt it to work on a raster stack of (99, 2000, 4000) and assume you are making a binary classifier with the label data in the label column of a sedex_mineral_deposits.shp shapefile

## Solution Architecture

### 1. RasterShapefileDataset Class
**File:** `torch_harmonics/examples/raster_shapefile_dataset.py`

A PyTorch Dataset class that:
- Loads raster data using `rasterio` (supports GeoTIFF and other formats)
- Rasterizes vector labels from shapefiles using `geopandas`
- Handles arbitrary number of input bands (99 in the target use case)
- Supports optional tiling for memory-efficient processing of large rasters
- Automatically handles CRS reprojection when needed
- Provides binary classification labels (0/1)

**Key Features:**
```python
RasterShapefileDataset(
    raster_path,          # Path to raster file
    shapefile_path,       # Path to shapefile with labels
    label_column='label', # Column in shapefile containing labels
    tile_size=None,       # Optional: (height, width) for tiling
    stride=None,          # Optional: stride for tile extraction
)
```

### 2. Adapted Training Script
**File:** `examples/segmentation/train_raster.py`

Modified training script that:
- Loads raster + shapefile data instead of Stanford 2D3DS dataset
- Configures models for arbitrary input channels (99 bands)
- Sets up binary classification (2 output classes)
- Computes normalization statistics from training data
- Supports tiled processing for large rasters
- Maintains all original training features (DDP, AMP, augmentation, etc.)

**Key Differences from Original:**
| Original `train.py` | New `train_raster.py` |
|---------------------|----------------------|
| Stanford 2D3DS dataset | Raster + Shapefile |
| 3-4 input channels (RGB/RGBA) | Arbitrary channels (e.g., 99) |
| Multi-class segmentation | Binary classification |
| Spherical images | Planar rasters |
| Fixed dataset | User-provided data paths |

### 3. Documentation and Examples
**Files:** 
- `examples/segmentation/README_raster.md` - Usage guide
- `examples/segmentation/example_usage.py` - Synthetic data generator
- `examples/segmentation/verify_implementation.py` - Implementation validator

## Usage Example

### Basic Usage
```bash
python train_raster.py \
    --raster_path /path/to/raster.tif \
    --shapefile_path /path/to/sedex_mineral_deposits.shp \
    --label_column label \
    --num_epochs 100 \
    --batch_size 4
```

### With Tiling (for large rasters)
```bash
python train_raster.py \
    --raster_path /path/to/raster.tif \
    --shapefile_path /path/to/sedex_mineral_deposits.shp \
    --label_column label \
    --tile_size 512 512 \
    --stride 256 256 \
    --num_epochs 100 \
    --batch_size 8
```

## Data Requirements

### Input Raster
- Format: GeoTIFF or any rasterio-supported format
- Shape: (bands, height, width) - e.g., (99, 2000, 4000)
- All bands used as input to the model
- Should have valid CRS information

### Shapefile
- Format: Shapefile (.shp + associated files)
- Geometry: Polygons or MultiPolygons
- Required column: Label values (specified by `--label_column`)
- Should be in same CRS as raster (or will be reprojected)

## Technical Implementation Details

### Data Pipeline
1. **Raster Loading:** Uses `rasterio` to read multi-band GeoTIFF
2. **Label Rasterization:** Uses `rasterio.features.rasterize` to convert vector polygons to raster mask
3. **Tiling (optional):** Splits large rasters into manageable tiles
4. **Normalization:** Computes per-channel statistics from training data
5. **Binary Labels:** Converts polygon presence to 0/1 classification

### Model Compatibility
The implementation reuses existing torch-harmonics models by:
- Dynamically configuring input channels (via `in_chans` parameter)
- Setting output channels to 2 for binary classification
- Maintaining spatial dimensions compatibility

### Memory Optimization
For large rasters:
- Optional tiling with configurable size and stride
- Lazy loading - data read only when needed
- Efficient rasterization with sparse polygon handling

## Dependencies

### Required
- `torch>=2.4.0`
- `numpy>=1.22.4`

### For Raster Support (install with: `pip install torch-harmonics[raster]`)
- `rasterio>=1.3.0`
- `geopandas>=0.12.0`
- `shapely>=2.0.0`

## Integration with torch-harmonics

The implementation integrates seamlessly with torch-harmonics:

1. **Dataset Interface:** Follows PyTorch Dataset conventions
2. **Model Registry:** Uses existing model architectures from `model_registry.py`
3. **Loss Functions:** Leverages S2 loss functions (CrossEntropyLossS2, etc.)
4. **Metrics:** Uses S2 metrics (IoU, Accuracy)
5. **Training Infrastructure:** Maintains DDP, AMP, checkpointing, etc.

## Verification

Run the verification script to check implementation:
```bash
cd examples/segmentation
python verify_implementation.py
```

Expected output: All checks should pass ✓

## Testing with Synthetic Data

Generate synthetic data for testing:
```bash
cd examples/segmentation
python example_usage.py --bands 99 --height 512 --width 512
```

This creates:
- Synthetic raster with 99 bands
- Synthetic shapefile with random polygons
- Example training command

## Limitations and Considerations

### 1. Spherical vs Planar Data (IMPORTANT)

**torch-harmonics is fundamentally a spherical library.** The models use spherical harmonics and spherical convolutions designed for data on a sphere.

#### Original Use Case: 360° Panoramas
The Stanford 2D3DS dataset uses **360° panoramic images** in equirectangular projection:
- Spherical data (panorama from a single point) unwrapped to a rectangle
- Rows represent latitude bands with equiangular spacing
- Columns represent longitude (wraps 360°)
- This is TRUE spherical data where spherical operations are geometrically correct

#### This Implementation: Planar Rasters
Our raster stack implementation treats data as **standard GIS rasters**:
- Cartesian (x,y) coordinates, not spherical (lat,lon) on a globe
- Planar projections: UTM, State Plane, local coordinates
- Standard satellite scenes (Landsat, Sentinel) - not panoramic
- Regional/local coverage (mineral deposits, land surveys)

#### Data Format Comparison

| Aspect | Stanford 2D3DS (Original) | Raster Stack (This Implementation) |
|--------|---------------------------|-------------------------------------|
| Format | Equirectangular panorama | Planar GIS raster |
| Coverage | 360° spherical view | Rectangular region |
| Coordinates | (lat, lon) on sphere | (x, y) in projection |
| Grid Type | Equiangular latitude | Regular Cartesian grid |
| Spherical Ops | Geometrically correct | Approximate |
| Use Case | Panoramic imagery, global data | Regional surveys, local mapping |

#### Recommendations

- **For local/planar data (mineral deposits, regional surveys):** This implementation works but uses spherical operations on planar data (approximate). Acceptable for many applications but not geometrically precise.

- **For true spherical data (360° panoramas, global climate):** Data must be in equirectangular format with proper lat/lon spacing. Use `--spherical` flag in predictions.

- **For purely planar applications requiring geometric precision:** Consider standard CNN architectures instead of spherical convolution models.

- **For global data:** This implementation can work if data is properly formatted in equirectangular projection, but verify the grid assumptions match.

### 2. Grid Assumptions

The implementation uses S2 (spherical) loss functions and metrics with equiangular grid assumptions. For planar rasters, these act as weighted losses/metrics but are not geometrically precise.

### 3. Point Geometry Support

Point geometries (e.g., mineral deposits) are supported by automatically buffering them to ~0.5 pixel radius. This ensures points are visible in the rasterized labels.

### 4. Memory Usage

Full raster loading requires sufficient RAM. Use tiling for large rasters (>2GB).

### 5. Binary Classification Only

Current implementation is for binary classification. Multi-class support would require minimal changes to label rasterization.

### 6. Coordinate Systems

Assumes raster and shapefile can be aligned in the same CRS. Complex coordinate transformations may need additional handling.

### 7. Platform Compatibility

Windows support is enabled via automatic multiprocessing method selection (spawn vs forkserver).

## Future Enhancements

Potential improvements:
- [ ] Multi-class classification support
- [ ] Planar-specific loss functions and metrics
- [ ] On-the-fly data augmentation for geospatial data
- [ ] Support for cloud-optimized GeoTIFFs (COGs)
- [ ] Integration with remote sensing preprocessing pipelines
- [ ] Class balancing for imbalanced datasets
- [ ] Support for additional vector formats (GeoJSON, GeoPackage)

## Security

All code has been checked with CodeQL and contains no security vulnerabilities.

## License

This implementation follows the same BSD-3-Clause license as torch-harmonics.
