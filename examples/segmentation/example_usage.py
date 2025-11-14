#!/usr/bin/env python3
# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Example script demonstrating how to use train_raster.py with synthetic data.

This script creates synthetic raster and shapefile data to demonstrate the 
training pipeline without requiring real data.
"""

import os
import tempfile
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
    import geopandas as gpd
    from shapely.geometry import box, Polygon
    import argparse
except ImportError:
    print("This example requires rasterio and geopandas.")
    print("Install them with: pip install rasterio geopandas")
    exit(1)


def create_synthetic_raster(output_path, bands=99, height=2000, width=4000):
    """
    Create a synthetic raster file with random data.
    
    Parameters
    ----------
    output_path : str
        Path to save the raster file
    bands : int
        Number of bands (channels)
    height : int
        Height in pixels
    width : int
        Width in pixels
    """
    print(f"Creating synthetic raster with shape ({bands}, {height}, {width})...")
    
    # Create random data
    data = np.random.randn(bands, height, width).astype(np.float32)
    
    # Add some spatial structure (make it look more realistic)
    for b in range(bands):
        # Add some gradients
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)
        data[b] += xx * 0.5 + yy * 0.5
        
        # Add some "features"
        data[b] += np.sin(xx * 10 + b * 0.1) * 0.2
        data[b] += np.cos(yy * 10 + b * 0.1) * 0.2
    
    # Define geographic extent (arbitrary coordinates)
    bounds = (0, 0, width, height)  # (left, bottom, right, top)
    transform = from_bounds(*bounds, width, height)
    
    # Write to file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform,
        compress='lzw',
    ) as dst:
        dst.write(data)
    
    print(f"✓ Created raster: {output_path}")
    return output_path


def create_synthetic_shapefile(output_path, raster_path):
    """
    Create a synthetic shapefile with polygon labels.
    
    Parameters
    ----------
    output_path : str
        Path to save the shapefile
    raster_path : str
        Path to the raster file (to get bounds)
    """
    print("Creating synthetic shapefile with labels...")
    
    # Get raster bounds
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
    
    # Create some random polygons with labels
    polygons = []
    labels = []
    
    # Create about 10-20 random polygons
    np.random.seed(42)
    n_polygons = 15
    
    for i in range(n_polygons):
        # Random polygon within raster bounds
        x1 = np.random.uniform(bounds.left, bounds.right - 100)
        y1 = np.random.uniform(bounds.bottom, bounds.top - 100)
        x2 = x1 + np.random.uniform(50, 200)
        y2 = y1 + np.random.uniform(50, 200)
        
        # Make sure it's within bounds
        x2 = min(x2, bounds.right)
        y2 = min(y2, bounds.top)
        
        # Create polygon (box for simplicity, but could be more complex)
        poly = box(x1, y1, x2, y2)
        
        polygons.append(poly)
        # Binary labels: 0 or 1
        labels.append(1 if i % 2 == 0 else 0)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {'label': labels, 'geometry': polygons},
        crs=crs
    )
    
    # Save to shapefile
    gdf.to_file(output_path)
    
    print(f"✓ Created shapefile with {len(polygons)} polygons: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data and demonstrate train_raster.py usage"
    )
    parser.add_argument(
        "--output_dir",
        default=tempfile.mkdtemp(prefix="torch_harmonics_demo_"),
        help="Directory to store synthetic data"
    )
    parser.add_argument(
        "--bands",
        type=int,
        default=99,
        help="Number of bands in synthetic raster"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,  # Smaller for demo
        help="Height of synthetic raster (use 2000 for full size)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,  # Smaller for demo
        help="Width of synthetic raster (use 4000 for full size)"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Only generate data, don't run training"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using output directory: {args.output_dir}")
    
    # Generate synthetic data
    raster_path = os.path.join(args.output_dir, "synthetic_raster.tif")
    shapefile_path = os.path.join(args.output_dir, "synthetic_labels.shp")
    
    create_synthetic_raster(
        raster_path,
        bands=args.bands,
        height=args.height,
        width=args.width
    )
    create_synthetic_shapefile(shapefile_path, raster_path)
    
    print("\n" + "="*80)
    print("Synthetic data created successfully!")
    print("="*80)
    print(f"\nRaster file: {raster_path}")
    print(f"Shapefile: {shapefile_path}")
    
    if not args.skip_training:
        print("\n" + "="*80)
        print("Example training command:")
        print("="*80)
        training_cmd = f"""
python train_raster.py \\
    --raster_path {raster_path} \\
    --shapefile_path {shapefile_path} \\
    --label_column label \\
    --num_epochs 5 \\
    --batch_size 2 \\
    --output_path {os.path.join(args.output_dir, 'checkpoints')}
"""
        print(training_cmd)
        
        print("\nNote: This would train on synthetic data. For actual training,")
        print("replace the paths with your real raster stack and shapefile.")
    else:
        print("\n" + "="*80)
        print("Training skipped. Use the following command to train:")
        print("="*80)
        print(f"\npython train_raster.py \\")
        print(f"    --raster_path {raster_path} \\")
        print(f"    --shapefile_path {shapefile_path} \\")
        print(f"    --label_column label \\")
        print(f"    --num_epochs 5 \\")
        print(f"    --batch_size 2")


if __name__ == "__main__":
    main()
