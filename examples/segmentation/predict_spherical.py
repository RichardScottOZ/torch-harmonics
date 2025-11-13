# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Prediction script for TRUE spherical/equirectangular raster data.

This script properly handles spherical topology for predictions on
equirectangular format data with proper topology handling.

Use this for: 360° panoramas, global climate data
Use predict_raster.py for: Regional planar mapping (mineral deposits, etc.)
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

try:
    import rasterio
except ImportError:
    print("This script requires rasterio. Install with: pip install rasterio")
    sys.exit(1)

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from torch_harmonics.examples.spherical_raster_dataset import SphericalRasterDataset, compute_stats_spherical

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_registry import get_baseline_models

from torchvision.transforms import v2


def load_model(checkpoint_path, model_name, in_channels, num_classes, img_size, device):
    """Load a trained model from checkpoint."""
    baseline_models = get_baseline_models(
        img_size=img_size,
        in_chans=in_channels,
        out_chans=num_classes,
        drop_path_rate=0.1
    )
    
    if model_name not in baseline_models:
        raise ValueError(f"Model {model_name} not found in registry")
    
    model = baseline_models[model_name]().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model


def predict_spherical(
    model,
    raster_path,
    output_path,
    shapefile_path=None,
    grid_type='equiangular',
    normalization=None,
    device=torch.device("cpu"),
):
    """
    Make predictions on equirectangular raster data with proper spherical handling.
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    raster_path : str
        Path to equirectangular raster (EPSG:4326)
    output_path : str
        Path to save predictions
    shapefile_path : str, optional
        Shapefile path (for dataset validation, labels not used in prediction)
    grid_type : str
        Grid type: 'equiangular' or 'legendre-gauss'
    normalization : callable, optional
        Normalization transform
    device : torch.device
        Device for inference
    """
    
    print("Loading and validating SPHERICAL/equirectangular raster...")
    
    # Create temporary shapefile path if not provided
    if shapefile_path is None:
        # Create dummy shapefile for validation purposes only
        import tempfile
        import geopandas as gpd
        from shapely.geometry import Point
        
        temp_dir = tempfile.mkdtemp()
        shapefile_path = os.path.join(temp_dir, "dummy.shp")
        
        # Create empty shapefile
        gdf = gpd.GeoDataFrame({'geometry': [Point(0, 0)], 'label': [0]}, crs='EPSG:4326')
        gdf.to_file(shapefile_path)
        print("Note: Using dummy shapefile for validation (no labels needed for prediction)")
    
    # Load dataset with validation
    dataset = SphericalRasterDataset(
        raster_path=raster_path,
        shapefile_path=shapefile_path,
        grid_type=grid_type,
        validate_spacing=True,
        require_global=False,
    )
    
    print(f"\nValidated spherical data:")
    print(f"  Latitude range: [{dataset.lat_min:.2f}, {dataset.lat_max:.2f}]")
    print(f"  Longitude range: [{dataset.lon_min:.2f}, {dataset.lon_max:.2f}]")
    print(f"  Shape: {dataset.nlat} x {dataset.nlon}")
    print(f"  Global coverage: {dataset.is_global}")
    
    # Load raster
    raster_data, _ = dataset[0]
    
    if normalization is not None:
        raster_data = normalization(raster_data)
    
    print("\nMaking predictions on SPHERICAL data...")
    
    # Process
    with torch.no_grad():
        input_batch = raster_data.unsqueeze(0).to(device)  # (1, C, H, W)
        output = model(input_batch)
        predictions = nn.functional.softmax(output, dim=1)
        predictions = torch.argmax(predictions, dim=1).squeeze(0).cpu().numpy()
    
    # Save predictions
    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()
    
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(predictions.astype(np.uint8), 1)
    
    print(f"\nSPHERICAL predictions saved to: {output_path}")
    print("Output maintains equirectangular projection and geographic coordinates")
    
    # Optionally save spherical visualization
    try:
        import matplotlib.pyplot as plt
        from torch_harmonics.plotting import imshow_sphere
        
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': 'mollweide'})
        imshow_sphere(predictions, ax=ax, cmap='gray', vmin=0, vmax=1)
        ax.set_title('Spherical Prediction (Mollweide Projection)')
        
        viz_path = output_path.replace('.tif', '_spherical_viz.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Spherical visualization saved to: {viz_path}")
    except Exception as e:
        print(f"Could not create spherical visualization: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions on SPHERICAL/equirectangular raster data"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Model architecture name (e.g., unet_sc2_layers4_e32)"
    )
    parser.add_argument(
        "--raster_path",
        required=True,
        help="Path to input equirectangular raster (EPSG:4326)"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save prediction output"
    )
    parser.add_argument(
        "--shapefile_path",
        default=None,
        help="Path to shapefile (optional, for validation only)"
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=99,
        help="Number of input channels (must match training)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of output classes (must match training)"
    )
    parser.add_argument(
        "--grid_type",
        default="equiangular",
        choices=["equiangular", "legendre-gauss"],
        help="Grid type for quadrature"
    )
    parser.add_argument(
        "--normalize_stats",
        type=str,
        default=None,
        help="Path to normalization statistics (mean/std) from training"
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get raster dimensions
    with rasterio.open(args.raster_path) as src:
        if not src.crs.is_geographic:
            raise ValueError(
                f"Raster must be in geographic coordinates. Found: {src.crs}. "
                f"Use gdalwarp -t_srs EPSG:4326 to reproject."
            )
        img_size = (src.height, src.width)
    
    print(f"\nLoading model: {args.model_name}")
    print(f"Input shape: ({args.in_channels}, {img_size[0]}, {img_size[1]})")
    print(f"Output classes: {args.num_classes}")
    print(f"Grid type: {args.grid_type}")
    
    # Load model
    model = load_model(
        args.checkpoint,
        args.model_name,
        args.in_channels,
        args.num_classes,
        img_size,
        device
    )
    
    # Setup normalization if provided
    normalization = None
    if args.normalize_stats is not None:
        stats = torch.load(args.normalize_stats)
        means = stats['means']
        stds = stats['stds']
        normalization = v2.Normalize(mean=means.tolist(), std=stds.tolist())
        print(f"Using normalization from {args.normalize_stats}")
    
    # Make predictions
    predict_spherical(
        model,
        args.raster_path,
        args.output_path,
        shapefile_path=args.shapefile_path,
        grid_type=args.grid_type,
        normalization=normalization,
        device=device
    )
    
    print("\nPrediction complete!")


if __name__ == "__main__":
    main()
