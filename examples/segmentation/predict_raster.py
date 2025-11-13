#!/usr/bin/env python3
# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""
Inference script for making predictions on planar raster data with trained models.

IMPORTANT: This script treats ALL rasters as planar rectangular images.
The --spherical flag does NOT perform any data transformations or special
spherical processing - it only validates CRS. The models apply spherical
convolutions to rectangular data in both modes.

This script handles:
- Loading trained models from checkpoints
- Making predictions on raster data (treated as planar rectangles)
- Saving predictions back to GeoTIFF format

For true spherical/equirectangular data processing with proper topology
handling, additional preprocessing would be required (not implemented here).
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_bounds
except ImportError:
    print("This script requires rasterio. Install with: pip install rasterio")
    sys.exit(1)

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from torch_harmonics.examples.raster_shapefile_dataset import RasterShapefileDataset
from torchvision.transforms import v2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_registry import get_baseline_models


def load_model(checkpoint_path, model_name, in_channels, num_classes, img_size, device):
    """
    Load a trained model from checkpoint.
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint file
    model_name : str
        Name of the model architecture
    in_channels : int
        Number of input channels
    num_classes : int
        Number of output classes
    img_size : tuple
        (height, width) of input images
    device : torch.device
        Device to load model on
        
    Returns
    -------
    model : nn.Module
        Loaded model in eval mode
    """
    # Get model registry
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


def predict_on_raster(
    model,
    raster_path,
    output_path,
    tile_size=None,
    stride=None,
    normalization=None,
    device=torch.device("cpu"),
    batch_size=1,
):
    """
    Make predictions on a raster file.
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    raster_path : str
        Path to input raster
    output_path : str
        Path to save predictions
    tile_size : tuple, optional
        Size of tiles for processing large rasters
    stride : tuple, optional
        Stride for tile processing
    normalization : callable, optional
        Normalization transform
    device : torch.device
        Device for inference
    batch_size : int
        Batch size for processing
    """
    
    # Open input raster
    with rasterio.open(raster_path) as src:
        raster_data = src.read()  # (bands, height, width)
        profile = src.profile
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
    
    # Convert to torch tensor
    raster_tensor = torch.from_numpy(raster_data).float()
    
    if normalization is not None:
        raster_tensor = normalization(raster_tensor)
    
    # Process raster
    if tile_size is None:
        # Process full raster
        with torch.no_grad():
            input_batch = raster_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
            output = model(input_batch)
            predictions = nn.functional.softmax(output, dim=1)
            predictions = torch.argmax(predictions, dim=1).squeeze(0).cpu().numpy()
    else:
        # Process in tiles
        predictions = process_tiled(
            model, raster_tensor, tile_size, stride, device, batch_size
        )
    
    # Save predictions
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(predictions.astype(np.uint8), 1)
    
    print(f"Predictions saved to: {output_path}")


def process_tiled(model, raster_tensor, tile_size, stride, device, batch_size):
    """
    Process large raster in tiles with overlap handling.
    
    Parameters
    ----------
    model : nn.Module
        Model for inference
    raster_tensor : torch.Tensor
        Input raster (C, H, W)
    tile_size : tuple
        (tile_h, tile_w)
    stride : tuple
        (stride_h, stride_w)
    device : torch.device
        Device for inference
    batch_size : int
        Batch size
        
    Returns
    -------
    predictions : np.ndarray
        Full prediction array
    """
    C, H, W = raster_tensor.shape
    tile_h, tile_w = tile_size
    stride_h, stride_w = stride if stride is not None else tile_size
    
    # Initialize output with vote counting for overlapping tiles
    prediction_sum = torch.zeros((2, H, W), dtype=torch.float32)  # 2 classes
    count_map = torch.zeros((H, W), dtype=torch.float32)
    
    # Generate tile positions
    tiles = []
    for y in range(0, H - tile_h + 1, stride_h):
        for x in range(0, W - tile_w + 1, stride_w):
            tiles.append((y, x))
    
    # Add edge tiles if needed
    if (H - tile_h) % stride_h != 0:
        for x in range(0, W - tile_w + 1, stride_w):
            tiles.append((H - tile_h, x))
    if (W - tile_w) % stride_w != 0:
        for y in range(0, H - tile_h + 1, stride_h):
            tiles.append((y, W - tile_w))
    if (H - tile_h) % stride_h != 0 and (W - tile_w) % stride_w != 0:
        tiles.append((H - tile_h, W - tile_w))
    
    # Process tiles in batches
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i+batch_size]
            batch_inputs = []
            
            for y, x in batch_tiles:
                tile = raster_tensor[:, y:y+tile_h, x:x+tile_w]
                batch_inputs.append(tile)
            
            batch_tensor = torch.stack(batch_inputs).to(device)
            outputs = model(batch_tensor)
            predictions = nn.functional.softmax(outputs, dim=1)
            
            # Accumulate predictions
            for j, (y, x) in enumerate(batch_tiles):
                prediction_sum[:, y:y+tile_h, x:x+tile_w] += predictions[j].cpu()
                count_map[y:y+tile_h, x:x+tile_w] += 1
    
    # Average overlapping predictions
    prediction_sum = prediction_sum / count_map.unsqueeze(0)
    final_predictions = torch.argmax(prediction_sum, dim=0).numpy()
    
    return final_predictions


def predict_on_sphere(
    model,
    raster_path,
    output_path,
    normalization=None,
    device=torch.device("cpu"),
    lat_grid='equiangular',
):
    """
    Make predictions on raster data with CRS validation for geographic coordinates.
    
    NOTE: This function performs IDENTICAL processing to predict_on_raster().
    It does NOT perform any special spherical transformations. The only difference
    is it warns if the raster is not in geographic CRS. The model processes the
    data as a rectangular array regardless of the spherical flag.
    
    For true spherical data handling with proper topology (periodic boundaries,
    pole treatment, equiangular spacing), additional preprocessing is required
    which is NOT implemented in this script.
    
    Parameters
    ----------
    model : nn.Module
        Trained model (applies spherical ops to rectangular data)
    raster_path : str
        Path to input raster
    output_path : str
        Path to save predictions
    normalization : callable, optional
        Normalization transform
    device : torch.device
        Device for inference
    lat_grid : str
        Not used - present for API compatibility only
    """
    
    # Open input raster and verify it's in a geographic coordinate system
    with rasterio.open(raster_path) as src:
        if src.crs is None or not src.crs.is_geographic:
            print("Warning: Raster CRS is not geographic. ")
            print("For geographic data, raster should be in lat/lon (EPSG:4326 or similar)")
            print("Consider reprojecting your data with gdalwarp -t_srs EPSG:4326")
            print("\nNote: This warning is informational only. Processing will proceed")
            print("identically to planar mode - no special spherical handling occurs.")
        
        raster_data = src.read()
        profile = src.profile
        height, width = src.height, src.width
    
    # Convert to torch tensor
    raster_tensor = torch.from_numpy(raster_data).float()
    
    if normalization is not None:
        raster_tensor = normalization(raster_tensor)
    
    # Process data (identically to planar mode)
    with torch.no_grad():
        input_batch = raster_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
        output = model(input_batch)
        predictions = nn.functional.softmax(output, dim=1)
        predictions = torch.argmax(predictions, dim=1).squeeze(0).cpu().numpy()
    
    # Save predictions
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(predictions.astype(np.uint8), 1)
    
    print(f"Predictions saved to: {output_path}")
    print(f"Output maintains projection from input")


def main():
    parser = argparse.ArgumentParser(
        description="Make predictions on raster data with trained models"
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
        help="Path to input raster for prediction"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save prediction output"
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
        "--tile_size",
        type=int,
        nargs=2,
        default=None,
        help="Tile size (height width) for processing large rasters"
    )
    parser.add_argument(
        "--stride",
        type=int,
        nargs=2,
        default=None,
        help="Stride for tile processing"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for tile processing"
    )
    parser.add_argument(
        "--spherical",
        action="store_true",
        help="Validate geographic CRS (warning only - processing is identical to planar mode)"
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
        img_size = (src.height, src.width)
    
    print(f"Loading model: {args.model_name}")
    print(f"Input shape: ({args.in_channels}, {img_size[0]}, {img_size[1]})")
    print(f"Output classes: {args.num_classes}")
    
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
    
    # Convert tile_size and stride to tuples
    tile_size = tuple(args.tile_size) if args.tile_size is not None else None
    stride = tuple(args.stride) if args.stride is not None else None
    
    # Make predictions
    print("\nMaking predictions...")
    if args.spherical:
        print("Mode: SPHERICAL (CRS validation only)")
        print("Note: Processing is identical to planar mode - no special transformations occur")
        print("      Rasters are treated as rectangular arrays regardless of this flag")
        predict_on_sphere(
            model,
            args.raster_path,
            args.output_path,
            normalization=normalization,
            device=device
        )
    else:
        print("Mode: PLANAR (default)")
        if tile_size is not None:
            print(f"Processing with tiles: {tile_size}, stride: {stride}")
        predict_on_raster(
            model,
            args.raster_path,
            args.output_path,
            tile_size=tile_size,
            stride=stride,
            normalization=normalization,
            device=device,
            batch_size=args.batch_size
        )
    
    print("\nPrediction complete!")


if __name__ == "__main__":
    main()
