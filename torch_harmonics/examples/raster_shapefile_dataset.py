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

import os
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
    from rasterio.features import rasterize
    import geopandas as gpd
except ImportError:
    raise ImportError(
        "This dataset requires rasterio and geopandas. "
        "Install them with: pip install rasterio geopandas"
    )


class RasterShapefileDataset(Dataset):
    """
    Dataset for raster stack data with shapefile labels for binary classification.
    
    This dataset loads standard GIS rasters (planar coordinates) and creates binary 
    labels from a shapefile. Designed for regional/local studies (e.g., mineral 
    deposits) in standard map projections, NOT for 360° panoramic/equirectangular data.
    
    Supports both point geometries (e.g., mineral deposit locations) and polygon
    geometries. Points are automatically buffered to be visible at raster resolution.
    
    Parameters
    ----------
    raster_path : str
        Path to the raster file (GeoTIFF or similar format) in any standard projection
        (UTM, State Plane, geographic, etc.)
    shapefile_path : str
        Path to the shapefile containing labels (supports Points, MultiPoints, 
        Polygons, MultiPolygons, LineStrings)
    label_column : str, optional
        Column name in shapefile to use for labels (default: 'label')
    transform : callable, optional
        Optional transform to be applied on the input data
    tile_size : tuple, optional
        Size of tiles to extract from raster (height, width). 
        If None, uses the full raster as a single sample.
    stride : tuple, optional
        Stride for tile extraction. If None, defaults to tile_size (no overlap)
    
    Returns
    -------
    tuple
        (input_tensor, label_tensor) where:
        - input_tensor: torch.Tensor of shape (C, H, W)
        - label_tensor: torch.Tensor of shape (1, H, W) with binary labels
        
    Notes
    -----
    - Point geometries are buffered by ~0.5 pixel radius to ensure they are visible
      in the rasterized output. This is important for sparse point labels like
      mineral deposits.
    - This dataset treats rasters as planar images with Cartesian coordinates,
      not as spherical/equirectangular projections. For 360° panoramic data,
      different preprocessing is needed.
    - When used with torch-harmonics spherical models, spherical operations are
      applied to planar data (geometrically approximate but functional for local areas).
    """
    
    def __init__(
        self,
        raster_path,
        shapefile_path,
        label_column='label',
        transform=None,
        tile_size=None,
        stride=None,
    ):
        self.raster_path = raster_path
        self.shapefile_path = shapefile_path
        self.label_column = label_column
        self.transform = transform
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size
        
        # Load raster metadata
        with rasterio.open(raster_path) as src:
            self.raster_meta = src.meta
            self.raster_shape = (src.count, src.height, src.width)
            self.raster_transform = src.transform
            self.raster_crs = src.crs
            self.raster_bounds = src.bounds
            
        # Load shapefile
        self.gdf = gpd.read_file(shapefile_path)
        
        # Ensure shapefile is in same CRS as raster
        if self.gdf.crs != self.raster_crs:
            self.gdf = self.gdf.to_crs(self.raster_crs)
            
        # Setup tiling if requested
        if tile_size is not None:
            self.tiles = self._compute_tiles()
        else:
            # Single sample - full raster
            self.tiles = [(0, 0, self.raster_shape[1], self.raster_shape[2])]
            
        self._raster_data = None
        self._label_data = None
        
    def _compute_tiles(self):
        """Compute tile positions for the raster."""
        tiles = []
        h, w = self.raster_shape[1], self.raster_shape[2]
        tile_h, tile_w = self.tile_size
        stride_h, stride_w = self.stride
        
        for y in range(0, h - tile_h + 1, stride_h):
            for x in range(0, w - tile_w + 1, stride_w):
                tiles.append((y, x, tile_h, tile_w))
                
        return tiles
    
    def _load_raster(self):
        """Load the full raster into memory."""
        if self._raster_data is None:
            with rasterio.open(self.raster_path) as src:
                self._raster_data = src.read()  # Shape: (bands, height, width)
        return self._raster_data
    
    def _load_labels(self):
        """Rasterize shapefile labels, supporting both polygons and points."""
        if self._label_data is None:
            # Create binary labels from shapefile
            shapes = []
            points = []
            
            for idx, row in self.gdf.iterrows():
                if self.label_column in row and row[self.label_column]:
                    # Use label value from column, or default to 1 for presence
                    label_value = 1
                    if isinstance(row[self.label_column], (int, float)):
                        label_value = int(row[self.label_column])
                    
                    geom_type = row.geometry.geom_type
                    if geom_type in ['Point', 'MultiPoint']:
                        # Store points separately for buffer-based rasterization
                        points.append((row.geometry, label_value))
                    else:
                        # Polygons, LineStrings, etc.
                        shapes.append((row.geometry, label_value))
            
            # Initialize empty label array
            self._label_data = np.zeros(
                (self.raster_shape[1], self.raster_shape[2]),
                dtype=np.uint8
            )
            
            # Rasterize polygon/line geometries
            if shapes:
                self._label_data = rasterize(
                    shapes,
                    out_shape=(self.raster_shape[1], self.raster_shape[2]),
                    transform=self.raster_transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=False
                )
            
            # Rasterize point geometries with a buffer
            if points:
                # Buffer points to make them visible at raster resolution
                # Use ~1 pixel buffer (adjust based on resolution)
                from shapely.geometry import Point
                pixel_size = abs(self.raster_transform[0])  # Approximate pixel size
                buffer_size = pixel_size * 0.5  # Half pixel radius
                
                buffered_shapes = []
                for geom, label_val in points:
                    if geom.geom_type == 'Point':
                        buffered = geom.buffer(buffer_size)
                        buffered_shapes.append((buffered, label_val))
                    elif geom.geom_type == 'MultiPoint':
                        for point in geom.geoms:
                            buffered = point.buffer(buffer_size)
                            buffered_shapes.append((buffered, label_val))
                
                if buffered_shapes:
                    point_raster = rasterize(
                        buffered_shapes,
                        out_shape=(self.raster_shape[1], self.raster_shape[2]),
                        transform=self.raster_transform,
                        fill=0,
                        dtype=np.uint8,
                        all_touched=True  # Use all_touched for points
                    )
                    # Merge point labels with existing labels
                    self._label_data = np.maximum(self._label_data, point_raster)
                
        return self._label_data
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        # Load data
        raster = self._load_raster()
        labels = self._load_labels()
        
        # Get tile coordinates
        y, x, h, w = self.tiles[idx]
        
        # Extract tile
        raster_tile = raster[:, y:y+h, x:x+w]
        label_tile = labels[y:y+h, x:x+w]
        
        # Convert to torch tensors
        raster_tensor = torch.from_numpy(raster_tile).float()
        label_tensor = torch.from_numpy(label_tile).long()
        
        # Add channel dimension to labels if needed
        if label_tensor.dim() == 2:
            label_tensor = label_tensor.unsqueeze(0)
        
        if self.transform:
            raster_tensor = self.transform(raster_tensor)
            
        return raster_tensor, label_tensor
    
    @property
    def input_shape(self):
        """Return shape of input data."""
        if self.tile_size is not None:
            return (self.raster_shape[0], self.tile_size[0], self.tile_size[1])
        return self.raster_shape
    
    @property
    def num_classes(self):
        """Return number of classes (binary classification)."""
        return 2
    
    @property
    def num_channels(self):
        """Return number of input channels."""
        return self.raster_shape[0]


def compute_stats_raster(dataset):
    """
    Compute mean and standard deviation statistics for raster dataset.
    
    Parameters
    ----------
    dataset : RasterShapefileDataset
        Dataset to compute statistics for
        
    Returns
    -------
    tuple
        (means, stds) as torch tensors
    """
    # Accumulate statistics
    sum_vals = torch.zeros(dataset.num_channels)
    sum_sq_vals = torch.zeros(dataset.num_channels)
    n_pixels = 0
    
    for i in range(len(dataset)):
        data, _ = dataset[i]
        # data shape: (C, H, W)
        sum_vals += data.sum(dim=(1, 2))
        sum_sq_vals += (data ** 2).sum(dim=(1, 2))
        n_pixels += data.shape[1] * data.shape[2]
    
    means = sum_vals / n_pixels
    stds = torch.sqrt(sum_sq_vals / n_pixels - means ** 2)
    
    return means, stds
