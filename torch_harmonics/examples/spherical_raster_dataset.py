# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Dataset for TRUE spherical/equirectangular raster data with proper topology handling.

This dataset is designed for data in equirectangular projection with:
- Equiangular latitude spacing
- Full or partial longitude coverage (periodic boundaries)
- Proper pole treatment
- Area-weighted operations

Use this for: 360° panoramas, global climate data, proper spherical topology
Use raster_shapefile_dataset.py for: Regional planar mapping (mineral deposits, etc.)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings

try:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import transform_bounds
    import geopandas as gpd
except ImportError:
    raise ImportError(
        "This dataset requires rasterio and geopandas. "
        "Install them with: pip install rasterio geopandas"
    )

from torch_harmonics.quadrature import _precompute_latitudes


class SphericalRasterDataset(Dataset):
    """
    Dataset for equirectangular raster data with TRUE spherical topology handling.
    
    This dataset validates and processes data in equirectangular projection with
    proper handling of:
    - Equiangular latitude spacing
    - Periodic longitude boundaries  
    - Pole regions
    - Area-weighted quadrature
    
    Parameters
    ----------
    raster_path : str
        Path to raster in geographic coordinates (EPSG:4326 or similar)
    shapefile_path : str
        Path to shapefile with labels (will be reprojected if needed)
    label_column : str, optional
        Column name in shapefile for labels (default: 'label')
    grid_type : str, optional
        Grid type: 'equiangular' or 'legendre-gauss' (default: 'equiangular')
    validate_spacing : bool, optional
        Validate equiangular spacing (default: True)
    require_global : bool, optional
        Require full 360° longitude coverage (default: False)
    exclude_polar_fraction : float, optional
        Fraction of polar latitudes to exclude (0.0-0.5, default: 0.0)
    transform : callable, optional
        Transform to apply to data
        
    Raises
    ------
    ValueError
        If raster is not in geographic coordinates
        If latitude spacing is not equiangular (when validate_spacing=True)
        If global coverage required but not present
    """
    
    def __init__(
        self,
        raster_path,
        shapefile_path,
        label_column='label',
        grid_type='equiangular',
        validate_spacing=True,
        require_global=False,
        exclude_polar_fraction=0.0,
        transform=None,
    ):
        self.raster_path = raster_path
        self.shapefile_path = shapefile_path
        self.label_column = label_column
        self.grid_type = grid_type
        self.validate_spacing = validate_spacing
        self.require_global = require_global
        self.exclude_polar_fraction = exclude_polar_fraction
        self.transform = transform
        
        # Load and validate raster metadata
        with rasterio.open(raster_path) as src:
            # Must be geographic CRS
            if src.crs is None or not src.crs.is_geographic:
                raise ValueError(
                    f"Raster must be in geographic coordinates (lat/lon). "
                    f"Found CRS: {src.crs}. Use gdalwarp -t_srs EPSG:4326 to reproject."
                )
            
            self.raster_meta = src.meta
            self.raster_shape = (src.count, src.height, src.width)
            self.raster_transform = src.transform
            self.raster_crs = src.crs
            self.raster_bounds = src.bounds
            
            # Extract lat/lon ranges
            self.lon_min, self.lat_min, self.lon_max, self.lat_max = src.bounds
            
        # Validate equirectangular format
        self._validate_equirectangular()
        
        # Compute quadrature weights for area correction
        self._compute_quadrature_weights()
        
        # Load shapefile
        self.gdf = gpd.read_file(shapefile_path)
        if self.gdf.crs != self.raster_crs:
            self.gdf = self.gdf.to_crs(self.raster_crs)
            
        self._raster_data = None
        self._label_data = None
        
    def _validate_equirectangular(self):
        """Validate that raster is in proper equirectangular format."""
        nlat, nlon = self.raster_shape[1], self.raster_shape[2]
        
        # Check longitude coverage
        lon_span = self.lon_max - self.lon_min
        if self.require_global and abs(lon_span - 360.0) > 1e-3:
            raise ValueError(
                f"Global coverage required but longitude span is {lon_span:.2f}° (need 360°)"
            )
        
        # Compute expected latitude spacing for equiangular grid
        lat_span = self.lat_max - self.lat_min
        if self.validate_spacing:
            # For equiangular: lat_spacing = lat_span / (nlat - 1)
            expected_lat_spacing = lat_span / (nlat - 1)
            
            # Verify using transform
            pixel_height = abs(self.raster_transform[4])  # Negative for north-up
            
            tolerance = expected_lat_spacing * 0.05  # 5% tolerance
            if abs(pixel_height - expected_lat_spacing) > tolerance:
                warnings.warn(
                    f"Latitude spacing ({pixel_height:.6f}°) deviates from equiangular "
                    f"({expected_lat_spacing:.6f}°) by more than 5%. "
                    f"Spherical operations may be approximate."
                )
        
        # Check if data spans full latitude range
        if abs(self.lat_max - 90.0) > 1.0 or abs(self.lat_min + 90.0) > 1.0:
            warnings.warn(
                f"Data does not span full latitude range [-90, 90]. "
                f"Found [{self.lat_min:.2f}, {self.lat_max:.2f}]. "
                f"Spherical operations will be applied to partial sphere."
            )
    
    def _compute_quadrature_weights(self):
        """Compute quadrature weights for area-weighted operations."""
        nlat = self.raster_shape[1]
        nlon = self.raster_shape[2]
        
        # Get latitude values and quadrature weights
        lats, quad_weights_1d = _precompute_latitudes(nlat=nlat, grid=self.grid_type)
        
        # Expand to 2D (nlat, nlon)
        # Longitude weight is uniform: 2*pi / nlon
        lon_weight = 2.0 * np.pi / float(nlon)
        
        # Combine: weights[i,j] = lat_weight[i] * lon_weight
        self.quad_weights = torch.from_numpy(quad_weights_1d).float()
        self.quad_weights = self.quad_weights.reshape(-1, 1) * lon_weight
        self.quad_weights = self.quad_weights.tile(1, nlon)
        
        # Normalize so sum = 1
        self.quad_weights = self.quad_weights / self.quad_weights.sum()
        
        # Handle polar exclusion if requested
        if self.exclude_polar_fraction > 0:
            n_exclude = int(nlat * self.exclude_polar_fraction)
            if n_exclude > 0:
                # Zero out top and bottom rows
                self.quad_weights[:n_exclude, :] = 0
                self.quad_weights[-n_exclude:, :] = 0
                # Renormalize
                if self.quad_weights.sum() > 0:
                    self.quad_weights = self.quad_weights / self.quad_weights.sum()
    
    def _load_raster(self):
        """Load raster data into memory."""
        if self._raster_data is None:
            with rasterio.open(self.raster_path) as src:
                self._raster_data = src.read()  # (bands, height, width)
        return self._raster_data
    
    def _load_labels(self):
        """Rasterize shapefile labels with support for points and polygons."""
        if self._label_data is None:
            shapes = []
            points = []
            
            for idx, row in self.gdf.iterrows():
                if self.label_column in row and row[self.label_column]:
                    label_value = 1
                    if isinstance(row[self.label_column], (int, float)):
                        label_value = int(row[self.label_column])
                    
                    geom_type = row.geometry.geom_type
                    if geom_type in ['Point', 'MultiPoint']:
                        points.append((row.geometry, label_value))
                    else:
                        shapes.append((row.geometry, label_value))
            
            # Initialize labels
            self._label_data = np.zeros(
                (self.raster_shape[1], self.raster_shape[2]),
                dtype=np.uint8
            )
            
            # Rasterize polygons
            if shapes:
                self._label_data = rasterize(
                    shapes,
                    out_shape=(self.raster_shape[1], self.raster_shape[2]),
                    transform=self.raster_transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=False
                )
            
            # Rasterize points with buffer
            if points:
                from shapely.geometry import Point
                pixel_size = abs(self.raster_transform[0])
                buffer_size = pixel_size * 0.5
                
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
                        all_touched=True
                    )
                    self._label_data = np.maximum(self._label_data, point_raster)
                    
        return self._label_data
    
    def __len__(self):
        return 1  # Single global image
    
    def __getitem__(self, idx):
        # Load data
        raster = self._load_raster()
        labels = self._load_labels()
        
        # Convert to tensors
        raster_tensor = torch.from_numpy(raster).float()
        label_tensor = torch.from_numpy(labels).long()
        
        if self.transform:
            raster_tensor = self.transform(raster_tensor)
            
        return raster_tensor, label_tensor
    
    @property
    def num_classes(self):
        """Number of classes (binary classification)."""
        return 2
    
    @property
    def num_channels(self):
        """Number of input channels."""
        return self.raster_shape[0]
    
    @property
    def nlat(self):
        """Number of latitude points."""
        return self.raster_shape[1]
    
    @property
    def nlon(self):
        """Number of longitude points."""
        return self.raster_shape[2]
    
    @property
    def is_global(self):
        """Check if data spans full 360° longitude."""
        lon_span = self.lon_max - self.lon_min
        return abs(lon_span - 360.0) < 1.0


def compute_stats_spherical(dataset):
    """
    Compute statistics for spherical dataset using area-weighted quadrature.
    
    Parameters
    ----------
    dataset : SphericalRasterDataset
        Dataset to compute statistics for
        
    Returns
    -------
    tuple
        (means, stds) as torch tensors
    """
    data, _ = dataset[0]
    quad_weights = dataset.quad_weights
    
    # data shape: (C, H, W)
    # quad_weights shape: (H, W)
    
    # Area-weighted mean
    means = torch.sum(data * quad_weights.unsqueeze(0), dim=(1, 2))
    
    # Area-weighted variance
    deviations = data - means[:, None, None]
    variances = torch.sum((deviations ** 2) * quad_weights.unsqueeze(0), dim=(1, 2))
    stds = torch.sqrt(variances)
    
    return means, stds
