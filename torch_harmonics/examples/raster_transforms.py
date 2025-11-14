# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Utilities for transforming arbitrary raster data into formats suitable for spherical processing.

This module provides tools to convert rasters from any projection into:
1. Equirectangular projection (EPSG:4326) with equiangular spacing
2. HEALPix grid format
3. Automated preprocessing pipelines

Functions handle reprojection, resampling, and format conversion.
"""

import os
import numpy as np
import warnings
from typing import Tuple, Optional, Union

try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.transform import from_bounds
    import geopandas as gpd
    from shapely.geometry import box
except ImportError:
    raise ImportError(
        "This module requires rasterio and geopandas. "
        "Install with: pip install rasterio geopandas"
    )


def transform_to_equirectangular(
    input_raster_path: str,
    output_raster_path: str,
    target_resolution: float = 0.1,
    extent: Optional[Tuple[float, float, float, float]] = None,
    resampling_method: str = 'bilinear',
    compress: bool = True,
) -> str:
    """
    Transform any raster into equirectangular projection with equiangular spacing.
    
    This function:
    - Reprojects to EPSG:4326 (geographic coordinates)
    - Ensures equiangular spacing (uniform pixel size in degrees)
    - Optionally clips to specified extent
    - Handles multi-band rasters
    
    Parameters
    ----------
    input_raster_path : str
        Path to input raster (any CRS/projection)
    output_raster_path : str
        Path to save equirectangular output
    target_resolution : float, optional
        Pixel size in degrees (default: 0.1° ≈ 11km at equator)
    extent : tuple, optional
        Target extent as (lon_min, lat_min, lon_max, lat_max) in degrees
        Default is global: (-180, -90, 180, 90)
    resampling_method : str, optional
        Resampling method: 'nearest', 'bilinear', 'cubic', 'average' (default: 'bilinear')
    compress : bool, optional
        Compress output with LZW (default: True)
        
    Returns
    -------
    str
        Path to output equirectangular raster
        
    Examples
    --------
    >>> # Convert UTM raster to global equirectangular
    >>> transform_to_equirectangular('utm_raster.tif', 'equirect_global.tif')
    
    >>> # Convert to regional equirectangular
    >>> transform_to_equirectangular(
    ...     'input.tif', 
    ...     'equirect_regional.tif',
    ...     target_resolution=0.01,  # 0.01° ≈ 1km
    ...     extent=(-120, 30, -110, 40)  # Western US
    ... )
    """
    # Set default global extent
    if extent is None:
        extent = (-180, -90, 180, 90)
    
    lon_min, lat_min, lon_max, lat_max = extent
    
    # Map resampling method string to enum
    resampling_map = {
        'nearest': Resampling.nearest,
        'bilinear': Resampling.bilinear,
        'cubic': Resampling.cubic,
        'cubic_spline': Resampling.cubic_spline,
        'average': Resampling.average,
        'mode': Resampling.mode,
    }
    
    if resampling_method not in resampling_map:
        raise ValueError(f"Invalid resampling method: {resampling_method}. "
                        f"Choose from: {list(resampling_map.keys())}")
    
    resampling_enum = resampling_map[resampling_method]
    
    print(f"Transforming {input_raster_path} to equirectangular...")
    print(f"  Target resolution: {target_resolution}° ({target_resolution * 111:.1f} km at equator)")
    print(f"  Extent: [{lon_min}, {lat_min}] to [{lon_max}, {lat_max}]")
    print(f"  Resampling: {resampling_method}")
    
    with rasterio.open(input_raster_path) as src:
        # Calculate output dimensions
        width = int((lon_max - lon_min) / target_resolution)
        height = int((lat_max - lat_min) / target_resolution)
        
        print(f"  Input: {src.width}x{src.height}, CRS={src.crs}")
        print(f"  Output: {width}x{height}, CRS=EPSG:4326")
        
        # Create output transform (equiangular grid)
        dst_transform = from_bounds(
            lon_min, lat_min, lon_max, lat_max,
            width, height
        )
        
        # Setup output profile
        dst_profile = src.profile.copy()
        dst_profile.update({
            'crs': 'EPSG:4326',
            'transform': dst_transform,
            'width': width,
            'height': height,
        })
        
        if compress:
            dst_profile['compress'] = 'lzw'
        
        # Reproject each band
        with rasterio.open(output_raster_path, 'w', **dst_profile) as dst:
            for band_idx in range(1, src.count + 1):
                src_array = src.read(band_idx)
                dst_array = np.empty((height, width), dtype=src_array.dtype)
                
                reproject(
                    source=src_array,
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs='EPSG:4326',
                    resampling=resampling_enum
                )
                
                dst.write(dst_array, band_idx)
                
                if band_idx % 10 == 0:
                    print(f"  Processed {band_idx}/{src.count} bands")
    
    print(f"✓ Saved equirectangular raster to: {output_raster_path}")
    
    # Validate output
    with rasterio.open(output_raster_path) as dst:
        if dst.crs.to_string() != 'EPSG:4326':
            warnings.warn(f"Output CRS is {dst.crs}, expected EPSG:4326")
        
        # Check pixel size uniformity
        pixel_width = abs(dst.transform[0])
        pixel_height = abs(dst.transform[4])
        
        if abs(pixel_width - target_resolution) > target_resolution * 0.01:
            warnings.warn(f"Pixel width {pixel_width}° differs from target {target_resolution}°")
        if abs(pixel_height - target_resolution) > target_resolution * 0.01:
            warnings.warn(f"Pixel height {pixel_height}° differs from target {target_resolution}°")
    
    return output_raster_path


def transform_shapefile_to_epsg4326(
    input_shapefile_path: str,
    output_shapefile_path: str,
) -> str:
    """
    Transform shapefile to EPSG:4326 (geographic coordinates).
    
    Parameters
    ----------
    input_shapefile_path : str
        Path to input shapefile (any CRS)
    output_shapefile_path : str
        Path to save reprojected shapefile
        
    Returns
    -------
    str
        Path to output shapefile
        
    Examples
    --------
    >>> transform_shapefile_to_epsg4326('utm_labels.shp', 'geographic_labels.shp')
    """
    print(f"Transforming shapefile to EPSG:4326...")
    
    gdf = gpd.read_file(input_shapefile_path)
    print(f"  Input CRS: {gdf.crs}")
    print(f"  Features: {len(gdf)}")
    
    if gdf.crs is None:
        raise ValueError("Input shapefile has no CRS defined")
    
    if gdf.crs.to_string() != 'EPSG:4326':
        gdf_reprojected = gdf.to_crs('EPSG:4326')
    else:
        gdf_reprojected = gdf
        print("  Shapefile already in EPSG:4326")
    
    gdf_reprojected.to_file(output_shapefile_path)
    print(f"✓ Saved reprojected shapefile to: {output_shapefile_path}")
    
    return output_shapefile_path


def raster_to_healpix(
    input_raster_path: str,
    output_path: str,
    nside: int = 64,
    nest: bool = False,
) -> str:
    """
    Convert raster to HEALPix grid format.
    
    HEALPix (Hierarchical Equal Area isoLatitude Pixelization) provides
    equal-area pixelization of the sphere, useful for global data analysis.
    
    Parameters
    ----------
    input_raster_path : str
        Path to input raster (must be in EPSG:4326)
    output_path : str
        Path to save HEALPix output (numpy .npz or FITS)
    nside : int, optional
        HEALPix resolution parameter (default: 64)
        Number of pixels = 12 * nside^2
        Common values: 16, 32, 64, 128, 256, 512
    nest : bool, optional
        Use nested pixel ordering (default: False = ring ordering)
        
    Returns
    -------
    str
        Path to output HEALPix file
        
    Notes
    -----
    Requires healpy: pip install healpy
    
    Examples
    --------
    >>> # Convert to HEALPix with resolution nside=128
    >>> raster_to_healpix('equirect.tif', 'healpix.npz', nside=128)
    """
    try:
        import healpy as hp
    except ImportError:
        raise ImportError(
            "HEALPix support requires healpy. "
            "Install with: pip install healpy"
        )
    
    print(f"Converting raster to HEALPix (nside={nside})...")
    
    with rasterio.open(input_raster_path) as src:
        if not src.crs.is_geographic:
            raise ValueError(
                f"Input raster must be in geographic CRS (EPSG:4326). "
                f"Found: {src.crs}. Use transform_to_equirectangular() first."
            )
        
        num_bands = src.count
        npix = hp.nside2npix(nside)
        
        print(f"  Input: {src.width}x{src.height}, {num_bands} bands")
        print(f"  Output: {npix} HEALPix pixels")
        print(f"  Pixel size: {hp.nside2resol(nside, arcmin=True):.2f} arcmin")
        
        # Initialize HEALPix arrays
        healpix_data = np.full((num_bands, npix), np.nan, dtype=np.float32)
        
        # Get pixel coordinates
        theta, phi = hp.pix2ang(nside, np.arange(npix), nest=nest)
        # Convert to lat/lon (degrees)
        lat = 90.0 - np.degrees(theta)
        lon = np.degrees(phi)
        lon[lon > 180] -= 360  # Convert to [-180, 180]
        
        # Sample raster at HEALPix pixel centers
        bounds = src.bounds
        transform = src.transform
        
        for band_idx in range(1, num_bands + 1):
            band_data = src.read(band_idx)
            
            # Convert lat/lon to pixel coordinates
            x = (lon - bounds.left) / (bounds.right - bounds.left) * src.width
            y = (bounds.top - lat) / (bounds.top - bounds.bottom) * src.height
            
            # Clip to valid range
            valid = (x >= 0) & (x < src.width) & (y >= 0) & (y < src.height)
            
            # Sample using bilinear interpolation
            x_valid = x[valid]
            y_valid = y[valid]
            
            x0 = np.floor(x_valid).astype(int)
            x1 = np.minimum(x0 + 1, src.width - 1)
            y0 = np.floor(y_valid).astype(int)
            y1 = np.minimum(y0 + 1, src.height - 1)
            
            wx = x_valid - x0
            wy = y_valid - y0
            
            # Bilinear interpolation
            values = (
                band_data[y0, x0] * (1 - wx) * (1 - wy) +
                band_data[y0, x1] * wx * (1 - wy) +
                band_data[y1, x0] * (1 - wx) * wy +
                band_data[y1, x1] * wx * wy
            )
            
            healpix_data[band_idx - 1, valid] = values
            
            if band_idx % 10 == 0:
                print(f"  Processed {band_idx}/{num_bands} bands")
    
    # Save HEALPix data
    if output_path.endswith('.fits') or output_path.endswith('.fit'):
        # Save as FITS file
        for band_idx in range(num_bands):
            hp.write_map(
                output_path if num_bands == 1 else output_path.replace('.fits', f'_band{band_idx+1}.fits'),
                healpix_data[band_idx],
                nest=nest,
                overwrite=True
            )
        print(f"✓ Saved HEALPix data to FITS: {output_path}")
    else:
        # Save as numpy npz
        np.savez_compressed(
            output_path,
            healpix_data=healpix_data,
            nside=nside,
            nest=nest,
            num_bands=num_bands,
        )
        print(f"✓ Saved HEALPix data to NPZ: {output_path}")
    
    return output_path


def preprocess_for_spherical_training(
    raster_path: str,
    shapefile_path: str,
    output_dir: str,
    target_resolution: float = 0.1,
    extent: Optional[Tuple[float, float, float, float]] = None,
    format: str = 'equirectangular',
    nside: int = 64,
) -> dict:
    """
    Automated preprocessing pipeline: transforms arbitrary data for spherical training.
    
    This function:
    1. Reprojects raster to EPSG:4326 (equirectangular or HEALPix)
    2. Reprojects shapefile to EPSG:4326
    3. Validates format for spherical processing
    4. Returns paths to processed data
    
    Parameters
    ----------
    raster_path : str
        Path to input raster (any CRS/projection)
    shapefile_path : str
        Path to input shapefile (any CRS)
    output_dir : str
        Directory to save processed outputs
    target_resolution : float, optional
        Pixel size in degrees for equirectangular (default: 0.1°)
    extent : tuple, optional
        Target extent as (lon_min, lat_min, lon_max, lat_max)
        Default is global: (-180, -90, 180, 90)
    format : str, optional
        Output format: 'equirectangular' or 'healpix' (default: 'equirectangular')
    nside : int, optional
        HEALPix resolution (only used if format='healpix')
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'raster': path to processed raster
        - 'shapefile': path to processed shapefile
        - 'format': output format
        - 'metadata': additional metadata
        
    Examples
    --------
    >>> # Preprocess for equirectangular training
    >>> outputs = preprocess_for_spherical_training(
    ...     'utm_raster.tif',
    ...     'utm_labels.shp',
    ...     'processed/',
    ...     target_resolution=0.05,
    ...     extent=(-180, -90, 180, 90)
    ... )
    >>> 
    >>> # Use with train_spherical.py
    >>> # python train_spherical.py --raster_path {outputs['raster']} ...
    
    >>> # Preprocess for HEALPix
    >>> outputs = preprocess_for_spherical_training(
    ...     'raster.tif',
    ...     'labels.shp',
    ...     'processed/',
    ...     format='healpix',
    ...     nside=128
    ... )
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("AUTOMATED PREPROCESSING FOR SPHERICAL TRAINING")
    print("="*80)
    
    results = {
        'format': format,
        'metadata': {}
    }
    
    # Step 1: Process raster
    print("\n[1/2] Processing raster...")
    
    base_name = os.path.splitext(os.path.basename(raster_path))[0]
    
    if format == 'equirectangular':
        output_raster = os.path.join(output_dir, f"{base_name}_equirect.tif")
        transform_to_equirectangular(
            raster_path,
            output_raster,
            target_resolution=target_resolution,
            extent=extent,
        )
        results['raster'] = output_raster
        results['metadata']['resolution'] = target_resolution
        results['metadata']['extent'] = extent
        
    elif format == 'healpix':
        # First convert to equirectangular
        temp_equirect = os.path.join(output_dir, f"{base_name}_temp_equirect.tif")
        transform_to_equirectangular(
            raster_path,
            temp_equirect,
            target_resolution=target_resolution,
            extent=extent,
        )
        
        # Then convert to HEALPix
        output_healpix = os.path.join(output_dir, f"{base_name}_healpix.npz")
        raster_to_healpix(
            temp_equirect,
            output_healpix,
            nside=nside,
        )
        results['raster'] = output_healpix
        results['metadata']['nside'] = nside
        
        # Clean up temp file
        os.remove(temp_equirect)
        print(f"  Removed temporary file: {temp_equirect}")
    else:
        raise ValueError(f"Invalid format: {format}. Choose 'equirectangular' or 'healpix'")
    
    # Step 2: Process shapefile
    print("\n[2/2] Processing shapefile...")
    
    shp_base = os.path.splitext(os.path.basename(shapefile_path))[0]
    output_shapefile = os.path.join(output_dir, f"{shp_base}_epsg4326.shp")
    transform_shapefile_to_epsg4326(shapefile_path, output_shapefile)
    results['shapefile'] = output_shapefile
    
    # Summary
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Format: {format}")
    print(f"Processed raster: {results['raster']}")
    print(f"Processed shapefile: {results['shapefile']}")
    print("\nNext steps:")
    
    if format == 'equirectangular':
        print(f"  python examples/segmentation/train_spherical.py \\")
        print(f"      --raster_path {results['raster']} \\")
        print(f"      --shapefile_path {results['shapefile']} \\")
        print(f"      --label_column label \\")
        print(f"      --num_epochs 100")
    elif format == 'healpix':
        print("  HEALPix format requires custom dataset loader (not yet implemented)")
        print("  Consider using equirectangular format for now")
    
    return results


def validate_equirectangular_format(raster_path: str, tolerance: float = 0.05) -> dict:
    """
    Validate that a raster is in proper equirectangular format.
    
    Checks:
    - CRS is geographic (EPSG:4326 or similar)
    - Latitude spacing is uniform (equiangular)
    - Longitude spacing is uniform
    - Provides recommendations if validation fails
    
    Parameters
    ----------
    raster_path : str
        Path to raster to validate
    tolerance : float, optional
        Tolerance for spacing uniformity (default: 0.05 = 5%)
        
    Returns
    -------
    dict
        Validation results with keys:
        - 'valid': bool, overall validation status
        - 'crs_valid': bool
        - 'spacing_valid': bool
        - 'messages': list of validation messages
        - 'recommendations': list of recommendations if invalid
    """
    results = {
        'valid': True,
        'crs_valid': False,
        'spacing_valid': False,
        'messages': [],
        'recommendations': []
    }
    
    with rasterio.open(raster_path) as src:
        # Check CRS
        if src.crs is None:
            results['valid'] = False
            results['messages'].append("❌ No CRS defined")
            results['recommendations'].append("Define CRS using gdalwarp or transform_to_equirectangular()")
        elif not src.crs.is_geographic:
            results['valid'] = False
            results['messages'].append(f"❌ CRS is not geographic: {src.crs}")
            results['recommendations'].append(
                f"Reproject to EPSG:4326 using transform_to_equirectangular('{raster_path}', 'output.tif')"
            )
        else:
            results['crs_valid'] = True
            results['messages'].append(f"✓ CRS is geographic: {src.crs}")
        
        # Check pixel spacing
        pixel_width = abs(src.transform[0])
        pixel_height = abs(src.transform[4])
        
        spacing_diff = abs(pixel_width - pixel_height) / pixel_width
        
        if spacing_diff > tolerance:
            results['valid'] = False
            results['messages'].append(
                f"❌ Non-uniform pixel spacing: {pixel_width:.6f}° x {pixel_height:.6f}° "
                f"(difference: {spacing_diff*100:.1f}%)"
            )
            results['recommendations'].append(
                "Resample to uniform spacing using transform_to_equirectangular()"
            )
        else:
            results['spacing_valid'] = True
            results['messages'].append(
                f"✓ Uniform pixel spacing: {pixel_width:.6f}° x {pixel_height:.6f}°"
            )
        
        # Additional info
        bounds = src.bounds
        results['messages'].append(f"  Extent: [{bounds.left:.2f}, {bounds.bottom:.2f}] to [{bounds.right:.2f}, {bounds.top:.2f}]")
        results['messages'].append(f"  Dimensions: {src.width} x {src.height}")
        results['messages'].append(f"  Bands: {src.count}")
    
    return results


if __name__ == "__main__":
    print("Raster transformation utilities for spherical processing")
    print("=" * 60)
    print("\nAvailable functions:")
    print("  1. transform_to_equirectangular() - Convert any raster to equirectangular")
    print("  2. transform_shapefile_to_epsg4326() - Reproject shapefile")
    print("  3. raster_to_healpix() - Convert to HEALPix grid")
    print("  4. preprocess_for_spherical_training() - Automated pipeline")
    print("  5. validate_equirectangular_format() - Validate format")
    print("\nExample usage:")
    print("  from torch_harmonics.examples.raster_transforms import preprocess_for_spherical_training")
    print("  outputs = preprocess_for_spherical_training('input.tif', 'labels.shp', 'processed/')")
