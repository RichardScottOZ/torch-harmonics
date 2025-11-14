#!/usr/bin/env python3
# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Command-line tool for preprocessing raster data for spherical training.

This script provides an easy-to-use interface for transforming arbitrary
raster data into formats suitable for spherical processing (equirectangular or HEALPix).

Usage:
    python preprocess_for_spherical.py \
        --raster input.tif \
        --shapefile labels.shp \
        --output_dir processed/ \
        --format equirectangular \
        --resolution 0.1

See --help for all options.
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from torch_harmonics.examples.raster_transforms import (
    preprocess_for_spherical_training,
    validate_equirectangular_format,
    transform_to_equirectangular,
    transform_shapefile_to_epsg4326,
    raster_to_healpix,
)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raster data for spherical training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full preprocessing pipeline (equirectangular)
  python preprocess_for_spherical.py \\
      --raster utm_raster.tif \\
      --shapefile labels.shp \\
      --output_dir processed/ \\
      --format equirectangular \\
      --resolution 0.1
  
  # Global coverage
  python preprocess_for_spherical.py \\
      --raster input.tif \\
      --shapefile labels.shp \\
      --output_dir processed/ \\
      --format equirectangular \\
      --resolution 0.05 \\
      --extent -180 -90 180 90
  
  # Regional coverage (Western US)
  python preprocess_for_spherical.py \\
      --raster input.tif \\
      --shapefile labels.shp \\
      --output_dir processed/ \\
      --extent -125 30 -100 50
  
  # HEALPix format
  python preprocess_for_spherical.py \\
      --raster input.tif \\
      --shapefile labels.shp \\
      --output_dir processed/ \\
      --format healpix \\
      --nside 128
  
  # Validate existing equirectangular raster
  python preprocess_for_spherical.py \\
      --validate equirect_raster.tif
  
  # Transform raster only (no shapefile)
  python preprocess_for_spherical.py \\
      --raster input.tif \\
      --output_dir processed/ \\
      --format equirectangular
"""
    )
    
    # Input/output
    parser.add_argument(
        '--raster',
        type=str,
        help='Path to input raster (any CRS/projection)'
    )
    parser.add_argument(
        '--shapefile',
        type=str,
        help='Path to input shapefile (any CRS)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='processed_spherical',
        help='Directory to save processed outputs (default: processed_spherical/)'
    )
    
    # Format options
    parser.add_argument(
        '--format',
        type=str,
        choices=['equirectangular', 'healpix'],
        default='equirectangular',
        help='Output format: equirectangular or healpix (default: equirectangular)'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=0.1,
        help='Pixel size in degrees for equirectangular (default: 0.1° ≈ 11km)'
    )
    parser.add_argument(
        '--extent',
        type=float,
        nargs=4,
        metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'),
        help='Target extent: lon_min lat_min lon_max lat_max (default: global -180 -90 180 90)'
    )
    parser.add_argument(
        '--nside',
        type=int,
        default=64,
        help='HEALPix resolution parameter (default: 64). Common: 16, 32, 64, 128, 256'
    )
    parser.add_argument(
        '--resampling',
        type=str,
        choices=['nearest', 'bilinear', 'cubic', 'average'],
        default='bilinear',
        help='Resampling method (default: bilinear)'
    )
    
    # Validation mode
    parser.add_argument(
        '--validate',
        type=str,
        metavar='RASTER',
        help='Validate existing equirectangular raster (skips processing)'
    )
    
    # Other options
    parser.add_argument(
        '--no_compress',
        action='store_true',
        help='Disable LZW compression in output'
    )
    
    args = parser.parse_args()
    
    # Validation mode
    if args.validate:
        print("="*80)
        print("VALIDATING EQUIRECTANGULAR FORMAT")
        print("="*80)
        print(f"Raster: {args.validate}\n")
        
        results = validate_equirectangular_format(args.validate)
        
        for msg in results['messages']:
            print(msg)
        
        if not results['valid']:
            print("\n" + "="*80)
            print("VALIDATION FAILED")
            print("="*80)
            print("Recommendations:")
            for rec in results['recommendations']:
                print(f"  • {rec}")
            sys.exit(1)
        else:
            print("\n" + "="*80)
            print("VALIDATION PASSED ✓")
            print("="*80)
            print("Raster is properly formatted for spherical training")
            sys.exit(0)
    
    # Processing mode
    if not args.raster:
        parser.error("--raster is required (or use --validate for validation only)")
    
    # Parse extent
    extent = tuple(args.extent) if args.extent else None
    
    # Full pipeline if shapefile provided
    if args.shapefile:
        outputs = preprocess_for_spherical_training(
            raster_path=args.raster,
            shapefile_path=args.shapefile,
            output_dir=args.output_dir,
            target_resolution=args.resolution,
            extent=extent,
            format=args.format,
            nside=args.nside,
        )
        
        print("\n" + "="*80)
        print("SUCCESS ✓")
        print("="*80)
        print(f"Processed data saved to: {args.output_dir}")
        print(f"  Raster: {outputs['raster']}")
        print(f"  Shapefile: {outputs['shapefile']}")
        
    # Raster only
    else:
        print("="*80)
        print("PROCESSING RASTER ONLY (no shapefile)")
        print("="*80)
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        if args.format == 'equirectangular':
            base_name = os.path.splitext(os.path.basename(args.raster))[0]
            output_raster = os.path.join(args.output_dir, f"{base_name}_equirect.tif")
            
            transform_to_equirectangular(
                args.raster,
                output_raster,
                target_resolution=args.resolution,
                extent=extent,
                resampling_method=args.resampling,
                compress=not args.no_compress,
            )
            
            print("\n" + "="*80)
            print("SUCCESS ✓")
            print("="*80)
            print(f"Processed raster: {output_raster}")
            
        elif args.format == 'healpix':
            # Need intermediate equirectangular
            temp_name = os.path.splitext(os.path.basename(args.raster))[0]
            temp_equirect = os.path.join(args.output_dir, f"{temp_name}_temp_equirect.tif")
            
            print("\n[1/2] Converting to equirectangular...")
            transform_to_equirectangular(
                args.raster,
                temp_equirect,
                target_resolution=args.resolution,
                extent=extent,
                resampling_method=args.resampling,
                compress=False,
            )
            
            print("\n[2/2] Converting to HEALPix...")
            output_healpix = os.path.join(args.output_dir, f"{temp_name}_healpix.npz")
            raster_to_healpix(
                temp_equirect,
                output_healpix,
                nside=args.nside,
            )
            
            os.remove(temp_equirect)
            print(f"Removed temporary file: {temp_equirect}")
            
            print("\n" + "="*80)
            print("SUCCESS ✓")
            print("="*80)
            print(f"Processed HEALPix data: {output_healpix}")


if __name__ == "__main__":
    main()
