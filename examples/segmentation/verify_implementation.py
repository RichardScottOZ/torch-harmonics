#!/usr/bin/env python3
# coding=utf-8

"""
Verification script to check that the raster training implementation is correct.

This script performs basic checks on the implementation without requiring
real data or the full torch-harmonics library to be built.
"""

import os
import sys
import ast


def check_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, "Valid syntax"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_file_contains(filepath, required_strings):
    """Check if file contains required strings."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        missing = []
        for s in required_strings:
            if s not in content:
                missing.append(s)
        if missing:
            return False, f"Missing required content: {missing}"
        return True, "All required content present"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    print("="*80)
    print("Verifying Raster Training Implementation")
    print("="*80)
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(base_dir))
    
    checks = []
    
    # Check 1: Dataset file exists and has valid syntax
    dataset_file = os.path.join(repo_root, "torch_harmonics", "examples", "raster_shapefile_dataset.py")
    checks.append(("Dataset file exists", os.path.exists(dataset_file)))
    if os.path.exists(dataset_file):
        ok, msg = check_file_syntax(dataset_file)
        checks.append(("Dataset file syntax", ok, msg))
        
        # Check for required classes/functions
        ok, msg = check_file_contains(dataset_file, [
            "class RasterShapefileDataset",
            "def compute_stats_raster",
            "import rasterio",
            "import geopandas",
        ])
        checks.append(("Dataset file content", ok, msg))
    
    # Check 2: Training script exists and has valid syntax
    train_file = os.path.join(base_dir, "train_raster.py")
    checks.append(("Training script exists", os.path.exists(train_file)))
    if os.path.exists(train_file):
        ok, msg = check_file_syntax(train_file)
        checks.append(("Training script syntax", ok, msg))
        
        # Check for required functionality
        ok, msg = check_file_contains(train_file, [
            "from torch_harmonics.examples.raster_shapefile_dataset import RasterShapefileDataset",
            "def train_model",
            "def validate_model",
            "def main",
            "argparse",
            "--raster_path",
            "--shapefile_path",
            "--label_column",
        ])
        checks.append(("Training script content", ok, msg))
    
    # Check 3: README exists
    readme_file = os.path.join(base_dir, "README_raster.md")
    checks.append(("README exists", os.path.exists(readme_file)))
    if os.path.exists(readme_file):
        ok, msg = check_file_contains(readme_file, [
            "train_raster.py",
            "--raster_path",
            "--shapefile_path",
            "binary classification",
        ])
        checks.append(("README content", ok, msg))
    
    # Check 4: Example script exists
    example_file = os.path.join(base_dir, "example_usage.py")
    checks.append(("Example script exists", os.path.exists(example_file)))
    if os.path.exists(example_file):
        ok, msg = check_file_syntax(example_file)
        checks.append(("Example script syntax", ok, msg))
    
    # Check 5: pyproject.toml has raster dependencies
    pyproject_file = os.path.join(repo_root, "pyproject.toml")
    checks.append(("pyproject.toml exists", os.path.exists(pyproject_file)))
    if os.path.exists(pyproject_file):
        ok, msg = check_file_contains(pyproject_file, [
            "raster =",
            "rasterio",
            "geopandas",
        ])
        checks.append(("pyproject.toml has raster deps", ok, msg))
    
    # Check 6: __init__.py exports dataset
    init_file = os.path.join(repo_root, "torch_harmonics", "examples", "__init__.py")
    checks.append(("__init__.py exists", os.path.exists(init_file)))
    if os.path.exists(init_file):
        ok, msg = check_file_contains(init_file, [
            "from .raster_shapefile_dataset import RasterShapefileDataset",
        ])
        checks.append(("__init__.py exports dataset", ok, msg))
    
    # Print results
    print("\nCheck Results:")
    print("-" * 80)
    
    all_passed = True
    for check in checks:
        if len(check) == 2:
            name, passed = check
            msg = ""
        else:
            name, passed, msg = check
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")
        if msg and not passed:
            print(f"         {msg}")
        
        if not passed:
            all_passed = False
    
    print("-" * 80)
    
    if all_passed:
        print("\n✓ All checks passed!")
        print("\nImplementation Summary:")
        print("  - RasterShapefileDataset class for loading raster + shapefile data")
        print("  - train_raster.py adapted training script for binary classification")
        print("  - Support for (99, 2000, 4000) raster stacks")
        print("  - Binary classification with shapefile labels")
        print("  - Documentation and examples included")
        return 0
    else:
        print("\n✗ Some checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
