# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2025 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Training script for TRUE spherical/equirectangular raster data.

This script properly handles spherical topology:
- Validates equirectangular format
- Applies area-weighted loss functions
- Handles periodic longitude boundaries
- Treats poles appropriately

Use this for: 360° panoramas, global climate data requiring proper spherical topology
Use train_raster.py for: Regional planar mapping (mineral deposits, etc.)
"""

import os, sys
import random
import time
import argparse
import platform
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from torch_harmonics.examples.spherical_raster_dataset import SphericalRasterDataset, compute_stats_spherical
from torch_harmonics.quadrature import _precompute_latitudes
from torch_harmonics.examples.losses import DiceLossS2, CrossEntropyLossS2, FocalLossS2
from torch_harmonics.examples.metrics import IntersectionOverUnionS2, AccuracyS2
from torch_harmonics.plotting import plot_sphere, imshow_sphere

from torchvision.transforms import v2

# Import baseline models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_registry import get_baseline_models

# wandb logging (optional)
try:
    import wandb
except ImportError:
    wandb = None


def count_parameters(model):
    """Count number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate_model(model, dataloader, loss_fn, metrics_fns, path_root, normalization=None, logging=True, device=torch.device("cpu")):
    """Validate model on validation dataset."""
    model.eval()

    num_examples = len(dataloader)

    if logging and not os.path.isdir(path_root):
        os.makedirs(path_root, exist_ok=True)

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])

    losses = torch.zeros(num_examples, dtype=torch.float32, device=device)
    metrics = {}
    for metric in metrics_fns:
        metrics[metric] = torch.zeros(num_examples, dtype=torch.float32, device=device)

    glob_off = 0
    if dist.is_initialized():
        glob_off = num_examples * dist.get_rank()

    with torch.no_grad():
        for idx, (inp, tar) in enumerate(dataloader):
            inpd = inp.to(device)
            tar = tar.to(device)

            if normalization is not None:
                inpd = normalization(inpd)

            prd = model(inpd)

            losses[idx] = loss_fn(prd, tar)

            for metric in metrics_fns:
                metric_buff = metrics[metric]
                metric_fn = metrics_fns[metric]
                metric_buff[idx] = metric_fn(prd, tar)

            prd = nn.functional.softmax(prd, dim=-3)
            prd = torch.argmax(prd, dim=-3).squeeze(0)

            # Save spherical visualization
            glob_idx = idx + glob_off
            
            pred_img = prd.cpu().numpy()
            tar_img = tar.cpu().squeeze(0).numpy()
            
            # Use spherical plotting if available
            try:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': 'mollweide'})
                
                # Prediction
                imshow_sphere(pred_img, ax=axes[0], cmap='gray', vmin=0, vmax=1)
                axes[0].set_title('Prediction (Spherical)')
                
                # Ground truth
                imshow_sphere(tar_img, ax=axes[1], cmap='gray', vmin=0, vmax=1)
                axes[1].set_title('Ground Truth (Spherical)')
                
                # Input (first 3 bands as RGB)
                inp_vis = inp.cpu().squeeze(0)[:3].permute(1, 2, 0).numpy()
                inp_vis = (inp_vis - inp_vis.min()) / (inp_vis.max() - inp_vis.min() + 1e-8)
                if inp_vis.shape[2] >= 3:
                    imshow_sphere(inp_vis[:,:,0], ax=axes[2], cmap='viridis')
                axes[2].set_title('Input (Spherical)')
                
                plt.tight_layout()
                plt.savefig(os.path.join(path_root, f"validation_spherical_{glob_idx}.png"), dpi=100)
                plt.close()
            except:
                # Fallback to regular plotting
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(pred_img, cmap='gray', vmin=0, vmax=1)
                plt.title('Prediction')
                plt.colorbar()
                
                plt.subplot(1, 3, 2)
                plt.imshow(tar_img, cmap='gray', vmin=0, vmax=1)
                plt.title('Ground Truth')
                plt.colorbar()
                
                plt.subplot(1, 3, 3)
                inp_vis = inp.cpu().squeeze(0)[:3].permute(1, 2, 0).numpy()
                inp_vis = (inp_vis - inp_vis.min()) / (inp_vis.max() - inp_vis.min() + 1e-8)
                plt.imshow(inp_vis)
                plt.title('Input (first 3 bands)')
                
                plt.tight_layout()
                plt.savefig(os.path.join(path_root, f"validation_{glob_idx}.png"))
                plt.close()

    return losses, metrics


def train_model(
    model,
    train_dataloader,
    test_dataloader,
    loss_fn,
    metrics_fns,
    optimizer,
    gscaler,
    scheduler=None,
    max_grad_norm=0.0,
    normalization=None,
    augmentation=None,
    nepochs=20,
    amp_mode="none",
    exp_dir=None,
    logging=True,
    device=torch.device("cpu"),
):
    """Train the model with spherical data."""
    
    train_start = time.time()

    # Set AMP type
    amp_dtype = torch.float32
    if amp_mode == "fp16":
        amp_dtype = torch.float16
    elif amp_mode == "bf16":
        amp_dtype = torch.bfloat16

    for epoch in range(nepochs):
        epoch_start = time.time()

        accumulated_loss = torch.zeros(2, dtype=torch.float32, device=device)

        model.train()

        for inp, tar in train_dataloader:
            inp = inp.to(device)
            tar = tar.to(device)

            if normalization is not None:
                inp = normalization(inp)

            if augmentation is not None:
                inp = augmentation(inp)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_mode != "none")):
                prd = model(inp)
                loss = loss_fn(prd, tar)

            optimizer.zero_grad(set_to_none=True)
            gscaler.scale(loss).backward()

            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            gscaler.step(optimizer)
            gscaler.update()

            accumulated_loss[0] += loss.detach() * inp.size(0)
            accumulated_loss[1] += inp.size(0)

        if dist.is_initialized():
            dist.all_reduce(accumulated_loss)

        accumulated_loss = (accumulated_loss[0] / accumulated_loss[1]).item()

        # Validation
        valid_loss = torch.zeros(2, dtype=torch.float32, device=device)
        valid_metrics = {}
        for metric in metrics_fns:
            valid_metrics[metric] = torch.zeros(2, dtype=torch.float32, device=device)

        model.eval()

        with torch.no_grad():
            for inp, tar in test_dataloader:
                inp = inp.to(device)
                tar = tar.to(device)

                if normalization is not None:
                    inp = normalization(inp)

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_mode != "none")):
                    prd = model(inp)
                    loss = loss_fn(prd, tar)

                valid_loss[0] += loss * inp.size(0)
                valid_loss[1] += inp.size(0)

                for metric in metrics_fns:
                    metric_buff = valid_metrics[metric]
                    metric_fn = metrics_fns[metric]
                    metric_buff[0] += metric_fn(prd, tar) * inp.size(0)
                    metric_buff[1] += inp.size(0)

            if dist.is_initialized():
                dist.all_reduce(valid_loss)
                for metric in metrics_fns:
                    dist.all_reduce(valid_metrics[metric])

        valid_loss = (valid_loss[0] / valid_loss[1]).item()
        for metric in valid_metrics:
            valid_metrics[metric] = (valid_metrics[metric][0] / valid_metrics[metric][1]).item()

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        if logging:
            print(f"={'='*80}")
            print(f"Epoch {epoch} summary (SPHERICAL MODE):")
            print(f"time taken: {epoch_time:.2f}")
            print(f"accumulated training loss: {accumulated_loss}")
            print(f"relative validation loss: {valid_loss}")
            for metric in valid_metrics:
                print(f"{metric}: {valid_metrics[metric]}")

            if wandb is not None and wandb.run is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                log_dict = {"loss": accumulated_loss, "validation loss": valid_loss, "learning rate": current_lr}
                for metric in valid_metrics:
                    log_dict[metric] = valid_metrics[metric]
                wandb.log(log_dict)

    train_time = time.time() - train_start

    if logging:
        print(f"={'='*80}")
        print(f"Done. Training took {train_time:.2f} seconds.")

    return valid_loss


def main(
    models,
    root_path,
    raster_path,
    shapefile_path,
    label_column='label',
    num_epochs=100,
    batch_size=1,  # Typically 1 for global data
    learning_rate=1e-4,
    label_smoothing=0.0,
    max_grad_norm=0.0,
    train=True,
    load_checkpoint=False,
    amp_mode="none",
    ddp=False,
    enable_data_augmentation=False,
    grid_type='equiangular',
    validate_spacing=True,
    require_global=False,
    exclude_polar_fraction=0.0,
):
    """Main training function for spherical data."""

    # Initialize distributed
    local_rank = 0
    logging = True
    if ddp:
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        local_rank = dist.get_rank() % torch.cuda.device_count()
        logging = dist.get_rank() == 0

    # Set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    if logging:
        print(f"{'='*80}")
        print(f"SPHERICAL TRAINING MODE")
        print(f"{'='*80}")
        print(f"Initializing SPHERICAL raster dataset from {raster_path}...")
        print(f"Using shapefile labels from {shapefile_path}...")
        print(f"Grid type: {grid_type}")
        print(f"Validate equiangular spacing: {validate_spacing}")
        print(f"Require global coverage: {require_global}")
        print(f"Exclude polar fraction: {exclude_polar_fraction}")

    # Initialize dataset with spherical validation
    dataset = SphericalRasterDataset(
        raster_path=raster_path,
        shapefile_path=shapefile_path,
        label_column=label_column,
        grid_type=grid_type,
        validate_spacing=validate_spacing,
        require_global=require_global,
        exclude_polar_fraction=exclude_polar_fraction,
    )

    if logging:
        print(f"\nDataset validated as proper equirectangular format:")
        print(f"  Latitude range: [{dataset.lat_min:.2f}, {dataset.lat_max:.2f}]")
        print(f"  Longitude range: [{dataset.lon_min:.2f}, {dataset.lon_max:.2f}]")
        print(f"  Shape: {dataset.nlat} x {dataset.nlon}")
        print(f"  Global coverage: {dataset.is_global}")
        print(f"  Input channels: {dataset.num_channels}")
        print(f"  Classes: {dataset.num_classes}")

    # For single global image, we don't split - use whole dataset
    # In practice, you might load multiple equirectangular images
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    valid_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Compute statistics with area weighting
    if logging:
        print("\nComputing dataset statistics (area-weighted)...")
    means, stds = compute_stats_spherical(dataset)
    if logging:
        print(f"Computed stats: means shape={means.shape}, stds shape={stds.shape}")

    # Setup normalization
    normalization = v2.Normalize(mean=means.tolist(), std=stds.tolist())
    
    # Setup augmentation (limited for spherical data)
    augmentation = None
    if enable_data_augmentation:
        if logging:
            print("Note: Augmentation for spherical data is limited")
        # Could add longitude shift, but not vertical flip (breaks poles)
        augmentation = v2.GaussianNoise(mean=0.0, sigma=0.1, clip=True)

    in_channels = dataset.num_channels
    num_classes = dataset.num_classes
    img_size = (dataset.nlat, dataset.nlon)

    # Get baseline models
    baseline_models = get_baseline_models(img_size=img_size, in_chans=in_channels, out_chans=num_classes, drop_path_rate=0.1)

    if models is None:
        models = ["unet_sc2_layers4_e32"]
    elif isinstance(models, str):
        models = [models]
    
    models = {k: baseline_models[k] for k in models if k in baseline_models}

    if len(models) == 0:
        raise ValueError("No valid models selected")

    # Create area-weighted loss with quadrature weights
    # Use dataset's quadrature weights
    loss_fn = CrossEntropyLossS2(
        nlat=img_size[0],
        nlon=img_size[1],
        grid=grid_type,
        weight=None,
        smooth=label_smoothing
    ).to(device=device)

    # Metrics with quadrature weights
    metrics_fns = {
        "mean IoU": IntersectionOverUnionS2(
            nlat=img_size[0],
            nlon=img_size[1],
            grid=grid_type,
            weight=None,
        ).to(device=device),
        "mean Accuracy": AccuracyS2(
            nlat=img_size[0],
            nlon=img_size[1],
            grid=grid_type,
            weight=None,
        ).to(device=device),
    }

    metrics = {}

    # Train each model
    for model_name, model_handle in models.items():

        model = model_handle().to(device)

        if logging:
            print(f"\n{model_name}:")
            print(model)

        if dist.is_initialized():
            model = DDP(model, device_ids=[device.index])

        metrics[model_name] = {}

        num_params = count_parameters(model)
        if logging:
            print(f"number of trainable params: {num_params}")

        metrics[model_name]["num_params"] = num_params

        exp_dir = os.path.join(root_path, model_name)
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir, exist_ok=True)

        if load_checkpoint:
            checkpoint_path = os.path.join(exp_dir, "checkpoint_spherical.pt")
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path))
                if logging:
                    print(f"Loaded checkpoint from {checkpoint_path}")

        # Run training
        if train:
            if logging and wandb is not None:
                run = wandb.init(project="spherical raster binary segmentation", group=model_name, name=model_name + "_spherical_" + str(time.time()), config=model_handle.keywords)
            else:
                run = None

            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1, foreach=torch.cuda.is_available())
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
            gscaler = torch.GradScaler("cuda", enabled=(amp_mode == "fp16"))

            start_time = time.time()

            if logging:
                print(f"\nTraining {model_name} on SPHERICAL data...")

            train_model(
                model,
                train_dataloader,
                test_dataloader,
                loss_fn,
                metrics_fns,
                optimizer,
                gscaler,
                scheduler,
                max_grad_norm=max_grad_norm,
                normalization=normalization,
                augmentation=augmentation,
                nepochs=num_epochs,
                amp_mode=amp_mode,
                exp_dir=exp_dir,
                logging=logging,
                device=device,
            )

            training_time = time.time() - start_time

            if logging:
                if run is not None:
                    run.finish()
                torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoint_spherical.pt"))
                print(f"Saved checkpoint to {os.path.join(exp_dir, 'checkpoint_spherical.pt')}")

        # Validation
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        with torch.inference_mode():
            losses, metric_results = validate_model(
                model, valid_dataloader, loss_fn, metrics_fns,
                os.path.join(exp_dir, "figures_spherical"),
                normalization=normalization,
                logging=logging,
                device=device
            )

            if dist.is_initialized():
                world_size = dist.get_world_size()
                losses_dist = torch.zeros(world_size * losses.shape[0], dtype=losses.dtype, device=device)
                dist.all_gather_into_tensor(losses_dist, losses)
                losses = losses_dist
                for metric_name, metric in metric_results.items():
                    metric_dist = torch.zeros(world_size * metric.shape[0], dtype=metric.dtype, device=device)
                    dist.all_gather_into_tensor(metric_dist, metric)
                    metric_results[metric_name] = metric_dist

            metrics[model_name]["loss mean"] = torch.mean(losses).item()
            metrics[model_name]["loss std"] = torch.std(losses).item()
            for metric in metric_results:
                metrics[model_name][metric + " mean"] = torch.mean(metric_results[metric]).item()
                metrics[model_name][metric + " std"] = torch.std(metric_results[metric]).item()

            if train:
                metrics[model_name]["training_time"] = training_time

        if logging:
            df = pd.DataFrame(metrics)
            output_dir = os.path.join(exp_dir, "output_data")
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            df.to_pickle(os.path.join(output_dir, "metrics_spherical.pkl"))
            print(f"\nMetrics for {model_name}:")
            print(df[model_name])

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])


if __name__ == "__main__":
    import torch.multiprocessing as mp

    # Cross-platform multiprocessing
    try:
        if platform.system() == "Windows":
            mp.set_start_method("spawn", force=True)
        else:
            mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    # Try to login to wandb if available
    if wandb is not None:
        try:
            wandb.login()
        except:
            print("Warning: wandb login failed, continuing without wandb logging")

    parser = argparse.ArgumentParser(description="Train binary segmentation model on SPHERICAL/equirectangular raster data")
    parser.add_argument(
        "--output_path",
        default=os.path.join(os.path.dirname(__file__), "checkpoints_spherical"),
        type=str,
        help="Path where checkpoints and run information are stored"
    )
    parser.add_argument(
        "--raster_path",
        required=True,
        type=str,
        help="Path to equirectangular raster in geographic coordinates (EPSG:4326)"
    )
    parser.add_argument(
        "--shapefile_path",
        required=True,
        type=str,
        help="Path to shapefile with labels"
    )
    parser.add_argument(
        "--label_column",
        default="label",
        type=str,
        help="Column name in shapefile containing labels"
    )
    parser.add_argument(
        "--grid_type",
        default="equiangular",
        type=str,
        choices=["equiangular", "legendre-gauss"],
        help="Grid type for quadrature weights"
    )
    parser.add_argument(
        "--validate_spacing",
        action="store_true",
        default=True,
        help="Validate equiangular latitude spacing"
    )
    parser.add_argument(
        "--require_global",
        action="store_true",
        help="Require full 360° longitude coverage"
    )
    parser.add_argument(
        "--exclude_polar_fraction",
        default=0.0,
        type=float,
        help="Fraction of polar latitudes to exclude (0.0-0.5)"
    )
    parser.add_argument("--models", default=None, type=str, nargs='+', help="List of models to run")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size (typically 1 for global data)")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm for clipping")
    parser.add_argument("--label_smoothing_factor", default=0.0, type=float, help="Label smoothing factor")
    parser.add_argument("--resume", action="store_true", help="Reload checkpoints")
    parser.add_argument("--amp_mode", default="none", type=str, choices=["none", "bf16", "fp16"], help="Enable AMP")
    parser.add_argument("--enable_ddp", action="store_true", help="Enable distributed data parallel")
    parser.add_argument("--enable_data_augmentation", action="store_true", help="Enable data augmentation")
    args = parser.parse_args()

    main(
        models=args.models,
        root_path=args.output_path,
        raster_path=args.raster_path,
        shapefile_path=args.shapefile_path,
        label_column=args.label_column,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing_factor,
        max_grad_norm=args.max_grad_norm,
        train=args.num_epochs > 0,
        load_checkpoint=args.resume,
        amp_mode=args.amp_mode,
        ddp=args.enable_ddp,
        enable_data_augmentation=args.enable_data_augmentation,
        grid_type=args.grid_type,
        validate_spacing=args.validate_spacing,
        require_global=args.require_global,
        exclude_polar_fraction=args.exclude_polar_fraction,
    )
