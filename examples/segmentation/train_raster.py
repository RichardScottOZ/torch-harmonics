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

import os, sys
import random
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from torch_harmonics.examples.raster_shapefile_dataset import RasterShapefileDataset, compute_stats_raster
from torch_harmonics.quadrature import _precompute_latitudes
from torch_harmonics.examples.losses import DiceLossS2, CrossEntropyLossS2, FocalLossS2
from torch_harmonics.examples.metrics import IntersectionOverUnionS2, AccuracyS2
from torch_harmonics.plotting import plot_sphere, imshow_sphere

from torchvision.transforms import v2

# import baseline models
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


def log_weights_and_grads(exp_dir, model, iters=1):
    """Log weights and gradients for debugging."""
    log_path = os.path.join(exp_dir, "weights_and_grads")
    if not os.path.isdir(log_path):
        os.makedirs(log_path, exist_ok=True)

    weights_and_grads_fname = os.path.join(log_path, f"weights_and_grads_step{iters:03d}.tar")
    print(weights_and_grads_fname)

    weights_dict = {k: v for k, v in model.named_parameters()}
    grad_dict = {k: v.grad for k, v in model.named_parameters()}

    store_dict = {"iteration": iters, "grads": grad_dict, "weights": weights_dict}
    torch.save(store_dict, weights_and_grads_fname)


def validate_model(model, dataloader, loss_fn, metrics_fns, path_root, normalization=None, logging=True, device=torch.device("cpu")):
    """Validate model on validation dataset."""
    model.eval()

    num_examples = len(dataloader)

    # make output directory
    if logging and not os.path.isdir(path_root):
        os.makedirs(path_root, exist_ok=True)

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])

    # accumulation buffers for metrics and losses
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
            num_classes = prd.shape[-3]

            losses[idx] = loss_fn(prd, tar)

            for metric in metrics_fns:
                metric_buff = metrics[metric]
                metric_fn = metrics_fns[metric]
                metric_buff[idx] = metric_fn(prd, tar)

            prd = nn.functional.softmax(prd, dim=-3)
            prd = torch.argmax(prd, dim=-3).squeeze(0)

            # Save prediction visualizations
            glob_idx = idx + glob_off
            
            # For regular images (not spherical), use standard plotting
            pred_img = prd.cpu().numpy()
            tar_img = tar.cpu().squeeze(0).numpy()
            
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
            # Show first 3 channels of input as RGB if available
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
    train_sampler,
    test_dataloader,
    test_sampler,
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
    log_grads=0,
    exp_dir=None,
    logging=True,
    device=torch.device("cpu"),
):
    """Train the model."""
    
    train_start = time.time()

    # set AMP type
    amp_dtype = torch.float32
    if amp_mode == "fp16":
        amp_dtype = torch.float16
    elif amp_mode == "bf16":
        amp_dtype = torch.bfloat16

    # count iterations
    iters = 0

    for epoch in range(nepochs):

        # time each epoch
        epoch_start = time.time()

        # do the training
        accumulated_loss = torch.zeros(2, dtype=torch.float32, device=device)

        model.train()

        if dist.is_initialized():
            train_sampler.set_epoch(epoch)

        for inp, tar in train_dataloader:
            inp = inp.to(device)
            tar = tar.to(device)

            if normalization is not None:
                inp = normalization(inp)

            if augmentation is not None:
                inp = augmentation(inp)

                # flip randomly horizontally
                if random.random() < 0.5:
                    inp = torch.flip(inp, dims=(-1,))
                    tar = torch.flip(tar, dims=(-1,))

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_mode != "none")):
                prd = model(inp)
                loss = loss_fn(prd, tar)

            optimizer.zero_grad(set_to_none=True)
            gscaler.scale(loss).backward()

            if log_grads and (iters % log_grads == 0) and (exp_dir is not None):
                log_weights_and_grads(exp_dir, model, iters=iters)

            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            gscaler.step(optimizer)
            gscaler.update()

            # accumulate loss
            accumulated_loss[0] += loss.detach() * inp.size(0)
            accumulated_loss[1] += inp.size(0)

            iters += 1

        if dist.is_initialized():
            dist.all_reduce(accumulated_loss)

        accumulated_loss = (accumulated_loss[0] / accumulated_loss[1]).item()

        # perform validation
        valid_loss = torch.zeros(2, dtype=torch.float32, device=device)

        valid_metrics = {}
        for metric in metrics_fns:
            valid_metrics[metric] = torch.zeros(2, dtype=torch.float32, device=device)

        model.eval()

        if dist.is_initialized():
            test_sampler.set_epoch(epoch)

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
            print(f"--------------------------------------------------------------------------------")
            print(f"Epoch {epoch} summary:")
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

    # wrapping up
    train_time = time.time() - train_start

    if logging:
        print(f"--------------------------------------------------------------------------------")
        print(f"done. Training took {train_time:.2f}.")

    return valid_loss


def main(
    models,
    root_path,
    raster_path,
    shapefile_path,
    label_column='label',
    num_epochs=100,
    batch_size=8,
    learning_rate=1e-4,
    label_smoothing=0.0,
    max_grad_norm=0.0,
    train=True,
    load_checkpoint=False,
    amp_mode="none",
    ddp=False,
    enable_data_augmentation=False,
    log_grads=0,
    tile_size=None,
    stride=None,
):
    """Main training function."""

    # initialize distributed
    local_rank = 0
    logging = True
    if ddp:
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        local_rank = dist.get_rank() % torch.cuda.device_count()
        logging = dist.get_rank() == 0

    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    # set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    # Initialize dataset
    if logging:
        print(f"Initializing raster dataset from {raster_path}...")
        print(f"Using shapefile labels from {shapefile_path}...")

    dataset = RasterShapefileDataset(
        raster_path=raster_path,
        shapefile_path=shapefile_path,
        label_column=label_column,
        tile_size=tile_size,
        stride=stride,
    )

    # Create train/test/validation splits
    rng = torch.Generator().manual_seed(333)
    split_ratios = [0.8, 0.1, 0.1]  # 80% train, 10% test, 10% valid
    
    total_samples = len(dataset)
    train_size = int(split_ratios[0] * total_samples)
    test_size = int(split_ratios[1] * total_samples)
    valid_size = total_samples - train_size - test_size
    
    train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size, valid_size], generator=rng
    )

    # Compute statistics on training data
    if logging:
        print("Computing dataset statistics...")
    means, stds = compute_stats_raster(train_dataset)
    if logging:
        print(f"Computed stats: means shape={means.shape}, stds shape={stds.shape}")

    # split dataset if distributed
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, drop_last=True)
        test_sampler = DistributedSampler(test_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=True)
        valid_sampler = DistributedSampler(valid_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=True)
    else:
        train_sampler = None
        test_sampler = None
        valid_sampler = None

    # create the dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True if train_sampler is None else False, sampler=train_sampler, num_workers=2, pin_memory=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=2, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, sampler=valid_sampler, num_workers=0, pin_memory=True)

    # Setup normalization
    normalization = v2.Normalize(mean=means.tolist(), std=stds.tolist())
    
    # Setup augmentation
    if enable_data_augmentation:
        if logging:
            print("Using data augmentation")
        augmentation = v2.Compose([
            v2.RandomAutocontrast(p=0.5),
            v2.GaussianNoise(mean=0.0, sigma=0.1, clip=True),
        ])
    else:
        augmentation = None

    in_channels = dataset.num_channels
    num_classes = dataset.num_classes

    # Get dataset shape info
    img_size = dataset.input_shape[1:]  # (H, W)

    if logging:
        print(f"Train dataset initialized with {len(train_dataset)} samples")
        print(f"Test dataset initialized with {len(test_dataset)} samples")
        print(f"Validation dataset initialized with {len(valid_dataset)} samples")
        print(f"Input shape: {dataset.input_shape} (channels: {in_channels}, spatial: {img_size})")
        print(f"Number of classes: {num_classes}")

    # get baseline model registry
    baseline_models = get_baseline_models(img_size=img_size, in_chans=in_channels, out_chans=num_classes, drop_path_rate=0.1)

    # specify which models to train
    if models is None:
        # Default to smaller models for large raster data
        models = [
            "unet_sc2_layers4_e32",
            "transformer_sc2_layers4_e128",
        ]
    elif isinstance(models, str):
        models = [models]
    
    # Filter to only available models
    models = {k: baseline_models[k] for k in models if k in baseline_models}

    if len(models) == 0:
        raise ValueError("No valid models selected")

    # Binary classification - no class weights for simplicity
    class_weights = None

    # create the loss object
    loss_fn = CrossEntropyLossS2(nlat=img_size[0], nlon=img_size[1], grid="equiangular", weight=class_weights, smooth=label_smoothing).to(device=device)

    # metrics
    metrics_fns = {
        "mean IoU": IntersectionOverUnionS2(
            nlat=img_size[0],
            nlon=img_size[1],
            grid="equiangular",
            weight=class_weights,
        ).to(device=device),
        "mean Accuracy": AccuracyS2(
            nlat=img_size[0],
            nlon=img_size[1],
            grid="equiangular",
            weight=class_weights,
        ).to(device=device),
    }

    metrics = {}

    # iterate over models and train each model
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
            checkpoint_path = os.path.join(exp_dir, "checkpoint.pt")
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path))
                if logging:
                    print(f"Loaded checkpoint from {checkpoint_path}")

        # run the training
        if train:
            if logging and wandb is not None:
                run = wandb.init(project="raster binary segmentation", group=model_name, name=model_name + "_" + str(time.time()), config=model_handle.keywords)
            else:
                run = None

            # optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1, foreach=torch.cuda.is_available())
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
            gscaler = torch.GradScaler("cuda", enabled=(amp_mode == "fp16"))

            start_time = time.time()

            if logging:
                print(f"Training {model_name}...")

            train_model(
                model,
                train_dataloader,
                train_sampler,
                test_dataloader,
                test_sampler,
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
                log_grads=log_grads,
                exp_dir=exp_dir,
                logging=logging,
                device=device,
            )

            training_time = time.time() - start_time

            if logging:
                if run is not None:
                    run.finish()
                torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoint.pt"))
                print(f"Saved checkpoint to {os.path.join(exp_dir, 'checkpoint.pt')}")

        # set seed
        torch.manual_seed(333)
        torch.cuda.manual_seed(333)

        with torch.inference_mode():

            # run the validation
            losses, metric_results = validate_model(
                model, valid_dataloader, loss_fn, metrics_fns, os.path.join(exp_dir, "figures"), normalization=normalization, logging=logging, device=device
            )

            # gather losses and metrics into a single tensor
            if dist.is_initialized():
                losses_dist = torch.zeros(world_size * losses.shape[0], dtype=losses.dtype, device=device)
                dist.all_gather_into_tensor(losses_dist, losses)
                losses = losses_dist
                for metric_name, metric in metric_results.items():
                    metric_dist = torch.zeros(world_size * metric.shape[0], dtype=metric.dtype, device=device)
                    dist.all_gather_into_tensor(metric_dist, metric)
                    metric_results[metric_name] = metric_dist

            # compute statistics
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
            df.to_pickle(os.path.join(output_dir, "metrics.pkl"))
            print(f"\nMetrics for {model_name}:")
            print(df[model_name])

    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])


if __name__ == "__main__":
    import torch.multiprocessing as mp
    import platform

    # Use spawn method for cross-platform compatibility (Windows, Linux, macOS)
    # forkserver is not available on Windows
    try:
        if platform.system() == "Windows":
            mp.set_start_method("spawn", force=True)
        else:
            # Prefer forkserver on Unix-like systems for better isolation
            mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        # If context has already been set, continue
        pass

    # Try to login to wandb if available
    if wandb is not None:
        try:
            wandb.login()
        except:
            print("Warning: wandb login failed, continuing without wandb logging")

    parser = argparse.ArgumentParser(description="Train binary segmentation model on raster stack with shapefile labels")
    parser.add_argument(
        "--output_path", 
        default=os.path.join(os.path.dirname(__file__), "checkpoints_raster"), 
        type=str, 
        help="Path where checkpoints and run information are stored"
    )
    parser.add_argument(
        "--raster_path",
        required=True,
        type=str,
        help="Path to raster file (e.g., GeoTIFF with shape (99, 2000, 4000))",
    )
    parser.add_argument(
        "--shapefile_path",
        required=True,
        type=str,
        help="Path to shapefile with labels (e.g., sedex_mineral_deposits.shp)",
    )
    parser.add_argument(
        "--label_column",
        default="label",
        type=str,
        help="Column name in shapefile containing labels",
    )
    parser.add_argument(
        "--tile_size",
        default=None,
        type=int,
        nargs=2,
        help="Tile size (height width) for splitting raster. If not provided, uses full raster.",
    )
    parser.add_argument(
        "--stride",
        default=None,
        type=int,
        nargs=2,
        help="Stride for tile extraction. If not provided, defaults to tile_size.",
    )
    parser.add_argument("--models", default=None, type=str, nargs='+', help="Provide a list of models to run")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size for training")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm for clipping")
    parser.add_argument("--label_smoothing_factor", default=0.0, type=float, help="Label smoothing factor [0, 1]")
    parser.add_argument("--resume", action="store_true", help="Reload checkpoints")
    parser.add_argument("--amp_mode", default="none", type=str, choices=["none", "bf16", "fp16"], help="Enable AMP")
    parser.add_argument("--enable_ddp", action="store_true", help="Enable distributed data parallel")
    parser.add_argument("--enable_data_augmentation", action="store_true", help="Enable data augmentation")
    args = parser.parse_args()

    # Convert tile_size and stride to tuples
    tile_size = tuple(args.tile_size) if args.tile_size is not None else None
    stride = tuple(args.stride) if args.stride is not None else None

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
        log_grads=0,
        tile_size=tile_size,
        stride=stride,
    )
