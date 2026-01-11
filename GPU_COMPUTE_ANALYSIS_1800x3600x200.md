# GPU Compute Requirements Analysis for torch-harmonics
## Raster Stack Configuration: 1800x3600x200

**Document Version:** 1.0  
**Date:** January 2026  
**Target Configuration:** 1800×3600 raster with 200 spectral layers

---

## Executive Summary

This document provides a comprehensive analysis of GPU compute requirements for running torch-harmonics on a high-resolution geospatial raster stack with dimensions 1800 (latitude) × 3600 (longitude) × 200 (spectral bands/layers). The analysis covers memory requirements, computational complexity, and hardware recommendations for both inference and training scenarios.

### Key Findings

- **Minimum GPU Memory (Inference, with SHT weights):** 32 GB (40 GB recommended)
- **Recommended GPU Memory (Training):** 40-80 GB
- **Optimal GPU:** NVIDIA A100 (40GB/80GB), H100
- **Compute Intensity:** ~30-100 TFLOPS per forward pass (model dependent)
- **Batch Processing:** Essential for memory management
- **Note:** SHT precomputed weights require ~22 GiB of fixed overhead

---

## Table of Contents

1. [Input Data Specifications](#1-input-data-specifications)
2. [Memory Requirements Analysis](#2-memory-requirements-analysis)
3. [Computational Complexity](#3-computational-complexity)
4. [GPU Architecture Considerations](#4-gpu-architecture-considerations)
5. [Hardware Recommendations](#5-hardware-recommendations)
6. [Performance Optimization Strategies](#6-performance-optimization-strategies)
7. [Scaling Considerations](#7-scaling-considerations)
8. [Appendix: Calculation Details](#appendix-calculation-details)

---

## 1. Input Data Specifications

### 1.1 Raster Stack Dimensions

| Parameter | Value | Description |
|-----------|-------|-------------|
| Latitude (nlat) | 1800 | Number of latitude grid points |
| Longitude (nlon) | 3600 | Number of longitude grid points |
| Spectral Bands | 200 | Number of input channels/layers |
| Total Grid Points | 6,480,000 | nlat × nlon |
| Grid Ratio | 2:1 | Standard equirectangular projection |

### 1.2 Data Precision Options

| Precision | Bytes/Value | Total Size (Single Sample) |
|-----------|-------------|---------------------------|
| Float32 (FP32) | 4 bytes | 4.83 GiB (5.18 GB) |
| Float16 (FP16) | 2 bytes | 2.42 GiB (2.59 GB) |
| BFloat16 (BF16) | 2 bytes | 2.42 GiB (2.59 GB) |

**Calculation:** 1800 × 3600 × 200 × bytes_per_value  
**Note:** GiB = binary gigabytes (1024³), GB = decimal gigabytes (10⁹)

---

## 2. Memory Requirements Analysis

### 2.1 Input Data Memory

#### Single Sample (Batch Size = 1)
- **FP32:** 1800 × 3600 × 200 × 4 bytes = **4.83 GiB** (5.18 GB)
- **FP16/BF16:** 1800 × 3600 × 200 × 2 bytes = **2.42 GiB** (2.59 GB)

#### Batched Processing
| Batch Size | FP32 (GiB) | FP16/BF16 (GiB) |
|------------|------------|-----------------|
| 1 | 4.83 | 2.42 |
| 2 | 9.66 | 4.83 |
| 4 | 19.31 | 9.66 |
| 8 | 38.62 | 19.31 |

**Recommendation:** For 1800×3600×200 input, batch size of 1-2 is practical for most consumer GPUs.

### 2.2 Spherical Harmonic Transform (SHT) Memory

The torch-harmonics library performs Spherical Harmonic Transforms as a core operation.

#### Precomputed Weights
For equiangular grid with nlat=1800, nlon=3600:
- **lmax** ≈ nlat = 1800
- **mmax** ≈ nlon/2 + 1 = 1801

**Legendre Polynomial Weights:**
- Shape: (mmax, lmax, nlat) = (1801, 1800, 1800)
- Size (FP32): 1801 × 1800 × 1800 × 4 bytes = 23,331,600,000 bytes = **21.73 GiB** (23.33 GB decimal)
- Size (FP64): 1801 × 1800 × 1800 × 8 bytes = 46,663,200,000 bytes = **43.46 GiB** (46.66 GB decimal)

**Note:** These weights are precomputed once and cached. They represent a significant one-time memory overhead.

#### Forward SHT Output
- Input: (batch, channels, nlat, nlon) = (1, 200, 1800, 3600)
- After FFT: (1, 200, 1800, 1801) complex values
- After Legendre: (1, 200, lmax, mmax) = (1, 200, 1800, 1801)
- **Size (Complex64):** 1 × 200 × 1800 × 1801 × 8 bytes = **4.83 GiB** (5.18 GB)

### 2.3 Model Memory Requirements

Typical model architectures and their approximate memory footprints:

#### U-Net (S2 U-Net with 4 layers)
- **Parameters:** 1-10M depending on embedding dimensions
- **Memory (FP32):** 
  - Weights: ~40-400 MB
  - Activations (per layer, batch=1): ~2-8 GB
  - Total: **8-15 GB** (with gradients in training)

#### Transformer (S2 Transformer)
- **Parameters:** 10-50M depending on configuration
- **Memory (FP32):**
  - Weights: 400 MB - 2 GB
  - Attention maps: Can be substantial for 1800×3600 resolution
  - Total: **15-30 GB** (with gradients in training)

#### Spherical Fourier Neural Operator (SFNO)
- **Parameters:** 5-20M
- **Memory (FP32):**
  - Weights: 200 MB - 800 MB
  - Spectral coefficients: Operates in harmonic space (smaller than spatial)
  - Total: **10-20 GB** (with gradients in training)

### 2.4 Training vs. Inference Memory

| Component | Inference | Training |
|-----------|-----------|----------|
| Input Data | 1× | 1× |
| Model Weights | 1× | 1× |
| Activations | 1× | 1× (per layer) |
| Gradients | 0 | 1× (all weights) |
| Optimizer State | 0 | 2× (Adam: momentum + variance) |
| **Total Multiplier** | **~1-2×** | **~4-5×** |

**Example (U-Net, FP32, Batch=1):**
- Inference: ~12-16 GB
- Training: ~30-48 GB

### 2.5 Total Memory Budget Breakdown

#### Inference (FP16, Batch Size = 1)
```
Input Data:              2.42 GiB
SHT Weights (cached):   21.73 GiB (FP32, one-time)
Model Forward Pass:      3-5 GiB
Intermediate Buffers:    2-4 GiB
─────────────────────────────────
Total Peak:             ~29-33 GiB (31-36 GB)
```

#### Training (FP16, Batch Size = 1)
```
Input Data:              2.42 GiB
SHT Weights (cached):   21.73 GiB (FP32, one-time)
Model Weights:           0.5-2 GiB
Activations:             3-8 GiB
Gradients:               0.5-2 GiB
Optimizer State:         1-4 GiB
Workspace Buffers:       2-4 GiB
─────────────────────────────────
Total Peak:             ~32-44 GiB (34-47 GB)
```

**Critical Note:** The SHT precomputed weights (~22 GiB / 23.3 GB) are a fixed overhead. With weight caching and efficient memory management, operational memory during forward/backward passes is more manageable.

---

## 3. Computational Complexity

### 3.1 Spherical Harmonic Transform (SHT)

The SHT is the core computational bottleneck for torch-harmonics operations.

#### Forward SHT Algorithm
1. **FFT in Longitude:** O(nlat × nlon × log(nlon))
2. **Legendre Transform:** O(channels × nlat × lmax × mmax)

**For 1800×3600×200:**

1. **FFT Component:**
   - Operations: 1800 × 3600 × log₂(3600) × 200 channels
   - ≈ 1800 × 3600 × 12 × 200 = **155.5 billion operations** (155.5 GFLOPS)

2. **Legendre Transform:**
   - Operations: 200 × 1800 × 1800 × 1801 (matrix multiplication)
   - ≈ 200 × 1800 × 1800 × 1801 = **1.17 trillion operations** (1.17 TFLOPS)

**Total SHT Forward Pass: ~1.3 TFLOPS per transform**

### 3.2 Model Architecture FLOPs

#### Convolutional Layers
For a standard 3×3 convolution:
- FLOPs per output pixel: 2 × C_in × kernel_h × kernel_w × C_out
- For full image: Above × nlat × nlon

**Example (3×3 conv, 200→128 channels):**
- 2 × 200 × 3 × 3 × 128 × 1800 × 3600 = **50 TFLOPS**

#### Attention Layers
Self-attention complexity: O(N² × d), where N = nlat × nlon

**For 1800×3600 (6.48M pixels):**
- N² = (6.48M)² = 4.2 × 10¹³
- With feature dimension d=128: **5.4 × 10¹⁵ operations = 5,400 TFLOPS**

**Critical:** Full self-attention is computationally prohibitive. Practical implementations use:
- Local/windowed attention
- Sparse attention patterns
- Hierarchical/multi-scale processing

### 3.3 Total Compute per Forward Pass

| Model Type | Estimated FLOPs (Single Forward Pass) |
|------------|--------------------------------------|
| U-Net (4 layers) | 30-80 TFLOPS |
| Transformer (local attention) | 50-150 TFLOPS |
| SFNO | 20-50 TFLOPS |

**Note:** SFNO operates primarily in spectral domain, which can be more efficient for global operations.

### 3.4 Training Iterations

Training requires:
- 1 forward pass: ~50 TFLOPS
- 1 backward pass: ~2× forward = ~100 TFLOPS
- **Total per training step: ~150 TFLOPS**

For typical training:
- Steps per epoch: dataset_size / batch_size
- Total epochs: 50-200
- **Total compute: 100s to 1000s of PFLOPS** (petaFLOPS)

---

## 4. GPU Architecture Considerations

### 4.1 Compute Capabilities

Modern GPU architectures provide different compute throughputs:

| GPU Model | FP32 TFLOPS | Tensor Core TFLOPS (FP16) | Memory | Bandwidth |
|-----------|-------------|---------------------------|---------|-----------|
| **NVIDIA A100 (40GB)** | 19.5 | 312 | 40 GB | 1.6 TB/s |
| **NVIDIA A100 (80GB)** | 19.5 | 312 | 80 GB | 2.0 TB/s |
| **NVIDIA H100** | 51 | 1000+ | 80 GB | 3.4 TB/s |
| **NVIDIA RTX 4090** | 82.6 | 330 (FP16) | 24 GB | 1.0 TB/s |
| **NVIDIA RTX 3090** | 35.6 | 142 | 24 GB | 936 GB/s |
| **NVIDIA V100 (32GB)** | 15.7 | 125 | 32 GB | 900 GB/s |
| **AMD MI250X** | 47.9 | 383 | 128 GB | 3.2 TB/s |

### 4.2 Performance Considerations

#### Memory Bandwidth
- **SHT operations** are memory-bandwidth bound
- Large tensor operations (1800×3600) require high bandwidth
- **Critical:** Choose GPU with high memory bandwidth (>1 TB/s)

#### Tensor Cores
- **Mixed precision training** (FP16/BF16) can leverage Tensor Cores
- Potential **4-8× speedup** for matrix multiplications
- torch-harmonics supports AMP (Automatic Mixed Precision)

#### FFT Performance
- **cuFFT library** (NVIDIA) provides optimized FFT implementations
- FFT performance scales well on modern GPUs
- Size 3600 is NOT a power-of-2, which can impact FFT efficiency
  - Power-of-2 sizes (e.g., 2048, 4096) are optimal
  - Non-power-of-2 can be 1.5-3× slower

**Recommendation:** If possible, consider padding to 4096 for longitude dimension for optimal FFT performance.

---

## 5. Hardware Recommendations

### 5.1 Minimum Configuration (Inference Only)

**Target:** Single-image inference at FP16 precision

- **GPU:** NVIDIA A100 (40GB) or RTX 6000 Ada (48GB)
- **System RAM:** 64 GB
- **Storage:** 500 GB NVMe SSD
- **Use Case:** Prediction/inference on pre-trained models

**Constraints:**
- Batch size limited to 1
- SHT weights consume ~22 GiB of fixed overhead
- Requires at least 32 GB GPU memory (40 GB recommended for comfort margin)

### 5.2 Recommended Configuration (Training)

**Target:** Full model training with reasonable batch sizes

- **GPU:** NVIDIA A100 (40GB or 80GB) or H100
- **System RAM:** 128 GB
- **Storage:** 1-2 TB NVMe SSD
- **Use Case:** Full training pipeline, experimentation

**Benefits:**
- Batch sizes 2-4 possible
- Comfortable memory margins
- Fast iteration times

### 5.3 Optimal Configuration (Production/Large Scale)

**Target:** Distributed training, multiple experiments, production inference

- **GPUs:** 4-8× NVIDIA A100 (80GB) or H100
- **System RAM:** 256-512 GB
- **Storage:** 4+ TB NVMe SSD (RAID for redundancy)
- **Networking:** 200+ Gbps InfiniBand for multi-GPU
- **Use Case:** Large-scale training, hyperparameter sweeps, production deployment

**Benefits:**
- Data parallelism across GPUs
- Larger effective batch sizes
- Faster training (linear scaling with GPUs)
- High availability for inference

### 5.4 Cloud Computing Options

#### AWS
- **p4d.24xlarge:** 8× A100 (40GB), ~$32/hour
- **p5.48xlarge:** 8× H100, ~$100/hour (when available)

#### Google Cloud Platform
- **a2-highgpu-8g:** 8× A100 (40GB), ~$30/hour
- **a2-ultragpu-8g:** 8× A100 (80GB), ~$45/hour

#### Azure
- **ND96amsr_A100_v4:** 8× A100 (80GB), ~$35/hour

**Cost Estimate (Training):**
- 100 epochs × 1000 steps/epoch × 5 sec/step = 139 hours
- Single A100: ~$4,000-6,000 per full training run
- **Recommendation:** Use spot/preemptible instances for 60-80% cost savings

---

## 6. Performance Optimization Strategies

### 6.1 Memory Optimization

#### 1. Mixed Precision Training (AMP)
```python
with torch.autocast(device_type="cuda", enabled=True):
    x = model(input)
    x = x.to(torch.float32)  # Convert before SHT
    with torch.autocast(device_type="cuda", enabled=False):
        xt = sht(x)  # SHT in FP32 for numerical stability
```
**Benefit:** 40-50% memory reduction, 2-3× speedup

#### 2. Gradient Checkpointing
```python
from torch.utils.checkpoint import checkpoint
x = checkpoint(model_layer, x)
```
**Benefit:** 30-50% memory reduction for activations
**Trade-off:** 20-30% slower (recomputes activations during backward)

#### 3. Tiling/Patching
- Split 1800×3600 image into smaller tiles (e.g., 512×512)
- Process tiles independently
- Reconstruct full prediction
**Benefit:** Enables processing on lower-memory GPUs
**Trade-off:** May lose global context for some models

#### 4. Channel-wise Processing
- Process spectral bands in chunks (e.g., 50 at a time instead of all 200)
- Aggregate results
**Benefit:** Linear reduction in memory (4× reduction for 50/200)

### 6.2 Compute Optimization

#### 1. Optimize FFT Dimensions
- Pad longitude to 4096 (power-of-2) for optimal FFT
```python
nlon_padded = 4096  # Next power of 2
x_padded = F.pad(x, (0, nlon_padded - 3600))
```
**Benefit:** 1.5-2× faster FFT operations

#### 2. Use Compiled Models (PyTorch 2.0+)
```python
model = torch.compile(model, mode="reduce-overhead")
```
**Benefit:** 10-30% speedup with no code changes

#### 3. Data Loading Optimization
- Use multiple workers: `DataLoader(..., num_workers=4)`
- Pin memory: `DataLoader(..., pin_memory=True)`
- Pre-load to GPU asynchronously
**Benefit:** Eliminate data loading bottleneck

#### 4. Distributed Training
```python
torchrun --nproc_per_node=4 train.py --enable_ddp
```
**Benefit:** Near-linear scaling with GPU count (4 GPUs = ~3.5× speedup)

---

## 7. Scaling Considerations

### 7.1 Horizontal Scaling (Multiple GPUs)

#### Data Parallel (DDP)
- Split batch across GPUs
- Each GPU processes batch_size/n_gpus samples
- Gradients synchronized after backward pass

**Scaling Efficiency:**
- 2 GPUs: ~1.8× speedup (90% efficiency)
- 4 GPUs: ~3.5× speedup (87% efficiency)
- 8 GPUs: ~6.5× speedup (81% efficiency)

**Communication Overhead:**
- Increases with model size and GPU count
- Mitigated by gradient accumulation and efficient backends (NCCL)

#### Model Parallel
- For models too large for single GPU
- Split model layers across GPUs
- More complex, but necessary for very large models

**Use Case:** When model weights + single sample > GPU memory

### 7.2 Vertical Scaling (Larger Raster)

| Resolution | Memory (FP16, GiB) | Relative Compute | Relative Time |
|------------|-------------------|------------------|---------------|
| 900×1800×200 | 0.60 | 1× | 1× |
| 1800×3600×200 | 2.42 | 4× | 4× |
| 3600×7200×200 | 9.66 | 16× | 16× |

**Key Insight:** Doubling spatial resolution → 4× memory and compute
- Quadratic scaling in spatial dimensions
- Linear scaling in spectral channels

### 7.3 Algorithmic Alternatives for Larger Scale

#### 1. Sparse/Hierarchical SHT
- Compute full resolution SHT only where needed
- Use multi-scale representations
- **Potential:** 2-5× speedup for regional applications

#### 2. Approximate Methods
- Truncate high-frequency components (reduce lmax)
- Use fast approximate SHT algorithms
- **Potential:** 2-10× speedup with controlled error

#### 3. Hybrid CPU-GPU Processing
- Offload SHT weight precomputation to CPU
- Keep only active tiles on GPU
- **Benefit:** Reduce GPU memory pressure

---

## Appendix: Calculation Details

### A.1 Memory Size Calculations

**Single precision (FP32):** 4 bytes per value
**Half precision (FP16/BF16):** 2 bytes per value
**Double precision (FP64):** 8 bytes per value

**Base calculation:**
```
memory_bytes = height × width × channels × precision_bytes
memory_GB = memory_bytes / (1024³)
```

**Example (1800×3600×200 in FP32):**
```
= 1800 × 3600 × 200 × 4 bytes
= 5,184,000,000 bytes
= 4.83 GiB (binary: 5,184,000,000 / 1024³)
= 5.184 GB (decimal: 5,184,000,000 / 10⁹)
```

### A.2 FLOP Calculations

**Convolution FLOPs:**
```
FLOPs = 2 × C_in × K_h × K_w × C_out × H_out × W_out
```

**Matrix Multiplication FLOPs:**
```
FLOPs = 2 × M × N × K
(for matrix shapes M×K and K×N)
```

**FFT FLOPs:**
```
FLOPs ≈ 5 × N × log₂(N)
(for N-point FFT)
```

### A.3 SHT Complexity Derivation

**Forward SHT:**
1. **FFT per latitude:** O(nlon × log(nlon))
2. **All latitudes:** O(nlat × nlon × log(nlon))
3. **Legendre transform:** O(nlat × lmax × mmax) per channel
4. **All channels:** O(channels × nlat × lmax × mmax)

**For large nlat, nlon:** Legendre transform dominates (O(n³) vs O(n² log n))

**With 1800×3600:**
- FFT: 1800 × 3600 × 12 = 77.76M ops per channel
- Legendre: 1800 × 1800 × 1801 = 5.83B ops per channel
- **Legendre is ~75× more expensive than FFT**

### A.4 Training Time Estimation

**Single training step:**
```
time_per_step = (FLOPs_forward + FLOPs_backward) / GPU_TFLOPS
               = 150 TFLOPS / 300 TFLOPS (A100 w/ Tensor Cores)
               ≈ 0.5 seconds (theoretical peak)
               ≈ 2-5 seconds (realistic, including overhead)
```

**Full training:**
```
total_steps = epochs × (dataset_size / batch_size)
            = 100 epochs × (1000 samples / 2)
            = 50,000 steps
            
total_time = 50,000 steps × 4 sec/step
           = 200,000 seconds
           ≈ 55 hours ≈ 2.3 days
```

---

## Summary and Quick Reference

### Memory Requirements

| Scenario | Minimum GPU Memory | Recommended |
|----------|-------------------|-------------|
| Inference (FP16, batch=1) | 32 GB | 40 GB |
| Training (FP16, batch=1) | 40 GB | 80 GB |
| Training (FP16, batch=2) | 80 GB | 2×40 GB |

**Note:** Minimum values include ~22 GiB SHT weight overhead

### Compute Requirements

- **Single Forward Pass:** ~30-100 TFLOPS
- **Training Step:** ~150 TFLOPS (forward + backward)
- **Training Time (100 epochs):** 50-200 hours on single A100

### Key Recommendations

1. ✅ **Use mixed precision (FP16/BF16)** - Essential for memory and speed
2. ✅ **Start with batch_size=1** - Safest for 1800×3600×200
3. ✅ **Consider tiling** - If GPU memory < 40 GB
4. ✅ **Use A100 or better** - For comfortable training experience
5. ✅ **Optimize FFT dimensions** - Pad to power-of-2 for best performance
6. ✅ **Multi-GPU for production** - Necessary for efficient large-scale training

### Critical Limitations

- ⚠️ **SHT weights (23 GB)** are substantial fixed overhead
- ⚠️ **Non-power-of-2 dimensions (3600)** reduce FFT efficiency
- ⚠️ **Full attention** is computationally prohibitive at this resolution
- ⚠️ **Memory bandwidth** often more limiting than compute

---

## Additional Resources

### torch-harmonics Documentation
- GitHub: https://github.com/RichardScottOZ/torch-harmonics
- Original NVIDIA Repository: https://github.com/NVIDIA/torch-harmonics
- Examples: `/examples` directory in repository
- Notebooks: `/notebooks` directory for tutorials

### Research Papers
1. Bonev et al. "Spherical Fourier Neural Operators" (ICML 2023)
2. Schaeffer "Efficient spherical harmonic transforms" (G³ 2013)
3. Liu-Schiaffini et al. "Neural Operators with Localized Kernels" (ICML 2024)

### Performance Profiling Tools
- **NVIDIA Nsight Systems:** System-wide profiling
- **PyTorch Profiler:** Python-level profiling
- **nvidia-smi:** GPU utilization monitoring

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Profile PyTorch code
python -m torch.utils.bottleneck your_script.py

# Detailed profiling
nsys profile python train.py
```

---

**End of Document**

For questions or updates to this analysis, please open an issue in the torch-harmonics repository.
