# TMA vs cudaMemcpy — Comparative Benchmark Suite

A systematic comparison of NVIDIA's **Tensor Memory Accelerator (TMA)** against
`cudaMemcpy`/`cudaMemcpy2D`/`cudaMemcpyAsync` for data movement on H100 (Hopper).

> **Research framing:** Although both mechanisms move data, they operate at
> distinct tiers of the CUDA memory hierarchy.  `cudaMemcpy` drives the
> **host ↔ device transport tier** (PCIe/NVLink DMA), while TMA is an
> **on-device shared-memory staging primitive** (global memory → shared memory
> inside a running kernel).  This study is therefore best understood as a
> *hierarchical data-movement comparison*—measuring PCIe DMA, device global
> memory movement, and TMA-based shared-memory staging as complementary layers,
> not as competing replacements.  Readers should interpret the benchmarks
> through this lens rather than as a head-to-head equivalence test.

---

## What is TMA?

The **Tensor Memory Accelerator** is a hardware unit introduced in NVIDIA's Hopper
architecture (H100, sm\_90).  It performs bulk, asynchronous copies between
**global memory (HBM3)** and **shared memory** inside a kernel, controlled by a
*TMA descriptor* (`CUtensorMap`).  Key properties:

| Property | TMA | cudaMemcpy |
|---|---|---|
| Who initiates | A single thread inside a kernel | Host CPU |
| Direction | Global → Shared (or Shared → Global) | Host ↔ Device, Device ↔ Device |
| Addressing | Tensor descriptor (rank 1–5, automatic stride) | Flat byte pointer |
| Synchronisation | `mbarrier` (hardware-tracked) | Event / stream |
| Minimum granularity | 16 bytes, inner dim ≤ 256 bytes | 1 byte |
| Requires | sm\_90+ (H100), CUDA ≥ 12.0 | Any CUDA device |

TMA shines for **tiled workloads** (GEMM, attention, convolution) where every
iteration loads the same fixed-shape tile from a strided tensor.

---

## Repository Layout

```
.
├── benchmarks/
│   ├── bench_1d_copy.cu      # 1-D H2D bandwidth: pageable / pinned / TMA
│   ├── bench_2d_stride.cu    # 2-D strided copy: cudaMemcpy2D vs TMA 2D
│   ├── bench_overlap.cu      # Compute-transfer overlap: seq / async / TMA pipeline
│   ├── bench_gemm.cu         # GEMM-style tile load: global loads vs TMA
│   └── bench_small_xfer.cu   # Small-transfer latency: pageable / pinned / async / TMA
├── include/
│   └── tma_utils.cuh         # Shared: error macros, GpuTimer, TMA descriptor builders
├── results/                  # Auto-created; benchmark CSVs written here
├── run.sh                    # Build + run a single benchmark
├── run_all.sh                # Build + run all benchmarks, aggregate summary CSV
└── README.md
```

---

## Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | Any CUDA-capable | **H100 (sm\_90a)** for TMA paths |
| CUDA Toolkit | 12.0 | 12.4+ |
| nvcc | 12.0 | 12.4+ |
| OS | Linux | Ubuntu 22.04 |
| Host RAM | ≥ 8 GB | 32 GB |
| VRAM | ≥ 8 GB | 80 GB (H100 SXM) |

> **On non-H100 GPUs**: all benchmarks compile and run.  At runtime the program
> detects `sm < 90` and prints a warning; TMA columns are skipped (reported as
> `-1`) while cudaMemcpy columns are measured normally.

---

## Building & Running

### Option 1 — Run everything at once

```bash
./run_all.sh                  # results written to results/
./run_all.sh /path/to/output  # custom output directory
```

This script:
1. Detects whether your `nvcc` supports `sm_90a`; falls back to `sm_80` otherwise.
2. Compiles all five benchmarks.
3. Runs them sequentially.
4. Aggregates all CSVs into `results/summary.csv`.

### Option 2 — Run one benchmark at a time

```bash
./run.sh 1d       # bench_1d_copy
./run.sh 2d       # bench_2d_stride
./run.sh overlap  # bench_overlap
./run.sh gemm     # bench_gemm
./run.sh small    # bench_small_xfer
./run.sh all      # all five
```

### Option 3 — Manual compile

```bash
nvcc -arch=sm_90a -std=c++17 -O3 -lcuda \
     benchmarks/bench_1d_copy.cu -o bench_1d -Iinclude
./bench_1d results/my_run.csv
```

Replace `sm_90a` with `sm_80` (or your device's arch) if not on H100.

---

## Benchmark Descriptions

### `bench_1d_copy` — 1-D Contiguous Copy

Transfers sizes: 1 MB → 2 GB.

| Path | Direction | Notes |
|---|---|---|
| Pageable | Host (pageable) → Device | Baseline PCIe bandwidth |
| Pinned | Host (pinned) → Device | Avoids double-copy through pinned staging buffer |
| **TMA** | Global → Shared | On-device; data already on HBM3; measures HBM3→SMEM path |

**Key insight**: cudaMemcpy bandwidth is bound by PCIe (≈ 32–64 GB/s on H100).
TMA measures the on-device HBM3→Shared path (aggregate ≈ 3.35 TB/s on H100,
per-SM ≈ 25–50 GB/s).

---

### `bench_2d_stride` — 2-D Strided Copy

Matrix sizes: 512×512 → 4096×4096; row pitch = 2× width (50 % padding).

| Path | Direction | Notes |
|---|---|---|
| cudaMemcpy2D | Host (pinned) → Device | Hardware-accelerated strided H2D |
| **TMA 2D** | Global → Shared | TMA descriptor handles row stride automatically |

Reports payload bandwidth (useful bytes only) and allocated bandwidth
(including padding) to highlight efficiency differences.

---

### `bench_overlap` — Compute–Transfer Overlap

256 MB total, 8 tiles of 32 MB each; 100 compute iterations per element.

| Path | Overlap strategy |
|---|---|
| Sequential | `cudaMemcpy` then `computeKernel`, serialised |
| Async overlap | `cudaMemcpyAsync` + `computeKernel` on ping-pong streams |
| **TMA pipeline** | Data pre-staged on device; TMA double-buffer load + compute |

Speedup column is relative to the sequential baseline.

---

### `bench_gemm` — GEMM-Style Tile Loading

Square matrices N×N (N = 512, 1024, 2048).  Tile BM=BN=BK=32.

| Path | Tile load method |
|---|---|
| Global loads | Each thread reads directly from global memory (no shared staging) |
| **TMA tile loads** | Thread 0 per block issues TMA 2-D copy per A-tile and B-tile |

Reports GFLOPS (2N³ operations) and effective memory bandwidth.

---

### `bench_small_xfer` — Small-Transfer Latency

Sizes: 1 KB → 4 MB.  50 runs for stable average.

| Path | Notes |
|---|---|
| Pageable | Highlights copy-engine launch overhead |
| Pinned | Avoids OS page-lock delay |
| Async | `cudaMemcpyAsync` + stream sync |
| **TMA** | Per-kernel TMA call; shows CUDA kernel launch latency floor |

Latency shown in **µs**; bandwidth in GB/s.

---

## Output Format

Every benchmark writes a CSV to `results/` with at minimum these columns:

| Column | Description |
|---|---|
| `benchmark` | Benchmark name (`1d_copy`, `2d_stride`, `overlap`, `gemm`, `small_xfer`) |
| `method` | Transfer path (`cudaMemcpy_pageable`, `cudaMemcpy_pinned`, `TMA_global_shared`, …) |
| `avg_time_ms` | Average time over measurement runs (milliseconds) |
| `bw_gbs` | Bandwidth in GB/s (bytes / time) |

Additional benchmark-specific columns (matrix size, GFLOPS, speedup, etc.)
are described in the header row of each CSV.

`results/summary.csv` concatenates all five CSVs into one file for easy
analysis in Python/R/Excel.

---

## Understanding the Results

```
cudaMemcpy (PCIe)      :   30 – 64  GB/s   (PCIe 4/5 peak)
TMA global→shared      :   20 – 50  GB/s   (per SM, serial tiles)
  — with 132 SMs       : up to ~3 TB/s     (aggregate, all SMs in parallel)
```

TMA is **not** a replacement for `cudaMemcpy`.  It operates on a fundamentally
different path (on-device, within a kernel).  The comparison highlights:

* TMA has **zero PCIe overhead** for workloads whose data is already on-device.
* TMA handles **tensor strides** automatically in hardware.
* TMA enables **asynchronous staging** (overlap with compute) without explicit
  stream management.
* For **pure H2D transfers** (data not yet on device), `cudaMemcpy` with pinned
  memory remains the baseline.

---

## Build Flags Reference

| Flag | Purpose |
|---|---|
| `-arch=sm_90a` | Enable Hopper-specific instructions (TMA, WGMMA). Required for TMA. |
| `-std=c++17` | Required for libcu++ `<cuda/barrier>` headers. |
| `-O3` | Full optimisation; suppresses dead-code-elimination of TMA sinks. |
| `-lcuda` | Link against the CUDA Driver library (for `cuTensorMapEncodeTiled`). |
| `-Iinclude` | Include path for `tma_utils.cuh`. |

---

## Scope & Limitations

* **Different memory-hierarchy tiers.** `cudaMemcpy` and TMA operate at
  fundamentally different levels of the CUDA memory hierarchy.  `cudaMemcpy`
  crosses the host↔device boundary via PCIe/NVLink (I/O path), whereas TMA
  moves data *within the device*—from HBM3 global memory into per-SM shared
  memory inside a running kernel.  Direct bandwidth or latency comparisons
  between the two must be interpreted in context: they measure different
  bottlenecks and are not interchangeable primitives.  The benchmarks in this
  suite are designed to make this distinction visible, not to declare a winner.

* **Hardware specificity.** TMA benchmarks require an H100 (sm\_90a) GPU.
  On older hardware the TMA paths are disabled at runtime and reported as `-1`.
  `cudaMemcpy` results are valid on any CUDA-capable device.

* **Single-node, single-GPU.** Multi-GPU or NVLink peer-to-peer transfers are
  outside the current scope.

* **Microbenchmark limitations.** Results reflect isolated transfer operations.
  Real application performance depends on overlapping transfers with compute,
  memory-access patterns, and occupancy—factors that are studied in the
  `bench_overlap` and `bench_gemm` benchmarks but not fully generalised here.
