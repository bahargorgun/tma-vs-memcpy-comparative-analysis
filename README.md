# TMA vs cp.async — Comparative Benchmark Suite

A systematic comparison of NVIDIA's **Tensor Memory Accelerator (TMA)** against
**cp.async** for **HBM3 → Shared Memory** staging on H100 (Hopper). PCIe
`cudaMemcpy*` paths are shown **only as reference** and are **not** directly
comparable to on‑device staging.

> **Research framing:** `cp.async` and TMA both operate **inside kernels** and
> move data from **global memory (HBM3)** into **shared memory**.  This suite
> focuses on that *on‑device* path.  Host↔device transfers (`cudaMemcpy`) are
> included only to provide context for PCIe/NVLink latency and bandwidth, not as
> a head‑to‑head replacement.

---

## What is TMA?

The **Tensor Memory Accelerator** is a hardware unit introduced in NVIDIA's
Hopper architecture (H100, sm_90). It performs bulk, asynchronous copies between
**global memory (HBM3)** and **shared memory** inside a kernel, controlled by a
*TMA descriptor* (`CUtensorMap`). Key properties:

| Property | TMA | cp.async |
|---|---|---|
| Who initiates | A single thread inside a kernel | Each participating thread |
| Direction | Global ↔ Shared | Global → Shared |
| Addressing | Tensor descriptor (rank 1–5, automatic stride) | Flat byte pointer |
| Synchronisation | `mbarrier` (hardware‑tracked) | `cp.async.wait*` |
| Minimum granularity | 256 B (64 float32) | 16 B |
| Requires | sm_90+ (H100), CUDA ≥ 12.0 | sm_80+ (Ampere) |

TMA shines for **tiled workloads** (GEMM, attention, convolution) where every
iteration loads the same fixed‑shape tile from a strided tensor.

---

## Repository Layout

```
.
├── benchmarks/
│   ├── bench_1d_copy.cu      # 1-D HBM3→SMEM: cp.async vs TMA (+ PCIe refs)
│   ├── bench_2d_stride.cu    # 2-D strided HBM3→SMEM: cp.async vs TMA (+ PCIe ref)
│   ├── bench_overlap.cu      # On-device pipeline: cp.async vs TMA (+ PCIe refs)
│   ├── bench_gemm.cu         # GEMM tile loads: global loads vs cp.async vs TMA
│   └── bench_small_xfer.cu   # Small-latency: cp.async vs TMA (+ PCIe refs)
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
| GPU | sm_80+ for cp.async | **H100 (sm_90a)** for TMA paths |
| CUDA Toolkit | 12.0 | 12.4+ |
| nvcc | 12.0 | 12.4+ |
| OS | Linux | Ubuntu 22.04 |
| Host RAM | ≥ 8 GB | 32 GB |
| VRAM | ≥ 8 GB | 80 GB (H100 SXM) |

> **On non‑H100 GPUs**: benchmarks compile and run. At runtime the program
> detects `sm < 90` and prints a warning; TMA columns are skipped (reported as
> `-1`) while cp.async / cudaMemcpy columns are measured normally.

---

## Building & Running

### Option 1 — Run everything at once

```bash
./run_all.sh                  # results written to results/
./run_all.sh /path/to/output  # custom output directory
```

This script:
1. Detects whether your `nvcc` supports `sm_90a`; falls back to `sm_80` otherwise.
2. Compiles all benchmarks.
3. Runs them sequentially.
4. Aggregates all CSVs into `results/summary.csv`.

### Option 2 — Run one benchmark at a time

```bash
./run.sh 1d       # bench_1d_copy
./run.sh 2d       # bench_2d_stride
./run.sh overlap  # bench_overlap
./run.sh gemm     # bench_gemm
./run.sh small    # bench_small_xfer
./run.sh all      # all
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

### `bench_1d_copy` — 1‑D Contiguous Tile Staging

Transfers sizes: 1 MB → 2 GB.

| Path | Direction | Notes |
|---|---|---|
| **cp.async** | Global → Shared | Thread‑driven tile load (HBM3→SMEM) |
| **TMA** | Global → Shared | Hardware‑driven tile load (HBM3→SMEM) |
| PCIe refs | Host → Device | Pageable / pinned `cudaMemcpy` (context only) |

**Key insight:** cp.async and TMA use the same on‑device path; PCIe is a separate
transport layer and shown only for context.

---

### `bench_2d_stride` — 2‑D Strided Tile Staging

Matrix sizes: 512×512 → 4096×4096; row pitch = 2× width (50% padding).

| Path | Direction | Notes |
|---|---|---|
| **cp.async** | Global → Shared | Thread‑driven strided tile load |
| **TMA 2D** | Global → Shared | Descriptor handles row stride automatically |
| PCIe ref | Host → Device | `cudaMemcpy2D` (pinned) |

Reports payload bandwidth (useful bytes) to highlight stride efficiency.

---

### `bench_overlap` — Compute + Transfer Pipeline (On‑device)

256 MB total, 8 tiles of 32 MB each; 100 compute iterations per element.

| Path | Overlap strategy |
|---|---|
| **cp.async pipeline** | Per‑block load→wait→compute; overlap occurs across blocks |
| **TMA pipeline** | Same pattern using TMA + mbarrier |
| PCIe refs | Sequential `cudaMemcpy` vs async ping‑pong (context only) |

> Note: the pipeline is sequential **within a block**; overlap is achieved across
> blocks at the SM level.

---

### `bench_gemm` — GEMM‑Style Tile Loading

Square matrices N×N (N = 512, 1024, 2048). Tile BM=BN=BK=32.

| Path | Tile load method |
|---|---|
| Global loads | Each thread reads directly from global memory |
| **cp.async** | Thread‑driven shared‑memory staging |
| **TMA** | TMA 2‑D tile staging |

Reports GFLOPS (2N³ operations) and effective memory bandwidth.

---

### `bench_small_xfer` — Small‑Transfer Latency (On‑device)

Sizes: 1 KB → 4 MB. 50 runs for stable average.

| Path | Notes |
|---|---|
| **cp.async** | Per‑kernel tile staging latency |
| **TMA** | Per‑kernel TMA latency (min 256 B granularity) |
| PCIe refs | pageable / pinned / async `cudaMemcpy*` |

Latency shown in **µs**; bandwidth in GB/s.

---

## Output Format

Every benchmark writes a CSV to `results/` with at minimum these columns:

| Column | Description |
|---|---|
| `benchmark` | Benchmark name (`1d_copy`, `2d_stride`, `overlap`, `gemm`, `small_xfer`) |
| `method` | Transfer path (`cp.async`, `TMA_global_shared`, `cudaMemcpy_*`, …) |
| `avg_time_ms` | Average time over measurement runs (milliseconds) |
| `bw_gbs` | Bandwidth in GB/s (bytes / time) |

Additional benchmark‑specific columns (matrix size, speedup, etc.) are described
in each CSV header row. `results/summary.csv` concatenates all CSVs into one file
for easy analysis.

---

## Understanding the Results

```
PCIe cudaMemcpy (H2D) :   30 – 64  GB/s   (PCIe 4/5 peak)
HBM3→SMEM (per‑SM)    :   20 – 50  GB/s   (serial tiles)
Aggregate (all SMs)  : up to ~3 TB/s     (H100 SXM)
```

cp.async and TMA are **on‑device staging** mechanisms; `cudaMemcpy` is **host↔device I/O**.
Interpret comparisons only within the same tier.

---

## Build Flags Reference

| Flag | Purpose |
|---|---|
| `-arch=sm_90a` | Enable Hopper‑specific instructions (TMA, WGMMA). Required for TMA. |
| `-std=c++17` | Required for libcu++ `<cuda/barrier>` headers. |
| `-O3` | Full optimisation; suppresses dead‑code‑elimination of TMA sinks. |
| `-lcuda` | Link against the CUDA Driver library (for `cuTensorMapEncodeTiled`). |
| `-Iinclude` | Include path for `tma_utils.cuh`. |

---

## Scope & Limitations

* **Different hierarchy tiers.** `cudaMemcpy` (PCIe/NVLink) and cp.async/TMA
  (HBM3→SMEM) operate at different layers. This suite focuses on **on‑device
  staging**, while PCIe paths are shown only for context.
* **Hardware specificity.** TMA requires H100 (sm_90a). On older hardware TMA
  paths are disabled and reported as `-1`.
* **Microbenchmark limitations.** Results reflect isolated transfer operations.
  Real performance depends on overlap, occupancy, and access patterns.