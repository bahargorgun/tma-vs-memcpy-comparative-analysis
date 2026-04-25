# TMA vs Classical CUDA Memory Transfer Methods on NVIDIA H100

## Comparison of Tensor Memory Accelerator (TMA) Performance with cp.async on H100 Hopper Architecture

**Platform:** NVIDIA H100 80GB HBM3 · sm_90a · CUDA 12.3
**Ana Karşılaştırma:** cp.async (Ampere, sm_80+) vs TMA (Hopper, sm_90+)
**Referans:** cudaMemcpy pageable/pinned (PCIe yolu — farklı donanım, doğrudan karşılaştırılamaz)

---

## Projenin Amacı

Bu çalışma, NVIDIA H100 GPU üzerinde HBM3→Shared Memory veri taşıma yöntemlerini karşılaştırır. İki yöntem aynı donanım yolunu kullanır:

- **cp.async** — Thread'ler koordine eder, DMA benzeri asenkron kopyalama (Ampere'den beri mevcut)
- **TMA (Tensor Memory Accelerator)** — Hopper'a özel donanım birimi, tile'ları otomatik taşır

Bu karşılaştırma doğrudandır çünkü ikisi de aynı HBM3→SMEM yolunu kullanır. cudaMemcpy ise Host RAM→GPU (PCIe) yolunu kullandığı için yalnızca referans olarak gösterilir.

### Neden Önemli

LLM inference (GPT-4, LLaMA, DeepSeek) sırasında her token üretimi milyonlarca tile yüklemesi gerektirir. Attention mekanizmasında Q×K^T ve ×V çarpımları tiled GEMM'dir. FFN katmanları da aynı şekilde tile tabanlıdır. Model ağırlıkları HBM3'te durur ve her adımda shared memory'e taşınması gerekir. Bu taşıma hızı doğrudan token üretim hızını belirler.

FlashAttention-3 (Shah et al., arXiv:2407.08608) ve DeepGEMM (DeepSeek, 2025) gibi production kütüphaneler tam bu mekanizmayı — TMA tile fetch — kullanır. Bu çalışma, o kütüphanelerin altındaki temel katmanı izole ederek ölçer.

---

## Benchmark Seti

### 1. bench_1d_copy.cu — 1-D Contiguous Transfer

**Ne ölçüyor:** Ardışık (contiguous) float32 array'in HBM3→SMEM aktarım hızı.
**Tile:** 64 eleman (256 byte)
**Boyutlar:** 1 MB → 2 GB

| Boyut | cp.async BW | TMA BW | TMA/CPA |
|-------|------------|--------|---------|
| 1 MB | 66.6 GB/s | 62.8 GB/s | 0.94 |
| 16 MB | 150.5 GB/s | 138.5 GB/s | 0.92 |
| 256 MB | 78.9 GB/s | 75.2 GB/s | 0.95 |
| 2 GB | 79.1 GB/s | 75.4 GB/s | 0.95 |

**Bulgu:** cp.async %5-8 daha hızlı. 1D contiguous veride adres hesaplaması trivial — her thread ardışık adresi okuyor. TMA'nın descriptor ve mbarrier overhead'i boşa gidiyor. **Basit sıralı erişimde TMA gereksiz, cp.async yeterli.**

### 2. bench_2d_stride.cu — 2-D Strided Transfer ⭐

**Ne ölçüyor:** Stride'lı (padding'li) 2D matrisin HBM3→SMEM aktarımı. 2× row padding ile.
**Tile:** 32×32 eleman
**Boyutlar:** 512×512 → 4096×4096

| Matris | cp.async BW | TMA BW | TMA/CPA |
|--------|------------|--------|---------|
| 512² | 121 GB/s | 115 GB/s | 0.95 |
| 1024² | 372 GB/s | 471 GB/s | 1.27 |
| 2048² | 800 GB/s | 1558 GB/s | 1.95 |
| 4096×2048 | 794 GB/s | 2449 GB/s | **3.08** |
| 4096² | 901 GB/s | 2254 GB/s | **2.50** |

**Bulgu — çalışmanın en güçlü sonucu:** TMA, stride'lı veride 3×'e kadar hızlı. Sebebi: cp.async her thread için `row * pitch + col` adresi hesaplıyor. TMA ise stride bilgisini descriptor'a gömmüş, donanım otomatik hallediyor. LLM'lerdeki KV-cache tam bu yapıda — stride'lı 2D matris.

### 3. bench_gemm.cu — GEMM Tile Loading

**Ne ölçüyor:** Aynı matris çarpımı (C = A×B), üç farklı tile yükleme yöntemi.
**Tile:** 32×32
**Boyutlar:** N = 512, 1024, 2048, 4096

| N | Global GFLOPS | cp.async GFLOPS | TMA GFLOPS | TMA/CPA | TMA/Global |
|---|--------------|----------------|-----------|---------|-----------|
| 512 | 5,424 | 6,657 | 6,913 | 1.04 | 1.27 |
| 1024 | 6,423 | 8,238 | 9,140 | 1.11 | 1.42 |
| 2048 | 6,557 | 8,522 | 9,790 | 1.15 | **1.49** |
| 4096 | 6,483 | 8,552 | 8,867 | 1.04 | 1.37 |

**Bulgu:** TMA, global load'a göre 1.27-1.49× hızlı. cp.async'e göre %4-15 kazanç. N=2048'de en büyük fark — memory-bound rejimde TMA avantajlı. N=4096'da fark kapanıyor çünkü compute-bound hale geliyor, tile fetch artık bottleneck değil. **LLM inference genelde memory-bound olduğu için TMA burada değerli.**

### 4. bench_overlap.cu — Compute-Transfer Pipeline

**Ne ölçüyor:** 256 MB veri üzerinde tile yükleme + hesaplama pipeline'ı.
**Konfigürasyon:** 8 tile × 32 MB, 100 compute iterasyonu

| Yöntem | Süre | BW |
|--------|------|-----|
| cp.async pipeline | 0.352 ms | 762 GB/s |
| TMA pipeline | 0.387 ms | 693 GB/s |
| Sequential cudaMemcpy (referans) | 5.208 ms | 52 GB/s |
| Async cudaMemcpy (referans) | 4.874 ms | 55 GB/s |

**Bulgu:** cp.async %10 daha hızlı. Sebebi: her tile iterasyonunda TMA descriptor oluşturuluyor (host tarafında), bu overhead pipeline'ı yavaşlatıyor. Production'da (FA3 gibi) descriptor'lar bir kere oluşturulur — bu sorun orada yok. **Tight loop'larda descriptor overhead'ine dikkat.**

### 5. bench_small_xfer.cu — Small Transfer Latency

**Ne ölçüyor:** 1 KB → 4 MB arası küçük transferlerde latency.

| Boyut | cp.async lat | TMA lat | TMA/CPA |
|-------|-------------|---------|---------|
| 1 KB | 7.70 µs | 7.94 µs | 0.97 |
| 64 KB | 8.04 µs | 8.38 µs | 0.96 |
| 1 MB | 14.02 µs | 14.84 µs | 0.94 |
| 4 MB | 33.16 µs | 35.71 µs | 0.93 |

**Referans:** Pinned cudaMemcpy 1 KB'de 9.8 µs, 4 MB'de 84.8 µs — on-chip yollar PCIe'den 2-4× düşük latency.

**Bulgu:** cp.async %3-7 daha hızlı. Küçük boyutlarda kernel launch overhead (~8 µs) baskın, mbarrier setup TMA'yı hafif yavaşlatıyor.

---

## Özet Tablo

| Benchmark | Kazanan | Fark | Sebep |
|-----------|---------|------|-------|
| 1-D Contiguous | cp.async | ~1.05× | TMA descriptor overhead, basit erişim |
| **2-D Strided** | **TMA** | **3.1×** | **Stride donanımda, en büyük avantaj** |
| **GEMM** | **TMA** | **1.49× vs global** | **Thread'ler hesaplamaya odaklanıyor** |
| Pipeline | cp.async | ~1.10× | Descriptor oluşturma overhead'i |
| Small Latency | cp.async | ~1.05× | mbarrier setup maliyeti |

## Ana Mesaj

**TMA her yerde kazanmıyor — ama kazandığı yerler LLM'ler için en kritik yerler.**

1. **KV-cache erişimi** → stride'lı 2D veri → TMA 3× hızlı
2. **Attention GEMM** → memory-bound matris çarpımı → TMA 1.5× hızlı
3. **Basit sıralı erişim** → cp.async yeterli, TMA gereksiz
4. **Küçük/sık transfer** → cp.async daha iyi, TMA overhead'i baskın

Bu sonuçlar FlashAttention-3 ve DeepGEMM'in neden TMA kullandığını doğruluyor: bu kütüphaneler tam da stride'lı KV-cache ve memory-bound GEMM senaryolarında çalışıyor.

---

## Sunumda Değinilecek Noktalar

### 1. Motivasyon (Neden önemli)
- LLM inference memory-bandwidth bound
- Her token üretimi = milyonlarca tile yüklemesi
- Tile fetch hızı = token üretim hızı

### 2. Yöntem (Ne yaptık)
- cp.async vs TMA — aynı HBM3→SMEM yolu, adil karşılaştırma
- cudaMemcpy ile karşılaştırma yapılmaz (farklı donanım yolu)
- 5 farklı senaryo: 1D, 2D strided, GEMM, pipeline, latency

### 3. Tile Nedir
- GPU shared memory küçük ama hızlı (228 KB/SM)
- Büyük matris küçük parçalara (tile) bölünür
- Her tile HBM3→SMEM'e taşınır, orada işlenir
- cp.async: thread adres hesaplar + DMA
- TMA: donanım descriptor'dan otomatik taşır

### 4. Sonuçlar
- 2D strided: TMA 3× hızlı (en güçlü bulgu)
- GEMM: TMA 1.5× hızlı (memory-bound'da)
- 1D/küçük: cp.async daha iyi (overhead)

### 5. LLM Bağlantısı
- Attention: Q×K^T ve ×V = tiled GEMM (bench_gemm ile ölçtük)
- KV-cache: stride'lı 2D (bench_2d ile ölçtük)
- FlashAttention-3: TMA + warp specialization + WGMMA
- DeepGEMM: TMA + FP8 GEMM
- Bu çalışma o pipeline'ın temel katmanını izole ederek ölçtü

### 6. Ecosystem Pozisyonu
```
LLM Applications     (GPT-4, LLaMA, DeepSeek)
       ↑
Inference Frameworks  (vLLM, TensorRT-LLM)
       ↑
Kernel Libraries      (FlashAttention-3, DeepGEMM)
       ↑
Template Infra        (CUTLASS 3.x — TMA + WGMMA + warp spec.)
       ↑
★ Bu Çalışma ★       (TMA tile-fetch mekanizması izole ölçüm)
       ↑
Hardware              (H100 — TMA Unit, WGMMA, HBM3)
```

### 7. Hoca Sorusu: "Neden TMA her yerde kazanmıyor?"
- TMA'nın güçlü olduğu yer: karmaşık adres hesaplaması gereken stride'lı erişim
- Basit sıralı erişimde thread'ler zaten adresi trivial hesaplıyor
- TMA'nın descriptor + mbarrier setup maliyeti küçük transferlerde baskın
- Production'da (FA3) TMA, warp specialization ile birleştirilince tam potansiyeline ulaşıyor

---

## Build & Run

```bash
tar -xzf final-benchmarks.tar.gz
cd final
chmod +x run_all.sh
./run_all.sh
```

### Gereksinimler
- CUDA >= 12.0
- NVIDIA H100 (sm_90a) — TMA için zorunlu
- cp.async: sm_80+ (A100/H100)

### Proje Yapısı
```
final/
├── include/
│   └── tma_utils.cuh          # Error check, GpuTimer, TMA descriptor builders
├── benchmarks/
│   ├── bench_1d_copy.cu        # 1D contiguous: cp.async vs TMA
│   ├── bench_2d_stride.cu      # 2D strided: cp.async vs TMA
│   ├── bench_gemm.cu           # GEMM: global vs cp.async vs TMA
│   ├── bench_overlap.cu        # Pipeline: cp.async vs TMA
│   └── bench_small_xfer.cu     # Small latency: cp.async vs TMA
├── results/                    # CSV çıktıları
└── run_all.sh                  # Tek komutla build + run
```

---

## Referanslar

### TMA & H100 Architecture
- NVIDIA Hopper Architecture Whitepaper (2022) — nvidia.com/en-us/data-center/hopper-architecture
- NVIDIA PTX ISA 8.x — cp.async.bulk.tensor documentation
- CUTLASS 3.x — Production TMA GEMM library — github.com/NVIDIA/cutlass

### LLM Kernels Using TMA
- Shah et al. (2024) — FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision — arXiv:2407.08608
- DeepSeek (2025) — DeepGEMM: Clean and Efficient FP8 GEMM — github.com/deepseek-ai/DeepGEMM
- Dao et al. (2022) — FlashAttention: Fast and Memory-Efficient Exact Attention — arXiv:2205.14135

### LLM Memory Bandwidth
- Kwon et al. (2023) — Efficient Memory Management for Large Language Model Serving — arXiv:2309.06180
- Williams et al. (2009) — Roofline Model — Communications of the ACM
