# GLM-5.2 heterogeneous-GPU weighted placement experiment (2026-07-13)

Status: **failed / non-win**. The weighting mechanism reduced measured cache
transfer time, but its 1.45% median end-to-end gain was smaller than run-to-run
spread and one of three pairs regressed. The runtime implementation is retained
on the failed experiment branch and is not merged into `main`.

## Hypothesis and acceptance contract

Hypothesis: assigning startup experts, streamed work, and cache slots by measured
low-row expert service rate will improve decode tok/s because equal-byte placement
sends too much cache traffic to the slower PCIe links on this heterogeneous box.

- Baseline: `120ad5c` (`main`).
- Candidate code: `a316d0b` (opt-in `COLI_CUDA_DEVICE_WEIGHTS`; unset preserves
  the baseline placement policy).
- Primary metric: median decode-only tok/s over three interleaved fixed-token
  replay repetitions, candidate versus control from the same binary.
- Workload: `c/bench/glm52_replay_greedy_192.json`, 192 decode tokens, GLM-5.2
  int4 experts and int4 dense weights, `DRAFT=0`, `TEMP=0`, `TOPP=0.7`.
- Equal memory budget: `CUDA_EXPERT_GB=58`, including an 8 GB cache;
  `PIN_GB=180`, `RAM_GB=200`. Further weight quantization is out of scope.
- Correctness: `make check`, the CUDA backend on CUDA ordinals 0/1/2, cache
  ownership/accounting, and a real-model replay without fallback or errors.
- Rollback: any correctness/accounting failure, OOM/fallback, RAM above 200 GB,
  expert VRAM above 58 GB, or a median gain no larger than run spread.

The previous equal-budget matrix at `120ad5c` measured static
`0.58/0.59/0.59` (median 0.59), direct streaming `0.61/0.73/0.73` (median
0.73), and preallocated cache `0.89/0.85/0.88` (median 0.88 tok/s). The new
matrix repeats the cache control rather than treating that historical median as
the A/B baseline.

## Machine and toolchain

- CPU/RAM: 2x Xeon E5-2680 v4, 28 cores / 56 threads, 251 GiB RAM, two NUMA
  nodes (128 GiB each; distance 21 across sockets).
- CUDA ordinal 0: RTX 3090 24 GiB, sm_86, PCI `0000:82:00.0`, x8 link,
  NUMA node 1.
- CUDA ordinal 1: RTX 3090 24 GiB, sm_86, PCI `0000:83:00.0`, x16 link,
  NUMA node 1.
- CUDA ordinal 2: RTX 5060 Ti 16 GiB, sm_120, PCI `0000:03:00.0`, x8 link,
  NUMA node 0.
- Topology: no NVLink; GPU 0/1 are PHB peers, and GPU 2 crosses the socket
  interconnect (`SYS`) to both 3090s.
- Storage/model: XPG GAMMIX S70 BLADE NVMe, ext4, model at `c/glm52`
  (358 GiB, `mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp`).
- OS/toolchain: Ubuntu kernel 6.8.0-100, GCC 11.4.0, CUDA 12.8.93,
  NVIDIA driver 595.71.05.

CUDA and `nvidia-smi` enumerate the cards differently; all experiment weights
and commands use the CUDA ordinals printed above.

## Per-device measurements and derived weights

Build:

```bash
make -C c cuda-bench \
  NVCCFLAGS='-O3 -std=c++17 -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_120,code=sm_120 -Xcompiler=-Wall,-Wextra'
```

Each device was run three times with 50 iterations and host allocation bound to
its local NUMA node. Medians:

| CUDA device | pinned H2D GB/s | resident W4 ms, 1/4/8 rows | streamed W4 ms, 1/4/8 rows |
|---|---:|---:|---:|
| 0, RTX 3090 x8 | 6.013 | 0.110 / 0.357 / 0.683 | 4.531 / 4.774 / 5.083 |
| 1, RTX 3090 x16 | 12.073 | 0.110 / 0.349 / 0.658 | 2.958 / 3.187 / 3.484 |
| 2, RTX 5060 Ti x8 | 6.730 | 0.111 / 0.383 / 0.752 | 4.176 / 4.449 / 4.813 |

The resident row-1 kernel is tied; the PCIe path creates the material
difference. The previous cache decode had 24,552 misses and 9,470 hits, so the
miss fraction is 0.7216. For each device, the service cost is
`0.7216 * streamed_row1 + 0.2784 * resident_row1`; inverse costs normalized to
CUDA device 1 produce:

```text
COLI_CUDA_DEVICE_WEIGHTS=0.656,1.000,0.711
```

## Candidate and gates

The candidate is opt-in and affects three scheduling decisions:

1. Startup-ranked experts are assigned by placed bytes divided by service
   weight instead of placed bytes alone.
2. Preallocated cache slots are divided by service weight subject to each
   device's safe capacity.
3. Direct-stream groups are assigned by queued groups divided by service
   weight.

The microbenchmark now accepts a CUDA ordinal and iteration count, prints the
PCI address and pinned-H2D bandwidth, and retains the 1/2/4/8-row expert output.
The checked-in `c/tools/run_weighted_placement_ab.sh` runs the interleaved
real-model matrix, records maximum RSS, monitors per-device VRAM/utilization,
and validates equal decode work and cache counters.

Completed gates:

```text
make -C c check                                      PASS (59 Python tests + C tests)
make -C c cuda-test NVCCFLAGS='<sm_86 + sm_120>'     PASS
c/backend_cuda_test 0 1 2                            PASS on all three devices
git diff --check                                     PASS
```

The canonical tiny GLM oracle was unavailable because `c/glm_tiny` is absent;
the repository's canonical `c/ref_glm.json` was not regenerated. The fixed
real-model replay completed without fallback, invariant errors, or OOM.

## Real-model A/B result

The matrix ran in AB/BA/AB order. Rates below are calculated from the unrounded
192-token decode wall times, not the engine's two-decimal display.

| Config | Rep | Prefill s | Decode s | Decode tok/s | Cache H2D ms | Routed rows | Max RSS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 1 | 12.266 | 224.299 | 0.8560 | 128199.5 | 65550 | 180.01 |
| weighted | 1 | 12.070 | 212.123 | 0.9051 | 116276.0 | 65585 | 179.62 |
| weighted | 2 | 10.927 | 212.126 | 0.9051 | 115349.5 | 65585 | 180.31 |
| control | 2 | 12.059 | 215.201 | 0.8922 | 120070.5 | 65550 | 180.37 |
| control | 3 | 12.297 | 213.606 | 0.8989 | 119658.2 | 65550 | 180.38 |
| weighted | 3 | 12.087 | 214.286 | 0.8960 | 119000.8 | 65585 | 180.31 |

- Control median: **0.8922 tok/s**; range 0.8560-0.8989.
- Weighted median: **0.9051 tok/s**; range 0.8960-0.9051.
- Median gain: **1.45%**; paired gains: +5.74%, +1.45%, -0.32%.
- Median cache-weight H2D time: 120070.5 ms control versus 116276.0 ms
  weighted (**-3.16%**).
- Equal budget held: 422 cache slots / 7.98 GB occupied, 42.11 GB permanent
  expert tier, and 59.25 GB total CUDA resident tensors in both configurations.
- Weighted did slightly more work: 65,585 versus 65,550 routed rows and
  464,409 versus 463,369 MB cache-weight H2D. This small difference penalizes
  the candidate and does not explain its apparent gain.
- Peak VRAM from the NVML monitor was 13,970 MiB on the 5060 Ti and
  21,080/22,238 MiB on the 3090s. No error/fallback/OOM marker was present.

Raw machine-local artifacts remain ignored under:

- `c/bench/logs/weighted_placement_20260713_summary.txt`
- `c/bench/logs/bench_weighted_placement_20260713_{control,weighted}_rN.log.1`
- `c/bench/logs/weighted_placement_20260713_gpu.csv`

A two-repetition variance follow-up was started because the apparent gain was
below the control spread, then deliberately stopped before its first repetition
completed: four more full model runs were not justified for a marginal policy
whose effect was already below noise. Its partial logs are excluded.

## Decision and next experiment

Decision: **failed**. The expected transfer improvement appeared, but it did not
produce a repeatable end-to-end win larger than noise, and the added placement
policy is not justified by a possible ~1% effect. Do not merge the runtime code.

Next, capture one sequential routing trace and simulate global LRU, per-layer
LRU, and LFU offline at the same 422-slot budget. Only implement and benchmark a
policy if the simulation projects a material miss/weight-H2D reduction (suggested
gate: at least 10%), avoiding another expensive real-model matrix for a weak
hypothesis.
