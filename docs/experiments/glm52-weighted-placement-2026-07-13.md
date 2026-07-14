# GLM-5.2 heterogeneous-GPU weighted placement experiment (2026-07-13)

Status: A/B launched; result pending. This note must not be read as a performance
claim until all six replay runs finish and the summary is inspected.

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
real-model replay is the remaining smoke/correctness gate.

## Long A/B handoff

- tmux: `colibri-weighted-placement-ab`
- attach: `tmux attach -t colibri-weighted-placement-ab`
- summary: `c/bench/logs/weighted_placement_20260713_summary.txt`
- raw logs: `c/bench/logs/bench_weighted_placement_20260713_{control,weighted}_rN.log.1`
- GPU monitor: `c/bench/logs/weighted_placement_20260713_gpu.csv`
- expected work: six full model loads/replays in AB/BA/AB order, then a parsed
  summary with every tok/s, decode wall time, prefill time, cache traffic,
  grouped rows, maximum RSS, median, and range.

Decision remains **inconclusive** until the tmux job completes and its result is
read.
