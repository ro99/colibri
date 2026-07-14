# Performance Research Charter

## Scope

These instructions apply to the entire repository. More specific `AGENTS.md`
files may refine them for a subtree, but may not weaken the measurement,
correctness, safety, or Git-hygiene requirements below.

## Mission

This repository is now an independent performance-research fork. Its primary
goal is to maximize **measured token throughput** on RAM-rich, CUDA-capable,
multi-GPU machines. Optimize for machines with enough RAM to retain a large
expert working set and enough VRAM to make explicit placement, caching,
streaming, sharding, and overlap worthwhile. Homogeneous GPU topologies are the
preferred target; always record the actual topology used by an experiment.

The upstream project's minimalism is no longer a design constraint. Additional
code, dependencies, generated kernels, platform specialization, and research
instrumentation are acceptable when they enable a repeatable performance win or
answer an important performance question.

"Best tok/s at all costs" does **not** permit fabricated measurements, hidden
quality changes, invalid comparisons, unsafe memory overcommit, silent fallback, further quantized weights (int4 is the minimum),
data loss, or an unmaintainable Git history. Complexity must buy evidence.

## Repository authority

- Treat this fork's `origin` and `main` as authoritative.
- Treat the old `upstream` remote as historical context only.
- Do not merge, rebase, or reset to upstream unless the user explicitly asks.
- `main` is the best validated research baseline, not an experiment scratchpad.
- Keep `main` buildable and benchmarkable. Land only validated wins, explicit
  policy changes, shared research infrastructure, and concise experiment notes.

## Default autonomous task

When invoked without a narrower task, execute **one bounded experiment cycle**:

1. Inspect current results and select the highest-leverage unanswered hypothesis.
2. Create a fresh experiment branch from `main`.
3. Establish or confirm a baseline.
4. Add only the instrumentation needed to test the hypothesis.
5. Implement the experiment in reversible commits.
6. Run short correctness gates.
7. Launch long measurements in `tmux` and hand them back to the user.
8. On the next invocation, read the completed results and decide: succeed, fail,
   or run one justified follow-up.
9. Record the result, preserve useful knowledge, and leave Git in a clean state.

Do not start several speculative implementations in one branch. Do not continue
indefinitely after the hypothesis has been answered.

## Before changing code

Always begin with:

```bash
git status --short --branch
git log --oneline --decorate -8
git branch -vv
```

Then:

- Read this file and any more specific instructions.
- Preserve unrelated user changes. Never discard a dirty tree to make room.
- Identify the current baseline commit and the exact target hardware.
- Read the latest relevant notes in `docs/experiments/` and local benchmark
  summaries before repeating old work.
- State the hypothesis in one sentence, including the expected mechanism.
- Define the primary metric, correctness gate, memory ceiling, and rollback
  condition before implementation.

## Branch discipline

Never develop an experiment directly on `main`.

Use branches such as:

- `exp/YYYYMMDD-short-hypothesis` for measured experiments;
- `feat/short-name` for a result already proven worth productizing;
- `fix/short-name` for correctness fixes;
- `infra/short-name` for reusable benchmark or profiling infrastructure.

Create each experiment from the latest validated `main` unless the experiment
explicitly depends on another unmerged branch. If it does, document that parent
commit and do not present the result as an independent win.

Use a separate Git worktree for concurrent experiments. Never mix two hypotheses
in one worktree or branch.

## Commit discipline and rollback

Commit frequently enough that every important state is recoverable:

- baseline or benchmark harness;
- instrumentation;
- each independently testable implementation slice;
- correctness fix;
- final result note.

Commit before a risky rewrite and whenever short tests pass. Experimental
checkpoint commits are allowed; prefix them with `exp:` and describe what is and
is not working. Do not hide a failed direction inside a later giant commit.

Each commit should have one purpose and a useful message. Avoid unrelated
formatting churn. Run `git diff --check` before every final/result commit.

Do not rewrite shared history or force-push `main`. Do not use destructive reset
or checkout commands to recover from an experiment. Prefer a new commit, revert,
new branch, or worktree.

### Successful experiment

After validation, clean the branch history only if doing so does not destroy
useful experimental checkpoints. Merge the winning branch into `main` with its
tests, reusable tooling, and result note. Record both baseline and winning SHAs.

### Failed experiment

A failed experiment is a result, not garbage:

1. Commit a concise note under `docs/experiments/` explaining the hypothesis,
   implementation, measurements, and why it failed.
2. Tag or retain the failed branch so its code can be recovered.
3. Do not merge failed runtime code into `main`.
4. Cherry-pick or otherwise land only the small result note on `main` when it
   prevents future agents from repeating the same dead end.
5. Return to a fresh branch from `main` for the next hypothesis.

Suggested tags: `experiment/failed/YYYYMMDD-name` and
`experiment/win/YYYYMMDD-name`.

## Measurement contract

Performance claims require reproducible A/B evidence. Use the same:

- model and quantization;
- fixed prompt or replay token sequence;
- build type and compiler flags;
- GPU visibility and device ordering;
- RAM, VRAM, cache, pin, and context budgets;
- routing policy and generation settings;
- warm/cold state;
- correctness mode.

Prefer a fixed real-model replay for throughput comparisons. A replay guarantees
the same token inputs and decode length; it does not guarantee bit-identical
hidden states or routing across different kernels. Use real generation separately
for output and quality checks.

For normal comparisons:

- separate prefill and decode time;
- use at least three interleaved repetitions per configuration;
- report every run and the median, not only the best run;
- rotate A/B order to reduce thermal, page-cache, and background-load bias;
- if the claimed gain is comparable to run-to-run spread, run five to seven
  repetitions or report the result as inconclusive;
- include initialization and warm-up policy explicitly;
- compare equal total RAM and expert-VRAM budgets unless memory usage itself is
  the experimental variable.

Record at least:

- decode tok/s and raw wall seconds;
- prefill wall time;
- tokens, forwards, and routed expert rows;
- RAM RSS and per-device VRAM use;
- cache hits, misses, fills, evictions, and occupied bytes;
- weight H2D separately from activation H2D/D2H;
- per-device kernel or wall time when available;
- GPU utilization and synchronization counts when relevant;
- any fallback, allocation failure, OOM warning, or invariant error.

Do not infer CPU-only time from a mixed CPU/GPU timer. Do not call staged expert
weights "activation H2D." Validate counter semantics before using them in a
conclusion.

Every report must identify:

- baseline and candidate commit SHA;
- exact commands or a checked-in reusable script;
- CPU, RAM, GPUs, VRAM, PCIe/NVLink topology, and NUMA layout;
- storage device, filesystem, and model location;
- OS, compiler, CUDA toolkit, and driver;
- model identity and quantization;
- run count and aggregation method.

## Correctness and acceptance gates

Performance is invalid if the computation is accidentally skipped, corrupted,
or silently downgraded.

Run proportionate gates before long benchmarks and full gates before landing a
win:

```bash
make check
```

For GLM engine changes, also run the canonical tiny oracle when assets are
available:

- teacher forcing: 32/32;
- greedy generation: 20/20.

Use the repository's canonical `c/ref_glm.json`; regenerating references with an
unreviewed Transformers version can change the oracle itself.

For CUDA changes:

- run `make -C c cuda-test` or an equivalent multi-device backend test;
- test every CUDA architecture affected by the change;
- test cache/ownership/accounting invariants without requiring the full model;
- run at least one real-model smoke or fixed replay;
- verify requested GPU paths did not silently fall back to CPU;
- keep enough RAM/VRAM headroom to prevent the OOM killer.

Quality-reducing changes are allowed only as explicit, named modes. Report the
quality/performance tradeoff and never change routing, quantization, sampling, or
model semantics silently.

## Long jobs and tmux handoff

Do not hold an agent turn open for a long model load, replay matrix, conversion,
profile, or benchmark.

1. Put the exact command in a reusable script when practical.
2. Write output to an ignored log and a deterministic summary path.
3. Start the run in a named `tmux` session, for example
   `colibri-<experiment>-<gate>`.
4. Verify that the process started, owns the intended GPUs, and is producing a
   log.
5. Hand the session name, attach command, log path, expected outputs, and current
   stage back to the user.
6. Stop the agent turn. On the next invocation, inspect the completed session and
   logs before taking the next step.

Never launch two memory-heavy experiments that can contaminate each other's
results unless concurrency is itself the experiment.

## Artifact hygiene

Track:

- source code and tests;
- generally useful benchmark/profile scripts;
- small deterministic fixtures when licensing and size permit;
- concise experiment notes and small aggregate result files.

Do not track:

- model weights or converted shards;
- generated binaries and object files;
- raw profiler captures;
- large logs, routing histories, caches, dumps, or per-run temporary files;
- secrets, API keys, machine credentials, or private absolute paths.

Keep machine-specific artifacts under ignored paths such as `c/bench/logs/` or
another documented local results directory. Add new generated artifacts to
`.gitignore` before they become a recurring source of dirty-tree noise.

Never overwrite an immutable baseline fixture or pin profile during a run. Use a
separate per-run `STATS` output.

## Research priorities

Choose work from evidence, not this list's order. High-value directions include:

- low-row routed-expert kernels and launch overhead;
- persistent/fused quantized expert kernels;
- expert placement and cache policies driven by routing traces;
- reducing or overlapping weight H2D;
- overlapping NVMe, RAM, H2D, GPU compute, and D2H;
- topology-aware multi-GPU placement, P2P, and NUMA affinity;
- larger RAM-resident tiers and explicit page/NUMA management;
- continuous or multi-request batching, reporting aggregate throughput and
  per-request latency separately;
- quantization and speculative decoding when quality impact is measured;
- replacing synchronous hot-path allocation and synchronization;
- profiler-guided removal of serial CPU orchestration.

External libraries and generated code are allowed. Before adding a heavyweight
dependency, quantify what it replaces, deployment cost, and measured gain.

## Decision rules

Call an experiment a **win** only when:

- the gain is repeatable and larger than measurement noise;
- correctness gates pass;
- memory usage stays within the declared ceiling;
- no hidden fallback or workload reduction explains the speedup;
- the relevant quality contract is unchanged or explicitly reported;
- the result is better than the current `main` baseline on the target metric.

Call it **inconclusive** when variance, instrumentation ambiguity, or unequal work
could explain the result. Improve the measurement once before adding more code.

Call it **failed** when the hypothesis is disproven, the gain is below noise, or
the costs dominate. Record it and move on instead of polishing a non-win.

There is no minimum absolute tok/s improvement required for a merge. Complexity
is acceptable in this fork, but it must be justified by repeatable evidence or
reusable research capability.

## Required handover

At the end of every agent cycle, report:

- current branch and commit SHA;
- hypothesis tested;
- files and behavior changed;
- commands and gates run;
- all performance runs and median;
- correctness, RAM, and VRAM status;
- tmux session and log paths, if anything is running;
- decision: win, failed, or inconclusive;
- exact next recommended experiment.

Leave the tracked worktree clean whenever possible. Never claim completion while
a required gate is still running or a result has not been read.
