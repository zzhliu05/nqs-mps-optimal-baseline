# NQS MPS Optimal Baseline

## Importance-Sampling Templates (2026-11-21)

New standalone entry points are included without changing the full-neighbor
baseline below. These use complete Hamiltonian rows BEFORE unbiased neighbor
subsampling; they do not restore the historical incomplete-H bug.

| Entry | Exact top terms | Tail draws | Total chains | chain_batch |
| --- | ---: | ---: | ---: | ---: |
| [importance_baseline](importance_baseline/README.md) | 32 | 32 | 16384 | 16384 default; 4096 sector launcher |
| [importance_top64](importance_top64/README.md) | 64 | 32 | 16384 | 4096 |

Both use Nphi51, nu1/3, H397x397, D10, fused18-thread CPU sampling, compact GPU
dedup, tile1024, valid-proposal-only MCMC forward, burn40 and KFAC lr0.005,
damping0.001, constant clip20 (NOT the historical10-to20 schedule below).
Fisher projection, PT and state-row caching are off. Defaults are3000 steps,
tk17, seed512039. Full-neighbor execution remains available via `--exact`;
`--no-mask` restores all-proposal forward. See template docs for validation
limits, resume semantics and source/estimator manifests.

```bash
cd importance_baseline
python run_importance.py --tk 17 --steps 3000 --output results/run1
# Or cd ../importance_top64 and use its run_importance.py for top64+32.
```

IMPORTANT: original runtime history/resolved can label the underlying evaluator
`exact`; actual neighbor estimator is recorded in importance_manifest.json.
Conditional unbiasedness does not mean zero extra variance or certified long
training convergence. Top64 is an experimental comparison, not a proven win.
No checkpoints, raw results, SSH keys or credentials are included.

Production template for sector-free Backflow MPS variational Monte Carlo on
the torus. The current Nphi=51 baseline (2026-11-19) uses the complete
Hamiltonian row reduction and a two-layer 0.3M-parameter MLP.

## Current Nphi=51 baseline

Use `configs/nphi51_tk0_full_h_0p3m_2000.json` for new runs:

| Setting | Value |
| --- | --- |
| System | Nphi=51, Ne=17, filling 1/3, LLL, square torus |
| Network | Backflow MPS, H397 x H397, 304103 parameters, D=10 |
| Sampling | conserved_pair, 4096 chains, burn=40, between=1; no PT |
| Optimizer | KFAC, constant lr=0.005, damping=0.001 |
| Direction-norm clipping | 10 for steps 1-300; 20 for steps 301-2000 |
| Execution | backend=m4 selects M3 for Nphi=51; compact_gpu_dedup (150 chunks), reuse_fused_tail |
| Batches | chain_batch=4096, forward_batch=32768, sampler_scan=1 |
| Neighbor execution tiles | 1024; skip unused tail blocks using runtime dedup count |
| Sparse-row allocation | row_width=1856; not the reduction tile width |
| Training | 2000 steps, seed=512039, fresh initialization |

The complete-row fix is built into `compact_row_reduce_pallas`; no experiment
wrapper or `ROW_REDUCE_COMPLETE` environment variable is needed. Earlier
versions reduced only the first tile of a compact row (1024 entries by
default). Rows longer than that silently lost contributions. The corrected
kernel loops over every tile, masks the final partial tile, and accumulates
in FP32. This removes that omission, not floating-point rounding. Old
wavefunctions must be re-evaluated with the corrected Hamiltonian before
comparing their energies with new runs.

The physical and training configuration matches complete-H job 168509 (tk=0) and spectrum
array 168717 (tk=1-4). Those jobs use an isolated wrapper with the same
reduction algorithm; the repository now integrates it directly. Existing
cluster jobs and their frozen source directories are not modified by this
repository update. GPU throughput depends on hardware and workload; single
GPU two-process concurrency is still experimental, not a default.

### Neighbor activation streaming (canonical Nphi=51 default)

`execution.neighbor_forward_tile` is an optional internal execution tile for
the cached compact-GPU-dedup neighbor forward. It defaults to `null`, which
preserves the original path for historical configs. The canonical Nphi=51 config enables 1024.
The streaming-only comparison configuration
`configs/nphi51_tk0_full_h_0p3m_stream1024_2000.json` sets it to 1024.

This does not change `forward_batch=32768`, dedup capacity (4915200 slots),
Hamiltonian completeness, network dimensions, precision, sampler, gradient
batch or KFAC. It splits only the neighbor forward into smaller execution
tiles to improve activation locality. It is NOT fewer neighbors or less
padding. Set it back to `null` to roll back. Configuration is fixed at process
startup, as with the other JIT execution switches.

RTX4090D fixed-input tests (2026-11-19) found about 10% faster full forward,
7% faster local energy and 7.5% faster loss/gradient, with identical results
on the tested inputs. Job168832 completed a 90-step real-training ABBA check:
two comparisons improved whole-step throughput by 6.15% and 6.31%
(3.57s to 3.36-3.37s median). Compilation and fixed-replay overhead were
excluded. Fixed-input locE and gradient differences were zero in this test.
Job168837 passed seven configuration/full-row tests and 30 GPU training
steps using this standalone template. These are short performance and
correctness checks, not a 2000-step convergence claim. Other systems and
networks need separate measurements; their configuration defaults remain unchanged.

### Count-controlled neighbor execution (canonical Nphi=51 default)

`execution.neighbor_skip_unused` defaults to `false` when omitted for backward compatibility.
The canonical Nphi=51 config explicitly enables it; the equivalent named configuration
`configs/nphi51_tk0_full_h_0p3m_counted1024_2000.json` combines it with the1024
internal tile. It requires backflow generation, `compact_gpu_dedup`, and an
explicit tile. Fixed buffer capacities, padding allocation, inverse indices,
and Hamiltonian terms are unchanged. Only unused tail blocks are skipped,
using the GPU dedup count as a runtime loop bound. The last partial block and
the padding sentinel group are evaluated. Overflow remains an error.
`test_runtime_count_does_not_retrace` also checks that changing only the
count value reuses the same JIT trace. Changing shape, dtype or static config
can still require another compilation; the assertion is not universal.
Nsight on the current XLA runtime also shows a host condition transfer after
each dynamic-loop iteration. Thus no-retracing does not mean host-free GPU
loop control. Grouped-loop alternatives remain separate experiments.

Job168844 compared this against full-capacity tile1024. Identical-input locE
and loss/gradient replays had zero measured error and speedups of about3.7%
and3.2%, when about95% of the capacity was needed. During real-training ABBA,
the candidate needed about88-89% of the blocks: whole-step medians fell from
3.36s to3.06-3.07s (9.4-9.8% higher throughput). Savings depend on the valid
count; these numbers are not a fixed speed guarantee or long-convergence
validation. Set `neighbor_skip_unused=false` to return to full-capacity tile
execution, and set `neighbor_forward_tile=null` for the original baseline.
Independent job168848 passed13configuration/structure/full-row tests,
7real-wavefunction boundary cases plus invalid-count rejection, and30GPU
training steps. Its warm steps11-30 had a3.233s median and9684MiB sampled
device-memory peak. This earlier training window has a different valid-count
distribution from the ABBA candidate windows; do not equate their timings.

Frozen-checkpoint job168849 directly compared all three paths on identical
parameters/states/rows from completed2000-step checkpoint168509. With4208370
valid groups, counted execution evaluates4110 of4800 blocks. Two rounds give
locE medians3.104/3.109s (original),2.950/2.959s (tile1024),2.561/2.567s
(counted1024). Loss/gradient medians are3.273/3.271s,3.117/3.109s,2.728/2.722s.
Both candidates have zero observed locE/gradient differences on these inputs.
The combined path reduces locE time by about17.5% and loss/gradient by16.7%.
This replay performs zero optimizer updates: these are not full-step timings
or a new convergence validation, and the gain still depends on valid count.

Direct full-step ABBA job168857 subsequently compared original32768 against
counted1024 in90fresh steps. Reference medians3.516/3.511s fell to2.954/2.899s
(1.1904/1.2109x throughput). Sample and optimizer times remained about0.122s
and0.049s. Candidate windows evaluated85.98%/82.89% of capacity. Fixed replay
checks again showed zero observed locE/gradient error. This combined path was
promoted to the canonical Nphi=51 configuration on 2026-11-19. These short tests
do not replace long-run convergence checks. Grouped loops and cache-policy
experiments are not included. Roll back with
`configs/nphi51_tk0_full_h_0p3m_rollback_2000.json`; this preserves the full-H fix
while disabling both execution optimizations. Existing running jobs are unchanged.

## Production path

- Packed `uint64` bitstrings; no materialized sector table.
- Symmetric particle-number and momentum-conserving pair proposals.
- Exact Hamiltonian neighbors for local energy.
- Compact GPU deduplication and contiguous row reduction.
- Structured first-layer reuse with fused tail.
- Fused M=4/M=3 MPS contraction selected from `Nphi`.
- JIT KFAC factors and GPU solve.

The retained historical two-layer and three-layer Nphi=51 configurations are:

```text
configs/nphi51_tk0_params_0p2m_2000.json
configs/nphi51_tk0_0p2m_h233x3_direct_2000.json
```

The two-layer model has hidden sizes `296 x 296` and 200376 parameters. The
controlled three-layer comparison uses `233 x 233 x 233` and 200995
parameters. All other physics, sampling, optimizer, batching, and kernel
settings are unchanged.

## Stochastic reconfiguration

Set `training.optimizer` to `"sr"` to use exact shifted stochastic
reconfiguration through a sample-space MinSR solve. The implementation uses
the same sampled states and exact local-energy gradient as KFAC, centers the
complex log-wavefunction derivatives, and solves the equivalent real
`2 * estimator_chains` sample kernel. `training.damping`,
`training.learning_rate`, and `training.max_step_norm` retain their existing
meanings. The production default remains `"kfac"`.

`training.sr_samples` selects how many estimator chains build the SR metric;
the energy and gradient still use all `sampling.estimator_chains`.
`training.sr_jacobian_batch_size` bounds the reverse-mode output batch so JAX
does not materialize an impractically large vectorized pullback. The Nphi=51
example uses 512 SR samples in blocks of 32 while retaining 4096 estimator
chains.

The canonical opt-in example is
`configs/nphi51_tk0_params_0p2m_sr_2000.json`. Its Jacobian storage scales as
`O(2 * estimator_chains * parameters)` and its dense sample solve as
`O((2 * estimator_chains)^3)`, so benchmark memory and optimizer time before a
long run.

## Run

Use Python 3.10+ with a CUDA-compatible JAX installation. Launch from the
repository root so legacy packages in a parent directory cannot shadow this
snapshot:

```bash
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
python -B -u -m nqs_mps \
  --config configs/nphi51_tk0_full_h_0p3m_2000.json \
  --output outputs/nphi51_tk0_full_h_0p3m
```

Every output directory records the resolved configuration, environment, code
hashes, history, and final parameters. Nonempty output directories are not
overwritten.

For a Slurm scan of tk=0..4, activate your JAX environment first, then run:

```bash
mkdir -p logs
export NQS_PYTHON="$(command -v python)"
sbatch scripts/submit_nphi51_full_h_spectrum.sbatch
```

The script requests one GPU, 18 CPU cores and 8 GiB host memory per sector,
with up to five sectors running concurrently. Adapt the partition name for
other clusters. To reuse an existing tk=0 run, submit only the other sectors
with `sbatch --array=1-4%4 scripts/submit_nphi51_full_h_spectrum.sbatch`.
Do not submit both commands for the same intended scan.

See [TEMPLATE.md](TEMPLATE.md) for the frozen settings and launch invariants.
`SHA256SUMS` records the source-file integrity manifest.

## Validation

`tests/test_compact_row_reduce.py` covers empty rows, tile boundaries, rows
up to 8193 entries, and a regression where all nonzero contributions lie
after entry 1024. CPU-only interpretation does not measure GPU performance:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python tests/test_compact_row_reduce.py
```

The three-layer RTX4090D validation run is Slurm job `156252`. Attempts
`156248` through `156251` exposed wrapper/core version mixing in older launch
paths and are not numerical results. Always launch with this repository as the
current working directory.
