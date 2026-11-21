# NQS MPS Optimal Baseline

Production template for sector-free Backflow MPS variational Monte Carlo on
the torus. The current Nphi=51 baseline (2026-11-19) uses the complete
Hamiltonian row reduction and a two-layer 0.3M-parameter MLP.

## Current Nphi=51 baseline

CPU update (2026-11-21): fixed-width packed-bitstring rows default to
`execution.cpu_row_builder="hash"`. This preserves term order, neighbor order,
and coefficients while replacing linear neighbor lookup with a row-local hash.
Rollback: set `execution.cpu_row_builder="linear"` in the input JSON. The
canonical launcher records the mode in resolved settings and environment.json;
shell NQS overrides are cleared by that launcher. This does not change network,
sampling, clipping, row width, or GPU deduplication settings.

Measured with Top64+32, 32768 chains, chain_batch4096 and18 CPU threads:
CPU rows plus packing about1.19s ->0.62s, total step about4.2s ->3.6s.
Two ABBA tests excluded compilation; real-chain row checks and same-process
fixed-input energy/gradient replay were exact. These timings do not claim
the same speedup for full-neighbor or other-size workloads, or bitwise
reproducibility across independent training processes.

The further optimized deterministic Top256 path (fused row construction and
selection, precomputed parity masks, fixed Hamiltonian coefficient reuse) is
available in the sibling `NQS_MPS_importance_baseline_20261121` template via
`run_importance.py --config config_top256_cpu.json --top256 --output ...`.
It is deliberately not substituted for this template's full-neighbor estimator:
deterministic truncation changes the estimator and must be selected explicitly.

### Nphi51 Full Neighbors, 16384 Chains

`configs/nphi51_fullneighbor_16384_cpu.json` selects the optimized full-H CPU
builder, `execution.cpu_row_builder="masked_full"`. It retains every neighbor:
there is no Top-K selection, random tail, or changed estimator. The preset uses
tk3, H397x397, D10, 16384 chains, chain_batch4096, forward_batch32768,
neighbor tile1024, burn40, KFAC lr0.005/clip20, 3000 steps and one GPU.

```bash
PYTHONPATH=src python -m nqs_mps --config configs/nphi51_fullneighbor_16384_cpu.json --output results/full51_cpu
```

Implementation: row-local hash, precomputed fermion-parity masks, fixed-H
coefficient reuse, CPU-chunk row-major packing, and counts-based compact output.
The full-batch mask and its count scan are avoided. Unused scratch slots are
never copied or sent to GPU. No wavefunction value or sampled row is cached.
The existing coefficient dtype, row width1856 and overflow checks are retained.
The GPU unique capacity is232*32768, enough for the maximum fixed row slots;
compact staging also retains its independent capacity check and fails on
overflow instead of dropping neighbors. This reserve is not a sector table.

Cached Hamiltonian arrays become read-only. Replacing the matrix or term array
invalidates the cache; forcibly re-enabling writable access fails closed.
Choose cpu_row_builder=hash to disable the mask cache and chunk packing, or
linear for the original row search. Other existing presets remain unchanged.
masked_full requires conserved_pair and compact_gpu_dedup, because it returns
an internal packed row object instead of a dense neighbor/mask tensor.

CPU tests: `PYTHONPATH=src python -m unittest discover -s tests -p "test_cpu_full_rows.py"`.
Tests include more than1024 valid neighbors per row, chunk boundaries, exact
neighbor order/coefficients, both coefficient dtypes, overflow and cache rebuild.
Integration job170764 completed normally on one4090D: same-process full-H
replay had energy_delta=0 and gradient_max_abs=0 for both warm repetitions.
CPU row/packing time was0.252/0.251s (hash) versus0.118/0.119s (masked_full).
A fresh3-step training smoke also completed and saved params.pkl; step3 was
11.722s. These are short validation timings, not a long-convergence guarantee
or a two-GPU performance comparison. Existing generic presets still default
to hash; this explicit16384-chain preset defaults to masked_full.

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

### Full-neighbor chain-batch device parallelism

`configs/nphi51_fullneighbor_16384_cpu_2gpu.json` enables
`execution.local_energy_devices=2`. The default remains 1. Both presets use
16384 total chains, not 16384 per GPU. With `chain_batch=4096`, GPU0 evaluates
batches 0 and 2, while GPU1 evaluates batches 1 and 3. Full Hamiltonian rows
are retained; this is not Top-K truncation or neighbor importance sampling.

Only row construction and local-energy evaluation run concurrently. Parameters
are replicated for each evaluation, local energies return to host in their
original chain order, and the global mean, VMC gradient, KFAC and optimizer
update remain on the primary device. Sampling also remains on that device.
This is not distributed KFAC or a data-parallel optimizer.

The immutable CPU coefficient cache is prepared before worker launch. An
18-thread CPU budget is split into 9 Numba threads per worker. Two visible
GPUs and an estimator sample count divisible by `2 * chain_batch` are required.
This opt-in path currently supports exact streamed bitstring backflow, not
site tensor banks. Set `local_energy_devices=1` to roll back without changing
the network, chain count or CPU row builder.

`CHAIN_DEVICE_TIMING` log records contain the elapsed parallel local-energy
stage. The existing CPU-row and local-energy component fields are **summed
worker time**, not elapsed stage time; do not add them to infer dual-GPU
training wall time. Use `time_step` for actual full-step comparisons.

From the template root, after activating the JAX environment:

```bash
mkdir -p logs
export NQS_PYTHON="$(command -v python)"
sbatch scripts/submit_nphi51_fullneighbor_2gpu.sbatch
```

For a short allocated two-GPU test, use `python -m nqs_mps --config
configs/nphi51_fullneighbor_16384_cpu_2gpu.json --steps 8 --output outputs/test2gpu`
(as one command).

RTX4090D job `170765` completed successfully. Two interleaved fixed-input
repeats had identical energies, gradients and KFAC inputs: mean `loss_grad`
was 10.9405 s on one GPU versus 5.6521 s on two (1.936x). Separate 8-step
training smoke tests, excluding step 1, averaged 11.7776 versus 6.4724 s per
complete step (1.820x); medians were 11.8057 and 6.4663 s. Both saved final
checkpoints. Peak observed memory over the entire validation job was
16854/9617 MiB on GPU0/GPU1. Independent training trajectories differed after
initialization, so these short tests do not establish long-run convergence
equivalence. The fixed-input comparison, not equal seeds alone, is the
numerical-equivalence check.

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
