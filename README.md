# NQS MPS Optimal Baseline

Production template for sector-free Backflow MPS variational Monte Carlo on
the torus. The current Nphi=51 baseline (2026-11-18) uses the complete
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
| Execution | M4, compact_gpu_dedup (150 chunks), reuse_fused_tail |
| Batches | chain_batch=4096, forward_batch=32768, sampler_scan=1 |
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

The configuration matches complete-H training job 168509 (tk=0) and spectrum
array 168717 (tk=1-4). Those jobs use an isolated wrapper with the same
reduction algorithm; the repository now integrates it directly. Existing
cluster jobs and their frozen source directories are not modified by this
repository update. GPU throughput depends on hardware and workload; single
GPU two-process concurrency is still experimental, not a default.

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
