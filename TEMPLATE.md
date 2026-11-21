# NQS MPS Optimal Baseline (updated 2026-11-19)

This directory is the versioned production template for bitstring,
sector-free Backflow MPS VMC jobs. Historical configs remain available.

## Canonical settings

For subsequent Nphi=51 experiments, use this fastest canonical configuration
as the control and change only the explicitly tested factors. Older execution
paths are diagnostic controls, not the default benchmark baseline.

- Packed `uint64` bitstrings; no materialized sector table.
- Symmetric `conserved_pair` proposal, one sampler scan per update.
- Exact local-energy neighbors with complete multi-tile row reduction.
- Nphi=51, tk=0, H397 x H397 (304103 parameters), D=10.
- 4096 chains, burn=40, between=1; no parallel tempering.
- `compact_gpu_dedup` with 150 chunks.
- Structured first-layer reuse with fused tail.
- `backend=m4`, selecting fused M3/triplet-tail contraction for Nphi=51.
- `chain_batch=4096`, `forward_batch=32768`.
- Neighbor internal tile 1024 with runtime-count skipping of unused tail blocks.
- KFAC/natural gradient, constant `lr=5e-3`, damping `1e-3`.
- Maximum natural-gradient direction norm 10 through step 300, then 20.
- 2000 steps, seed=512039.

KFAC remains the default. An exact sample-space stochastic reconfiguration
path is available by setting `training.optimizer` to `"sr"`; it uses the same
learning rate, damping, and direction-norm controls. See
`configs/nphi51_tk0_params_0p2m_sr_2000.json` and the memory warning in the
README before using it for production runs.

## Canonical configs

- Full-neighbor 16384-chain CPU optimization:
  `configs/nphi51_fullneighbor_16384_cpu.json` (`masked_full`, clip20).
- The same configuration with local-energy chain batches on two GPUs:
  `configs/nphi51_fullneighbor_16384_cpu_2gpu.json`.
  Total chains remain 16384; each GPU evaluates two 4096-chain batches.
  Gradient, KFAC and sampling remain global on the primary GPU.
  Only `execution.local_energy_devices` changes between these two presets.
- Current complete-H two-layer 0.3M baseline:
  `configs/nphi51_tk0_full_h_0p3m_2000.json` (`397 x 397`, 304103 params).
- Streaming-only comparison, unchanged physical settings:
  `configs/nphi51_tk0_full_h_0p3m_stream1024_2000.json`.
- Equivalent explicit name for the current counted1024 baseline, same capacity:
  `configs/nphi51_tk0_full_h_0p3m_counted1024_2000.json`.
- Original execution rollback (complete Hamiltonian preserved):
  `configs/nphi51_tk0_full_h_0p3m_rollback_2000.json`.
- Historical two-layer 0.2M baseline:
  `configs/nphi51_tk0_params_0p2m_2000.json` (`296 x 296`, 200376 params).
- Three-layer controlled comparison:
  `configs/nphi51_tk0_0p2m_h233x3_direct_2000.json`
  (`233 x 233 x 233`, 200995 params).

The three-layer config changes only the user-visible hidden layers. Sampling,
optimizer, MPS bond dimension, batches, and execution kernels remain fixed.
The canonical Nphi=51 config explicitly enables both execution optimizations.
Omitted options still default off for historical/other-architecture configs.
Short RTX4090D fixed-input and training timing checks are documented in README;
they are not new 2000-step convergence runs. No running job is modified.
Grouped loops and experimental cache policies are excluded from this baseline.

## Launch invariant

Always launch from this template directory so an older package in the parent
project cannot shadow the frozen source:

```bash
cd /path/to/NQS_MPS_optimal_baseline_20261106
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
python -B -u -m nqs_mps --config configs/CONFIG.json --output OUTPUT_DIR
```

Do not launch from a legacy NQS project root. Python places the current working
directory before `PYTHONPATH`, which can silently mix wrapper and core versions.

## Integrity

`SHA256SUMS` records all frozen Python sources. Generated `__pycache__`, `.pyc`,
logs, and numerical results are intentionally excluded from this template.

The Nphi=51 three-layer RTX4090D production validation is Slurm job `156252`.
