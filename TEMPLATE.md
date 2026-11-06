# NQS MPS Optimal Baseline (2026-11-06)

This directory is the frozen production template for bitstring, sector-free
Backflow MPS VMC jobs.

## Canonical settings

- Packed `uint64` bitstrings; no materialized sector table.
- Symmetric `conserved_pair` proposal, one sampler scan per update.
- Exact local-energy neighbors.
- `compact_gpu_dedup` with 150 chunks.
- Structured first-layer reuse with fused tail.
- Fused M4 MPS contraction.
- `chain_batch=4096`, `forward_batch=32768`.
- KFAC/natural gradient, constant `lr=5e-3`, damping `1e-3`.
- Maximum natural-gradient direction norm 10.

## Canonical configs

- Two-layer 0.2M baseline:
  `configs/nphi51_tk0_params_0p2m_2000.json` (`296 x 296`, 200376 params).
- Three-layer controlled comparison:
  `configs/nphi51_tk0_0p2m_h233x3_direct_2000.json`
  (`233 x 233 x 233`, 200995 params).

The three-layer config changes only the user-visible hidden layers. Sampling,
optimizer, MPS bond dimension, batches, and execution kernels remain fixed.

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
