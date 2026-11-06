# NQS MPS Optimal Baseline

Frozen production template for sector-free Backflow MPS variational Monte
Carlo on the torus. This snapshot corresponds to the code and execution path
used for the 2026-11-06 Nphi=51 experiments.

## Production path

- Packed `uint64` bitstrings; no materialized sector table.
- Symmetric particle-number and momentum-conserving pair proposals.
- Exact Hamiltonian neighbors for local energy.
- Compact GPU deduplication and contiguous row reduction.
- Structured first-layer reuse with fused tail.
- Fused M=4/M=3 MPS contraction selected from `Nphi`.
- JIT KFAC factors and GPU solve.

The canonical two-layer and three-layer Nphi=51 configurations are:

```text
configs/nphi51_tk0_params_0p2m_2000.json
configs/nphi51_tk0_0p2m_h233x3_direct_2000.json
```

The two-layer model has hidden sizes `296 x 296` and 200376 parameters. The
controlled three-layer comparison uses `233 x 233 x 233` and 200995
parameters. All other physics, sampling, optimizer, batching, and kernel
settings are unchanged.

## Run

Use Python 3.10+ with a CUDA-compatible JAX installation. Launch from the
repository root so legacy packages in a parent directory cannot shadow this
snapshot:

```bash
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
python -B -u -m nqs_mps \
  --config configs/nphi51_tk0_params_0p2m_2000.json \
  --output outputs/nphi51_tk0_0p2m
```

Every output directory records the resolved configuration, environment, code
hashes, history, and final parameters. Nonempty output directories are not
overwritten.

See [TEMPLATE.md](TEMPLATE.md) for the frozen settings and launch invariants.
`SHA256SUMS` records the source-file integrity manifest.

## Validation

The three-layer RTX4090D validation run is Slurm job `156252`. Attempts
`156248` through `156251` exposed wrapper/core version mixing in older launch
paths and are not numerical results. Always launch with this repository as the
current working directory.
