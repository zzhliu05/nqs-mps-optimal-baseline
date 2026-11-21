#!/bin/bash
#SBATCH --job-name=N51_FULL_16384_2GPU
#SBATCH --partition=RTX4090D
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=18
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/N51_FULL_16384_2GPU_%j.out
#SBATCH --error=logs/N51_FULL_16384_2GPU_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="$PWD/src"
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-18}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
"${NQS_PYTHON:-python}" -u -m nqs_mps \
  --config configs/nphi51_fullneighbor_16384_cpu_2gpu.json \
  --output "outputs/nphi51_fullneighbor_16384_2gpu_${SLURM_JOB_ID}"
