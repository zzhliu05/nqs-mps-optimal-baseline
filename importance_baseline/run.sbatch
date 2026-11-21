#!/bin/bash
#SBATCH --job-name=N51_IMPORTANCE_BASELINE
#SBATCH --partition=RTX4090D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=importance_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
python -u run_importance.py --output "results/job_${SLURM_JOB_ID}" "$@"
