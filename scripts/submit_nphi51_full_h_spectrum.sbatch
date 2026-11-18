#!/bin/bash
#SBATCH --job-name=N51_FULLH_03_SPEC
#SBATCH --partition=RTX4090D
#SBATCH --array=0-4%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/N51_FULLH_03_%A_%a.out
#SBATCH --error=logs/N51_FULLH_03_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
PYTHON="${NQS_PYTHON:-python}"
RUN="$PWD/results/nphi51_full_h_0p3m_2000_${SLURM_ARRAY_JOB_ID}/tk_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total --format=csv -l 5 > "$RUN/gpu_memory.csv" &
MONITOR=$!
trap 'kill "$MONITOR" 2>/dev/null || true; wait "$MONITOR" 2>/dev/null || true' EXIT
/usr/bin/time -v -o "$RUN/host_memory.txt" "$PYTHON" -B -u -m nqs_mps \
    --config configs/nphi51_tk0_full_h_0p3m_2000.json \
    --tk "$SLURM_ARRAY_TASK_ID" --output "$RUN/training" \
    > "$RUN/run.log" 2> "$RUN/run.err"
touch "$RUN/COMPLETED"
