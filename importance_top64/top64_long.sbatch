#!/bin/bash
#SBATCH --job-name=N51_TK17_TOP64_CB4096_3K
#SBATCH --partition=RTX4090D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/liujp/zzhliu/NQS_LLL/logs/N51_TOP64_%j.out
#SBATCH --error=/share/home/liujp/zzhliu/NQS_LLL/logs/N51_TOP64_%j.err
set -euo pipefail
root=/share/home/liujp/zzhliu/NQS_LLL
template="$root/code/NQS_MPS_importance_top64_20261121"
result="$root/results/nphi51_tk17_top64_cb4096_3000_${SLURM_JOB_ID}"
mkdir -p "$result/source"
cp -a "$template/src" "$template/implementation" "$result/source/"
cp "$template/config.json" "$template/run_importance.py" "$result/source/"
cd "$result/source"
export OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
python=/share/home/liujp/miniconda3/envs/pytorch_new/bin/python
"$python" -u run_importance.py --top-exact 64 --steps 3 --output "$result/smoke"
"$python" - "$result/smoke/training" <<'PY'
import csv,math,sys
from pathlib import Path
p=Path(sys.argv[1])
with (p/'history.csv').open() as f:r=list(csv.DictReader(f))
assert len(r)==3
for row in r:
 for key in ('energy','variance','grad_norm','time_step'):assert math.isfinite(float(row[key]))
assert (p/'params.pkl').exists()
print('Top64 smoke gate passed',flush=True)
PY
"$python" -u run_importance.py --top-exact 64 --steps 3000 --output "$result/main"
