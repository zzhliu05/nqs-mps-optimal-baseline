#!/bin/bash
#SBATCH --job-name=N51_ISFUSED_CB4096_3TK
#SBATCH --partition=RTX4090D
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=24G
#SBATCH --array=0-2%3
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/liujp/zzhliu/NQS_LLL/logs/N51_ISFUSED_CB4096_%A_%a.out
#SBATCH --error=/share/home/liujp/zzhliu/NQS_LLL/logs/N51_ISFUSED_CB4096_%A_%a.err
set -euo pipefail
root=/share/home/liujp/zzhliu/NQS_LLL
template="$root/code/NQS_MPS_importance_baseline_20261121"
sectors=(0 17 34)
export TASK_TK=${sectors[$SLURM_ARRAY_TASK_ID]}
result="$root/results/nphi51_is_fused_cb4096_3sectors_3000_${SLURM_ARRAY_JOB_ID}/tk_${TASK_TK}"
mkdir -p "$result/source"
cp -a "$template/src" "$template/implementation" "$result/source/"
cp "$template/run_importance.py" "$template/config.json" "$result/source/"
cd "$result/source"
export OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=18 XLA_PYTHON_CLIENT_PREALLOCATE=false
python=/share/home/liujp/miniconda3/envs/pytorch_new/bin/python
"$python" - <<'PY'
import json,os
from pathlib import Path
p=Path('config.json');config=json.loads(p.read_text())
config['system']['tk']=int(os.environ['TASK_TK'])
config['execution']['chain_batch']=4096
config['sampling']['chains']=16384
config['training']['steps']=3000
Path('task_config.json').write_text(json.dumps(config,indent=2))
PY
"$python" -u run_importance.py --config task_config.json --output "$result/run"
