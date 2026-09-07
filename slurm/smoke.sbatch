#!/bin/bash
#SBATCH --job-name=gap-smoke
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/home/kzhao2/nobackup/autodelete/gap_measurement/logs/smoke.out

source /home/kzhao2/gap_measurement/slurm/env.sh
set -e

echo "=== nvidia-smi ==="; nvidia-smi -L

echo "=== [1/7] gen moe smoke ==="
python src/gen_vllm.py --arch moe --shard 0 --smoke

echo "=== [2/7] recompute moe bf16 smoke ==="
python src/recompute_train.py --arch moe --shard 0 --precision bf16 --smoke

echo "=== [3/7] recompute moe fp32 smoke ==="
python src/recompute_train.py --arch moe --shard 0 --precision fp32 --smoke

echo "=== [4/7] c2 reruns moe smoke ==="
python src/recompute_train.py --arch moe --shard 0 --precision bf16 --smoke --tag c2a --subsample 1.0
python src/recompute_train.py --arch moe --shard 0 --precision bf16 --smoke --tag c2b --subsample 1.0

echo "=== [5/7] c1 moe smoke ==="
python src/rescore_prefill_c1.py --arch moe --shard 0 --smoke --subsample 1.0

echo "=== [6/7] gen+recompute dense smoke ==="
python src/gen_vllm.py --arch dense --shard 0 --smoke
python src/recompute_train.py --arch dense --shard 0 --precision bf16 --smoke

echo "=== [7/7] join check ==="
python scripts/smoke_check.py moe
python scripts/smoke_check.py dense

echo "ALL SMOKE STAGES PASSED"
