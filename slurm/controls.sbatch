#!/bin/bash
#SBATCH --job-name=gap-ctrl
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-31
#SBATCH --output=/home/kzhao2/nobackup/autodelete/gap_measurement/logs/ctrl_%a.out

# Array layout: idx = control(2) x arch(2) x shard(8)
#   control = idx / 16    (0 = C2 bf16 rerun 5%, 1 = C1 prefill rescore 5%)
#   arch    = (idx / 8) % 2 (0=moe, 1=dense)
#   shard   = idx % 8
source /home/kzhao2/gap_measurement/slurm/env.sh

IDX=$SLURM_ARRAY_TASK_ID
SHARD=$((IDX % 8))
if [ $(((IDX / 8) % 2)) -eq 0 ]; then ARCH=moe; else ARCH=dense; fi

if [ $((IDX / 16)) -eq 0 ]; then
  # C2 determinism floor: two IDENTICAL recompute runs (same subsample,
  # same deterministic batching) compared against each other.
  python src/recompute_train.py --arch $ARCH --shard $SHARD \
    --precision bf16 --tag c2a --subsample 0.05
  python src/recompute_train.py --arch $ARCH --shard $SHARD \
    --precision bf16 --tag c2b --subsample 0.05
else
  python src/rescore_prefill_c1.py --arch $ARCH --shard $SHARD --subsample 0.05
fi
