#!/bin/bash
#SBATCH --job-name=gap-ent
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/home/kzhao2/nobackup/autodelete/gap_measurement/logs/ent_%a.out
#SBATCH --exclude=dw-2-4

source /home/kzhao2/gap_measurement/slurm/env.sh
cd /home/kzhao2/gap_measurement

# 0-7 = moe shards, 8-15 = dense shards; bf16 + entropy column, tag ent
ID=$SLURM_ARRAY_TASK_ID
if [ "$ID" -lt 8 ]; then ARCH=moe; SHARD=$ID; else ARCH=dense; SHARD=$((ID-8)); fi
python src/recompute_train.py --arch $ARCH --shard $SHARD \
    --precision bf16 --entropy --tag ent
