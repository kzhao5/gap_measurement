#!/bin/bash
#SBATCH --job-name=gap-recomp
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-31
#SBATCH --output=/home/kzhao2/nobackup/autodelete/gap_measurement/logs/recomp_%a.out

# Array layout: idx = arch(2) x precision(2) x shard(8)
#   arch  = idx / 16      (0=moe, 1=dense)
#   prec  = (idx / 8) % 2 (0=bf16, 1=fp32)
#   shard = idx % 8
source /home/kzhao2/gap_measurement/slurm/env.sh

IDX=$SLURM_ARRAY_TASK_ID
SHARD=$((IDX % 8))
if [ $((IDX / 16)) -eq 0 ]; then ARCH=moe; else ARCH=dense; fi
if [ $(((IDX / 8) % 2)) -eq 0 ]; then PREC=bf16; else PREC=fp32; fi

python src/recompute_train.py --arch $ARCH --shard $SHARD --precision $PREC
