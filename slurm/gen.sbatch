#!/bin/bash
#SBATCH --job-name=gap-gen
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-15
#SBATCH --output=/home/kzhao2/nobackup/autodelete/gap_measurement/logs/gen_%a.out

# Array layout: 0-7 = moe shards 0-7, 8-15 = dense shards 0-7
source /home/kzhao2/gap_measurement/slurm/env.sh

IDX=$SLURM_ARRAY_TASK_ID
if [ $IDX -lt 8 ]; then ARCH=moe; SHARD=$IDX; else ARCH=dense; SHARD=$((IDX-8)); fi

python src/gen_vllm.py --arch $ARCH --shard $SHARD --num-shards 8
