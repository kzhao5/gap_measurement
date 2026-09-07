#!/bin/bash
#SBATCH --job-name=kt-model
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/home/kzhao2/nobackup/autodelete/gap_measurement/logs/kt_%a.out
#SBATCH --array=0-1

source /home/kzhao2/gap_measurement/slurm/env.sh
cd /home/kzhao2/gap_measurement
ARCH=$([ "$SLURM_ARRAY_TASK_ID" -eq 0 ] && echo moe || echo dense)
python analysis/kt_model_selection.py "$ARCH" --split-both
