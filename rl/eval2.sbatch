#!/bin/bash
#SBATCH --job-name=kt-eval2
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/kzhao2/nobackup/autodelete/areal_rl/logs/eval2_%a.out
#SBATCH --array=0-3

source /home/kzhao2/gap_measurement/rl/env.sh
CK=/home/kzhao2/nobackup/autodelete/areal_rl/experiments/checkpoints/kzhao2/kt-lgrid
TAGS=(sigmatis_v2_inf23 sigmatis_v2_inf16 sigmatis_v2_inf31 sigmatis_v2_1023)
A=$SLURM_ARRAY_TASK_ID
P="$CK/lg${A}-e3/default/epoch2epochstep28globalstep86"
cd /home/kzhao2/gap_measurement
source .venv/bin/activate
OPENSSL_CONF=/dev/null python rl/eval_gsm8k.py "$P" "${TAGS[$A]}"
