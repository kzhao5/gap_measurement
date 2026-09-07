#!/bin/bash
#SBATCH --job-name=kt-eval3
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/kzhao2/nobackup/autodelete/areal_rl/logs/eval3_%j.out

source /home/kzhao2/gap_measurement/rl/env.sh
cd /home/kzhao2/gap_measurement
source .venv/bin/activate
OPENSSL_CONF=/dev/null python rl/eval_gsm8k.py "/home/kzhao2/nobackup/autodelete/areal_rl/experiments/checkpoints/kzhao2/kt-recipe/nocorr-moe-e3/default/epoch2epochstep28globalstep86" nocorr
