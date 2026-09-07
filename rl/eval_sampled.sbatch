#!/bin/bash
#SBATCH --job-name=kt-suite
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/%u/nobackup/autodelete/areal_rl/logs/suite_%j.out
source $HOME/gap_measurement/rl/env.sh
cd $HOME/gap_measurement
source .venv/bin/activate
OPENSSL_CONF=/dev/null python rl/eval_sampled.py "$EVAL_PATH" "$EVAL_TAG" 4
