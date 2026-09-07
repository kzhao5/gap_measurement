#!/bin/bash
#SBATCH --job-name=kt-sgeval
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/%u/nobackup/autodelete/areal_rl/logs/sgeval_%j.out
source $HOME/gap_measurement/rl/env.sh
cd $HOME/AReaL && source .venv/bin/activate
OPENSSL_CONF=/dev/null python $HOME/gap_measurement/rl/eval_sampled_sglang.py "$EVAL_PATH" "$EVAL_TAG"
