#!/bin/bash
#SBATCH --job-name=kt-diag
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --exclude=dw-2-4,dw-1-3,dw-2-1
#SBATCH --output=/home/%u/nobackup/autodelete/areal_rl/logs/diag_%j.out
source $HOME/gap_measurement/rl/env.sh
cd $HOME/AReaL && source .venv/bin/activate
OPENSSL_CONF=/dev/null python $HOME/gap_measurement/rl/diag_areal_path.py "$MODEL" "$TAG"
