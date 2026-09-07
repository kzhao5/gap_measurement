#!/bin/bash
#SBATCH --job-name=kt-eval
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/%u/nobackup/autodelete/areal_rl/logs/eval_%a.out
#SBATCH --array=0-5

source $HOME/gap_measurement/rl/env.sh
CK=$HOME/nobackup/autodelete/areal_rl/experiments/checkpoints/$USER/kt-recipe
TAGS=(base fullis tis icepop kpop sigmatis)
PATHS=(
  "Qwen/Qwen1.5-MoE-A2.7B-Chat"
  "$CK/fullis-moe-e3/default/epoch2epochstep28globalstep86"
  "$CK/tis-moe-e3/default/epoch2epochstep28globalstep86"
  "$CK/icepop-moe-e3/default/epoch2epochstep28globalstep86"
  "$CK/kpop-moe-e3/default/epoch2epochstep28globalstep86"
  "$CK/sigmatis-moe-e3b/default/epoch1epochstep28globalstep57"
)
A=$SLURM_ARRAY_TASK_ID
cd $HOME/gap_measurement
source .venv/bin/activate
OPENSSL_CONF=/dev/null python rl/eval_gsm8k.py "${PATHS[$A]}" "${TAGS[$A]}"
