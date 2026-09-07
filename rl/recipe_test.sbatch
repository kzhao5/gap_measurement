#!/bin/bash
#SBATCH --job-name=kt-recipe
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1500G
#SBATCH --time=04:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/kzhao2/nobackup/autodelete/areal_rl/logs/recipe_%a_%j.out
#SBATCH --array=0-1

source /home/kzhao2/gap_measurement/rl/env.sh
cd "$AREAL"
source .venv/bin/activate
python -c "import torch, transformers, datasets, sglang; \
import areal.trainer.rl_trainer" || true

# Recipe test on the official-MoE scalars (lr 3e-6, wd 0.01, clip .2/.28):
#   0 fullis   : never-binding clamp (fragility probe)
#   1 sigmatis : anisotropic clip, center (lam-, lam+) = (1.0, 2.3)
A=$SLURM_ARRAY_TASK_ID
ARMS=(fullis sigmatis)
ARM=${ARMS[$A]}
if [ "$A" -eq 1 ]; then
  export KT_SIGMA_TIS=1 KT_SIGMA_C1=1.0 KT_SIGMA_A=1.0 KT_SIGMA_B=2.3
fi
export KT_DUMP_DIR="$RLROOT/ktdump/recipe_$ARM"
export KT_DUMP_EVERY=10

python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_icepop.yaml \
    experiment_name=kt-recipe \
    trial_name=$ARM-moe-r1 \
    seed=1 \
    scheduler.type=local \
    cluster.fileroot="$RLROOT/experiments" \
    cluster.name_resolve.nfs_record_root="$RLROOT/name_resolve" \
    actor.path=Qwen/Qwen1.5-MoE-A2.7B-Chat \
    actor.optimizer.lr=3e-6 \
    actor.optimizer.weight_decay=0.01 \
    ++actor.eps_clip=0.2 \
    ++actor.eps_clip_higher=0.28 \
    actor.rejection_sampling.action=clamp \
    actor.rejection_sampling.lower=1e-6 \
    actor.rejection_sampling.upper=1e6 \
    actor.mb_spec.max_tokens_per_mb=4096 \
    ref.mb_spec.max_tokens_per_mb=4096 \
    ++train_dataset.setup_timeout=900 \
    ++valid_dataset.setup_timeout=900 \
    total_train_epochs=1
