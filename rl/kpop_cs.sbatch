#!/bin/bash
#SBATCH --job-name=kt-kpop-cs
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=08:00:00
#SBATCH 
#SBATCH --output=/home/%u/nobackup/autodelete/areal_rl/logs/rec3_%a_%j.out
#SBATCH --array=2

source $HOME/gap_measurement/rl/env.sh
cd "$AREAL"
source .venv/bin/activate
python -c "import torch, transformers, datasets, sglang; \
import areal.trainer.rl_trainer" || true

# Recipe test on the official-MoE scalars (lr 3e-6, wd 0.01, clip .2/.28):
#   0 fullis   : never-binding clamp (fragility probe)
#   1 sigmatis : anisotropic clip, center (lam-, lam+) = (1.0, 2.3)
A=$SLURM_ARRAY_TASK_ID
ARMS=(tis icepop kpop)
ARM=${ARMS[$A]}
case $A in
  0) RS="actor.rejection_sampling.action=clamp actor.rejection_sampling.lower=1e-6 actor.rejection_sampling.upper=2.0" ;;
  1) RS="" ;;
  2) RS="actor.rejection_sampling.metric=binary_kl actor.rejection_sampling.upper=2.0 ~actor.rejection_sampling.lower" ;;
esac
export KT_DUMP_DIR="$RLROOT/ktdump/recipe_$ARM"
export KT_DUMP_EVERY=10

python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_icepop.yaml \
    experiment_name=kt-recipe \
    trial_name=$ARM-moe-r1d \
    seed=1 \
    scheduler.type=local \
    cluster.fileroot="$RLROOT/experiments" \
    cluster.name_resolve.nfs_record_root="$RLROOT/name_resolve" \
    actor.path=Qwen/Qwen1.5-MoE-A2.7B-Chat \
    actor.optimizer.lr=3e-6 \
    actor.optimizer.weight_decay=0.01 \
    ++actor.eps_clip=0.2 \
    ++actor.eps_clip_higher=0.28 \
    $RS \
    actor.mb_spec.max_tokens_per_mb=4096 \
    ref.mb_spec.max_tokens_per_mb=4096 \
    ++rollout.setup_timeout=900 \
    ++train_dataset.setup_timeout=900 \
    ++valid_dataset.setup_timeout=900 \
    total_train_epochs=1
