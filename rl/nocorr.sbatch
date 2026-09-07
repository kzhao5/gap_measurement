#!/bin/bash
#SBATCH --job-name=kt-nocorr
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1500G
#SBATCH --time=08:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/%u/nobackup/autodelete/areal_rl/logs/nocorr_%j.out

source $HOME/gap_measurement/rl/env.sh
cd "$AREAL"
source .venv/bin/activate
python -c "import torch, transformers, datasets, sglang; \
import areal.trainer.rl_trainer" || true

# TRUE no-correction baseline (literature GRPO): decoupled loss OFF,
# ratios ignored entirely; rejection_sampling inert without decoupled.
export KT_DUMP_DIR="$RLROOT/ktdump/e3_nocorr"
export KT_DUMP_EVERY=10

python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_icepop.yaml \
    experiment_name=kt-recipe \
    trial_name=nocorr-moe-e3 \
    seed=1 \
    scheduler.type=local \
    cluster.fileroot="$RLROOT/experiments" \
    cluster.name_resolve.nfs_record_root="$RLROOT/name_resolve" \
    actor.path=Qwen/Qwen1.5-MoE-A2.7B-Chat \
    actor.use_decoupled_loss=false \
    actor.optimizer.lr=3e-6 \
    actor.optimizer.weight_decay=0.01 \
    ++actor.eps_clip=0.2 \
    ++actor.eps_clip_higher=0.28 \
    actor.mb_spec.max_tokens_per_mb=4096 \
    ref.mb_spec.max_tokens_per_mb=4096 \
    ++train_dataset.setup_timeout=900 \
    ++valid_dataset.setup_timeout=900 \
    ++rollout.setup_timeout=900 \
    total_train_epochs=3
