#!/bin/bash
#SBATCH --job-name=kt-v3
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1500G
#SBATCH --time=08:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/kzhao2/nobackup/autodelete/areal_rl/logs/v3_%a_%j.out
#SBATCH --array=2

source /home/kzhao2/gap_measurement/rl/env.sh
cd "$AREAL"
source .venv/bin/activate
python -c "import torch, transformers, datasets, sglang; \
import areal.trainer.rl_trainer" || true

# Controlled single-variable arms vs champion (inf,2.3)+floor:
#   0 affine : upper = 2.3*(1-p) + 0.12, no floor, lower inf  (band SHAPE only)
#   1 lowb   : upper identical to champion, lower = const w>=0.001 (LOWER only)
# lower bound w >= 0.1 (log K >= -2.303), upper identical to champion
ARM=lowb01
export KT_SIGMA_TIS=1 KT_SIGMA_C1=1.0 KT_SIGMA_B=2.3
export KT_SIGMA_A=1e9 KT_SIGMA_FLOOR=5e-3 KT_SIGMA_ETA=0 KT_SIGMA_LOWER_ABS=2.303
export KT_DUMP_DIR="$RLROOT/ktdump/v3_$ARM"
export KT_DUMP_EVERY=10

python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_icepop.yaml \
    experiment_name=kt-v3 \
    trial_name=$ARM-e3 \
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
    ++rollout.setup_timeout=900 \
    total_train_epochs=3
