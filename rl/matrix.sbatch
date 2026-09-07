#!/bin/bash
#SBATCH --job-name=kt-matrix
#SBATCH --partition=dw
#SBATCH --qos=dw87
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1500G
#SBATCH --time=10:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/%u/nobackup/autodelete/areal_rl/logs/matrix_%a_%j.out
#SBATCH --array=0-4

source $HOME/gap_measurement/rl/env.sh
cd "$AREAL"
source .venv/bin/activate
python -c "import torch, transformers, datasets, sglang; \
import areal.trainer.rl_trainer" || true

# Arms (identical pipeline; only the modification M differs):
#   0 full-IS   : clamp with never-binding bounds (no-modification baseline)
#   1 TIS       : clamp, min(k, 2.0)
#   2 IcePop    : official mask ratio [0.5, 5.0]  (yaml as-is)
#   3 KPop      : official mask binary_kl upper=2.0
#   4 sigma-TIS : scaled clamp via KT_SIGMA_TIS=1 (c1=0.195, floor 3e-7, a=20 b=12)
A=$SLURM_ARRAY_TASK_ID
ARMS=(fullis tis icepop kpop sigmatis)
ARM=${ARMS[$A]}
OVR=""
case $A in
  0) OVR="actor.rejection_sampling.action=clamp actor.rejection_sampling.lower=1e-6 actor.rejection_sampling.upper=1e6" ;;
  1) OVR="actor.rejection_sampling.action=clamp actor.rejection_sampling.lower=1e-6 actor.rejection_sampling.upper=2.0" ;;
  2) OVR="" ;;
  3) OVR="actor.rejection_sampling.metric=binary_kl actor.rejection_sampling.upper=2.0 ~actor.rejection_sampling.lower" ;;
  4) OVR="actor.rejection_sampling.action=clamp actor.rejection_sampling.lower=1e-6 actor.rejection_sampling.upper=1e6"
     export KT_SIGMA_TIS=1 ;;
esac

export KT_DUMP_DIR="$RLROOT/ktdump/$ARM"
export KT_DUMP_EVERY=20

python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_icepop.yaml \
    experiment_name=kt-matrix \
    trial_name=$ARM-s1 \
    seed=1 \
    scheduler.type=local \
    cluster.fileroot="$RLROOT/experiments" \
    cluster.name_resolve.nfs_record_root="$RLROOT/name_resolve" \
    actor.path=Qwen/Qwen1.5-MoE-A2.7B-Chat \
    +actor.attn_impl=sdpa \
    +sglang.disable_cuda_graph=true \
    actor.mb_spec.max_tokens_per_mb=4096 \
    ref.mb_spec.max_tokens_per_mb=4096 \
    total_train_epochs=3 \
    $OVR
