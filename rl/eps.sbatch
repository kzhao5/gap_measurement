#!/bin/bash
#SBATCH --job-name=kt-eps
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=700G
#SBATCH --time=08:00:00
#SBATCH --exclude=dw-2-4
#SBATCH --output=/home/kzhao2/nobackup/autodelete/areal_rl/logs/eps_%j.out

# Multi-seed significance: METHOD x SEED via --export, exact e3-recipe flags.
source /home/kzhao2/gap_measurement/rl/env.sh
cd "$AREAL"
source .venv/bin/activate
python -c "import torch, transformers, datasets, sglang; import areal.trainer.rl_trainer" || true

OVR=""
case $METHOD in
  nocorr) OVR="actor.use_decoupled_loss=false" ;;
  fullis) OVR="actor.rejection_sampling.action=clamp actor.rejection_sampling.lower=1e-6 actor.rejection_sampling.upper=1e6" ;;
  tis)    OVR="actor.rejection_sampling.action=clamp actor.rejection_sampling.lower=1e-6 actor.rejection_sampling.upper=2.0" ;;
  icepop) OVR="" ;;
  kpop)   OVR="actor.rejection_sampling.metric=binary_kl actor.rejection_sampling.upper=2.0 ~actor.rejection_sampling.lower" ;;
  ours)   OVR="actor.rejection_sampling.action=clamp actor.rejection_sampling.lower=1e-6 actor.rejection_sampling.upper=1e6"
          export KT_SIGMA_TIS=1 KT_SIGMA_C1=1.0 KT_SIGMA_B=2.3 KT_SIGMA_A=1e9 KT_SIGMA_FLOOR=${EPS0} ;;
esac
export KT_DUMP_DIR="$RLROOT/ktdump/eps_${EPS0}_s${SEED}"
export KT_DUMP_EVERY=20

python examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_icepop.yaml \
    experiment_name=kt-eps \
    trial_name=eps${EPS0}-s${SEED} \
    seed=$SEED \
    scheduler.type=local \
    cluster.fileroot="$RLROOT/experiments" \
    cluster.name_resolve.nfs_record_root="$RLROOT/name_resolve" \
    actor.path=Qwen/Qwen1.5-MoE-A2.7B-Chat \
    actor.optimizer.lr=3e-6 \
    actor.optimizer.weight_decay=0.01 \
    ++actor.eps_clip=0.2 \
    ++actor.eps_clip_higher=0.28 \
    actor.mb_spec.max_tokens_per_mb=4096 \
    ref.mb_spec.max_tokens_per_mb=4096 \
    ++train_dataset.setup_timeout=900 \
    ++valid_dataset.setup_timeout=900 \
    ++rollout.setup_timeout=900 \
    total_train_epochs=3 \
    $OVR

CK=$RLROOT/experiments/checkpoints/kzhao2/kt-eps/eps${EPS0}-s${SEED}/default
E=$(ls "$CK" 2>/dev/null | tail -1)
[ -n "$E" ] && sbatch --export=ALL,EVAL_PATH="$CK/$E",EVAL_TAG="eps_${EPS0}_s${SEED}" /home/kzhao2/gap_measurement/rl/eval_tp.sbatch
